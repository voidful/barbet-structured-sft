"""Megatron integration for Barbet long-context continued pretraining.

The position policy is deliberately stage- and lane-aware:

* E1--E3 keep the imported checkpoint's native RoPE bit-for-bit.
* E4 long examples scale only global-attention layers.
* E4 short replay examples and every local-attention layer keep native RoPE.

The released UltraLong checkpoint provides one useful frequency-selective
scaling reference, but Barbet's local-attention/Mamba hybrid is materially
different.  Applying the transform to every attention layer would therefore
be an untested architecture change, not a faithful reproduction.

This module keeps the validation and routing math importable without Megatron.
Pinned runtime classes are imported only while constructing the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch


NATIVE_POSITION_POLICY = "native"
SCALED_ALL_POSITION_POLICY = "scaled_all"
E4_GLOBAL_SCALED_LOCAL_NATIVE_POLICY = "e4_global_scaled_local_native"
LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY = (
    "longrope2_per_dimension_global_only"
)
POSITION_POLICIES = (
    NATIVE_POSITION_POLICY,
    SCALED_ALL_POSITION_POLICY,
    E4_GLOBAL_SCALED_LOCAL_NATIVE_POLICY,
    LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY,
)
E4_AUTHORIZED_POSITION_POLICIES = (
    E4_GLOBAL_SCALED_LOCAL_NATIVE_POLICY,
    SCALED_ALL_POSITION_POLICY,
    LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY,
    NATIVE_POSITION_POLICY,
)
E4_GLOBAL_ONLY_POSITION_POLICIES = (
    E4_GLOBAL_SCALED_LOCAL_NATIVE_POLICY,
    LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY,
)

POSITION_POLICY_PAYLOAD_SCHEMA_VERSION = (
    "barbet-position-policy-runtime-payload-v1"
)
BARBET_ROPE_HEAD_DIM = 128
BARBET_ROPE_FREQUENCY_DIMENSIONS = BARBET_ROPE_HEAD_DIM // 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_POLICY_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "runtime_policy",
        "type",
        "attention_scope",
        "factor",
        "original_max_position_embeddings",
        "target_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
        "short_context_policy",
        "head_dim",
        "scale_factors",
        "scale_factors_sha256",
        "canonical_sha256",
    }
)

_FACTOR4 = 4.0
_ORIGINAL_CONTEXT_LENGTH = 262_144
_TARGET_CONTEXT_LENGTH = 1_048_576
_LOW_FREQUENCY_FACTOR = 1.0
_HIGH_FREQUENCY_FACTOR = 4.0

E4_PHYSICAL_TOKENS = 1_048_576
E4_SHORT_MAX_SEQUENCE = 32_768
E4_CP_PADDING_MULTIPLE = 8
_DIAGNOSTIC_E4_RUN_ROLE = "diagnostic_e4_training"

EXPECTED_GLOBAL_ATTENTION_LAYERS = 7
EXPECTED_LOCAL_ATTENTION_LAYERS = 14
EXPECTED_MAMBA_LAYERS = 7
EXPECTED_MLP_LAYERS = 28
MAMBA_FP32_PARAMETER_NAMES = ("A_log", "D")


class PackedLane(str, Enum):
    """The only two legal E4 packed-sequence layouts."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class PositionRouteCounts:
    """Auditable counts for the exact Barbet hybrid stack."""

    global_attention: int
    local_attention: int
    mamba: int
    mlp: int

    def validate_barbet_1b(self) -> None:
        expected = PositionRouteCounts(
            global_attention=EXPECTED_GLOBAL_ATTENTION_LAYERS,
            local_attention=EXPECTED_LOCAL_ATTENTION_LAYERS,
            mamba=EXPECTED_MAMBA_LAYERS,
            mlp=EXPECTED_MLP_LAYERS,
        )
        if self != expected:
            raise ValueError(
                "unexpected Barbet hybrid layer routing: "
                f"observed={self!r}, expected={expected!r}"
            )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize one policy artifact in the campaign's canonical JSON form."""

    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("position-policy payload is not canonical JSON") from error


def _normalized_longrope2_scale_vector(value: Any) -> tuple[float, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError("LongRoPE2 scale_vector must be a JSON-style array")
    if len(value) != BARBET_ROPE_FREQUENCY_DIMENSIONS:
        raise ValueError(
            "LongRoPE2 scale_vector must contain exactly "
            f"{BARBET_ROPE_FREQUENCY_DIMENSIONS} values for head_dim="
            f"{BARBET_ROPE_HEAD_DIM}"
        )
    normalized: list[float] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
        ):
            raise ValueError(
                "LongRoPE2 scale_vector values must be finite and positive; "
                f"index {index} is invalid"
            )
        normalized.append(float(item))
    if any(
        current < previous
        for previous, current in zip(normalized, normalized[1:])
    ):
        raise ValueError(
            "LongRoPE2 scale_vector must be monotonic nondecreasing"
        )
    return tuple(normalized)


def canonical_longrope2_scale_vector_sha256(value: Any) -> str:
    """Hash the exact 64-value Barbet LongRoPE2 vector canonically."""

    vector = _normalized_longrope2_scale_vector(value)
    return hashlib.sha256(
        json.dumps(
            list(vector),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def validate_longrope2_scale_vector(
    value: Any,
    supplied_sha256: Any,
) -> tuple[float, ...]:
    """Return a normalized vector only when its canonical digest matches."""

    vector = _normalized_longrope2_scale_vector(value)
    if (
        type(supplied_sha256) is not str
        or _SHA256_RE.fullmatch(supplied_sha256) is None
        or supplied_sha256 == "0" * 64
    ):
        raise ValueError(
            "LongRoPE2 scale_vector_sha256 must be a nonzero lowercase SHA-256"
        )
    expected = canonical_longrope2_scale_vector_sha256(vector)
    if not secrets.compare_digest(supplied_sha256, expected):
        raise ValueError("LongRoPE2 scale_vector canonical digest mismatch")
    return vector


def per_dimension_scaled_inv_freq(
    inv_freq: torch.Tensor,
    *,
    scale_vector: Any,
    scale_vector_sha256: Any,
) -> torch.Tensor:
    """Apply the LongRoPE2-style wavelength multiplier per RoPE dimension.

    Each scale factor multiplies that dimension's wavelength, equivalently
    dividing its inverse frequency.  Validation happens before tensor
    construction, so malformed metadata can never silently fall back to native
    RoPE.
    """

    if (
        not isinstance(inv_freq, torch.Tensor)
        or not torch.is_floating_point(inv_freq)
        or inv_freq.ndim != 1
        or inv_freq.numel() != BARBET_ROPE_FREQUENCY_DIMENSIONS
    ):
        raise ValueError(
            "LongRoPE2 inv_freq must be a one-dimensional floating tensor "
            f"with {BARBET_ROPE_FREQUENCY_DIMENSIONS} values"
        )
    if not bool(torch.isfinite(inv_freq).all()) or not bool((inv_freq > 0).all()):
        raise ValueError("inv_freq values must be finite and positive")
    vector = validate_longrope2_scale_vector(
        scale_vector,
        scale_vector_sha256,
    )
    scales = torch.tensor(
        vector,
        dtype=inv_freq.dtype,
        device=inv_freq.device,
    )
    result = inv_freq / scales
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError("LongRoPE2 transformed inv_freq is not finite and positive")
    return result


def build_position_policy_payload(
    *,
    position_policy: str,
    factor: Any = _FACTOR4,
    original_context_length: Any = _ORIGINAL_CONTEXT_LENGTH,
    low_frequency_factor: Any = _LOW_FREQUENCY_FACTOR,
    high_frequency_factor: Any = _HIGH_FREQUENCY_FACTOR,
    scale_vector: Any = None,
    scale_vector_sha256: Any = None,
) -> dict[str, Any]:
    """Build the exact self-digesting runtime payload for one policy."""

    if type(position_policy) is not str or position_policy not in POSITION_POLICIES:
        raise ValueError(f"unsupported Barbet position policy: {position_policy!r}")

    if position_policy == NATIVE_POSITION_POLICY:
        if scale_vector is not None or scale_vector_sha256 is not None:
            raise ValueError("native position policy forbids a scale vector")
        transform = "native"
        attention_scope = "all_attention"
        factor_value = 1.0
        original_value = _ORIGINAL_CONTEXT_LENGTH
        low_value = 1.0
        high_value = 1.0
        short_context_policy = "native_all_lengths"
        vector_value = None
        vector_digest = None
    else:
        numeric = (
            ("factor", factor, _FACTOR4),
            (
                "low_frequency_factor",
                low_frequency_factor,
                _LOW_FREQUENCY_FACTOR,
            ),
            (
                "high_frequency_factor",
                high_frequency_factor,
                _HIGH_FREQUENCY_FACTOR,
            ),
        )
        for label, observed, expected in numeric:
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    f"{position_policy} requires {label}={expected}"
                )
        if (
            type(original_context_length) is not int
            or original_context_length != _ORIGINAL_CONTEXT_LENGTH
        ):
            raise ValueError(
                f"{position_policy} requires original_context_length="
                f"{_ORIGINAL_CONTEXT_LENGTH}"
            )
        factor_value = float(factor)
        original_value = original_context_length
        low_value = float(low_frequency_factor)
        high_value = float(high_frequency_factor)
        attention_scope = (
            "all_attention"
            if position_policy == SCALED_ALL_POSITION_POLICY
            else "global_only"
        )
        short_context_policy = (
            "native_at_or_below_262144_scaled_above_262144"
        )
        if position_policy == LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY:
            vector = validate_longrope2_scale_vector(
                scale_vector,
                scale_vector_sha256,
            )
            transform = "longrope2_per_dimension"
            vector_value = list(vector)
            vector_digest = str(scale_vector_sha256)
        else:
            if scale_vector is not None or scale_vector_sha256 is not None:
                raise ValueError(
                    f"{position_policy} forbids a LongRoPE2 scale vector"
                )
            transform = "llama3_frequency_selective"
            vector_value = None
            vector_digest = None

    unsigned = {
        "schema_version": POSITION_POLICY_PAYLOAD_SCHEMA_VERSION,
        "runtime_policy": position_policy,
        "type": transform,
        "attention_scope": attention_scope,
        "factor": factor_value,
        "original_max_position_embeddings": original_value,
        "target_max_position_embeddings": _TARGET_CONTEXT_LENGTH,
        "low_freq_factor": low_value,
        "high_freq_factor": high_value,
        "short_context_policy": short_context_policy,
        "head_dim": BARBET_ROPE_HEAD_DIM,
        "scale_factors": vector_value,
        "scale_factors_sha256": vector_digest,
    }
    return {
        **unsigned,
        "canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(unsigned)
        ).hexdigest(),
    }


def validate_position_policy_payload(
    value: Any,
    *,
    expected_policy: str | None = None,
    require_e4_authorized: bool = False,
) -> dict[str, Any]:
    """Recompute an exact policy payload; never trust its selected label."""

    if type(value) is not dict or frozenset(value) != _POSITION_POLICY_PAYLOAD_KEYS:
        raise ValueError("position-policy runtime payload keys changed")
    policy = value.get("runtime_policy")
    if type(policy) is not str or policy not in POSITION_POLICIES:
        raise ValueError("position-policy runtime payload has an unknown policy")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError("position-policy runtime payload policy mismatch")
    if require_e4_authorized and policy not in E4_AUTHORIZED_POSITION_POLICIES:
        raise ValueError(
            "E4 authorization rejects an unknown position policy"
        )
    if value.get("schema_version") != POSITION_POLICY_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("position-policy runtime payload schema changed")
    if value.get("head_dim") != BARBET_ROPE_HEAD_DIM or type(
        value.get("head_dim")
    ) is not int:
        raise ValueError("position-policy runtime payload head_dim changed")
    rebuilt = build_position_policy_payload(
        position_policy=policy,
        factor=value.get("factor"),
        original_context_length=value.get(
            "original_max_position_embeddings"
        ),
        low_frequency_factor=value.get("low_freq_factor"),
        high_frequency_factor=value.get("high_freq_factor"),
        scale_vector=value.get("scale_factors"),
        scale_vector_sha256=value.get("scale_factors_sha256"),
    )
    digest = value.get("canonical_sha256")
    if (
        type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
        or digest == "0" * 64
        or not secrets.compare_digest(digest, rebuilt["canonical_sha256"])
        or value != rebuilt
    ):
        raise ValueError("position-policy runtime payload canonical digest mismatch")
    return rebuilt


def position_policy_payload_from_args(
    args: Any,
    *,
    require_e4_authorized: bool = False,
) -> dict[str, Any]:
    """Reconstruct policy evidence from the narrow Megatron CLI surface."""

    payload = build_position_policy_payload(
        position_policy=getattr(args, "barbet_position_policy", None),
        factor=getattr(args, "barbet_rope_factor", None),
        original_context_length=getattr(
            args,
            "barbet_rope_original_context_length",
            None,
        ),
        low_frequency_factor=getattr(
            args,
            "barbet_rope_low_frequency_factor",
            None,
        ),
        high_frequency_factor=getattr(
            args,
            "barbet_rope_high_frequency_factor",
            None,
        ),
        scale_vector=getattr(args, "barbet_rope_scale_vector", None),
        scale_vector_sha256=getattr(
            args,
            "barbet_rope_scale_vector_sha256",
            None,
        ),
    )
    return validate_position_policy_payload(
        payload,
        expected_policy=getattr(args, "barbet_position_policy", None),
        require_e4_authorized=require_e4_authorized,
    )


def _require_int32_vector(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.int32 or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if value.numel() < 2:
        raise ValueError(f"{name} must contain at least two cumulative offsets")
    return value


def _tensors_equal(left: torch.Tensor, right: Any) -> bool:
    return (
        isinstance(right, torch.Tensor)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.shape == right.shape
        and bool(torch.equal(left, right))
    )


def classify_e4_packed_lane(packed_seq_params: Any) -> PackedLane:
    """Validate and classify an E4 THD layout, failing closed.

    Transformer Engine consumes both cumulative-length views: ``cu_seqlens``
    counts real tokens while ``cu_seqlens_padded`` locates those subsequences in
    the fixed physical tensor.  Mamba derives ``seq_idx`` from the padded view.
    Long batches have equal values in the two arrays; short batches may not.
    """

    if packed_seq_params is None:
        raise ValueError("E4 position routing requires PackedSeqParams")
    if getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise ValueError("E4 position routing requires qkv_format='thd'")

    padded_cu = _require_int32_vector(
        getattr(packed_seq_params, "cu_seqlens_q_padded", None),
        "cu_seqlens_q_padded",
    )
    if not _tensors_equal(
        padded_cu,
        getattr(packed_seq_params, "cu_seqlens_kv_padded", None),
    ):
        raise ValueError(
            "cu_seqlens_kv_padded must exactly equal "
            "cu_seqlens_q_padded"
        )
    real_cu = _require_int32_vector(
        getattr(packed_seq_params, "cu_seqlens_q", None),
        "cu_seqlens_q",
    )
    if not _tensors_equal(
        real_cu,
        getattr(packed_seq_params, "cu_seqlens_kv", None),
    ):
        raise ValueError("cu_seqlens_kv must exactly equal cu_seqlens_q")
    if real_cu.shape != padded_cu.shape:
        raise ValueError(
            "real and padded cumulative-length arrays must have equal shape"
        )
    if real_cu.data_ptr() == padded_cu.data_ptr():
        raise ValueError(
            "real and padded cumulative-length arrays must remain distinct"
        )

    if int(getattr(packed_seq_params, "total_tokens", -1)) != E4_PHYSICAL_TOKENS:
        raise ValueError(
            f"E4 total_tokens must equal {E4_PHYSICAL_TOKENS}"
        )
    max_q = getattr(packed_seq_params, "max_seqlen_q", None)
    max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
    if not isinstance(max_q, int) or isinstance(max_q, bool) or max_q <= 0:
        raise ValueError("max_seqlen_q must be a positive integer")
    if max_kv != max_q:
        raise ValueError("max_seqlen_kv must exactly equal max_seqlen_q")

    if (
        int(padded_cu[0].item()) != 0
        or int(padded_cu[-1].item()) != E4_PHYSICAL_TOKENS
    ):
        raise ValueError(
            "E4 padded cumulative lengths must start at zero and end at "
            f"{E4_PHYSICAL_TOKENS}"
        )
    padded_deltas = padded_cu[1:] - padded_cu[:-1]
    if not bool(torch.all(padded_deltas > 0).item()):
        raise ValueError("E4 padded cumulative lengths must be strictly increasing")
    observed_max = int(torch.max(padded_deltas).item())
    if observed_max != max_q:
        raise ValueError(
            "max_seqlen_q must equal the largest padded subsequence length"
        )

    if padded_cu.numel() == 2:
        if (
            max_q != E4_PHYSICAL_TOKENS
            or not _tensors_equal(real_cu, padded_cu)
        ):
            raise ValueError("the E4 long lane must be one exact 1M subsequence")
        return PackedLane.LONG

    if int(real_cu[0].item()) != 0:
        raise ValueError("E4 real cumulative lengths must start at zero")
    real_deltas = real_cu[1:] - real_cu[:-1]
    if (
        not bool(torch.all(real_deltas > 0).item())
        or not bool(torch.all(real_deltas <= padded_deltas).item())
    ):
        raise ValueError(
            "E4 real subsequences must be positive and fit their padded slots"
        )
    if max_q > E4_SHORT_MAX_SEQUENCE:
        raise ValueError(
            f"E4 short subsequences must be at most {E4_SHORT_MAX_SEQUENCE}"
        )
    if not bool(torch.all(padded_deltas <= E4_SHORT_MAX_SEQUENCE).item()):
        raise ValueError(
            f"E4 short subsequences must be at most {E4_SHORT_MAX_SEQUENCE}"
        )
    if not bool(
        torch.all(
            torch.remainder(
                padded_deltas,
                E4_CP_PADDING_MULTIPLE,
            )
            == 0
        ).item()
    ):
        raise ValueError(
            "E4 short padded subsequence lengths must be divisible by "
            f"{E4_CP_PADDING_MULTIPLE}"
        )
    return PackedLane.SHORT


def select_e4_rotary_pos_emb(
    *,
    position_policy: str,
    is_local_attention: bool,
    packed_seq_params: Any,
    scaled_rotary_pos_emb: torch.Tensor,
    native_rotary: Callable[..., torch.Tensor],
    require_packed_lane: bool = True,
    allow_diagnostic_nonpacked_long_lane: bool = False,
    context_parallel_size: int = 1,
) -> tuple[torch.Tensor, PackedLane | None]:
    """Select the exact RoPE tensor for one E4 attention layer."""

    if position_policy not in E4_AUTHORIZED_POSITION_POLICIES:
        raise ValueError("E4 rotary routing requires an authorized position policy")
    if type(is_local_attention) is not bool:
        raise ValueError("is_local_attention must be boolean")
    if not isinstance(scaled_rotary_pos_emb, torch.Tensor):
        raise ValueError("rotary_pos_emb must be a tensor in E4")
    if type(require_packed_lane) is not bool:
        raise ValueError("require_packed_lane must be boolean")
    if type(allow_diagnostic_nonpacked_long_lane) is not bool:
        raise ValueError(
            "allow_diagnostic_nonpacked_long_lane must be boolean"
        )
    if allow_diagnostic_nonpacked_long_lane and not require_packed_lane:
        raise ValueError(
            "diagnostic non-packed LONG requires packed-lane routing"
        )
    if type(context_parallel_size) is not int or context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be a positive integer")
    if require_packed_lane:
        if packed_seq_params is None:
            if not allow_diagnostic_nonpacked_long_lane:
                raise ValueError(
                    "packed E4 routing requires PackedSeqParams; only the "
                    "diagnostic LONG/SBHD path may explicitly opt in"
                )
            if scaled_rotary_pos_emb.ndim < 1:
                raise ValueError("non-packed E4 long rotary tensor has no sequence dimension")
            native_max_seqlen = (
                int(scaled_rotary_pos_emb.shape[0]) * context_parallel_size
            )
            if native_max_seqlen != E4_PHYSICAL_TOKENS:
                raise ValueError(
                    "non-packed E4 routing requires one exact 1M long sequence"
                )
            lane = PackedLane.LONG
            packed_seq = False
        else:
            lane = classify_e4_packed_lane(packed_seq_params)
            native_max_seqlen = int(packed_seq_params.max_seqlen_q)
            packed_seq = True
    else:
        if packed_seq_params is not None:
            raise ValueError(
                "non-packed position routing forbids PackedSeqParams"
            )
        lane = None
        if scaled_rotary_pos_emb.ndim < 1:
            raise ValueError("non-packed rotary tensor has no sequence dimension")
        # Megatron slices a non-packed RoPE tensor to the current CP rank
        # inside RotaryEmbedding.forward.  Reconstruct the global length before
        # asking the independent native module to perform that same one-time
        # CP selection.  Using the already-local length would shard twice.
        native_max_seqlen = (
            int(scaled_rotary_pos_emb.shape[0]) * context_parallel_size
        )
        packed_seq = False
    if native_max_seqlen <= 0:
        raise ValueError("rotary sequence length must be positive")
    if (
        require_packed_lane
        and packed_seq
        and (
            scaled_rotary_pos_emb.ndim < 1
            or scaled_rotary_pos_emb.shape[0] != native_max_seqlen
        )
    ):
        raise ValueError(
            "scaled rotary tensor length does not match packed max_seqlen"
        )

    short_context = (
        lane is PackedLane.SHORT
        if require_packed_lane
        else native_max_seqlen <= _ORIGINAL_CONTEXT_LENGTH
    )
    uses_native = (
        position_policy == NATIVE_POSITION_POLICY
        or short_context
        or (
            position_policy in E4_GLOBAL_ONLY_POSITION_POLICIES
            and is_local_attention
        )
    )
    if uses_native:
        native = native_rotary(native_max_seqlen, packed_seq=packed_seq)
        if not isinstance(native, torch.Tensor) or native.shape != scaled_rotary_pos_emb.shape:
            raise ValueError(
                "native and scaled rotary tensors must have identical shapes"
            )
        return native, lane
    return scaled_rotary_pos_emb, lane


def make_e4_position_pre_hook(
    *,
    position_policy: str,
    is_local_attention: bool,
    native_rotary: Callable[..., torch.Tensor],
    require_packed_lane: bool = True,
    allow_diagnostic_nonpacked_long_lane: bool = False,
    context_parallel_size: int = 1,
) -> Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]:
    """Create a kwargs-aware TransformerLayer hook for eager and recompute."""

    def hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if "rotary_pos_emb" not in kwargs:
            raise ValueError("E4 TransformerLayer call omitted rotary_pos_emb")
        if require_packed_lane and "packed_seq_params" not in kwargs:
            raise ValueError("E4 TransformerLayer call omitted packed_seq_params")
        selected, lane = select_e4_rotary_pos_emb(
            position_policy=position_policy,
            is_local_attention=is_local_attention,
            packed_seq_params=kwargs.get("packed_seq_params"),
            scaled_rotary_pos_emb=kwargs["rotary_pos_emb"],
            native_rotary=native_rotary,
            require_packed_lane=require_packed_lane,
            allow_diagnostic_nonpacked_long_lane=(
                allow_diagnostic_nonpacked_long_lane
            ),
            context_parallel_size=context_parallel_size,
        )
        routed = dict(kwargs)
        routed["rotary_pos_emb"] = selected
        # Runtime telemetry may read this non-checkpointed attribute.
        setattr(
            _module,
            "_barbet_last_packed_lane",
            lane.value
            if lane is not None
            else (
                "non_packed_native_window"
                if selected is not kwargs["rotary_pos_emb"]
                else "non_packed_scaled_window"
            ),
        )
        setattr(
            _module,
            "_barbet_last_position_route",
            "scaled" if selected is kwargs["rotary_pos_emb"] else "native",
        )
        setattr(_module, "_barbet_position_policy", position_policy)
        return args, routed

    return hook


def install_e4_position_router(
    model: torch.nn.Module,
    *,
    position_policy: str,
    native_rotary: torch.nn.Module,
    transformer_layer_type: type[torch.nn.Module],
    mamba_layer_type: type[torch.nn.Module],
    mlp_layer_type: type[torch.nn.Module],
    is_window_attention: Callable[[Any, Any, int], bool],
    require_packed_lane: bool = True,
    allow_diagnostic_nonpacked_long_lane: bool = False,
    context_parallel_size: int = 1,
) -> PositionRouteCounts:
    """Install exact global/local hooks without adding checkpoint state."""

    if position_policy not in E4_AUTHORIZED_POSITION_POLICIES:
        raise ValueError("E4 position router requires an authorized policy")
    if type(allow_diagnostic_nonpacked_long_lane) is not bool:
        raise ValueError(
            "allow_diagnostic_nonpacked_long_lane must be boolean"
        )
    if allow_diagnostic_nonpacked_long_lane and not require_packed_lane:
        raise ValueError(
            "diagnostic non-packed LONG requires packed-lane routing"
        )
    if type(context_parallel_size) is not int or context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be a positive integer")

    decoder = getattr(model, "decoder", None)
    layers: Sequence[torch.nn.Module] | None = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("HybridModel decoder did not expose its layer list")

    state_keys_before = tuple(model.state_dict().keys())
    # RotaryEmbedding has no parameters or persistent buffers.  Registering it
    # as a child module makes device/train-mode handling explicit while keeping
    # the DCP schema unchanged.
    model.add_module("barbet_native_rotary_pos_emb", native_rotary)
    if tuple(model.state_dict().keys()) != state_keys_before:
        raise ValueError("native rotary module unexpectedly changed state-dict keys")

    global_attention = 0
    local_attention = 0
    mamba = 0
    mlp = 0
    handles: list[Any] = []

    for layer in layers:
        # Pinned Megatron's MLPLayer subclasses TransformerLayer so it can
        # reuse the transformer-layer interface.  Classify that narrower type
        # first; otherwise every dense MLP is mistaken for global attention
        # and receives an invalid position-routing hook.
        if isinstance(layer, mlp_layer_type):
            mlp += 1
        elif isinstance(layer, transformer_layer_type):
            layer_number = getattr(layer, "layer_number", None)
            config = getattr(layer, "config", None)
            if (
                not isinstance(layer_number, int)
                or isinstance(layer_number, bool)
                or config is None
            ):
                raise ValueError("attention layer lacks an exact layer number/config")
            local = bool(
                is_window_attention(
                    getattr(config, "window_size", None),
                    getattr(config, "window_attn_skip_freq", None),
                    layer_number,
                )
            )
            if local:
                local_attention += 1
            else:
                global_attention += 1
            handles.append(
                layer.register_forward_pre_hook(
                    make_e4_position_pre_hook(
                        position_policy=position_policy,
                        is_local_attention=local,
                        native_rotary=native_rotary,
                        require_packed_lane=require_packed_lane,
                        allow_diagnostic_nonpacked_long_lane=(
                            allow_diagnostic_nonpacked_long_lane
                        ),
                        context_parallel_size=context_parallel_size,
                    ),
                    with_kwargs=True,
                )
            )
        elif isinstance(layer, mamba_layer_type):
            mamba += 1
        else:
            raise ValueError(
                "unexpected module in Barbet hybrid stack: "
                f"{type(layer).__module__}.{type(layer).__qualname__}"
            )

    counts = PositionRouteCounts(
        global_attention=global_attention,
        local_attention=local_attention,
        mamba=mamba,
        mlp=mlp,
    )
    counts.validate_barbet_1b()
    if len(handles) != EXPECTED_GLOBAL_ATTENTION_LAYERS + EXPECTED_LOCAL_ATTENTION_LAYERS:
        raise ValueError("unexpected number of E4 position-routing hooks")
    # Hook handles must remain alive for the model lifetime and are not state.
    model._barbet_position_hook_handles = tuple(handles)
    model._barbet_position_route_counts = counts
    model._barbet_position_router_policy = position_policy
    model._barbet_position_router_requires_packed_lane = require_packed_lane
    model._barbet_position_router_allows_diagnostic_nonpacked_long_lane = (
        allow_diagnostic_nonpacked_long_lane
    )
    model._barbet_position_router_context_parallel_size = context_parallel_size
    return counts


def install_mamba_fp32_parameter_guards(
    model: torch.nn.Module,
    *,
    mamba_layer_type: type[torch.nn.Module],
) -> int:
    """Preserve the canonical Mamba ``A_log``/``D`` checkpoint dtypes.

    Pinned Megatron initializes these two state-space parameters in fp32, but
    its outer ``Float16Module`` later calls ``model.bfloat16()`` recursively.
    Without an instance-local guard, a training save changes 14 canonical DCP
    tensors to bf16 (and silently changes the checkpoint schema).  The guard
    delegates every device/dtype transform first, then restores only these two
    parameters to fp32.  Optimizer construction happens later and therefore
    observes the intended mixed-precision parameter dtypes.
    """

    decoder = getattr(model, "decoder", None)
    layers: Sequence[torch.nn.Module] | None = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("HybridModel decoder did not expose its layer list")
    state_keys_before = tuple(model.state_dict().keys())
    guarded = 0

    for layer in layers:
        if not isinstance(layer, mamba_layer_type):
            continue
        mixer = getattr(layer, "mixer", None)
        if not isinstance(mixer, torch.nn.Module):
            raise ValueError("MambaLayer did not expose its mixer module")
        if getattr(mixer, "_barbet_fp32_parameter_guard_installed", False):
            raise ValueError("Mamba fp32 parameter guard was installed twice")
        for name in MAMBA_FP32_PARAMETER_NAMES:
            parameter = getattr(mixer, name, None)
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError(f"Mamba mixer lacks parameter {name}")
            if parameter.dtype != torch.float32:
                raise ValueError(
                    f"Mamba {name} must be fp32 before precision wrapping"
                )

        original_apply = mixer._apply

        def guarded_apply(
            self: torch.nn.Module,
            fn: Callable[[torch.Tensor], torch.Tensor],
            recurse: bool = True,
            *,
            _original_apply: Callable[..., torch.nn.Module] = original_apply,
        ) -> torch.nn.Module:
            result = _original_apply(fn, recurse=recurse)
            with torch.no_grad():
                for parameter_name in MAMBA_FP32_PARAMETER_NAMES:
                    parameter = getattr(self, parameter_name)
                    parameter.data = parameter.data.to(dtype=torch.float32)
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.to(
                            dtype=torch.float32
                        )
            return result

        mixer._apply = types.MethodType(guarded_apply, mixer)
        mixer._barbet_fp32_parameter_guard_installed = True
        guarded += 1

    if guarded != EXPECTED_MAMBA_LAYERS:
        raise ValueError(
            f"expected {EXPECTED_MAMBA_LAYERS} guarded Mamba mixers, got {guarded}"
        )
    if tuple(model.state_dict().keys()) != state_keys_before:
        raise ValueError("Mamba fp32 guards unexpectedly changed state-dict keys")
    model._barbet_mamba_fp32_guard_count = guarded
    return guarded


def frequency_selective_inv_freq(
    inv_freq: torch.Tensor,
    *,
    factor: float,
    original_context_length: int,
    low_frequency_factor: float = 1.0,
    high_frequency_factor: float = 4.0,
) -> torch.Tensor:
    """Apply the Llama-3/UltraLong frequency-selective RoPE transform.

    Frequencies with wavelengths shorter than
    ``original_context_length / high_frequency_factor`` are retained,
    frequencies with wavelengths longer than
    ``original_context_length / low_frequency_factor`` are divided by
    ``factor``, and the interval between them is smoothly interpolated.
    """

    if not torch.is_floating_point(inv_freq) or inv_freq.ndim != 1:
        raise ValueError("inv_freq must be a one-dimensional floating tensor")
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("factor must be finite and greater than one")
    if original_context_length <= 0:
        raise ValueError("original_context_length must be positive")
    if (
        not math.isfinite(low_frequency_factor)
        or not math.isfinite(high_frequency_factor)
        or low_frequency_factor <= 0.0
        or high_frequency_factor <= low_frequency_factor
    ):
        raise ValueError(
            "frequency factors must be finite and satisfy "
            "0 < low_frequency_factor < high_frequency_factor"
        )
    if not bool(torch.isfinite(inv_freq).all()) or not bool((inv_freq > 0).all()):
        raise ValueError("inv_freq values must be finite and positive")

    low_frequency_wavelength = original_context_length / low_frequency_factor
    high_frequency_wavelength = original_context_length / high_frequency_factor
    wavelength = (2.0 * math.pi) / inv_freq

    scaled = torch.where(
        wavelength > low_frequency_wavelength,
        inv_freq / factor,
        inv_freq,
    )
    smooth = (
        original_context_length / wavelength - low_frequency_factor
    ) / (high_frequency_factor - low_frequency_factor)
    smoothed = (1.0 - smooth) * scaled / factor + smooth * scaled
    is_medium_frequency = (wavelength >= high_frequency_wavelength) & (
        wavelength <= low_frequency_wavelength
    )
    return torch.where(is_medium_frequency, smoothed, scaled)


def inv_freq_for_position_policy(
    inv_freq: torch.Tensor,
    *,
    position_policy: str,
    factor: float,
    original_context_length: int,
    low_frequency_factor: float = 1.0,
    high_frequency_factor: float = 4.0,
    scale_vector: Any = None,
    scale_vector_sha256: Any = None,
) -> torch.Tensor:
    """Return native frequencies unchanged or the explicit scaled transform."""

    if position_policy == NATIVE_POSITION_POLICY:
        if scale_vector is not None or scale_vector_sha256 is not None:
            raise ValueError("native position policy forbids a scale vector")
        return inv_freq
    if position_policy == LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY:
        return per_dimension_scaled_inv_freq(
            inv_freq,
            scale_vector=scale_vector,
            scale_vector_sha256=scale_vector_sha256,
        )
    if position_policy not in (
        SCALED_ALL_POSITION_POLICY,
        E4_GLOBAL_SCALED_LOCAL_NATIVE_POLICY,
    ):
        raise ValueError(f"unsupported Barbet position policy: {position_policy!r}")
    if scale_vector is not None or scale_vector_sha256 is not None:
        raise ValueError(f"{position_policy} forbids a LongRoPE2 scale vector")
    return frequency_selective_inv_freq(
        inv_freq,
        factor=factor,
        original_context_length=original_context_length,
        low_frequency_factor=low_frequency_factor,
        high_frequency_factor=high_frequency_factor,
    )






def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a small CPU tensor without relying on NumPy."""

    import hashlib

    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(bytes(value.view(torch.uint8).tolist())).hexdigest()
