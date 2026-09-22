"""Configuration for Barbet models.

Barbet mirrors the Open Formosa R2 architecture (Taiwan-Omni R2): a hybrid
decoder with a repeating ``global, sliding, sliding, mamba2`` block motif,
tied embedding/LM-head, and the frozen voidful/PangolinTokenizer vocabulary.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

# Canonical PangolinTokenizer contract (mirrors open_formosa special_tokens.py).
# The base BPE vocab is 114688, special tokens run to id 114821 (effective
# 114822), and the Megatron-padded embedding size is 114944 (multiple of 128).
BASE_VOCAB_SIZE = 114688
EFFECTIVE_VOCAB_SIZE = 114822
MEGATRON_PADDED_VOCAB_SIZE = 114944

UNK_TOKEN_ID = 114688
BOS_TOKEN_ID = 114689
EOS_TOKEN_ID = 114690
PAD_TOKEN_ID = 114691

ROPE_SCALING_TYPES = (
    "none",
    "linear",
    "llama3",
    "longrope2",
    "yarn",
    "longrope",
)
ROPE_ATTENTION_SCOPES = ("all_attention", "global_only")


class BarbetConfig(PretrainedConfig):
    """Configuration class for Barbet decoder-only causal language models."""

    model_type = "barbet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = MEGATRON_PADDED_VOCAB_SIZE,
        hidden_size: int = 1536,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        max_position_embeddings: int = 262144,
        rope_theta: float = 10000000.0,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window_size: int = 8192,
        global_attention_layers: list[int] | tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24),
        mamba_layers: list[int] | tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27),
        qk_norm: bool = True,
        qk_logit_clip: bool = False,
        qk_clip_alpha: float = 0.5,
        qk_clip_threshold: float = 100.0,
        attention_sink: bool = False,
        rms_norm_eps: float = 1.0e-6,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_fallback_max_sequence_length: int = 4096,
        loss_chunk_size: int = 1024,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_prefill_chunk_size: int = 65536,
        mtp_enabled: bool = True,
        mtp_offsets: list[int] | tuple[int, ...] = (2, 3),
        mtp_loss_weights: dict[str, float] | dict[int, float] | None = None,
        pad_token_id: int | None = PAD_TOKEN_ID,
        bos_token_id: int | None = BOS_TOKEN_ID,
        eos_token_id: int | None = EOS_TOKEN_ID,
        unk_token_id: int | None = UNK_TOKEN_ID,
        **kwargs: Any,
    ) -> None:
        rope_scaling = self._normalize_rope_scaling(rope_scaling, kwargs.pop("rope_parameters", None))
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window_size = sliding_window_size
        self.global_attention_layers = [int(layer) for layer in global_attention_layers]
        self.mamba_layers = [int(layer) for layer in mamba_layers]
        self.qk_norm = qk_norm
        self.qk_logit_clip = qk_logit_clip
        self.qk_clip_alpha = qk_clip_alpha
        self.qk_clip_threshold = qk_clip_threshold
        self.attention_sink = attention_sink
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_fallback_max_sequence_length = attention_fallback_max_sequence_length
        self.loss_chunk_size = loss_chunk_size
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_prefill_chunk_size = mamba_prefill_chunk_size
        self.mtp_enabled = mtp_enabled
        self.mtp_offsets = [int(offset) for offset in mtp_offsets]
        weights = mtp_loss_weights or {2: 0.2, 3: 0.1}
        self.mtp_loss_weights = {str(int(k)): float(v) for k, v in dict(weights).items()}
        self.unk_token_id = unk_token_id
        self._validate()

    @classmethod
    def barbet_300m(
        cls,
        vocab_size: int = MEGATRON_PADDED_VOCAB_SIZE,
        max_position_embeddings: int = 8192,
    ) -> "BarbetConfig":
        """Taiwan-Omni-300M-R2 shape: tied vocab tables fund a 20-layer body."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=20,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=2048,
            global_attention_layers=(0, 4, 8, 12, 16),
            mamba_layers=(3, 7, 11, 15, 19),
            qk_logit_clip=False,
            attention_sink=False,
            tie_word_embeddings=True,
        )

    @classmethod
    def barbet_1b(
        cls,
        vocab_size: int = MEGATRON_PADDED_VOCAB_SIZE,
        max_position_embeddings: int = 262144,
    ) -> "BarbetConfig":
        """Taiwan-Omni-1B-R2 shape: tied vocab tables fund a 28-layer body."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=1536,
            intermediate_size=5120,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=8192,
            global_attention_layers=(0, 4, 8, 12, 16, 20, 24),
            mamba_layers=(3, 7, 11, 15, 19, 23, 27),
            qk_logit_clip=False,
            attention_sink=False,
            tie_word_embeddings=True,
        )

    @classmethod
    def barbet_1b_1m_extension(cls) -> "BarbetConfig":
        """1M research extension candidate with global-attention-only RoPE."""
        config = cls.barbet_1b(max_position_embeddings=1048576)
        config.rope_scaling = {
            "type": "llama3",
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "attention_scope": "global_only",
        }
        config._validate()
        return config

    def layer_type(self, layer_idx: int) -> str:
        if layer_idx in self.global_attention_layers:
            return "global_attention"
        if layer_idx in self.mamba_layers:
            return "mamba"
        return "sliding_attention"

    def _validate(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_attention_heads * self.head_dim < self.hidden_size:
            raise ValueError("num_attention_heads * head_dim must cover hidden_size")
        if self.attention_fallback_max_sequence_length <= 0:
            raise ValueError("attention_fallback_max_sequence_length must be positive")
        if self.loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")
        if (
            self.mamba_prefill_chunk_size <= 0
            or self.mamba_prefill_chunk_size % 128 != 0
        ):
            raise ValueError(
                "mamba_prefill_chunk_size must be a positive multiple of 128"
            )
        layer_ids = set(range(self.num_hidden_layers))
        scheduled = set(self.global_attention_layers) | set(self.mamba_layers)
        invalid = scheduled - layer_ids
        if invalid:
            raise ValueError(f"scheduled layers out of range: {sorted(invalid)}")
        overlap = set(self.global_attention_layers) & set(self.mamba_layers)
        if overlap:
            raise ValueError(f"layers cannot be both attention and mamba: {sorted(overlap)}")
        if self.rope_scaling:
            scaling_type = str(self.rope_scaling.get("type", "linear"))
            if scaling_type not in ROPE_SCALING_TYPES:
                raise ValueError(
                    f"rope_scaling.type must be one of {ROPE_SCALING_TYPES}, got {scaling_type!r}"
                )
            if scaling_type != "none":
                factor = self.rope_scaling.get("factor")
                if not isinstance(factor, (int, float)) or float(factor) <= 1.0:
                    raise ValueError("rope_scaling.factor must be greater than one")
                attention_scope = self.rope_scaling.get(
                    "attention_scope", "all_attention"
                )
                if attention_scope not in ROPE_ATTENTION_SCOPES:
                    raise ValueError(
                        "rope_scaling.attention_scope must be one of "
                        f"{ROPE_ATTENTION_SCOPES}, got {attention_scope!r}"
                    )
            if scaling_type == "llama3":
                original = self.rope_scaling.get(
                    "original_max_position_embeddings",
                    self.rope_scaling.get("original_context_length"),
                )
                low = self.rope_scaling.get("low_freq_factor")
                high = self.rope_scaling.get("high_freq_factor")
                if not isinstance(original, int) or original <= 0:
                    raise ValueError(
                        "llama3 rope_scaling requires positive "
                        "original_max_position_embeddings"
                    )
                if (
                    not isinstance(low, (int, float))
                    or not isinstance(high, (int, float))
                    or float(low) <= 0.0
                    or float(high) <= float(low)
                ):
                    raise ValueError(
                        "llama3 rope_scaling requires "
                        "0 < low_freq_factor < high_freq_factor"
                    )
            elif scaling_type == "longrope2":
                if self.head_dim != 128:
                    raise ValueError(
                        "longrope2 rope_scaling requires Barbet head_dim=128"
                    )
                if self.rope_scaling.get("attention_scope") != "global_only":
                    raise ValueError(
                        "longrope2 rope_scaling is authorized only for "
                        "global_only attention"
                    )
                from .megatron_long_context import (
                    LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY,
                    build_position_policy_payload,
                )

                build_position_policy_payload(
                    position_policy=(
                        LONGROPE2_PER_DIMENSION_GLOBAL_ONLY_POLICY
                    ),
                    factor=self.rope_scaling.get("factor"),
                    original_context_length=self.rope_scaling.get(
                        "original_max_position_embeddings",
                        self.rope_scaling.get("original_context_length"),
                    ),
                    low_frequency_factor=self.rope_scaling.get(
                        "low_freq_factor"
                    ),
                    high_frequency_factor=self.rope_scaling.get(
                        "high_freq_factor"
                    ),
                    scale_vector=self.rope_scaling.get("scale_vector"),
                    scale_vector_sha256=self.rope_scaling.get(
                        "scale_vector_sha256"
                    ),
                )

    @staticmethod
    def _normalize_rope_scaling(
        rope_scaling: dict[str, Any] | None,
        rope_parameters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        scaling = rope_scaling if rope_scaling is not None else rope_parameters
        if not scaling:
            return None
        normalized = dict(scaling)
        rope_type = normalized.pop("rope_type", None)
        if "type" not in normalized and rope_type is not None:
            normalized["type"] = rope_type
        normalized.setdefault("type", "linear")
        return normalized


BarbetConfig.register_for_auto_class()
