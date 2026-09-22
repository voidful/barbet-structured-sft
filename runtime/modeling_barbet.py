"""PyTorch modeling code for Barbet.

Mirrors the Open Formosa R2 reference architecture: hybrid global/sliding
attention + Mamba-style mixer blocks, SwiGLU MLPs, QK RMSNorm, tied
embedding/LM head, and an optional multi-token prediction training loss.

Incremental decoding uses :class:`BarbetCache`, a hybrid cache holding
attention K/V states (a rolling window for sliding layers) and the causal-conv
tail state for Mamba-style layers.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_barbet import BarbetConfig
from .megatron_long_context import per_dimension_scaled_inv_freq


def _fused_create_block_mask(
    mask_mod: Any,
    batch_size: int | None,
    num_heads: int | None,
    query_length: int,
    kv_length: int,
    device: torch.device,
    block_size: int,
) -> Any:
    if not torch.compiler.is_compiling():
        raise RuntimeError(
            "compiled FlexAttention block-mask creation fell back to eager; "
            "refusing an unbounded mask materialization"
        )
    return eager_create_block_mask(
        mask_mod,
        batch_size,
        num_heads,
        query_length,
        kv_length,
        device,
        block_size,
        _compile=False,
    )


def _fused_flex_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    score_mod: Any,
    block_mask: Any,
    enable_gqa: bool,
) -> torch.Tensor:
    if not torch.compiler.is_compiling():
        raise RuntimeError(
            "compiled FlexAttention fell back to eager; refusing an unfused "
            "long-context attention path"
        )
    return eager_flex_attention(
        query_states,
        key_states,
        value_states,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )


try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as eager_create_block_mask,
    )
    from torch.nn.attention.flex_attention import flex_attention as eager_flex_attention

    # Compile both mask construction and the FlexAttention HOP with dynamic
    # sequence dimensions. The wrappers above deliberately raise if Dynamo
    # ever exhausts its guard cache and tries to execute them eagerly.
    compiled_create_block_mask = torch.compile(
        _fused_create_block_mask,
        dynamic=True,
        fullgraph=True,
    )
    compiled_flex_attention = torch.compile(
        _fused_flex_attention,
        dynamic=True,
        fullgraph=True,
    )
except (ImportError, AttributeError):
    eager_create_block_mask = None
    compiled_flex_attention = None
    compiled_create_block_mask = None

_FLEX_BLOCK_MASK_CACHE: dict[tuple[Any, ...], Any] = {}
_FLEX_BLOCK_MASK_CACHE_LIMIT = 4
_FLEX_BLOCK_SIZE = 128
_GROWING_KV_BLOCK_MASK_DIMS = (
    ("kv_indices", 3),
    ("full_kv_indices", 3),
    ("q_num_blocks", 2),
    ("q_indices", 2),
    ("full_q_num_blocks", 2),
    ("full_q_indices", 2),
)


def _mark_growing_kv_block_mask_dynamic(block_mask: Any) -> None:
    """Keep sparse-mask KV block dimensions symbolic across cache growth."""

    for attribute, dimension in _GROWING_KV_BLOCK_MASK_DIMS:
        tensor = getattr(block_mask, attribute, None)
        if tensor is not None:
            torch._dynamo.mark_dynamic(tensor, dimension)

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as MambaRMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except Exception:
    MambaRMSNormGated = None
    mamba_chunk_scan_combined = None


class BarbetCache:
    """Hybrid per-layer cache for incremental decoding.

    Attention layers store un-repeated GQA key/value states; sliding-window
    layers keep only the most recent ``sliding_window_size`` positions. Mamba
    layers store the trailing ``d_conv - 1`` causal-conv inputs. Duck-types the
    parts of the transformers ``Cache`` interface that ``generate()`` touches
    for stateful models (``get_seq_length`` and ``reorder_cache``).
    """

    def __init__(
        self,
        config: BarbetConfig,
        *,
        rope_scaling_branch: str | None = None,
    ) -> None:
        if rope_scaling_branch not in {None, "native", "scaled"}:
            raise ValueError(
                "rope_scaling_branch must be exactly 'native', 'scaled', or None"
            )
        self.sliding_window_size = config.sliding_window_size
        num_layers = config.num_hidden_layers
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self._key_cache_storage: list[torch.Tensor | None] = [None] * num_layers
        self._value_cache_storage: list[torch.Tensor | None] = [None] * num_layers
        self.conv_cache: list[torch.Tensor | None] = [None] * num_layers
        self.ssm_cache: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens = 0
        self.rope_scaling_branch = rope_scaling_branch

    def bind_rope_scaling_branch(
        self,
        position_ids: torch.Tensor,
        *,
        original_context_length: int,
    ) -> str:
        """Latch native/scaled RoPE before any cached K tensor is created.

        A cache first built with native RoPE cannot cross the frozen boundary:
        those historical keys cannot be re-encoded in place.  Callers must
        discard it and prefill the complete long prompt under the scaled
        branch.  A scaled cache never downgrades to native.
        """

        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.numel() <= 0
            or position_ids.dtype not in {torch.int32, torch.int64}
            or type(original_context_length) is not int
            or original_context_length <= 0
        ):
            raise ValueError("RoPE cache branch requires valid integer positions")
        desired = (
            "scaled"
            if bool((position_ids >= original_context_length).any().item())
            else "native"
        )
        if self.rope_scaling_branch is None:
            self.rope_scaling_branch = desired
        elif self.rope_scaling_branch == "native" and desired == "scaled":
            raise ValueError(
                "cached native RoPE cannot cross the long-context boundary; "
                "discard the cache and rebuild the full prompt with scaled RoPE"
            )
        elif self.rope_scaling_branch not in {"native", "scaled"}:
            raise ValueError("cached RoPE scaling branch is invalid")
        return self.rope_scaling_branch

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def update_attention(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sliding: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_keys = self.key_cache[layer_idx]
        previous_values = self.value_cache[layer_idx]
        if not sliding:
            previous_length = (
                0 if previous_keys is None else previous_keys.shape[2]
            )
            required_length = previous_length + key_states.shape[2]
            key_storage = self._key_cache_storage[layer_idx]
            value_storage = self._value_cache_storage[layer_idx]
            needs_growth = (
                key_storage is None
                or value_storage is None
                or key_storage.shape[2] < required_length
            )
            if needs_growth:
                capacity = 1 << max(required_length - 1, 0).bit_length()
                new_key_storage = key_states.new_empty(
                    key_states.shape[0],
                    key_states.shape[1],
                    capacity,
                    key_states.shape[3],
                )
                new_value_storage = value_states.new_empty(
                    value_states.shape[0],
                    value_states.shape[1],
                    capacity,
                    value_states.shape[3],
                )
                if previous_length > 0:
                    new_key_storage[:, :, :previous_length].copy_(
                        previous_keys
                    )
                    new_value_storage[:, :, :previous_length].copy_(
                        previous_values
                    )
                key_storage = new_key_storage
                value_storage = new_value_storage
                self._key_cache_storage[layer_idx] = key_storage
                self._value_cache_storage[layer_idx] = value_storage
            key_storage[:, :, previous_length:required_length].copy_(key_states)
            value_storage[:, :, previous_length:required_length].copy_(
                value_states
            )
            self.key_cache[layer_idx] = key_storage[:, :, :required_length]
            self.value_cache[layer_idx] = value_storage[:, :, :required_length]
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if previous_keys is not None:
            key_states = torch.cat([previous_keys, key_states], dim=2)
            value_states = torch.cat([previous_values, value_states], dim=2)
        # Return the full states for the current block (early queries in a
        # prefill chunk still need keys beyond the window tail); store only the
        # rolling window for future steps.
        if self.sliding_window_size > 0 and key_states.shape[2] > self.sliding_window_size:
            self.key_cache[layer_idx] = key_states[:, :, -self.sliding_window_size :]
            self.value_cache[layer_idx] = value_states[:, :, -self.sliding_window_size :]
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def update_conv(self, layer_idx: int, conv_inputs: torch.Tensor, tail_len: int) -> torch.Tensor:
        """Prepend the cached conv tail (zeros initially) and store the new tail.

        ``conv_inputs`` has shape ``(batch, channels, seq_len)``; the returned
        tensor has ``tail_len`` extra leading positions so a valid (unpadded)
        causal conv produces exactly ``seq_len`` outputs.
        """
        previous = self.conv_cache[layer_idx]
        if previous is None:
            previous = conv_inputs.new_zeros(conv_inputs.shape[0], conv_inputs.shape[1], tail_len)
        full = torch.cat([previous, conv_inputs], dim=-1)
        self.conv_cache[layer_idx] = full[..., full.shape[-1] - tail_len :]
        return full

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx, key_storage in enumerate(self._key_cache_storage):
            value_storage = self._value_cache_storage[layer_idx]
            if key_storage is None or value_storage is None:
                for tensors in (self.key_cache, self.value_cache):
                    tensor = tensors[layer_idx]
                    if tensor is not None:
                        tensors[layer_idx] = tensor.index_select(
                            0, beam_idx.to(tensor.device)
                        )
                continue
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            assert key_states is not None and value_states is not None
            used_length = key_states.shape[2]
            capacity = key_storage.shape[2]
            reordered_keys = key_states.index_select(
                0, beam_idx.to(key_states.device)
            )
            reordered_values = value_states.index_select(
                0, beam_idx.to(value_states.device)
            )
            new_key_storage = key_storage.new_empty(
                reordered_keys.shape[0],
                reordered_keys.shape[1],
                capacity,
                reordered_keys.shape[3],
            )
            new_value_storage = value_storage.new_empty(
                reordered_values.shape[0],
                reordered_values.shape[1],
                capacity,
                reordered_values.shape[3],
            )
            new_key_storage[:, :, :used_length].copy_(reordered_keys)
            new_value_storage[:, :, :used_length].copy_(reordered_values)
            self._key_cache_storage[layer_idx] = new_key_storage
            self._value_cache_storage[layer_idx] = new_value_storage
            self.key_cache[layer_idx] = new_key_storage[:, :, :used_length]
            self.value_cache[layer_idx] = new_value_storage[:, :, :used_length]

        for tensors in (self.conv_cache, self.ssm_cache):
            for idx, tensor in enumerate(tensors):
                if tensor is not None:
                    tensors[idx] = tensor.index_select(0, beam_idx.to(tensor.device))


class BarbetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return hidden_states.to(dtype=self.weight.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class BarbetRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: BarbetConfig,
        *,
        is_global_attention: bool = True,
    ) -> None:
        super().__init__()
        self.dim = config.head_dim
        self.base = config.rope_theta
        self.scaling_type = "none"
        self.attention_scope = "all_attention"
        self.is_global_attention = is_global_attention
        scale = None
        self.original_context_length = None
        self.low_frequency_factor = None
        self.high_frequency_factor = None
        self.scale_vector = None
        self.scale_vector_sha256 = None
        if config.rope_scaling:
            scaling_type = str(config.rope_scaling.get("type", "linear")).lower()
            self.attention_scope = str(
                config.rope_scaling.get(
                    "attention_scope", "all_attention"
                )
            )
            if (
                self.attention_scope == "global_only"
                and not is_global_attention
            ):
                scaling_type = "none"
            factor = config.rope_scaling.get("factor")
            self.scaling_type = scaling_type
            if scaling_type in {"linear", "llama3"} and factor:
                scale = float(factor)
            if scaling_type == "llama3":
                self.original_context_length = int(
                    config.rope_scaling.get(
                        "original_max_position_embeddings",
                        config.rope_scaling.get("original_context_length"),
                    )
                )
                self.low_frequency_factor = float(
                    config.rope_scaling["low_freq_factor"]
                )
                self.high_frequency_factor = float(
                    config.rope_scaling["high_freq_factor"]
                )
            elif scaling_type == "longrope2":
                self.original_context_length = int(
                    config.rope_scaling.get(
                        "original_max_position_embeddings",
                        config.rope_scaling.get("original_context_length"),
                    )
                )
                self.scale_vector = list(
                    config.rope_scaling["scale_vector"]
                )
                self.scale_vector_sha256 = str(
                    config.rope_scaling["scale_vector_sha256"]
                )
            elif scaling_type not in {"none", "linear"}:
                raise ValueError(
                    f"rope_scaling.type={scaling_type!r} is metadata-only in the "
                    "HF reference runtime; use 'linear', 'llama3', or "
                    "'longrope2'"
                )
        self.scale = scale

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
        *,
        scaling_branch: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.float()
        if self.scaling_type == "linear" and self.scale and self.scale > 1.0:
            positions = positions / self.scale
        # Computed per call instead of a non-persistent buffer: meta-device
        # checkpoint loading materializes such buffers uninitialized.
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device) / self.dim)
        )
        if scaling_branch not in {None, "native", "scaled"}:
            raise ValueError("RoPE scaling_branch must be native or scaled")
        scale_long_context = self.scaling_type in {"llama3", "longrope2"} and (
            scaling_branch == "scaled"
            or (
                scaling_branch is None
                and self.original_context_length is not None
                and bool(
                    (position_ids >= self.original_context_length).any().item()
                )
            )
        )
        if self.scaling_type == "llama3" and scale_long_context:
            wavelength = (2.0 * math.pi) / inv_freq
            low_wavelength = (
                self.original_context_length / self.low_frequency_factor
            )
            high_wavelength = (
                self.original_context_length / self.high_frequency_factor
            )
            scaled = torch.where(
                wavelength > low_wavelength,
                inv_freq / self.scale,
                inv_freq,
            )
            smooth = (
                self.original_context_length / wavelength
                - self.low_frequency_factor
            ) / (self.high_frequency_factor - self.low_frequency_factor)
            smoothed = (1.0 - smooth) * scaled / self.scale + smooth * scaled
            medium = (wavelength >= high_wavelength) & (
                wavelength <= low_wavelength
            )
            inv_freq = torch.where(medium, smoothed, scaled)
        elif self.scaling_type == "longrope2" and scale_long_context:
            inv_freq = per_dimension_scaled_inv_freq(
                inv_freq,
                scale_vector=self.scale_vector,
                scale_vector_sha256=self.scale_vector_sha256,
            )
        freqs = torch.einsum("bs,d->bsd", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[:, None, :, :]) + (rotate_half(x) * sin[:, None, :, :])


class BarbetAttention(nn.Module):
    def __init__(self, config: BarbetConfig, layer_idx: int, sliding_window: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window_size = config.sliding_window_size if sliding_window else None
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = BarbetRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = BarbetRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.rotary_emb = BarbetRotaryEmbedding(
            config,
            is_global_attention=not sliding_window,
        )
        self.dropout = nn.Dropout(config.attention_dropout)
        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads)) if config.attention_sink else None

    def _shape(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _dense_attention_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        total = past_len + q_len
        q_pos = torch.arange(past_len, total, device=device)[:, None]
        # Cached keys are the most recent kv_len positions, in order.
        k_pos = torch.arange(total - kv_len, total, device=device)[None, :]
        mask = k_pos <= q_pos
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            mask &= k_pos >= (q_pos - self.sliding_window_size + 1)
        mask = mask.view(1, 1, q_len, kv_len).expand(batch_size, 1, q_len, kv_len)
        if attention_mask is not None:
            if attention_mask.shape[1] < total:
                # Mask covers only the newest tokens; treat older history as visible.
                pad = attention_mask.new_ones(batch_size, total - attention_mask.shape[1])
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
            key_mask = attention_mask[:, -kv_len:][:, None, None, :].bool()
            mask = mask & key_mask
        return mask

    def _key_padding_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
            raise ValueError(
                "attention_mask must have shape (batch, sequence); "
                f"got {tuple(attention_mask.shape)}"
            )
        total = past_len + q_len
        if attention_mask.shape[1] < total:
            # Generation callers occasionally pass a mask for only the newest
            # chunk. Cache history predates that mask and is treated as visible.
            pad = attention_mask.new_ones(batch_size, total - attention_mask.shape[1])
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
        return attention_mask[:, -kv_len:].bool()

    def _dense_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded correctness path used when attention weights are requested."""
        batch_size, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        max_len = int(self.config.attention_fallback_max_sequence_length)
        if max(q_len, kv_len) > max_len:
            raise ValueError(
                "The dense attention fallback is disabled above "
                f"{max_len} tokens (received q_len={q_len}, kv_len={kv_len}). "
                "Set output_attentions=False and attention_dropout=0 to use "
                "the memory-efficient SDPA/FlexAttention path."
            )

        repeated_keys = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        repeated_values = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, repeated_keys.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        if self.config.qk_logit_clip:
            threshold = float(self.config.qk_clip_threshold)
            attn_weights = threshold * torch.tanh(attn_weights / threshold)

        allowed = self._dense_attention_mask(
            batch_size,
            q_len,
            kv_len,
            past_len,
            attention_mask,
            query_states.device,
        )
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~allowed, min_value)

        if self.sink_logits is None:
            softmax_input = attn_weights if attn_weights.is_cuda else attn_weights.float()
            attn_probs = torch.softmax(softmax_input, dim=-1).to(query_states.dtype)
        else:
            sink = self.sink_logits.view(1, self.num_heads, 1, 1).float()
            max_score = torch.maximum(attn_weights.float().max(dim=-1, keepdim=True).values, sink)
            real_exp = torch.exp(attn_weights.float() - max_score)
            sink_exp = torch.exp(sink - max_score)
            attn_probs = (real_exp / (real_exp.sum(dim=-1, keepdim=True) + sink_exp)).to(
                query_states.dtype
            )

        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, repeated_values), attn_probs

    def _flex_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        past_len: int,
    ) -> torch.Tensor:
        if (
            compiled_flex_attention is None
            or compiled_create_block_mask is None
        ):
            raise RuntimeError(
                "This attention pattern requires torch.nn.attention.flex_attention "
                "(PyTorch >= 2.5) to avoid a dense attention matrix."
            )
        if not query_states.is_cuda:
            raise RuntimeError("Fused FlexAttention is only used on CUDA tensors.")

        batch_size, _, q_len, _ = query_states.shape
        real_kv_len = key_states.shape[2]
        sliding_window = self.sliding_window_size
        expected_history_len = (
            past_len
            if sliding_window is None
            else min(past_len, sliding_window)
        )
        expected_kv_len = expected_history_len + q_len
        if real_kv_len != expected_kv_len:
            raise ValueError(
                "cached attention state is not contiguous: expected "
                f"kv_len={expected_kv_len} from past_len={past_len}, "
                f"q_len={q_len}, and sliding_window={sliding_window}; "
                f"received kv_len={real_kv_len}"
            )
        history_len = real_kv_len - q_len
        has_sink = self.sink_logits is not None
        flex_kv_len = real_kv_len + int(has_sink)

        if has_sink:
            sink_key = key_states.new_zeros(batch_size, self.num_key_value_heads, 1, self.head_dim)
            sink_value = value_states.new_zeros(
                batch_size, self.num_key_value_heads, 1, self.head_dim
            )
            key_states = torch.cat([key_states, sink_key], dim=2)
            value_states = torch.cat([value_states, sink_value], dim=2)

        def causal_window_mask(
            batch_idx: torch.Tensor,
            _head_idx: torch.Tensor,
            query_idx: torch.Tensor,
            key_idx: torch.Tensor,
        ) -> torch.Tensor:
            is_real_key = key_idx < real_kv_len
            allowed = is_real_key & (
                key_idx <= query_idx + history_len
            )
            if sliding_window is not None and sliding_window > 0:
                allowed = allowed & (
                    key_idx
                    >= query_idx + history_len - sliding_window + 1
                )
            if key_padding_mask is not None:
                safe_key_idx = torch.minimum(
                    key_idx, key_idx.new_tensor(max(real_kv_len - 1, 0))
                )
                allowed = allowed & key_padding_mask[batch_idx, safe_key_idx]
            if has_sink:
                allowed = allowed | (key_idx == real_kv_len)
            return allowed

        block_mask = None
        cache_key: tuple[Any, ...] | None = None
        if key_padding_mask is None:
            cache_key = (
                query_states.device.type,
                query_states.device.index,
                q_len,
                flex_kv_len,
                real_kv_len,
                sliding_window,
                has_sink,
            )
            block_mask = _FLEX_BLOCK_MASK_CACHE.get(cache_key)
        if block_mask is None:
            block_mask = compiled_create_block_mask(
                causal_window_mask,
                batch_size if key_padding_mask is not None else None,
                None,
                q_len,
                flex_kv_len,
                query_states.device,
                _FLEX_BLOCK_SIZE,
            )
            if cache_key is not None:
                if len(_FLEX_BLOCK_MASK_CACHE) >= _FLEX_BLOCK_MASK_CACHE_LIMIT:
                    _FLEX_BLOCK_MASK_CACHE.pop(next(iter(_FLEX_BLOCK_MASK_CACHE)))
                _FLEX_BLOCK_MASK_CACHE[cache_key] = block_mask
        _mark_growing_kv_block_mask_dynamic(block_mask)

        threshold = float(self.config.qk_clip_threshold)
        sink_logits = self.sink_logits
        if self.config.qk_logit_clip or has_sink:

            def score_mod(
                score: torch.Tensor,
                _batch_idx: torch.Tensor,
                head_idx: torch.Tensor,
                _query_idx: torch.Tensor,
                key_idx: torch.Tensor,
            ) -> torch.Tensor:
                if self.config.qk_logit_clip:
                    score = threshold * torch.tanh(score / threshold)
                if sink_logits is not None:
                    score = torch.where(key_idx == real_kv_len, sink_logits[head_idx], score)
                return score

        else:
            score_mod = None

        return compiled_flex_attention(
            query_states,
            key_states,
            value_states,
            score_mod,
            block_mask,
            self.num_key_value_heads != self.num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BarbetCache | None = None,
        past_len: int = 0,
        rope_scaling_branch: str | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        query_states = self._shape(self.q_proj(hidden_states), self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value_states = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = self.rotary_emb(
            position_ids,
            query_states.dtype,
            scaling_branch=rope_scaling_branch,
        )
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update_attention(
                self.layer_idx, key_states, value_states, sliding=self.sliding_window_size is not None
            )

        kv_len = key_states.shape[2]
        use_dense_fallback = output_attentions or (
            self.training and float(self.config.attention_dropout) > 0.0
        )
        if use_dense_fallback:
            attn_output, attn_probs = self._dense_attention(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                past_len=past_len,
            )
        else:
            key_padding_mask = self._key_padding_mask(
                batch_size, seq_len, kv_len, past_len, attention_mask
            )
            has_advanced_scores = self.config.qk_logit_clip or self.sink_logits is not None
            if self.sliding_window_size is None and past_len == 0 and not has_advanced_scores:
                sdpa_mask = (
                    key_padding_mask[:, None, None, :]
                    if key_padding_mask is not None
                    else None
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=seq_len > 1,
                    enable_gqa=self.num_key_value_heads != self.num_heads,
                )
            elif self.sliding_window_size is None and seq_len == 1 and not has_advanced_scores:
                sdpa_mask = (
                    key_padding_mask[:, None, None, :]
                    if key_padding_mask is not None
                    else None
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=self.num_key_value_heads != self.num_heads,
                )
            elif self.sliding_window_size is not None and seq_len == 1 and not has_advanced_scores:
                # A rolling cache may return window+1 states (the previous
                # full window plus the new token). Selecting the trailing
                # window keeps this single-token generation path exact.
                window = int(self.sliding_window_size)
                key_states = key_states[:, :, -window:, :]
                value_states = value_states[:, :, -window:, :]
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask[:, -window:]
                sdpa_mask = (
                    key_padding_mask[:, None, None, :]
                    if key_padding_mask is not None
                    else None
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=self.num_key_value_heads != self.num_heads,
                )
            elif (
                not has_advanced_scores
                and max(seq_len, kv_len)
                <= int(self.config.attention_fallback_max_sequence_length)
            ):
                # FlexAttention's decode-oriented kernel selection can reject
                # very short GQA prefills on some PyTorch/Triton builds. A
                # bounded boolean mask keeps short-context calls on fused SDPA.
                sdpa_mask = self._dense_attention_mask(
                    batch_size,
                    seq_len,
                    kv_len,
                    past_len,
                    attention_mask,
                    query_states.device,
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=self.num_key_value_heads != self.num_heads,
                )
            elif query_states.is_cuda and self.head_dim >= 16:
                attn_output = self._flex_attention(
                    query_states,
                    key_states,
                    value_states,
                    key_padding_mask=key_padding_mask,
                    past_len=past_len,
                )
            else:
                # FlexAttention's eager CPU fallback is dense. Keep CPU only
                # (and unsupported tiny CUDA head dimensions) as a bounded
                # correctness path for unit tests.
                attn_output, _ = self._dense_attention(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=attention_mask,
                    past_len=past_len,
                )
            attn_probs = None

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), attn_probs if output_attentions else None


class BarbetMambaMixer(nn.Module):
    """Megatron Mamba2-compatible mixer with a PyTorch selective-scan path."""

    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.inner_size = config.hidden_size * config.mamba_expand
        self.d_state = max(config.mamba_d_state, 1)
        self.d_conv = config.mamba_d_conv
        self.head_dim = config.head_dim
        self.num_heads = self.inner_size // self.head_dim
        self.num_groups = config.num_key_value_heads
        self.prefill_chunk_size = config.mamba_prefill_chunk_size
        if self.inner_size % self.head_dim != 0:
            raise ValueError("mamba inner size must be divisible by head_dim")
        if self.num_heads % self.num_groups != 0:
            raise ValueError("mamba heads must be divisible by mamba groups")
        self.group_size = self.inner_size // self.num_groups

        self.in_proj_z = nn.Linear(config.hidden_size, self.inner_size, bias=False)
        self.in_proj_x = nn.Linear(config.hidden_size, self.inner_size, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_groups * self.d_state, bias=False)
        self.in_proj_c = nn.Linear(config.hidden_size, self.num_groups * self.d_state, bias=False)
        self.in_proj_dt = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.conv_x = nn.Conv1d(
            self.inner_size,
            self.inner_size,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.inner_size,
        )
        self.conv_b = nn.Conv1d(
            self.num_groups * self.d_state,
            self.num_groups * self.d_state,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.num_groups * self.d_state,
        )
        self.conv_c = nn.Conv1d(
            self.num_groups * self.d_state,
            self.num_groups * self.d_state,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.num_groups * self.d_state,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        if MambaRMSNormGated is not None:
            self.norm = MambaRMSNormGated(
                self.inner_size,
                eps=1.0e-5,
                group_size=self.group_size,
                norm_before_gate=False,
            )
        else:
            self.norm = BarbetRMSNorm(self.inner_size, eps=1.0e-5)
        self.out_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

    def _conv_full(self, conv: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        seq_len = values.shape[1]
        values = values.transpose(1, 2)
        values = conv(values)[..., :seq_len]
        return F.silu(values.transpose(1, 2))

    def _rmsnorm_gated(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # mamba-ssm's fused RMSNorm is a Triton kernel.  Merely having the
        # optional package installed must not make CPU inference/tests enter
        # that CUDA-only path.
        if (
            hidden_states.is_cuda
            and MambaRMSNormGated is not None
            and isinstance(self.norm, MambaRMSNormGated)
        ):
            return self.norm(hidden_states, gate)
        hidden_states = hidden_states * F.silu(gate)
        shape = hidden_states.shape
        grouped = hidden_states.view(*shape[:-1], self.num_groups, self.group_size)
        variance = grouped.float().pow(2).mean(dim=-1, keepdim=True)
        grouped = grouped.float() * torch.rsqrt(variance + self.norm.eps)
        weight = self.norm.weight.view(1, 1, self.num_groups, self.group_size)
        return (grouped.to(dtype=self.norm.weight.dtype) * weight).view(shape)

    def _selective_scan(
        self,
        x: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        dt: torch.Tensor,
        z: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b_proj = b_proj.view(batch_size, seq_len, self.num_groups, self.d_state)
        c_proj = c_proj.view(batch_size, seq_len, self.num_groups, self.d_state)
        z = z.view(batch_size, seq_len, self.num_heads, self.head_dim)

        state = initial_state
        if state is None:
            state = x.new_zeros(batch_size, self.num_heads, self.head_dim, self.d_state)
        else:
            state = state.to(dtype=dtype)

        heads_per_group = self.num_heads // self.num_groups
        group_for_head = torch.arange(self.num_heads, device=x.device) // heads_per_group
        a = -torch.exp(self.A_log.float()).to(dtype=dtype)
        d = self.D.to(dtype=dtype)
        dt_bias = self.dt_bias.to(dtype=dtype)
        outputs: list[torch.Tensor] = []
        for pos in range(seq_len):
            dt_pos = F.softplus(dt[:, pos] + dt_bias)
            d_a = torch.exp(dt_pos * a)
            b_pos = b_proj[:, pos].index_select(1, group_for_head)
            c_pos = c_proj[:, pos].index_select(1, group_for_head)
            x_pos = x[:, pos]
            state = state * d_a[:, :, None, None] + (
                dt_pos[:, :, None, None] * b_pos[:, :, None, :] * x_pos[:, :, :, None]
            )
            y = (state * c_pos[:, :, None, :]).sum(dim=-1)
            y = y + d[None, :, None] * x_pos
            outputs.append(y.reshape(batch_size, self.inner_size))
        y = torch.stack(outputs, dim=1)
        return self._rmsnorm_gated(y, z.reshape(batch_size, seq_len, self.inner_size)), state

    def _selective_scan_kernel(
        self,
        x: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        dt: torch.Tensor,
        z: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mamba_chunk_scan_combined is None or not x.is_cuda:
            raise RuntimeError("mamba_ssm selective-scan kernel is unavailable")
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        b_proj = b_proj.view(batch_size, seq_len, self.num_groups, self.d_state).contiguous()
        c_proj = c_proj.view(batch_size, seq_len, self.num_groups, self.d_state).contiguous()
        dt = dt.contiguous()
        a = -torch.exp(self.A_log.float())
        preserve_autograd = self.training and torch.is_grad_enabled()
        outputs: list[torch.Tensor] = []
        normalized_output = (
            None
            if preserve_autograd
            else x.new_empty(batch_size, seq_len, self.inner_size)
        )
        state = initial_state
        for start in range(0, seq_len, self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, seq_len)
            needs_state = return_final_state or end < seq_len
            result = mamba_chunk_scan_combined(
                x[:, start:end],
                dt[:, start:end],
                a,
                b_proj[:, start:end],
                c_proj[:, start:end],
                chunk_size=128,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias.float(),
                initial_states=state,
                dt_softplus=True,
                return_final_states=needs_state,
            )
            if needs_state:
                y_chunk, state = result
            else:
                y_chunk = result
            y_chunk = self._rmsnorm_gated(
                y_chunk.reshape(batch_size, end - start, self.inner_size),
                z[:, start:end],
            )
            if preserve_autograd:
                outputs.append(y_chunk)
            else:
                assert normalized_output is not None
                normalized_output[:, start:end].copy_(y_chunk)
        if preserve_autograd:
            y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        else:
            assert normalized_output is not None
            y = normalized_output
        final_state = state if return_final_state else None
        return y, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: BarbetCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        z = self.in_proj_z(hidden_states)
        x = self.in_proj_x(hidden_states)
        b_proj = self.in_proj_b(hidden_states)
        c_proj = self.in_proj_c(hidden_states)
        dt = self.in_proj_dt(hidden_states)

        has_cached_state = (
            past_key_values is not None
            and past_key_values.conv_cache[layer_idx] is not None
            and past_key_values.ssm_cache[layer_idx] is not None
        )
        use_step_cache = has_cached_state and hidden_states.shape[1] == 1
        if use_step_cache:
            conv_inputs = torch.cat([x, b_proj, c_proj], dim=-1)
            conv_state = past_key_values.conv_cache[layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = conv_inputs[:, 0, :]
            weights = torch.cat([self.conv_x.weight, self.conv_b.weight, self.conv_c.weight], dim=0)
            bias = torch.cat([self.conv_x.bias, self.conv_b.bias, self.conv_c.bias], dim=0)
            conv_out = (conv_state * weights.squeeze(1)[None, :, :]).sum(dim=-1) + bias
            conv_out = F.silu(conv_out).unsqueeze(1).to(dtype=hidden_states.dtype)
            past_key_values.conv_cache[layer_idx] = conv_state
        elif has_cached_state:
            conv_inputs = torch.cat([x, b_proj, c_proj], dim=-1).transpose(1, 2)
            conv_state = past_key_values.conv_cache[layer_idx]
            # The cache stores the latest d_conv raw inputs. The oldest one is
            # not needed for the first position of this chunk, so prepend the
            # remaining d_conv - 1 inputs and run an unpadded depthwise conv.
            full_conv_inputs = torch.cat(
                [conv_state[..., 1:], conv_inputs],
                dim=-1,
            )
            weights = torch.cat(
                [self.conv_x.weight, self.conv_b.weight, self.conv_c.weight],
                dim=0,
            )
            bias = torch.cat(
                [self.conv_x.bias, self.conv_b.bias, self.conv_c.bias],
                dim=0,
            )
            conv_out = F.conv1d(
                full_conv_inputs,
                weights,
                bias,
                groups=full_conv_inputs.shape[1],
            )
            conv_out = F.silu(conv_out.transpose(1, 2)).to(
                dtype=hidden_states.dtype
            )
            past_key_values.conv_cache[layer_idx] = torch.cat(
                [conv_state, conv_inputs],
                dim=-1,
            )[..., -self.d_conv :]
        else:
            if past_key_values is not None:
                conv_tail = torch.cat(
                    [
                        x[:, -self.d_conv :],
                        b_proj[:, -self.d_conv :],
                        c_proj[:, -self.d_conv :],
                    ],
                    dim=-1,
                )
                padded = F.pad(
                    conv_tail.transpose(1, 2),
                    (max(self.d_conv - conv_tail.shape[1], 0), 0),
                )
                past_key_values.conv_cache[layer_idx] = padded[
                    ..., -self.d_conv :
                ]
            x = self._conv_full(self.conv_x, x)
            b_proj = self._conv_full(self.conv_b, b_proj)
            c_proj = self._conv_full(self.conv_c, c_proj)
            conv_out = torch.cat([x, b_proj, c_proj], dim=-1)

        x, b_proj, c_proj = torch.split(
            conv_out,
            [self.inner_size, self.num_groups * self.d_state, self.num_groups * self.d_state],
            dim=-1,
        )
        initial_state = (
            past_key_values.ssm_cache[layer_idx]
            if has_cached_state
            else None
        )
        if (
            mamba_chunk_scan_combined is not None
            and not use_step_cache
            and hidden_states.is_cuda
        ):
            y, final_state = self._selective_scan_kernel(
                x,
                b_proj,
                c_proj,
                dt,
                z,
                initial_state=initial_state,
                return_final_state=past_key_values is not None,
            )
        else:
            y, final_state = self._selective_scan(x, b_proj, c_proj, dt, z, initial_state=initial_state)
        if past_key_values is not None:
            past_key_values.ssm_cache[layer_idx] = final_state.detach()
        return self.out_proj(y)


class BarbetMLP(nn.Module):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class BarbetDecoderLayer(nn.Module):
    def __init__(self, config: BarbetConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.input_layernorm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.layer_type == "mamba":
            self.mixer = BarbetMambaMixer(config)
        else:
            self.mixer = BarbetAttention(config, layer_idx, sliding_window=self.layer_type == "sliding_attention")
        self.post_attention_layernorm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = BarbetMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BarbetCache | None = None,
        past_len: int = 0,
        rope_scaling_branch: str | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if isinstance(self.mixer, BarbetAttention):
            mixed, attn = self.mixer(
                normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                past_len=past_len,
                rope_scaling_branch=rope_scaling_branch,
                output_attentions=output_attentions,
            )
        else:
            mixed = self.mixer(normed, past_key_values=past_key_values, layer_idx=self.layer_idx)
            attn = None
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attn


class BarbetPreTrainedModel(PreTrainedModel):
    config_class = BarbetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BarbetDecoderLayer"]
    # Preserve the checkpoint's FP32 state parameters under HF BF16 loading.
    _keep_in_fp32_modules_strict = ["A_log", "D"]

    def mark_tied_weights_as_initialized(self, loading_info: Any) -> None:
        # transformers >= 5 additionally drops declared tie targets from
        # missing_keys for remote-code models (a module-tying heuristic), which
        # then stops tie_weights() from re-tying lm_head after loading a
        # deduplicated checkpoint. Barbet only ties parameters explicitly via
        # _tied_weights_keys, so keep the init-skip flag and skip that cleanup.
        for tied_param in getattr(self, "all_tied_weights_keys", {}):
            self.get_parameter(tied_param)._is_hf_initialized = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class BarbetModel(BarbetPreTrainedModel):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__(config)
        padding_idx = config.pad_token_id
        if padding_idx is not None and padding_idx >= config.vocab_size:
            padding_idx = None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx)
        self.layers = nn.ModuleList([BarbetDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BarbetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warn(
                "use_cache=True is incompatible with gradient checkpointing during training; "
                "disabling the cache for this forward pass.",
                stacklevel=2,
            )
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape

        if use_cache and not isinstance(past_key_values, BarbetCache):
            past_key_values = BarbetCache(self.config)
        if not use_cache:
            past_key_values = None
        past_len = past_key_values.seen_tokens if past_key_values is not None else 0

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError(
                    "attention_mask must have shape (batch, sequence); "
                    f"got {tuple(attention_mask.shape)}"
                )
            # Avoid carrying an all-ones key mask through every attention
            # layer. This preserves SDPA's causal fast path and lets a shared
            # structural FlexAttention mask remain batch-independent.
            if bool(attention_mask.bool().all().item()):
                attention_mask = None
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        rope_scaling_branch = None
        rope_scaling = self.config.rope_scaling
        if (
            past_key_values is not None
            and isinstance(rope_scaling, dict)
            and str(rope_scaling.get("type", "")).lower()
            in {"llama3", "longrope2"}
        ):
            original_context_length = rope_scaling.get(
                "original_max_position_embeddings",
                rope_scaling.get("original_context_length"),
            )
            if type(original_context_length) is not int:
                raise ValueError(
                    "cached scaled RoPE requires an exact original context length"
                )
            rope_scaling_branch = past_key_values.bind_rope_scaling_branch(
                position_ids,
                original_context_length=original_context_length,
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training and not output_attentions:

                def layer_forward(
                    states: torch.Tensor,
                    layer: BarbetDecoderLayer = decoder_layer,
                ) -> torch.Tensor:
                    return layer(
                        states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        past_len=0,
                        rope_scaling_branch=rope_scaling_branch,
                        output_attentions=False,
                    )[0]

                hidden_states = self._gradient_checkpointing_func(layer_forward, hidden_states)
                attn = None
            else:
                hidden_states, attn = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    past_len=past_len,
                    rope_scaling_branch=rope_scaling_branch,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                all_attentions += (attn,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if past_key_values is not None:
            past_key_values.seen_tokens += seq_len

        if not return_dict:
            return tuple(
                v for v in (hidden_states, past_key_values, all_hidden_states, all_attentions) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class BarbetMTPHead(nn.Module):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.offsets = list(config.mtp_offsets)
        self.weights = {int(k): float(v) for k, v in config.mtp_loss_weights.items()}
        self.proj = nn.ModuleDict(
            {
                str(offset): nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.SiLU(),
                )
                for offset in self.offsets
            }
        )

    def project(self, hidden_states: torch.Tensor, offset: int) -> torch.Tensor:
        return self.proj[str(offset)](hidden_states)


class BarbetForCausalLM(BarbetPreTrainedModel, GenerationMixin):
    # {target: source} mapping (transformers >= 5); 4.x ties via
    # get_output_embeddings() and only iterates these keys for bookkeeping.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # The hybrid cache (rolling sliding-window K/V + Mamba conv state) cannot
    # roll back, so generate() must not create a DynamicCache for this model.
    _is_stateful = True

    def __init__(self, config: BarbetConfig) -> None:
        super().__init__(config)
        self.model = BarbetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = BarbetMTPHead(config) if config.mtp_enabled else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def _chunked_shifted_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        offset: int = 1,
        projector: nn.Module | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Cross entropy without retaining a ``sequence x vocabulary`` tensor.

        During training, each vocabulary projection is activation-checkpointed
        independently. Backward recomputes a single chunk's logits, so peak
        saved-logit memory is bounded by ``chunk_size x vocab_size`` instead of
        growing with the context length.
        """
        if offset <= 0:
            raise ValueError("offset must be positive")
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                "labels must match the hidden-state batch and sequence dimensions; "
                f"got labels={tuple(labels.shape)}, hidden={tuple(hidden_states.shape[:2])}"
            )
        shifted_labels = torch.full_like(labels, -100)
        if offset < labels.shape[1]:
            shifted_labels[:, :-offset] = labels[:, offset:]
        valid = shifted_labels.ne(-100)
        denominator = valid.sum().clamp_min(1).to(dtype=torch.float32)
        chunk_size = int(chunk_size or self.config.loss_chunk_size)
        if chunk_size <= 0:
            raise ValueError("loss chunk size must be positive")

        def chunk_loss(
            states: torch.Tensor,
            targets: torch.Tensor,
            lm_head_weight: torch.Tensor,
        ) -> torch.Tensor:
            if projector is not None:
                states = projector(states)
            chunk_logits = F.linear(states, lm_head_weight)
            return F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden_states.shape[1], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[1])
            states = hidden_states[:, start:end]
            targets = shifted_labels[:, start:end]
            if self.training and torch.is_grad_enabled():
                partial_loss = activation_checkpoint(
                    chunk_loss,
                    states,
                    targets,
                    self.lm_head.weight,
                    use_reentrant=False,
                )
            else:
                partial_loss = chunk_loss(states, targets, self.lm_head.weight)
            total_loss = total_loss + partial_loss
        return total_loss / denominator

    @staticmethod
    def _select_logits_hidden_states(
        hidden_states: torch.Tensor,
        logits_to_keep: int | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be non-negative")
            return hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]
        if not isinstance(logits_to_keep, torch.Tensor):
            raise TypeError("logits_to_keep must be an int or a tensor of sequence indices")
        return hidden_states[:, logits_to_keep, :]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BarbetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        loss_chunk_size: int | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        loss = None
        if labels is not None:
            loss = self._chunked_shifted_loss(
                hidden_states,
                labels,
                offset=1,
                chunk_size=loss_chunk_size,
            )
            if self.mtp is not None:
                for offset in self.mtp.offsets:
                    loss = loss + self.mtp.weights.get(offset, 1.0) * self._chunked_shifted_loss(
                        hidden_states,
                        labels,
                        offset=offset,
                        projector=self.mtp.proj[str(offset)],
                        chunk_size=loss_chunk_size,
                    )

        selected_hidden_states = self._select_logits_hidden_states(
            hidden_states, logits_to_keep
        )
        logits = self.lm_head(selected_hidden_states)

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: BarbetCache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        past_len = past_key_values.seen_tokens if isinstance(past_key_values, BarbetCache) else 0
        if past_len > 0:
            input_ids = input_ids[:, past_len:]
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1] :]
        logits_to_keep = kwargs.get("logits_to_keep", 1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache if use_cache is not None else True,
            # Generation consumes only the newest-token logits. Keeping this
            # at one avoids a prompt_length x vocab allocation during a long
            # prefill; callers may override it explicitly.
            "logits_to_keep": logits_to_keep,
        }


BarbetModel.register_for_auto_class("AutoModel")
BarbetForCausalLM.register_for_auto_class("AutoModelForCausalLM")
