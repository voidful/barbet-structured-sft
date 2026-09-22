"""Load the published checkpoint without rounding its FP32 Mamba parameters."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def load_barbet(folder, device="cuda:0"):
    folder = Path(folder).resolve()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    cls = get_class_from_dynamic_module(
        "modeling_barbet.BarbetForCausalLM", str(folder), local_files_only=True
    )
    config = cls.config_class.from_pretrained(folder, local_files_only=True)
    if (config.num_hidden_layers != 29
            or config.global_attention_layers != [i for i in range(29) if i % 4 != 3]
            or config.mamba_layers != list(range(3, 28, 4))
            or config.max_position_embeddings != 1048576
            or config.rope_theta != 10000000 or config.rope_scaling is not None):
        raise ValueError("This loader expects the released native-1M architecture")
    state = load_file(str(folder / "model.safetensors"), device="cpu")
    with torch.device("meta"):
        model = cls(config)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys or set(result.missing_keys) - {"lm_head.weight"}:
        raise ValueError(str(result))
    model.lm_head.weight = model.model.embed_tokens.weight
    census = Counter()
    for name, p in model.named_parameters():
        if name not in state or p.dtype != state[name].dtype:
            raise ValueError(f"Unexpected parameter or precision: {name}")
        census[str(p.dtype)] += p.numel()
    if dict(census) != {"torch.bfloat16": 1118798760, "torch.float32": 336}:
        raise ValueError(f"Checkpoint dtype census differs: {dict(census)}")
    model.requires_grad_(False).eval().to(device)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    return model, tokenizer


@torch.inference_mode()
def generate_tokens(model, input_ids, max_new_tokens=64):
    """Greedy base continuation: one full prefill, then original-cache decoding."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not 0 < len(input_ids) <= model.config.max_position_embeddings:
        raise ValueError("Input must contain 1 through 1,048,576 tokens")
    device = model.model.embed_tokens.weight.device
    if len(input_ids) > 8192:
        import importlib
        runtime = importlib.import_module(type(model).__module__)
        if device.type != "cuda" or runtime.mamba_chunk_scan_combined is None:
            raise RuntimeError("Long inputs require CUDA and the Mamba scan kernel")
    context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    cache, generated = None, []
    ids = torch.as_tensor(input_ids, device=device, dtype=torch.long)[None]
    with context:
        for _ in range(max_new_tokens):
            output = model.model(input_ids=ids, past_key_values=cache,
                                 use_cache=True, return_dict=True)
            logits = model.lm_head(output.last_hidden_state[:, -1, :])[0].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite continuation logits")
            token = int(logits.argmax())
            cache = output.past_key_values
            generated.append(token)
            if token == model.config.eos_token_id:
                break
            ids = torch.tensor([[token]], device=device, dtype=torch.long)
    return generated
