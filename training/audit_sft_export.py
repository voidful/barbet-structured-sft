"""Verify every exported weight against its FP32 training checkpoint on CPU."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

# Optional Mamba dependencies import quack, whose autotuner probes CUDA at
# import time when GPUs are visible. Isolate this CPU-only audit before torch
# or model imports, as the checkpoint generation monitor already does.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from safetensors import safe_open

from sft_common import CHAT_TEMPLATE, sha256_file, write_json
from train_downstream_sft import export_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--base',default='outputs/sft_20260916/base')
    parser.add_argument('--export-first',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint_path,model_path,base = map(Path,[args.checkpoint,args.model,args.base])
    checkpoint = torch.load(checkpoint_path/'training.pt',map_location='cpu',weights_only=False)
    step = checkpoint['step']
    reference = checkpoint['model']
    del checkpoint['optimizer']
    sys.path.insert(0,str(base.resolve()))
    from load_barbet import load_barbet
    if args.export_first:
        if (model_path/'model.safetensors').exists():
            raise FileExistsError('Refusing to overwrite an existing model during rehearsal')
        model,_ = load_barbet(base,device='cpu')
        model.float()
        model.load_state_dict(reference,strict=True)
        export_model(model,base,model_path)
        del model
    expected = set(reference)-{'lm_head.weight'}
    dtypes = Counter()
    changed = 0
    with safe_open(str(model_path/'model.safetensors'),framework='pt',device='cpu') as actual:
        if set(actual.keys()) != expected:
            raise ValueError('Exported tensor names differ from the checkpoint')
        for name in sorted(expected):
            dtype = torch.float32 if name.endswith(('.A_log','.D')) else torch.bfloat16
            value = actual.get_tensor(name)
            if value.dtype != dtype or value.shape != reference[name].shape:
                raise ValueError('Exported dtype or shape differs: '+name)
            if not torch.isfinite(value).all() or not torch.equal(value,reference[name].to(dtype)):
                raise ValueError('Exported weight differs from checkpoint: '+name)
            dtypes[str(dtype)] += value.numel()
            changed += 1
    if dict(dtypes) != {'torch.bfloat16':1118798760,'torch.float32':336}:
        raise ValueError('Parameter census differs from the base architecture')
    if sha256_file(model_path/'tokenizer.json') != sha256_file(base/'tokenizer.json'):
        raise ValueError('Tokenizer vocabulary changed')
    config = json.loads((model_path/'tokenizer_config.json').read_text())
    if config['chat_template'] != CHAT_TEMPLATE:
        raise ValueError('Exported chat template differs from training')
    model,tokenizer = load_barbet(model_path,device='cpu')
    ids = torch.tensor(tokenizer.encode('請保留原文的數字與標點。',add_special_tokens=False).ids)[None]
    with torch.inference_mode():
        output = model(input_ids=ids,logits_to_keep=1,use_cache=True)
    if not torch.isfinite(output.logits).all() or output.past_key_values is None:
        raise ValueError('CPU inference/cache check failed')
    if torch.cuda.is_initialized():
        raise ValueError('CPU export audit unexpectedly initialized CUDA')
    report = dict(checkpoint=str(checkpoint_path.resolve()),step=step,model=str(model_path.resolve()),
                  scope='full_epoch_export' if step==7657 else 'smoke_export_rehearsal',
                  tensors_checked=changed,stored_parameters=sum(dtypes.values()),dtype_census=dict(dtypes),
                  all_weights_match_cast_training_checkpoint=True,all_weights_finite=True,
                  cpu_inference_finite=True,cpu_cache_present=True,cuda_initialized=False,
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  model_sha256=sha256_file(model_path/'model.safetensors'),
                  tokenizer_sha256=sha256_file(model_path/'tokenizer.json'),passed=True)
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
