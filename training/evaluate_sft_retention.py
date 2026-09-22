"""Paired, text-only retention and fresh-probe evaluation for the SFT release.

The existing development holdout is all-token BOS-conditioned BPB, without a
chat template or truncation. Fresh probes use the exact SFT chat template.
Raw development text is not copied into result artifacts.
"""
import argparse
from collections import Counter, defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tokenizers import Tokenizer

from sft_common import render_prompt, sha256_file, write_json
from train_downstream_sft import assistant_loss


def prepare_inputs(registry, tokenizer_path):
    holdout = Path(registry['retention_path'])
    probes_path = Path(registry['fresh_probe_path'])
    for path, key in [(holdout,'retention_sha256'),(probes_path,'fresh_probe_sha256')]:
        if sha256_file(path) != registry[key]:
            raise ValueError(f'Frozen holdout changed: {path}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = [json.loads(line) for line in holdout.read_text().splitlines()]
    if len(rows) != registry['retention_rows'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Holdout membership is missing or duplicated')
    encoded = tokenizer.encode_batch([r['text'] for r in rows],add_special_tokens=False)
    prepared = []
    for r, encoding in zip(rows,encoded):
        if len(encoding.ids) != r['token_count'] or not 0 < len(encoding.ids) <= 4095:
            raise ValueError(f"Holdout tokenization differs: {r['id']}")
        prepared.append(dict(id=r['id'],category=r['category'],
                             input_ids=[tokenizer.token_to_id('<s>')]+encoding.ids,
                             tokens=len(encoding.ids),utf8_bytes=len(r['text'].encode('utf-8'))))
    probe_doc = json.loads(probes_path.read_text())
    if len(probe_doc['probes']) != registry['fresh_probe_count']:
        raise ValueError('Fresh probe count differs')
    return prepared, probe_doc


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def aggregate(output, prepared, probes, registry, model_sha):
    rows = [r for p in sorted(output.glob('rank-*.jsonl')) for r in read_jsonl(p)]
    expected = {('retention',r['id']) for r in prepared} | {('probe',r['id']) for r in probes['probes']}
    keys = [(r['event'],r['id']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Evaluation output is incomplete or duplicated')
    launches = [json.loads(p.read_text()) for p in output.glob('launch-*.json')]
    if not launches or {x['model_sha256'] for x in launches} != {model_sha}:
        raise ValueError('Model identity mismatch')
    world = launches[0]['world']
    if sorted(x['rank'] for x in launches) != list(range(world)):
        raise ValueError('Missing rank provenance')
    for launch in launches:
        if launch['registry'] != registry or launch['world'] != world:
            raise ValueError('Protocol differs between workers')
        if not (output/f"complete-{launch['rank']}.json").exists():
            raise ValueError('Worker has not completed')
    inputs = {r['id']:r for r in prepared}
    golds = {r['id']:r for r in probes['probes']}
    totals = defaultdict(lambda: {'rows':0,'nll_sum':0.,'tokens':0,'utf8_bytes':0})
    probe_totals = defaultdict(lambda: {'exact':0,'n':0})
    for r in rows:
        if r['event'] == 'retention':
            source = inputs[r['id']]
            if any(r[k] != source[k] for k in ['category','tokens','utf8_bytes']):
                raise ValueError('Retention membership mismatch')
            if not math.isfinite(r['nll']) or r['nll'] < 0:
                raise ValueError('Invalid retention loss')
            t = totals[r['category']]
            t['rows'] += 1
            t['nll_sum'] += r['nll'] * r['tokens']
            t['tokens'] += r['tokens']
            t['utf8_bytes'] += r['utf8_bytes']
        else:
            gold = golds[r['id']]
            exact = r['prediction'].strip() == gold['expected']
            if r['domain'] != gold['domain'] or r['exact'] != exact:
                raise ValueError('Probe membership/scoring mismatch')
            for bucket in ['overall',r['domain']]:
                probe_totals[bucket]['n'] += 1
                probe_totals[bucket]['exact'] += int(exact)
    for t in totals.values():
        t['bits_per_byte'] = t['nll_sum'] / (math.log(2)*t['utf8_bytes'])
        t['nll'] = t['nll_sum'] / t['tokens']
    summary = dict(model_sha256=model_sha,retention=dict(totals),fresh_probes=dict(probe_totals),registry=registry,
                   measured=time.time(),retention_semantics='Existing development set, BOS-conditioned all-token BPB; no truncation; not a fresh release set',
                   fresh_probe_semantics='16 text-only probes, exact-string metric can reject valid alternate translations; inspect raw predictions')
    write_json(output/'summary.json',summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--registry',default='docs/sft_20260916/holdout_registry.json')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--aggregate',action='store_true')
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    model_path, output = Path(a.model).resolve(),Path(a.output).resolve()
    registry = json.loads(Path(a.registry).read_text())
    prepared, probes = prepare_inputs(registry,model_path/'tokenizer.json')
    if a.preflight:
        result = dict(rows=len(prepared),tokens=sum(r['tokens'] for r in prepared),utf8_bytes=sum(r['utf8_bytes'] for r in prepared),
                      categories=dict(Counter(r['category'] for r in prepared)),probes=len(probes['probes']),
                      registry=registry,model_inference_run=False,cuda_initialized=torch.cuda.is_initialized())
        write_json(output/'preflight.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    model_sha = sha256_file(model_path/'model.safetensors')
    if a.aggregate:
        print(json.dumps(aggregate(output,prepared,probes,registry,model_sha),ensure_ascii=False),flush=True)
        return
    rank,world,local = (int(os.environ.get(k,v)) for k,v in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(model_path))
    from load_barbet import load_barbet,generate_tokens
    model, tokenizer = load_barbet(model_path,device=f'cuda:{local}')
    runtime = importlib.import_module(type(model).__module__)
    runtime.BarbetForCausalLM._chunked_shifted_loss = assistant_loss
    metadata = dict(model_sha256=model_sha,rank=rank,world=world,registry=registry,
                    scorer_sha256=sha256_file(__file__),generation_max_tokens=256)
    path,launch = output/f'rank-{rank}.jsonl',output/f'launch-{rank}.json'
    done = set()
    if path.exists():
        if not a.resume or json.loads(launch.read_text()) != metadata:
            raise ValueError('Existing output: explicit resume with identical provenance required')
        done = {(r['event'],r['id']) for r in read_jsonl(path)}
    write_json(launch,metadata)
    with torch.inference_mode(),path.open('a') as out:
        for r in prepared[rank::world]:
            if ('retention',r['id']) in done:
                continue
            ids = torch.tensor(r['input_ids'],device=f'cuda:{local}',dtype=torch.long)[None]
            labels = ids.clone();labels[:,0] = -100
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss = model(input_ids=ids,labels=labels,use_cache=False,logits_to_keep=1,loss_chunk_size=256).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite BPB loss')
            result = dict(event='retention',**{k:v for k,v in r.items() if k!='input_ids'},nll=float(loss))
            out.write(json.dumps(result)+'\n');out.flush()
        for r in probes['probes'][rank::world]:
            if ('probe',r['id']) in done:
                continue
            messages = [dict(role='system',content=probes['system']),dict(role='user',content=r['user']),dict(role='assistant',content='')]
            prompt = tokenizer.encode(render_prompt(messages),add_special_tokens=False).ids
            generated = generate_tokens(model,prompt,max_new_tokens=256)
            ended = bool(generated and generated[-1]==model.config.eos_token_id)
            prediction = tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False)
            result = dict(event='probe',id=r['id'],domain=r['domain'],prediction=prediction,expected=r['expected'],
                          exact=prediction.strip()==r['expected'],eos=ended,generated_tokens=len(generated))
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
    write_json(output/f'complete-{rank}.json',dict(rank=rank,world=world,finished=time.time()))
    print(json.dumps(dict(rank=rank,status='complete')),flush=True)


if __name__ == '__main__':
    main()
