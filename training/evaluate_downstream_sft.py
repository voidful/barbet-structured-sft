"""Fixed, stratified SFT evaluation using the dataset's canonical parser.

Run once per model with torchrun after training, or as an independent baseline.
Never changes model weights. Partial per-rank files support explicit resume.
"""
import argparse
from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from sft_common import TokenCorpus, sha256_file, write_json
from train_downstream_sft import assistant_loss, collate


def domain(family):
    if family.startswith('ASR_') or family in ['TRANSCRIPTION_MODE','SPEECH_TEXT_ALIGNMENT']:
        return 'ASR'
    if family.startswith('TTS_') or family == 'WRITTEN_SPOKEN_ALIGNMENT':
        return 'TTS'
    if 'OCR' in family:
        return 'OCR'
    if family.startswith('MT_') or family == 'SOURCE_TARGET_ALIGNMENT':
        return 'MT'
    return 'other'


def summarize(records):
    buckets = defaultdict(list)
    for row in records:
        for key in ['overall', 'domain:'+domain(row['task_family']), 'family:'+row['task_family'], 'serializer:'+row['serializer']]:
            buckets[key].append(row)
    result = {}
    for key, rows in sorted(buckets.items()):
        nll = [r for r in rows if r['event']=='nll']
        generation = [r for r in rows if r['event']=='generation']
        metrics = {}
        if nll:
            count = sum(r['supervised_tokens'] for r in nll)
            metrics.update(nll=sum(r['nll']*r['supervised_tokens'] for r in nll)/count,
                           nll_examples=len(nll), supervised_tokens=count)
        if generation:
            metrics['generation_examples'] = len(generation)
            for field in ['parse_ok','canonical_exact','surface_exact','eos','decision_correct','evidence_correct']:
                applicable = [r[field] for r in generation if r.get(field) is not None]
                metrics[field] = {'correct':sum(applicable),'n':len(applicable),'rate':sum(applicable)/len(applicable)} if applicable else None
        result[key] = metrics
    return result


def score_prediction(prediction, gold, serializer, parse_target):
    """Recompute correctness from strings instead of trusting stored flags."""
    gold_target = parse_target(gold,serializer)
    error = None
    try:
        target = parse_target(prediction,serializer)
    except ValueError as exc:
        target,error = None,str(exc)
    return dict(parse_ok=target is not None,canonical_exact=target==gold_target,
                surface_exact=prediction==gold,
                decision_correct=(target is not None and target.get('decision')==gold_target['decision']) if 'decision' in gold_target else None,
                evidence_correct=(target is not None and target.get('evidence')==gold_target['evidence']) if 'evidence' in gold_target else None,
                parse_error=error)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--root',default='outputs/sft_20260916')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--aggregate',action='store_true')
    args = p.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    selections = json.loads((root/'tokens'/f'{args.split}_selection.json').read_text())
    corpus = TokenCorpus(root/'tokens',args.split)
    expected = {(event,int(i)) for event in ['nll','generation'] for i in selections[event]}
    if args.aggregate:
        from tokenizers import Tokenizer
        sys.path.insert(0,str(root/'dataset/code/src'))
        from barbet_bgd.serializers_v2 import parse_target_v2
        rows = [json.loads(line) for f in sorted(output.glob('rank-*.jsonl')) for line in f.read_text().splitlines()]
        keys = [(r['event'],r['index']) for r in rows]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError('Evaluation is incomplete, duplicated or uses a different selection')
        launches = [json.loads(p.read_text()) for p in output.glob('launch-*.json')]
        model_sha = sha256_file(Path(args.model)/'model.safetensors')
        if not launches or {x['model_sha256'] for x in launches} != {model_sha}:
            raise ValueError('Missing or mixed model provenance')
        world = launches[0]['world']
        if sorted(x['rank'] for x in launches) != list(range(world)):
            raise ValueError('Missing rank provenance')
        for launch in launches:
            if (launch['world'] != world or launch['split'] != args.split
                or launch['max_new_tokens'] != args.max_new_tokens
                or launch['selections_sha256'] != sha256_file(root/'tokens'/f'{args.split}_selection.json')
                or launch['scorer_sha256'] != sha256_file(__file__)
                or launch['data_manifest_sha256'] != sha256_file(root/'tokens/manifest.json')):
                raise ValueError('Evaluation protocol mismatch')
            if not (output/f"complete-{launch['rank']}.json").exists():
                raise ValueError('Worker has not completed')
        tokenizer = Tokenizer.from_file(str(root/'base/tokenizer.json'))
        for row in rows:
            i = row['index']
            if any(row[k] != v for k,v in corpus.metadata[i].items()):
                raise ValueError('Scored metadata differs from frozen example')
            if row['event'] == 'nll':
                if row['supervised_tokens'] != int(corpus.supervised_lengths[i]) or not math.isfinite(row['nll']) or row['nll'] < 0:
                    raise ValueError('Invalid NLL or token accounting')
            else:
                seq,prompt = corpus[i]
                gold = tokenizer.decode(seq[prompt:-1].tolist(),skip_special_tokens=False)
                if row['gold'] != gold or row['prompt_tokens'] != prompt:
                    raise ValueError('Gold/prompt differs from frozen example')
                actual = score_prediction(row['prediction'],gold,row['serializer'],parse_target_v2)
                if any(row[k] != v for k,v in actual.items()):
                    raise ValueError('Stored scores differ from recomputed correctness')
        result = dict(model_sha256=model_sha,split=args.split,
                      selections_sha256=sha256_file(root/'tokens'/f'{args.split}_selection.json'),
                      scorer_sha256=sha256_file(__file__),max_new_tokens=args.max_new_tokens,
                      data_manifest_sha256=sha256_file(root/'tokens/manifest.json'),
                      metrics=summarize(rows),generated=time.time())
        write_json(output/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    rank,world,local = (int(os.environ.get(k,v)) for k,v in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    model_path = Path(args.model).resolve()
    sys.path.insert(0,str(model_path))
    sys.path.insert(0,str(root/'dataset/code/src'))
    from load_barbet import load_barbet, generate_tokens
    from barbet_bgd.serializers_v2 import parse_target_v2
    model, tokenizer = load_barbet(model_path,device=f'cuda:{local}')
    runtime = importlib.import_module(type(model).__module__)
    runtime.BarbetForCausalLM._chunked_shifted_loss = assistant_loss
    metadata = dict(model=str(model_path),model_sha256=sha256_file(model_path/'model.safetensors'),
                    rank=rank,world=world,split=args.split,max_new_tokens=args.max_new_tokens,
                    selections_sha256=sha256_file(root/'tokens'/f'{args.split}_selection.json'),
                    scorer_sha256=sha256_file(__file__),data_manifest_sha256=sha256_file(root/'tokens/manifest.json'))
    launch_path = output/f'launch-{rank}.json'
    path = output/f'rank-{rank}.jsonl'
    done = set()
    if path.exists():
        if not args.resume or json.loads(launch_path.read_text()) != metadata:
            raise ValueError('Existing eval output: explicit resume with identical provenance required')
        done = {(r['event'],r['index']) for r in map(json.loads,path.read_text().splitlines())}
    write_json(launch_path,metadata)
    with torch.inference_mode(),path.open('a') as f:
        for event in ['nll','generation']:
            for i in selections[event][rank::world]:
                if (event,i) in done:
                    continue
                row = dict(event=event,index=i,**corpus.metadata[i])
                start = time.time()
                if event == 'nll':
                    ids,labels = collate(corpus,[i],tokenizer.token_to_id('<pad>'),f'cuda:{local}')
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        value = model(input_ids=ids,labels=labels,use_cache=False,logits_to_keep=1,loss_chunk_size=256).loss
                    if not torch.isfinite(value):
                        raise ValueError('Nonfinite eval loss')
                    row.update(nll=float(value),supervised_tokens=int(corpus.supervised_lengths[i]))
                else:
                    seq,prompt_length = corpus[i]
                    prompt_ids = seq[:prompt_length].tolist()
                    gold = tokenizer.decode(seq[prompt_length:-1].tolist(),skip_special_tokens=False)
                    generated = generate_tokens(model,prompt_ids,max_new_tokens=args.max_new_tokens)
                    ended = bool(generated and generated[-1]==model.config.eos_token_id)
                    prediction = tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False)
                    row.update(prediction=prediction,gold=gold,
                               **score_prediction(prediction,gold,row['serializer'],parse_target_v2),
                               eos=ended,generated_tokens=len(generated),prompt_tokens=prompt_length,
                               )
                row['seconds'] = time.time()-start
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(json.dumps(dict(rank=rank,event=event,index=i,seconds=row['seconds'])),flush=True)
    write_json(output/f'complete-{rank}.json',dict(rank=rank,world=world,finished=time.time()))


if __name__ == '__main__':
    main()
