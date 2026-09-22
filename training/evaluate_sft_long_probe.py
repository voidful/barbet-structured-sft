"""Small, fixed raw-text retrieval probe at 8K and 128K; no 1M claim."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend

from sft_common import sha256_file, write_json
from train_downstream_sft import assistant_loss


def prepare(root):
    target = root/'long_probe_inputs'
    if (target/'manifest.json').exists():
        return json.loads((target/'manifest.json').read_text())
    target.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(root/'base/tokenizer.json'))
    encode = lambda text: tokenizer.encode(text,add_special_tokens=False).ids
    header = [tokenizer.token_to_id('<s>')]+encode('以下是收件文件。請忠實保留正式收件編號中的英文字母、數字及連字號。\n')
    footer = encode('\n請只輸出文件中的正式收件編號，不加說明。\n答案：')
    filler = encode('附件摘要：本段僅記錄一般行政說明，未提供本次核對的有效編號。\n')
    period = encode('.')
    assert len(period)==1
    cases=[]
    for length in [8192,131072]:
        for index,(fraction,code) in enumerate(zip([.05,.33,.67,.95],['AX-3971-K','BZ-6428-P','CT-1853-R','DU-7264-M'])):
            evidence = encode('\n正式收件編號：'+code+'。\n')
            available = length-len(header)-len(footer)-len(evidence)
            left = int(available*fraction)
            def padding(n):
                return filler*(n//len(filler))+period*(n%len(filler))
            ids = header+padding(left)+evidence+padding(available-left)+footer
            assert len(ids)==length
            # Segment-wise BPE composition can differ from encoding the full
            # visible string. Canonicalize, then fill the shortfall with blank
            # lines. Each newline encodes independently with this tokenizer.
            prefix_text=tokenizer.decode(ids[:-len(footer)],skip_special_tokens=False)
            footer_text=tokenizer.decode(footer,skip_special_tokens=False)
            extra_lines=0
            for _ in range(8):
                ids=encode(prefix_text+'\n'*extra_lines+footer_text)
                difference=length-len(ids)
                if difference==0:
                    break
                extra_lines+=difference
                if extra_lines<0:
                    raise ValueError('Canonical prompt cannot be padded to target length')
            assert len(ids)==length
            assert encode(tokenizer.decode(ids,skip_special_tokens=False))==ids
            evidence_start=len(encode(prefix_text[:prefix_text.index('正式收件編號：'+code)]))
            case_id = f'{length}-{index}'
            path=target/(case_id+'.npy');np.save(path,np.asarray(ids,dtype=np.int32))
            cases.append(dict(id=case_id,length=length,requested_position=fraction,
                              evidence_start_token=evidence_start,evidence_position=evidence_start/length,expected=code,
                              answer_ids=encode(code),input_sha256=sha256_file(path)))
    manifest=dict(cases=cases,tokenizer_sha256=sha256_file(root/'base/tokenizer.json'),
                  max_new_tokens=64,protocol='BOS-conditioned raw continuation, no chat template; 4 fixed codes at 4 positions per length',
                  limitations='Eight synthetic repeated-background retrieval cases; no natural-document, general reasoning, or 1M capability claim',
                  prepared=time.time(),scorer_sha256=sha256_file(__file__))
    write_json(target/'manifest.json',manifest)
    write_json('docs/sft_20260916/long_probe_plan.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    args=parser.parse_args()
    root=Path('outputs/sft_20260916').resolve()
    manifest=prepare(root)
    if manifest['scorer_sha256']!=sha256_file(__file__):
        raise ValueError('Frozen probe scorer changed')
    if args.prepare:
        print(json.dumps(dict(cases=len(manifest['cases']),lengths=[8192,131072],model_inference_run=False)),flush=True)
        return
    output=root/'evaluation/long_probe'
    output.mkdir(parents=True,exist_ok=True)
    if args.aggregate:
        rows=[r for p in output.glob('rank-*.json') for r in json.loads(p.read_text())]
        expected={(label,c['id']) for label in ['base','candidate'] for c in manifest['cases']}
        keys=[(r['model'],r['case']['id']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError('Long-probe membership incomplete or duplicated')
        source={c['id']:c for c in manifest['cases']}
        hashes={label:sha256_file(folder/'model.safetensors') for label,folder in [('base',root/'base'),('candidate',root/'run/final')]}
        for row in rows:
            if row['case']!=source[row['case']['id']] or row['model_sha256']!=hashes[row['model']]:
                raise ValueError('Scored case or model differs')
            if row['manifest_sha256']!=sha256_file(root/'long_probe_inputs/manifest.json') or not np.isfinite(row['nll']):
                raise ValueError('Protocol or finite-loss check failed')
            if row['exact']!=(row['prediction'].strip()==row['case']['expected']):
                raise ValueError('Stored exact-match flag differs')
        summary={}
        for label in ['base','candidate']:
            summary[label]={}
            for length in [8192,131072]:
                subset=[r for r in rows if r['model']==label and r['case']['length']==length]
                summary[label][str(length)]=dict(exact=sum(r['exact'] for r in subset),n=len(subset),
                                                mean_answer_nll=sum(r['nll'] for r in subset)/len(subset))
        report=dict(summary=summary,model_sha256=hashes,plan=manifest,rows=rows)
        write_json(output/'summary.json',report);write_json('docs/sft_20260916/long_probe_summary.json',report)
        print(json.dumps(summary),flush=True)
        return
    rank,world,local=(int(os.environ.get(k,v)) for k,v in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    path=output/f'rank-{rank}.json'
    if path.exists():
        raise FileExistsError('Existing long-probe results must be inspected before rerunning')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    sys.path.insert(0,str(root/'base'))
    from load_barbet import load_barbet,generate_tokens
    results=[]
    for label,folder in [('base',root/'base'),('candidate',root/'run/final')]:
        model,tokenizer=load_barbet(folder,device=f'cuda:{local}')
        runtime=importlib.import_module(type(model).__module__)
        runtime.BarbetForCausalLM._chunked_shifted_loss=assistant_loss
        model_sha=sha256_file(folder/'model.safetensors')
        for case in manifest['cases'][rank::world]:
            input_path=root/'long_probe_inputs'/(case['id']+'.npy')
            if sha256_file(input_path)!=case['input_sha256']:
                raise ValueError('Frozen long input changed')
            prompt=np.load(input_path).tolist()
            ids=torch.tensor(prompt+case['answer_ids'],device=f'cuda:{local}')[None]
            labels=ids.clone();labels[:,:len(prompt)]=-100
            with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss=model(input_ids=ids,labels=labels,use_cache=False,logits_to_keep=1,loss_chunk_size=256).loss
            generated=generate_tokens(model,prompt,max_new_tokens=64)
            eos=bool(generated and generated[-1]==model.config.eos_token_id)
            prediction=tokenizer.decode(generated[:-1] if eos else generated,skip_special_tokens=False)
            results.append(dict(model=label,model_sha256=model_sha,case=case,nll=float(loss),prediction=prediction,
                                exact=prediction.strip()==case['expected'],eos=eos,
                                manifest_sha256=sha256_file(root/'long_probe_inputs/manifest.json')))
            write_json(output/f'.rank-{rank}.partial.json',results)
        del model
        torch.cuda.empty_cache()
    write_json(path,results)
    print(json.dumps(dict(rank=rank,completed=len(results))),flush=True)


if __name__=='__main__':
    main()
