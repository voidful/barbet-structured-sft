"""Full-parameter, assistant-only SFT with exact token-normalized DDP loss."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import save_file

from sft_common import CHAT_TEMPLATE, TokenCorpus, sha256_file, stratified_indices, write_json


def assistant_loss(self, hidden_states, labels, offset=1, projector=None, chunk_size=None):
    """Identical masked CE, projecting only supervised positions into vocab."""
    from torch.utils.checkpoint import checkpoint
    if projector is not None or offset != 1:
        raise ValueError('This SFT runtime requires MTP disabled and offset=1')
    targets = labels[:, 1:].reshape(-1)
    valid = targets != -100
    states = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])[valid]
    targets = targets[valid]
    if not len(targets):
        raise ValueError('No assistant tokens in batch')
    total = hidden_states.new_zeros((), dtype=torch.float32)
    def project_loss(x, y, weight):
        return torch.nn.functional.cross_entropy(torch.nn.functional.linear(x, weight).float(), y, reduction='sum')
    for start in range(0, len(targets), chunk_size or 256):
        end = start + (chunk_size or 256)
        if self.training and torch.is_grad_enabled():
            value = checkpoint(project_loss, states[start:end], targets[start:end], self.lm_head.weight, use_reentrant=False)
        else:
            value = project_loss(states[start:end], targets[start:end], self.lm_head.weight)
        total = total + value
    return total / len(targets)


def load_training_model(base, device):
    sys.path.insert(0, str(Path(base).resolve()))
    from load_barbet import load_barbet
    model, tokenizer = load_barbet(base, device)
    runtime = importlib.import_module(type(model).__module__)
    runtime.BarbetForCausalLM._chunked_shifted_loss = assistant_loss
    model.float().requires_grad_(True).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    return model, tokenizer


def collate(corpus, indices, pad, device):
    width = int(math.ceil(max(corpus.lengths[indices]) / 8) * 8)
    ids = torch.full((len(indices), width), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for row, i in enumerate(indices):
        seq, prompt = corpus[int(i)]
        ids[row, :len(seq)] = torch.from_numpy(seq.astype(np.int64))
        labels[row, prompt:len(seq)] = ids[row, prompt:len(seq)]
    # Causal right-padding cannot affect earlier valid tokens. Do not pass a
    # dense padding mask to Flash SDPA; labels exclude every padded position.
    return ids.to(device), labels.to(device)


def microbatches(corpus, indices, token_budget, max_micro):
    current, longest = [], 0
    for i in indices:
        length = int(corpus.lengths[i])
        new_max = max(longest, length)
        if current and (len(current) >= max_micro or new_max * (len(current) + 1) > token_budget):
            yield current
            current, longest = [], 0
        current.append(int(i))
        longest = max(longest, length)
    if current:
        yield current


def epoch_batches(corpus, batch_size, seed, world):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus))
    batches = []
    # Length sorting inside shuffled pools balances GPU work without a
    # long-first curriculum. Every example occurs exactly once per epoch.
    for start in range(0, len(order), batch_size * 64):
        pool = order[start:start + batch_size * 64]
        pool = pool[np.argsort(corpus.lengths[pool], kind='stable')]
        batches.extend(pool[i:i+batch_size] for i in range(0, len(pool), batch_size))
    if len(batches[-1]) < world:
        tail = batches.pop()
        batches[-1] = np.concatenate([batches[-1], tail])
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, corpus, indices, rank, world, device):
    model.eval()
    families = sorted({m['task_family'] for m in corpus.metadata})
    lookup = {k:i for i,k in enumerate(families)}
    totals = torch.zeros((len(families), 3), dtype=torch.float64, device=device)
    for i in indices[rank::world]:
        ids, labels = collate(corpus, [i], model.config.pad_token_id, device)
        count = int(corpus.supervised_lengths[i])
        with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = model(input_ids=ids, labels=labels, use_cache=False, logits_to_keep=1, loss_chunk_size=256)
        totals[lookup[corpus.metadata[i]['task_family']]] += torch.stack([
            result.loss.double() * count,
            torch.tensor(count, device=device, dtype=torch.float64),
            torch.ones((), device=device, dtype=torch.float64)])
    if world > 1:
        dist.all_reduce(totals)
    values = totals.cpu().tolist()
    summed = totals.sum(0).cpu().tolist()
    model.train()
    return {'nll': summed[0]/summed[1], 'supervised_tokens': int(summed[1]), 'examples': int(summed[2]),
            'by_family': {f: {'nll': v[0]/v[1], 'tokens': int(v[1]), 'examples': int(v[2])}
                          for f,v in zip(families, values) if v[1]}}


def save_checkpoint(model, optimizer, root, step, config, tokens, supervised, examples):
    destination = root / f'checkpoint-{step}'
    tmp = root / f'.checkpoint-{step}.tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'config': config, 'tokens': tokens, 'supervised_tokens': supervised,
                'examples': examples}, tmp / 'training.pt')
    write_json(tmp/'complete.json', dict(step=step, tokens=tokens, examples=examples))
    tmp.rename(destination)
    write_json(root/'latest.json', dict(path=str(destination.resolve()), step=step))
    checkpoints = sorted(root.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
    for old in checkpoints[:-3]:
        shutil.rmtree(old)


def export_model(model, base, output):
    output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, value in model.state_dict().items():
        if name == 'lm_head.weight':
            continue
        dtype = torch.float32 if name.endswith(('.A_log', '.D')) else torch.bfloat16
        tensors[name] = value.detach().to(device='cpu', dtype=dtype).contiguous()
    save_file(tensors, str(output/'model.safetensors'), metadata={'format':'pt'})
    names = ['config.json','configuration_barbet.py','modeling_barbet.py','megatron_long_context.py',
             'load_barbet.py','requirements-inference.txt','tokenizer.json','tokenizer_config.json',
             'special_tokens_map.json','added_tokens.json','vocab.json','merges.txt']
    for name in names:
        shutil.copy2(base/name, output/name)
    tok = json.loads((output/'tokenizer_config.json').read_text())
    tok['chat_template'] = CHAT_TEMPLATE
    write_json(output/'tokenizer_config.json', tok)
    (output/'chat_template.jinja').write_text(CHAT_TEMPLATE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='docs/sft_20260916/config.json')
    p.add_argument('--run-dir')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--skip-eval', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    root = Path(config['root']).resolve()
    run = Path(a.run_dir or root/'run').resolve()
    run.mkdir(parents=True, exist_ok=True)
    rank, world, local = (int(os.environ.get(k, v)) for k,v in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60), device_id=device)
    torch.manual_seed(config['seed'])
    manifest = json.loads((root/'tokens/manifest.json').read_text())
    if manifest['model_revision'] != config['model_revision'] or manifest['dataset_revision'] != config['dataset_revision']:
        raise ValueError('Revision mismatch')
    train, validation = TokenCorpus(root/'tokens','train'), TokenCorpus(root/'tokens','validation')
    batches = epoch_batches(train, config['global_batch_examples'], config['seed'], world)
    max_steps = min(a.max_steps or len(batches), len(batches))
    warmup = math.ceil(len(batches)*config['warmup_ratio'])
    model, tokenizer = load_training_model(root/'base', device)
    params = [dict(params=[v for k,v in model.named_parameters() if v.ndim >= 2], weight_decay=config['weight_decay']),
              dict(params=[v for k,v in model.named_parameters() if v.ndim < 2], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], betas=tuple(config['betas']), eps=1e-8, fused=True)
    start_step = tokens = supervised = examples = 0
    if a.resume:
        latest = json.loads((run/'latest.json').read_text())
        checkpoint = torch.load(Path(latest['path'])/'training.pt', map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume config differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step, tokens, supervised, examples = (checkpoint[k] for k in ['step','tokens','supervised_tokens','examples'])
        del checkpoint
    elif (run/'latest.json').exists():
        raise ValueError('Existing run: use --resume')
    if rank == 0:
        write_json(run/'launch.json',dict(pid=os.getpid(), world=world, config=config, steps=len(batches),
                                         start_step=start_step, warmup_steps=warmup, data_manifest_sha256=sha256_file(root/'tokens/manifest.json'),
                                         trainer_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                         trainer_sha256=sha256_file(__file__),
                                         common_sha256=sha256_file(Path(__file__).with_name('sft_common.py')),
                                         torch_version=torch.__version__, cuda_version=torch.version.cuda,
                                         device_name=torch.cuda.get_device_name(local)))
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
    eval_indices = stratified_indices(validation, config['validation_per_family'], config['seed'])

    def log(record):
        if rank == 0:
            record.update(time=time.time())
            with (run/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_json(run/'status.json', record)

    if not a.skip_eval and start_step == 0:
        log(dict(event='validation', step=0, **evaluate(model, validation, eval_indices, rank, world, device)))
    started = time.time()
    for step in range(start_step, max_steps):
        batch = batches[step]
        indices = batch[rank::world]
        assert len(indices) > 0
        global_count = int(train.supervised_lengths[batch].sum())
        global_tokens = int(train.lengths[batch].sum())
        lr_factor = (step+1)/warmup if step < warmup else config['min_lr_ratio'] + (1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warmup)/(len(batches)-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = config['learning_rate']*lr_factor
        optimizer.zero_grad(set_to_none=True)
        local_loss = torch.zeros((),device=device)
        micros = list(microbatches(train, indices, config['micro_token_budget'], config['max_micro_examples']))
        step_start = time.time()
        for index, micro in enumerate(micros):
            sync = ddp.no_sync() if world > 1 and index < len(micros)-1 else nullcontext()
            ids, labels = collate(train, micro, tokenizer.token_to_id('<pad>'), device)
            count = int(train.supervised_lengths[micro].sum())
            with sync, torch.autocast('cuda',dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = ddp(input_ids=ids, labels=labels, use_cache=False, logits_to_keep=1, loss_chunk_size=256)
                loss = result.loss * (count * world / global_count)
                local_loss += result.loss.detach() * count
                loss.backward()
            del result, loss, ids, labels
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(local_loss)
        if not torch.isfinite(local_loss):
            raise RuntimeError('Nonfinite training loss')
        tokens += global_tokens
        supervised += global_count
        examples += len(batch)
        elapsed = time.time()-step_start
        log(dict(event='train', step=step+1, total_steps=len(batches), nll=float(local_loss)/global_count,
                 grad_norm=float(norm), lr=optimizer.param_groups[0]['lr'], tokens=tokens,
                 supervised_tokens=supervised, examples=examples, step_tokens=global_tokens,
                 step_seconds=elapsed, tokens_per_second=global_tokens/elapsed,
                 max_length=int(train.lengths[batch].max()), peak_gb=torch.cuda.max_memory_allocated()/1e9))
        if (step+1) % config['save_steps'] == 0 or step+1 == max_steps:
            if rank == 0:
                save_checkpoint(model, optimizer, run, step+1, config, tokens, supervised, examples)
            if world > 1:
                dist.barrier()
        if not a.skip_eval and ((step+1) % config['eval_steps'] == 0 or step+1 == max_steps):
            log(dict(event='validation',step=step+1, **evaluate(model, validation, eval_indices, rank, world, device)))
    if max_steps == len(batches) and rank == 0:
        if examples != len(train) or tokens != manifest['splits']['train']['tokens'] or supervised != manifest['splits']['train']['supervised_tokens']:
            raise RuntimeError('Epoch coverage mismatch')
        export_model(model, root/'base', run/'final')
        write_json(run/'finished.json',dict(step=max_steps, tokens=tokens, supervised_tokens=supervised,
                                          examples=examples, seconds=time.time()-started, status='trained_pending_release_evaluation'))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
