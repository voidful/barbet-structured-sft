"""Audit and tokenize every row of the pinned dataset, preserving all tokens."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from sft_common import ROLES, TokenCorpus, render_prompt, sha256_file, stratified_indices, write_json


def prepare_shard(args):
    source, tokenizer_path, dest = map(Path, args)
    out = dest / source.stem
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id('</s>')
    offsets, prompt_lengths, meta = [0], [], []
    groups, prompts, message_hashes = set(), set(), set()
    families, primitives, serializers = Counter(), Counter(), Counter()
    with out.with_suffix('.bin.tmp').open('wb') as output:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=128):
            rows = batch.to_pylist()
            texts = []
            for r in rows:
                for m in r['messages']:
                    if not m['content'] or any(s in m['content'] for s in [*ROLES.values(), '<s>', '</s>', '<pad>']):
                        raise ValueError(f"Empty content or injected chat marker: {r['record_id']}")
                texts.extend([render_prompt(r['messages']), r['messages'][-1]['content']])
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for i, r in enumerate(rows):
                prompt, response = encodings[2 * i].ids, encodings[2 * i + 1].ids
                ids = prompt + response + [eos]
                if len(ids) > 1048576 or not response:
                    raise ValueError(f"Unsupported sequence: {r['record_id']}")
                np.asarray(ids, dtype=np.int32).tofile(output)
                offsets.append(offsets[-1] + len(ids))
                prompt_lengths.append(len(prompt))
                m = {k: r[k] for k in ['record_id', 'source_group', 'task_family', 'primitive', 'serializer', 'messages_sha256']}
                meta.append(m)
                groups.add(r['source_group'])
                import hashlib
                prompts.add(hashlib.sha256(texts[2 * i].encode()).hexdigest())
                message_hashes.add(r['messages_sha256'])
                families[r['task_family']] += 1
                primitives[r['primitive']] += 1
                serializers[r['serializer']] += 1
    out.with_suffix('.bin.tmp').replace(out.with_suffix('.bin'))
    np.savez(out.with_suffix('.npz'), offsets=np.array(offsets, dtype=np.int64), prompt_lengths=np.array(prompt_lengths, dtype=np.int32))
    write_json(out.with_suffix('.metadata.json'), meta)
    return dict(shard=source.stem, split=source.stem.split('-')[0], rows=len(meta),
                tokens=offsets[-1], supervised_tokens=offsets[-1]-sum(prompt_lengths),
                max_length=max(np.diff(offsets)).item(), source_sha256=sha256_file(source),
                token_sha256=sha256_file(out.with_suffix('.bin')), families=dict(families),
                primitives=dict(primitives), serializers=dict(serializers), groups=sorted(groups),
                prompt_hashes=sorted(prompts), message_hashes=sorted(message_hashes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/sft_20260916')
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    root = Path(a.root).resolve()
    dest = root / 'tokens'
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted((root / 'dataset/data').glob('*.parquet'))
    if len(sources) != 100:
        raise ValueError(f'Expected all 100 shards, found {len(sources)}')
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        futures = [pool.submit(prepare_shard, (str(s), str(root/'base/tokenizer.json'), str(dest))) for s in sources]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(json.dumps({k: r[k] for k in ['shard', 'rows', 'tokens', 'max_length']}), flush=True)
    split_hashes = {k: defaultdict(set) for k in ['groups', 'prompt_hashes', 'message_hashes']}
    for r in results:
        for k in split_hashes:
            split_hashes[k][r['split']].update(r.pop(k))
    overlap = {}
    for k, buckets in split_hashes.items():
        for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            overlap[f'{k}:{left}:{right}'] = len(buckets[left] & buckets[right])
    if any(overlap.values()):
        raise ValueError(f'Split leakage: {overlap}')
    summary = {}
    for split, expected in [('train', 980000), ('validation', 10000), ('test', 10000)]:
        ds = TokenCorpus(dest, split)
        assert len(ds) == expected
        summary[split] = dict(rows=len(ds), tokens=int(ds.lengths.sum()),
                              supervised_tokens=int(ds.supervised_lengths.sum()),
                              length_quantiles=dict(zip(['min','p50','p90','p95','p99','max'], np.quantile(ds.lengths,[0,.5,.9,.95,.99,1]).tolist())),
                              families=dict(Counter(m['task_family'] for m in ds.metadata)))
        if split != 'train':
            write_json(dest/f'{split}_selection.json', dict(nll=stratified_indices(ds, 20), generation=stratified_indices(ds, 8), seed=42))
    manifest = dict(model_revision='4dbb35216a32de59570a3f0d164804f25c82009d',
                    dataset_revision='0bf2389cbcd6e828af9df0676aed56abc61c4c2e',
                    tokenizer_sha256=sha256_file(root/'base/tokenizer.json'),
                    truncation=False, assistant_only=True, overlaps=overlap, splits=summary,
                    shards=sorted(results, key=lambda r:r['shard']))
    write_json(dest/'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
