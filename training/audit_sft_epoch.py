"""Compare every logged training step with the frozen epoch ordering."""
import json
import math
from pathlib import Path

import numpy as np

from sft_common import sha256_file, write_json
from train_downstream_sft import epoch_batches


def main():
    root = Path('outputs/sft_20260916')
    launch = json.loads((root/'run/launch.json').read_text())
    for path, key in [('scripts/train_downstream_sft.py', 'trainer_sha256'),
                      ('scripts/sft_common.py', 'common_sha256'),
                      (root/'tokens/manifest.json', 'data_manifest_sha256')]:
        if sha256_file(path) != launch[key]:
            raise ValueError('Executed source or data manifest changed: '+str(path))
    if launch['start_step'] != 0:
        raise ValueError('This audit expects the recorded fresh full-epoch run')

    class Lengths:
        def __len__(self):
            return len(self.lengths)

    corpus = Lengths()
    arrays = [np.load(p) for p in sorted((root/'tokens').glob('train-*.npz'))]
    corpus.lengths = np.concatenate([np.diff(a['offsets']) for a in arrays])
    supervised = corpus.lengths - np.concatenate([a['prompt_lengths'] for a in arrays])
    config = launch['config']
    batches = epoch_batches(corpus, config['global_batch_examples'], config['seed'], launch['world'])
    order = np.concatenate(batches)
    if not np.array_equal(np.sort(order), np.arange(len(corpus))):
        raise ValueError('Epoch ordering repeats or omits training rows')
    records = [json.loads(line) for line in (root/'run/metrics.jsonl').read_text().splitlines()]
    records = [r for r in records if r['event'] == 'train']
    if len(records) != len(batches) or len(batches) != launch['steps']:
        raise ValueError('Training step count differs from the frozen epoch')
    totals = dict(examples=0, tokens=0, supervised_tokens=0)
    for step, (batch, row) in enumerate(zip(batches, records), 1):
        step_tokens = int(corpus.lengths[batch].sum())
        totals['examples'] += len(batch)
        totals['tokens'] += step_tokens
        totals['supervised_tokens'] += int(supervised[batch].sum())
        expected = dict(step=step, total_steps=len(batches), step_tokens=step_tokens,
                        max_length=int(corpus.lengths[batch].max()), **totals)
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError(f'Logged coverage differs at step {step}')
        if not all(math.isfinite(row[k]) for k in ['nll', 'grad_norm', 'lr']):
            raise ValueError(f'Nonfinite training value at step {step}')
    finished = json.loads((root/'run/finished.json').read_text())
    manifest = json.loads((root/'tokens/manifest.json').read_text())['splits']['train']
    if any(finished[k] != v for k, v in totals.items()):
        raise ValueError('Final receipt differs from step accounting')
    if totals != dict(examples=manifest['rows'], tokens=manifest['tokens'],
                      supervised_tokens=manifest['supervised_tokens']):
        raise ValueError('Final accounting differs from data manifest')
    report = dict(passed=True, steps_checked=len(records), unique_examples=len(order),
                  every_row_once=True, every_step_coverage_matches=True,
                  all_logged_losses_gradients_learning_rates_finite=True, **totals,
                  trainer_sha256=launch['trainer_sha256'], common_sha256=launch['common_sha256'],
                  data_manifest_sha256=launch['data_manifest_sha256'],
                  metrics_sha256=sha256_file(root/'run/metrics.jsonl'),
                  scope='Deterministic epoch ordering and logged counters; not an independent replay of optimizer updates')
    write_json('docs/sft_20260916/epoch_coverage_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
