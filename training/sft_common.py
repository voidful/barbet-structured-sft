"""Pinned Barbet SFT data format; no truncation or cross-example state sharing."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROLES = {"system": "<|system|>", "user": "<|user_channel|>", "assistant": "<|assistant_channel|>"}
CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{{ {'system': '<|system|>', 'user': '<|user_channel|>', 'assistant': '<|assistant_channel|>'}[message['role']] }}{{ message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant_channel|>' }}{% endif %}"


def render_prompt(messages):
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected one system/user/assistant exchange')
    return '<s>' + ''.join(ROLES[m['role']] + m['content'] + '\n' for m in messages[:2]) + ROLES['assistant']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


class TokenCorpus:
    def __init__(self, root, split):
        self.root = Path(root)
        self.shards = sorted(self.root.glob(split + '-*.npz'))
        self.maps, self.offsets, self.prompts, self.metadata = [], [], [], []
        self.ends = []
        lengths = []
        for p in self.shards:
            a = np.load(p)
            self.maps.append(np.memmap(p.with_suffix('.bin'), dtype=np.int32, mode='r'))
            self.offsets.append(a['offsets'])
            self.prompts.append(a['prompt_lengths'])
            lengths.extend(np.diff(a['offsets']).tolist())
            self.ends.append(len(lengths))
            self.metadata.extend(json.loads(p.with_suffix('.metadata.json').read_text()))
        self.lengths = np.array(lengths)
        self.prompt_lengths = np.concatenate(self.prompts)
        self.supervised_lengths = self.lengths - self.prompt_lengths
        assert len(self.metadata) == len(self.lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        shard = int(np.searchsorted(self.ends, i, side='right'))
        j = i - (self.ends[shard - 1] if shard else 0)
        ids = self.maps[shard][self.offsets[shard][j]:self.offsets[shard][j + 1]].copy()
        return ids, int(self.prompts[shard][j])


def stratified_indices(corpus, per_family, seed=42):
    rng = np.random.default_rng(seed)
    families = {}
    for i, m in enumerate(corpus.metadata):
        families.setdefault(m['task_family'], []).append(i)
    return sorted(int(i) for k in sorted(families) for i in rng.choice(families[k], min(per_family, len(families[k])), replace=False))
