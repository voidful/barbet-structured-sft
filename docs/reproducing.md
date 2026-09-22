# Reproduction and training

## What is publicly runnable

- Load the public SFT checkpoint and run the eight bundled example requests.
- Inspect all published metrics, exact source hashes and training settings.
- Re-evaluate the SFT checkpoint on the pinned public synthetic dataset,
  using its original parser and the archived evaluation source.

**Exact full retraining also requires the original pre-SFT base**, hosted
in a separate access-controlled repository. Publishing the SFT checkpoint
does not make that base public. Starting from the SFT weights would be
continued fine-tuning, not a reproduction of the original experiment.
The historical general-text retention corpus is not distributed; its
aggregate results and data identity remain available in the reports.

## Frozen identifiers

- Base: `OpenFormosa/barbet-1b-base` at `4dbb35216a32de59570a3f0d164804f25c82009d`.
- Dataset: `voidful/barbet-sft` at `0bf2389cbcd6e828af9df0676aed56abc61c4c2e`.
- Public release weights SHA-256:
  `ea16e76ea8539ee9ad307ef4e202a94a2f86a96ccdbf22b2c34b95be0523895d`.

## Original recipe

8 H200 NVL; full-parameter AdamW; FP32 parameters/gradient reduction/Adam
states; BF16 autocast. Peak LR 1e-5, 230-step warmup (3%), cosine decay to
1e-6, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 for matrices
and zero for vectors, gradient clipping 1.0. Global batch 128 examples;
the final partial batch is shuffled into the epoch. Adaptive microbatches
use at most four examples or 8,192 tokens; longer examples run alone,
in full. Assistant answer and EOS tokens contribute to loss. There is
no cross-example packing and no sequence truncation.

## Commands, when base access is available

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_reproduction.py --include-base
python training/prepare_downstream_sft.py --root outputs/sft_20260916
torchrun --standalone --nproc_per_node=8 training/train_downstream_sft.py \
  --config training/config.json
```

`prepare_reproduction.py` downloads only the pinned dataset and, when
explicitly requested, the pinned base. It does not launch training.
Authenticate with HF only if needed for the base's existing permissions.
Preparation requires all 100 dataset shards and verifies split overlap.
Training all 980,000 examples yields 7,657 optimizer steps. It is a
substantial GPU/disk workload; CI never executes these commands.

For evaluation of the public SFT alone:

```bash
python scripts/prepare_reproduction.py --include-sft
python training/prepare_downstream_sft.py --root outputs/sft_20260916
torchrun --standalone --nproc_per_node=8 training/evaluate_downstream_sft.py \
  --model outputs/sft_20260916/public_sft \
  --output outputs/sft_20260916/evaluation/public_sft
python training/evaluate_downstream_sft.py \
  --model outputs/sft_20260916/public_sft \
  --output outputs/sft_20260916/evaluation/public_sft --aggregate
```

In SFT-only mode, only tokenizer/config assets are placed under `base/`
to support data preparation. No pre-SFT weights are substituted there.
The manifest's historical base revision identifies the original run;
the evaluator separately hashes the actual model being evaluated.
Private retention evaluation needs your own authorized registry/data;
it cannot reproduce the historical numbers without the original corpus.
