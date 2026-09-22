# Barbet 1B Structured SFT

A 1.12B-parameter Traditional Chinese text model for **structured,
evidence-grounded editing and assessment** in ASR, TTS text preparation,
OCR and translation workflows.

[Model weights](https://huggingface.co/OpenFormosa/barbet-1b-structured-sft) · [Documentation website](https://voidful.github.io/barbet-structured-sft/) ·
[繁體中文](README.zh-TW.md) · [Evaluation](docs/evaluation.md) · [Examples](docs/examples.md)

**Experimental, specialized checkpoint.** The synthetic structured test
reaches 408/408 parseable responses and 214/408 whole-answer exact matches.
All 16 fresh free-form instruction probes fail. Code and mathematics
bits-per-byte regress by 27.36% and 16.21%. Use the full structured request
format and evaluate on your own workload. This release processes text;
it does not transcribe audio, recognize image pixels, or synthesize audio.

## Quick start

Tested with Python 3.12, PyTorch 2.11.0, CUDA 13.0 and H200 GPUs. Native
Mamba/causal-convolution packages need a compatible CUDA build toolchain.
A different CUDA wheel or kernel stack can change numerical results.

```bash
git clone https://github.com/voidful/barbet-structured-sft.git
cd barbet-structured-sft
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/infer.py --example ocr_correction --device cuda:0
```

The inference command downloads the pinned `v0.1.0` HF release, preserves
the 336 FP32 Mamba parameters, applies the published chat template, and
prints the model's raw answer. No HF login is required for this public
checkpoint. Short CPU inference is also supported with `--device cpu`.

```bash
python scripts/infer.py --example tts_spoken_form
python scripts/infer.py --messages examples/mt_post_edit.json
python scripts/infer.py --prompt '請將 5083 逐位讀出。' --max-new-tokens 256
```

The final command illustrates the API; reliable free-form instruction
following is **not** established. [Full usage and precision notes](docs/usage.md).

## What was trained

| Item | Recorded value |
|---|---:|
| Parameters | 1,118,799,096 |
| Architecture | 22 global-attention + 7 Mamba2 layers |
| Dataset | `voidful/barbet-sft`, 51 task families, 4 serializers |
| Training examples / epochs | 980,000 / 1 |
| Optimizer steps | 7,657 |
| Input / supervised tokens | 1,490,165,039 / 225,601,357 |
| Longest training sample | 168,442 tokens, no truncation |
| Test answer NLL, base → SFT | 1.830924 → 0.003356 |
| Strict whole-answer match, base → SFT | 0/408 → 214/408 |

Every training example appears once. The exported tensors match the final
FP32 checkpoint after the intended precision conversion. Configuration
retains a 1,048,576-token maximum; this is **not a demonstrated SFT 1M
capability score**. [Training details and reproducibility boundaries](docs/reproducing.md).

## Measured behavior

With a visible reference and the full request contract, the recorded examples
repair `數位千章` to `數位簽章`, normalize `GPU 使用率為 92.5%。` to
`G P U 使用率為百分之九十二點五。`, and fix OCR/date errors.
The four additional natural-language examples all fail their instructions.
[Read the actual inputs and outputs](docs/examples.md); examples are not a benchmark.

## Repository guide

- `scripts/infer.py`: public inference entry point.
- `examples/`: four structured requests and four natural-language probes.
- `runtime/`: exact inference code distributed with the weights.
- `training/`: original training, preparation and evaluation source.
- `reports/`: metrics, provenance, weight audits and recorded generations.
- `site/`: dependency-free documentation website, deployed by GitHub Pages.
- `docs/`: usage, evaluation, reproduction, architecture and licensing.

## Development and contributions

```bash
python scripts/validate_release.py
python -m http.server 8000 --directory site
```

Open `http://localhost:8000` to preview the site. CI validates packaged
evidence, examples, syntax and local documentation links without downloading
weights. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [CHANGELOG.md](CHANGELOG.md).

## License and citation

**Public access currently does not include an explicit open-source license
grant.** The original model metadata is `other`; the dataset has a public-access
statement. No Apache/MIT/CC license is inferred from visibility. See
[LICENSE](LICENSE) and [licensing details](docs/licensing.md).
Cite this software/model release using [CITATION.cff](CITATION.cff).
