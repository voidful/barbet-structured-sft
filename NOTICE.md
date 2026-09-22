# Provenance and third-party notices

Barbet Structured SFT is distributed by voidful / OpenFormosa. The
architecture follows the Open Formosa R2 family; the original tokenizer
derives from `voidful/PangolinTokenizer`. The SFT dataset is
`voidful/barbet-sft`. Exact revisions and source-report hashes are recorded
in `reports/provenance.json`.

The HF runtime uses PyTorch, Transformers, Safetensors, Tokenizers,
Hugging Face Hub, Mamba-SSM, causal-conv1d and Triton. These packages
are installed as dependencies and retain their own upstream terms;
this publication does not relicense them. Review their package notices
when redistributing an environment.

Dataset names and people in the bundled examples come from deterministic
synthetic scenarios and do not describe real events. The original
data-access statement is preserved in `notices/LICENSE_DATA.md`.
Historical private development text is not included in this repository.
