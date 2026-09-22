# Inference

## Inputs and outputs

`python scripts/infer.py --example asr_correction` uses a complete recorded
request. The system role, request envelope, evidence unit identifiers,
`task_family`, `task_spec` and `response_format` are all part of the input.
The bundled assistant reference is never sent to the model.

Use `--messages FILE` with either a JSON array of system/user messages or
an object containing a `messages` array. Use `--prompt TEXT` for exploratory
plain instructions; failures on such inputs are documented, not hidden.
Prompts must end in a user message. Output is the model's raw text; no
repair, schema coercion or retry is applied.

```bash
python scripts/infer.py --example mt_post_edit --output outputs/prediction.json
python scripts/infer.py --messages examples/ocr_correction.json --device cpu
```

`--output` records the actual resolved HF commit (or local model path),
prompt/output token counts, EOS and length-limit status. A response that
hits the limit is not silently reported as complete.

## Precision and runtime

Default model: `OpenFormosa/barbet-1b-structured-sft`, revision `v0.1.1`. The verified loader
preserves BF16 weights plus the original 336 FP32 Mamba parameters.
Do not globally cast the model. The original tokenizer and role tokens
are unchanged. Transformers 5 chat tokenization uses `return_dict=False`.

Install the pinned environment from the repository root, with PyTorch
available before building native packages:

```bash
python -m pip install torch==2.11.0 packaging ninja setuptools wheel
python -m pip install --no-build-isolation -r requirements-inference.txt
```

Build isolation is disabled so native extensions use the installed PyTorch;
see the [Mamba installation instructions](https://github.com/state-spaces/mamba#installation).
Use a CUDA-enabled wheel compatible with your host, matching CUDA headers
and a compiler. The recorded environment used CUDA 13.0. CPU works for short
prompts but is slow; long prompts require CUDA, Mamba scan kernels and
sufficient memory. Eight H200 GPUs were used for training, not a claim
that eight are required for short inference.

## Interpretation

`decision: FAIL` means the **candidate input** failed its task checks; it
does not mean the model process crashed. Read the proposed edit and cited
evidence. `confidence_target` comes from a synthetic prior and is not a
calibrated probability. Examples often supply a correct reference, so
successful correction is not evidence of standalone transcription or translation.
