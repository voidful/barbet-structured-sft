# Evaluation and limitations

## Fixed synthetic test

Both models use the same 1,020 answer-NLL examples and 408 greedy generations
(20 and 8 per task family). Maximum generation length is 2,048 tokens. The
dataset's original canonical parser scores the entire answer object. Format
success is distinct from content correctness; parse failures count as wrong.

| Domain | Base NLL | SFT NLL | SFT whole-answer exact | SFT parse rate |
|---|---:|---:|---:|---:|
| ASR | 1.6994 | 0.0029 | 27/48 | 100% |
| TTS | 1.6405 | 0.0027 | 19/32 | 100% |
| OCR | 1.6231 | 0.0026 | 24/40 | 100% |
| MT | 1.5686 | 0.0024 | 13/32 | 100% |

Across all 51 task families: NLL **1.830924 → 0.003356**, whole-answer exact
**0/408 → 214/408 (52.45%)**. Decision and evidence-list exact are each
**408/408**. Among 194 strict errors, 189 differ only in synthetic confidence;
five also differ in quality score and human-review reason. Strict scoring was
not relaxed. Shared templates across distinct source groups limit generalization claims.

## Existing development retention corpus

6,955 rows; BOS-conditioned all-token loss; no truncation. BPB is negative
log-likelihood bits per UTF-8 byte; lower is better. These are loss changes,
not percentage-point changes in benchmark accuracy. The underlying historical
development texts are not distributed and are not a fresh private release set.

| Category | Base BPB | SFT BPB | Relative change |
|---|---:|---:|---:|
| zh_tw_zh | 1.1399 | 1.1756 | +3.13% |
| math | 0.6890 | 0.8007 | +16.21% |
| ja_ko | 0.9680 | 0.9833 | +1.58% |
| english_general | 0.8615 | 0.8727 | +1.30% |
| code | 0.7154 | 0.9111 | +27.36% |
| multilingual | 1.1374 | 1.1679 | +2.69% |

## Additional instruction probes

Both models score **0/16** on the original fresh text instructions. Manual
inspection finds prompt repetition or unrelated continuation in candidate
outputs, not valid alternate translations. The additional 2026-09-19 demo
uses four new natural instructions; all four fail. These are small diagnostic
sets, not estimates of broad product accuracy.

## Controlled 8K / 128K retrieval

Four evidence positions per input length, repeated synthetic background,
raw continuation and a 64-token generation limit. Strict exact is 0/4 for
each model at both lengths. SFT 8K outputs contain the right codes plus a
full stop. At 128K, only the near-end case contains the correct code, with
unrelated continuation; the other positions claim no document/record.

This does not establish reliable long-document retrieval or SFT 1M capability.
The numerical context configuration is not an evaluated capability guarantee.
No raw-audio WER, speech quality, or image OCR accuracy was measured.

## Evidence

[Full metrics](../reports/evaluation_summary.json),
[manual inspection](../reports/evaluation_inspection.json),
[weight validation](../reports/package_validation.json),
[epoch coverage](../reports/epoch_coverage_audit.json),
[provenance](../reports/provenance.json).
