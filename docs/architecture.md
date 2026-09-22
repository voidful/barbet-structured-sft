# Architecture

Released checkpoint: 1,118,799,096 parameters, 29 layers with motif
`[G, G, G, M] × 7 + G`: 22 global-attention blocks and seven Mamba2 blocks.
Hidden size 1,536, FFN size 5,120, attention/KV heads 16/2, padded vocabulary
114,944. Embedding and LM-head weights are tied. RoPE theta is 10,000,000.

The export stores 1,118,798,760 BF16 parameters and 336 FP32 Mamba
parameters. `runtime/load_barbet.py` preserves this census. A global
`.half()` or `.bfloat16()` cast loses the intended precision.

The hybrid cache stores attention KV and Mamba convolution/recurrent
state. The generation helper performs one full prefill followed by
cached single-token decoding and projects only the final hidden position.
This is a causal text model; modality encoders/decoders are not included.

The configured maximum is 1,048,576 tokens. SFT training reached 168,442
tokens for its longest sample; the published controlled retrieval probes
do not establish reliable 128K/1M natural-document understanding.
