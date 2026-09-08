# C002: fresh GPU paired gain protocol

`C002` is a fresh GPU-only frozen-CAM++ comparison for Vast instance `50288952`.
It never resumes C001, reads no C001 output/cache path, and writes every mutable
file below `artifacts/training/cuda_gain_c002/C002_<timestamp>_<uuid>`.

The four recipes are deliberately separate:

1. **C002a** reproduces historical S008c scoring. It is a non-selectable control.
2. **C002b** extracts a new CUDA identity frontend from the signed source models.
3. **C002c** extracts the paired fixed RMS-gain frontend with the same models.
4. **C002d** chooses C002b or C002c, plus calibration, from group-excluded inner
   queries only. Both choices are serialized and read back before fresh outer
   predictions are written.

Fresh CUDA identity embeddings may differ numerically from historical S008c
embeddings because they are a separate extraction identity. C002 records
cosine/L2/max-absolute diagnostics without an acceptance threshold and never
uses bitwise historical embedding parity as a gate. The decision contrast is
`C002d − C002b`; C002a→C002b is diagnostic only, not gain evidence.

The launcher defaults to metadata validation:

```bash
uv run python scripts/score_gain_cuda.py --config configs/train/campp_gain_cuda_c002.json
```

Actual extraction/scoring requires `--execute`, the C002 readiness contract, a
visible matching RTX 3090 with at least 24 GiB capacity and 10 GiB free, and
`VAST_INSTANCE_ID=50288952`. CPU fallback, cache reuse, resume, encoder updates
and embedding uploads to MLflow are rejected. On failure the partial C002 files
remain as evidence and a new invocation must create a new C002 output directory.
