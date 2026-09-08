# C001: matched CPU waveform-gain comparison

C001 tests one fixed frontend hypothesis on the existing Vast instance50079023.
It keeps both pretrained CAM++ encoders frozen (512 and192 dimensions). The
original S010 CUDA gate is unchanged; this experiment makes no GPU-health claim.

The completed CP001 pilot used one four-thread CPU worker and seven representatives
selected from file positions, signal presence, duration and RMS, without labels or
error outcomes. Fourteen frontend pairs completed with unchanged model hashes.
Its nominal duration-weighted full extraction estimate is3.56 hours; the conservative
extrema formula gives27.33 hours. These are estimates from a small nonrandom sample.
C001 explicitly uses one worker and an eight-hour soft extraction budget, checked
between file pairs. It may fail to complete within that budget. It does not retry,
resume, substitute a device, reuse CP001 features or discard difficult recordings.

## Comparisons

| Recipe | Purpose |
|---|---|
| C001a | Reproduce S008c exactly using original verified caches before new extraction. |
| C001b | Recompute both models with the unchanged frontend on CPU. |
| C001c | Recompute both models on the same CPU with the fixed RMS boost. |
| C001d | Choose between C001b/c using only group-excluded inner queries. |

The boost targets−20dBFS, caps gain at60dB and peak at0.95, never attenuates or
clips, and leaves exact zeros unchanged. It runs after mono selection/resampling,
before the unchanged180-second/one-view inference. The alpha and rejection-gate
grids are identical across both fresh CPU frontends. Both folds' complete inner
choices are uploaded and read back before any fresh outer evaluation.

C001d−C001b is the primary matched contrast. C001c−C001b measures gain under equal
calibration opportunity. C001b−C001a is a CPU/backend diagnostic, never a gain
effect. The historical GPU cache is not a selectable frontend. The development
decision requires at least0.003 pooled Macro-F1 improvement with neither fold
declining by more than0.005. Repeated development-fold results do not establish a
hidden-test or leaderboard score, and P002 remains intact.

## Execution and evidence

Default CLI operation is metadata-only:

```sh
python scripts/score_gain_cpu.py --config configs/train/campp_gain_cpu.json
```

Actual execution additionally requires `--execute`, the exact authorized Linux
host/workspace, clean committed source, its matching operational instance marker,
the complete hash-pinned CP001 result and fresh owned MLflow readback. Numerical
thread variables must each be4; Torch uses four intra-op and one inter-op thread.
All source changes are committed/pushed locally and pulled on the server first.

Each file produces independently signed fresh identity/gain caches and a paired
receipt. Actual raw input and model files are hashed; frozen parameters/buffers
are checked before and after extraction. No-op gain requires bitwise-identical
CPU embeddings. Partial outputs remain as failure evidence; no implicit recovery
or overwrite is allowed. Every50 pairs, scalar progress and JSON evidence are
strictly synchronized to the existing separate project experiment.

MLflow receives source/config snapshots, hashes, numerical diagnostics and
evaluation reports, including prediction/support arrays. Embedding/cache NPZs
are kept on the server and transferred directly to the user's local workspace;
they are never uploaded to MLflow. No raw audio or credentials are published.
