#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export VAST_INSTANCE_ID=50288952
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /workspace/Speaker-identification-2-c002
exec .venv/bin/python scripts/train_f005.py "$@"
