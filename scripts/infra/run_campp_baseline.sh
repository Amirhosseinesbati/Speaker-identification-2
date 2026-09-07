#!/usr/bin/env bash
# Manual start only. Supervisor is configured with autostart/autorestart disabled.
# Source changes arrive through Git; this wrapper never installs or edits code.
set -euo pipefail

readonly PROJECT_ROOT=/workspace/Speaker-identification-2
if [ "$#" -ne 0 ]; then
    printf '%s\n' 'B001 has a fixed configuration; unexpected arguments are forbidden.' >&2
    exit 2
fi
test "$(uname -s)" = Linux
cd "$PROJECT_ROOT"
test -x .venv/bin/python
test -f artifacts/infrastructure/readiness.json
test -f .env
test ! -L .env
umask 077

# The Python entry point independently checks the measured instance/hostname,
# current code, complete data hashes, readiness evidence and live MLflow access.
export VAST_INSTANCE_ID=50079023
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

exec "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/infra/with_project_env.py" \
    "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/train.py" \
    --config "$PROJECT_ROOT/configs/train/campp_baseline.json" \
    --execute-training
