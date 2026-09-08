#!/usr/bin/env bash
# Start exactly one committed, allowlisted experiment after explicit user approval.
# This script never installs packages, edits source, or starts another experiment.
set -euo pipefail

readonly PROJECT_ROOT=/workspace/Speaker-identification-2
if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: run_campp_experiment.sh <allowlisted configuration.json>' >&2
    exit 2
fi
case "$1" in
    configs/train/campp_adapted_candidate_fusion.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_adapted_candidate_fusion.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_candidate_fusion.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_candidate_fusion.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_adaptation_comparison.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_adaptation_comparison.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_advanced_scoring.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_candidate.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_expanded_gallery.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_expanded.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_adapted_scoring.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_adapted.py
        EXECUTION_FLAG=--execute
        ;;
    configs/package/campp_selected.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/package_selected.py
        EXECUTION_FLAG=--execute
        ;;
    configs/package/campp_s002f.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/package_frozen.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_dualview_scoring.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_fusion.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_scoring_suite.json|configs/train/campp_coverage_scoring.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/score_frozen.py
        EXECUTION_FLAG=--execute
        ;;
    configs/train/campp_coverage.json|configs/train/campp_finetune.json|configs/train/campp_finetune_warmup.json|configs/train/campp_finetune_fp32.json|configs/train/campp_finetune_head600.json)
        CONFIG_PATH=$1
        ENTRYPOINT=scripts/train.py
        EXECUTION_FLAG=--execute-training
        ;;
    *)
        printf '%s\n' 'Experiment configuration is not allowlisted.' >&2
        exit 2
        ;;
esac
readonly CONFIG_PATH ENTRYPOINT EXECUTION_FLAG

test "$(uname -s)" = Linux
cd "$PROJECT_ROOT"
test -x .venv/bin/python
test -f artifacts/infrastructure/readiness.json
test -f "$CONFIG_PATH"
test ! -L "$CONFIG_PATH"
test -f "$ENTRYPOINT"
test ! -L "$ENTRYPOINT"
test -f .env
test ! -L .env
umask 077

export VAST_INSTANCE_ID=50079023
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/infra/with_project_env.py" \
    "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/$ENTRYPOINT" \
    --config "$PROJECT_ROOT/$CONFIG_PATH" \
    "$EXECUTION_FLAG"
