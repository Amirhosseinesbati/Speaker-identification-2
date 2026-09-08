#!/usr/bin/env bash
# This file must arrive through git clone/pull, never by remote source editing.
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$(uname -s)" = Linux
expected_git_branch="${EXPECTED_GIT_BRANCH:-develop}"
test "$(git branch --show-current)" = "$expected_git_branch"
test -z "$(git status --porcelain --untracked-files=normal)"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
python3 -m venv .venv-bootstrap
.venv-bootstrap/bin/python -m pip install 'uv==0.11.28'
.venv-bootstrap/bin/uv sync --locked --no-default-groups --group train
mkdir -p data/incoming artifacts/infrastructure artifacts/models/campp
.venv/bin/python -m compileall -q src scripts
# No data extraction, model fitting, or experiment execution happens here.
