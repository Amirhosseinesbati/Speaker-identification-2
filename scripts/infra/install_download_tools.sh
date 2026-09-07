#!/usr/bin/env bash
# Explicit optional infrastructure installation; no download or training starts.
set -euo pipefail
test "$(uname -s)" = Linux
if ! command -v aria2c >/dev/null 2>&1; then
    test "$(id -u)" -eq 0
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install --no-install-recommends -y aria2
fi
aria2c --version
