#!/usr/bin/env bash
# Install only the project's committed manual-start program; never start a job.
set -euo pipefail

readonly PROJECT_ROOT=/workspace/Speaker-identification-2
readonly PROGRAM=speaker_id_campp_b001
readonly TARGET=/etc/supervisor/conf.d/speaker_id_campp_b001.conf
test "$(uname -s)" = Linux
cd "$PROJECT_ROOT"
test -z "$(git status --porcelain --untracked-files=normal)"
test -d /etc/supervisor/conf.d
test ! -L "$TARGET"
grep -Eq '^files[[:space:]]*=[[:space:]]*/etc/supervisor/conf.d/\*.conf[[:space:]]*$' /etc/supervisor/supervisord.conf

# Refuse to alter an active program. An absent or already stopped one is safe.
status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
if ! printf '%s\n' "$status" | grep -Eq '(no such process|STOPPED)'; then
    printf '%s\n' "Refusing to change program in unexpected state: $status" >&2
    exit 1
fi
install -d -m 0750 "$PROJECT_ROOT/artifacts/logs"
install -m 0644 configs/infra/supervisor_campp_baseline.conf "$TARGET"
supervisorctl reread
supervisorctl update "$PROGRAM"
status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
printf '%s\n' "$status"
printf '%s\n' "$status" | grep -Eq '^speaker_id_campp_b001[[:space:]]+STOPPED([[:space:]]|$)'
cmp -- configs/infra/supervisor_campp_baseline.conf "$TARGET"
printf '%s\n' 'Manual-start program registered and verified STOPPED. No training started.'
