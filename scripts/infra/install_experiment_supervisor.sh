#!/usr/bin/env bash
# Register one inactive experiment program. This installer never starts a job.
set -euo pipefail

readonly PROJECT_ROOT=/workspace/Speaker-identification-2
if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: install_experiment_supervisor.sh <S001|B002|F001|F002>' >&2
    exit 2
fi
case "$1" in
    S001)
        PROGRAM=speaker_id_campp_s001
        SOURCE=configs/infra/supervisor_campp_scoring_suite.conf
        ;;
    B002)
        PROGRAM=speaker_id_campp_b002
        SOURCE=configs/infra/supervisor_campp_coverage.conf
        ;;
    F001)
        PROGRAM=speaker_id_campp_f001
        SOURCE=configs/infra/supervisor_campp_finetune.conf
        ;;
    F002)
        PROGRAM=speaker_id_campp_f002
        SOURCE=configs/infra/supervisor_campp_finetune_warmup.conf
        ;;
    *)
        printf '%s\n' 'Experiment program is not allowlisted.' >&2
        exit 2
        ;;
esac
readonly PROGRAM SOURCE
readonly TARGET="/etc/supervisor/conf.d/$PROGRAM.conf"

test "$(uname -s)" = Linux
cd "$PROJECT_ROOT"
test -z "$(git status --porcelain --untracked-files=normal)"
test -d /etc/supervisor/conf.d
test ! -L "$TARGET"
if [ -e "$TARGET" ]; then
    test -f "$TARGET"
fi
test -f "$SOURCE"
test ! -L "$SOURCE"
grep -Eq '^files[[:space:]]*=[[:space:]]*/etc/supervisor/conf.d/\*.conf[[:space:]]*$' /etc/supervisor/supervisord.conf

# A completed EXITED program is inactive and can be re-registered. Never change
# a RUNNING, STARTING, STOPPING, BACKOFF, FATAL, or unrecognized program state.
status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
if ! printf '%s\n' "$status" | grep -Eq "^${PROGRAM}[[:space:]]+(STOPPED|EXITED)([[:space:]]|$)|^${PROGRAM}: ERROR \\(no such process\\)$"; then
    printf '%s\n' "Refusing to change program in unexpected state: $status" >&2
    exit 1
fi
install -d -m 0750 "$PROJECT_ROOT/artifacts/logs"
install -m 0644 "$SOURCE" "$TARGET"
supervisorctl reread
supervisorctl update "$PROGRAM"
status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
printf '%s\n' "$status"
printf '%s\n' "$status" | grep -Eq "^${PROGRAM}[[:space:]]+(STOPPED|EXITED)([[:space:]]|$)"
cmp -- "$SOURCE" "$TARGET"
printf '%s\n' 'Manual-start program registered and verified inactive. No experiment started.'
