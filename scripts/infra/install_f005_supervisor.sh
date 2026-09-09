#!/usr/bin/env bash
# Register F005 as an inactive Supervisor job. This script never starts it.
set -euo pipefail

readonly PROJECT_ROOT=/workspace/Speaker-identification-2-c002
readonly PROGRAM=speaker_id_campp_f005
readonly TARGET=/etc/supervisor/conf.d/speaker_id_campp_f005.conf
readonly TEMPORARY="$PROJECT_ROOT/artifacts/logs/.speaker_id_campp_f005.conf.$$"

test "$(uname -s)" = Linux
cd "$PROJECT_ROOT"
test -x .venv/bin/python
test -f .env
test ! -L .env
test -f scripts/run_f005_experiment.py
test ! -L scripts/run_f005_experiment.py
test -f artifacts/infrastructure/C002_preparation/mlflow_state.json
test -d /etc/supervisor/conf.d
test ! -L "$TARGET"
grep -Eq '^files[[:space:]]*=[[:space:]]*/etc/supervisor/conf.d/\*.conf[[:space:]]*$' /etc/supervisor/supervisord.conf

status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
if ! printf '%s\n' "$status" | grep -Eq "^${PROGRAM}[[:space:]]+(STOPPED|EXITED)([[:space:]]|$)|^${PROGRAM}: ERROR \\(no such process\\)$"; then
    printf '%s\n' "Refusing to replace an active or unexpected program: $status" >&2
    exit 1
fi

install -d -m 0750 "$PROJECT_ROOT/artifacts/logs"
trap 'rm -f -- "$TEMPORARY"' EXIT
umask 077
cat >"$TEMPORARY" <<'EOF'
[program:speaker_id_campp_f005]
directory=/workspace/Speaker-identification-2-c002
command=/workspace/Speaker-identification-2-c002/.venv/bin/python /workspace/Speaker-identification-2-c002/scripts/infra/with_project_env.py /workspace/Speaker-identification-2-c002/.venv/bin/python /workspace/Speaker-identification-2-c002/scripts/run_f005_experiment.py --config /workspace/Speaker-identification-2-c002/configs/train/campp_f005_consistency.json --binding /workspace/Speaker-identification-2-c002/artifacts/infrastructure/C002_preparation/mlflow_state.json --execute
user=root
numprocs=1
autostart=false
autorestart=false
startsecs=0
startretries=0
exitcodes=0
stopsignal=TERM
stopwaitsecs=120
stopasgroup=true
killasgroup=true
environment=VAST_INSTANCE_ID="50288952",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONUNBUFFERED="1",MPLBACKEND="Agg"
redirect_stderr=false
stdout_logfile=/workspace/Speaker-identification-2-c002/artifacts/logs/f005.stdout.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
stderr_logfile=/workspace/Speaker-identification-2-c002/artifacts/logs/f005.stderr.log
stderr_logfile_maxbytes=20MB
stderr_logfile_backups=5
EOF
chmod 0644 "$TEMPORARY"
install -m 0644 "$TEMPORARY" "$TARGET"
supervisorctl reread
supervisorctl update "$PROGRAM"
status=$(supervisorctl status "$PROGRAM" 2>&1 || true)
printf '%s\n' "$status"
printf '%s\n' "$status" | grep -Eq "^${PROGRAM}[[:space:]]+(STOPPED|EXITED)([[:space:]]|$)"
cmp -- "$TEMPORARY" "$TARGET"
grep -Fx 'autostart=false' "$TARGET"
grep -Fx 'autorestart=false' "$TARGET"
printf '%s\n' 'F005 registered and verified inactive. Start it manually with supervisorctl start speaker_id_campp_f005.'
