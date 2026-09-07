#!/usr/bin/env bash
# Fixed public competition source; source code arrives through Git.
# Run install_download_tools.sh explicitly first if aria2 is unavailable.
# This creates a candidate ZIP only. Every member still needs manifest verification.
set -euo pipefail
test "$(uname -s)" = Linux
readonly PROJECT_ROOT=/workspace/Speaker-identification-2
readonly SOURCE_URL='https://iaaa-contest-speaker.s3.ir-thr-at1.arvanstorage.ir/iaaa-contest-speaker.zip?versionId='
readonly SOURCE_ETAG='"726c7582d0338030036f7534a288a0e7-292"'
readonly EXPECTED_BYTES=9764950603
cd "$PROJECT_ROOT"
test -x .venv/bin/python
command -v aria2c >/dev/null
command -v flock >/dev/null
test ! -L data/incoming
mkdir -p data/incoming artifacts/infrastructure
test ! -L data/incoming/competition_parallel.zip
test ! -L data/incoming/competition_parallel.zip.aria2

# This script must never have two writers to the same output/control file.
exec 9>data/incoming/.competition_parallel.download.lock
flock -n 9

aria2c --no-conf=true \
    --max-connection-per-server=8 --split=8 --min-split-size=8M \
    --continue=true --allow-overwrite=false --auto-file-renaming=false \
    --check-certificate=true --file-allocation=none \
    --header="If-Match: $SOURCE_ETAG" \
    --header='Accept-Encoding: identity' \
    --summary-interval=60 --show-console-readout=false --enable-color=false \
    --console-log-level=notice --download-result=full \
    --auto-save-interval=30 --max-tries=5 --retry-wait=5 \
    --connect-timeout=30 --timeout=60 \
    --dir="$PROJECT_ROOT/data/incoming" --out=competition_parallel.zip \
    "$SOURCE_URL"

.venv/bin/python - "$EXPECTED_BYTES" "$SOURCE_ETAG" "$SOURCE_URL" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

path = Path("data/incoming/competition_parallel.zip")
expected_bytes = int(sys.argv[1])
if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_bytes:
    raise SystemExit("Downloaded ZIP size differs from the official object; payload verification is blocked")
if not zipfile.is_zipfile(path):
    raise SystemExit("Downloaded object is not a complete ZIP container")
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
report = {"status": "downloaded_payload_verification_pending", "training_started": False,
          "source_url": sys.argv[3], "source_etag": sys.argv[2],
          "archive_path": path.as_posix(), "archive_size_bytes": path.stat().st_size,
          "archive_sha256": digest, "zip_container_detected": True,
          "payload_crc_and_audio_manifest_verified": False,
          "checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "next_step": "Verify every archive member CRC, all 4529 audio hashes and labels before using or deleting the ZIP"}
target = Path("artifacts/infrastructure/competition_download.json")
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(report, indent=2), flush=True)
PY
