"""Safely close an explicitly named interrupted MLflow run.

This command uploads only a small interruption receipt and never calls the
normal durable-run flush path, so it cannot accidentally upload pending caches
or other historical artifacts from the interrupted attempt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.data import confined_path
from speaker_id.tracking.interruption import close_interrupted_run
from speaker_id.tracking.security import Redactor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", required=True,
                        help="Existing tracking directory under artifacts/.")
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--cause", required=True)
    parser.add_argument("--preservation-archive-sha256", required=True)
    parser.add_argument("--preservation-archive-path", required=True)
    args = parser.parse_args(argv)
    try:
        tracking_dir = confined_path(ROOT, args.tracking_dir)
        if not tracking_dir.is_relative_to(ROOT / "artifacts"):
            raise ValueError("Tracking directory must stay under artifacts/.")
        result = close_interrupted_run(
            tracking_dir=tracking_dir,
            expected_run_id=args.expected_run_id,
            interruption_cause=args.cause,
            preservation_archive_sha256=args.preservation_archive_sha256,
            preservation_archive_path=args.preservation_archive_path,
        )
        print(json.dumps(Redactor()(result), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        message = Redactor().text(f"{type(error).__name__}: {error}")
        print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
