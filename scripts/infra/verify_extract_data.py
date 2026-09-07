"""Run full server dataset verification; never starts model training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.data import verify_extract_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, default=Path("data/incoming/raw.zip"))
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/eda_v1/audio_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/infrastructure/data_readiness.json"))
    parser.add_argument("--delete-archive-after-verification", action="store_true")
    arguments = vars(parser.parse_args())
    try:
        result = verify_extract_data(**arguments)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
