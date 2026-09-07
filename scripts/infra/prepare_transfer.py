"""Prepare exactly deployment.json's metadata_files for private ZIP transfer."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.metadata import prepare_metadata_transfer
from speaker_id.tracking.security import Redactor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--deployment", type=Path, default=Path("configs/infra/deployment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/infrastructure/data_metadata.zip"))
    parser.add_argument("--identity", type=Path, default=Path("artifacts/infrastructure/data_metadata.identity.json"))
    try:
        result = prepare_metadata_transfer(**vars(parser.parse_args()))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": Redactor().text(str(error))}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
