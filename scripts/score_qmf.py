"""Validate S017 by default; --execute runs the sealed nested QMF experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/postprocessing/campp_s017_qmf.json"),
    )
    parser.add_argument(
        "--binding-state",
        type=Path,
        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from speaker_id.postprocessing import qmf_suite

    result = (
        qmf_suite.execute(ROOT, args.config, args.binding_state)
        if args.execute
        else qmf_suite.validate(ROOT, args.config)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
