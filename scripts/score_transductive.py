from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tracked local S014 batch-alignment suite")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/postprocessing/campp_s014.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "artifacts/infrastructure/mlflow_state.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    from speaker_id.postprocessing.transductive_suite import execute_suite, validate_inputs

    if args.validate_only:
        print(validate_inputs(ROOT, args.config))
    else:
        print(execute_suite(ROOT, args.config, args.binding))


if __name__ == "__main__":
    main()
