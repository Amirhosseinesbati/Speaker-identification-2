"""Validate S011 by default; explicitly execute local immutable-cache scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-state", type=Path, default=Path("artifacts/infrastructure/mlflow_state.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.postprocessing.suite import execute_suite, validate_inputs
    if not args.execute:
        result = validate_inputs(ROOT, args.config)
    else:
        result = execute_suite(ROOT, args.config, args.experiment_state)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
