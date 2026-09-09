"""Validate S012 by default; --execute starts local cached CPU decision fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/postprocessing/campp_s012.json"))
    parser.add_argument("--experiment-state", type=Path, default=Path("artifacts/infrastructure/mlflow_state.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.postprocessing.decision_suite import execute_suite, validate_inputs
    result = execute_suite(ROOT, args.config, args.experiment_state) if args.execute else validate_inputs(ROOT, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
