"""Run a preregistered remote scoring suite; default validates configuration only."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_scoring_suite.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    if not config.is_relative_to(ROOT / "configs/train"):
        parser.error("Suite configuration must be committed under configs/train")
    suite = json.loads(config.read_text())
    from speaker_id.training.frozen_suite import validate_suite
    validate_suite(suite)
    from speaker_id.training.contracts import load_contract
    contract = load_contract(ROOT / suite["baseline_config"], ROOT, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "recipes": [r["id"] for r in suite["recipes"]],
                          "source_run": suite["source_run"]}))
        return
    from speaker_id.training.frozen_suite import execute_suite
    print(json.dumps(execute_suite(ROOT, config, contract, ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
