"""Validate S006 by default; execute only with completed source identities on the server."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.training.adaptation_comparison import execute_comparison, load_comparison_contracts
    from speaker_id.training.fusion_suite import project_path
    path = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(path.read_text(encoding="utf-8"))
    contracts = load_comparison_contracts(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "experiment": "S006",
            "recipes": [row["id"] for row in suite["recipes"]], "source_cache_files_verified": False,
            "readiness_config": suite["readiness_config"], "expanded_arm": suite["expanded_arm"]}))
        return
    print(json.dumps(execute_comparison(ROOT, path, suite, contracts,
                                       ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
