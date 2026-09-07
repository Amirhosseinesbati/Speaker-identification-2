"""Validate S007 by default; manually execute only after the S006 decision."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_advanced_scoring.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.training.candidate_comparison import execute_candidate_comparison, load_candidate_contract
    from speaker_id.training.fusion_suite import project_path
    path = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(path.read_text(encoding="utf-8"))
    contract, identity = load_candidate_contract(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "experiment": "S007",
            "candidate_signature": identity["signature"], "embedding_dim": 192,
            "weights_and_cache_verified": False, "execution_policy": suite["execution_policy"]}))
        return
    print(json.dumps(execute_candidate_comparison(ROOT, path, suite, contract, identity,
                                                  ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
