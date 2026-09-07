"""Validate S008 by default; actual completed S007 identities are mandatory."""
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
    from speaker_id.training.candidate_fusion import execute_candidate_fusion, load_fusion_inputs
    from speaker_id.training.fusion_suite import project_path
    path = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(path.read_text(encoding="utf-8"))
    contract = load_fusion_inputs(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "experiment": "S008", "source_caches_verified": False,
                          "alphas": suite["alphas"], "execution_policy": suite["execution_policy"]}))
        return
    print(json.dumps(execute_candidate_fusion(ROOT, path, suite, contract, ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
