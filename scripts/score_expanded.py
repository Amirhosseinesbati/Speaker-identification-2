"""Validate S005 by default; run only with completed S004 controls and server readiness."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_expanded_gallery.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.training.adapted_scoring import execute_adapted_suite, load_adapted_contracts
    from speaker_id.training.expanded_gallery import validate_expanded_suite
    from speaker_id.training.fusion_suite import project_path
    path = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(path.read_text(encoding="utf-8"))
    validate_expanded_suite(suite)
    contracts = load_adapted_contracts(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "experiment": "S005",
            "recipes": [row["id"] for row in suite["recipes"]], "source_cache_files_verified": False,
            "completed_control_artifacts_verified": False, "calibration_protocol": suite["calibration_protocol"]}))
        return
    print(json.dumps(execute_adapted_suite(ROOT, path, suite, contracts,
                                         ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
