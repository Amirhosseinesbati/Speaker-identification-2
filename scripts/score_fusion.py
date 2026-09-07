"""Validate S003 by default; execute the frozen dual-view suite only on the server."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_dualview_scoring.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from speaker_id.training.fusion_suite import execute_fusion, load_fusion_contracts, project_path
    config = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(config.read_text(encoding="utf-8"))
    contracts = load_fusion_contracts(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_no_experiment_started", "experiment": suite["experiment_code"],
                          "candidates": [row["id"] for row in suite["candidates"]],
                          "source_cache_files_verified": False, "readiness_config": suite["readiness_config"]}))
        return
    print(json.dumps(execute_fusion(ROOT, config, suite, contracts,
                                   ROOT / "artifacts/infrastructure/mlflow_state.json"), indent=2))


if __name__ == "__main__":
    main()
