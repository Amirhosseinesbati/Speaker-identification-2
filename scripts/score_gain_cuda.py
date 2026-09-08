"""Validate C002; fresh CUDA extraction/scoring requires explicit --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from speaker_id.training.cuda_gain_suite import execute_cuda_gain_suite
    from speaker_id.training.cuda_pair_contract import load_cuda_gain_inputs
    from speaker_id.training.fusion_suite import project_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--binding-state",
        type=Path,
        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"),
        help="C002-specific MLflow binding receipt, relative to the project root by default",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = project_path(ROOT, args.config, "configs/train")
    suite = json.loads(path.read_text(encoding="utf-8"))
    contract, source_config, _ = load_cuda_gain_inputs(ROOT, suite)
    if not args.execute:
        print(json.dumps({
            "status": "validated_no_experiment_started",
            "experiment": "C002",
            "execution": suite["execution"],
            "target_identity": suite["target_identity"],
            "candidate_frontends": 2,
            "encoder_training": False,
            "embedding_artifacts_uploaded": False,
        }))
        return
    binding_state = args.binding_state if args.binding_state.is_absolute() else ROOT / args.binding_state
    print(json.dumps(
        execute_cuda_gain_suite(ROOT, path, suite, contract, source_config, binding_state),
        indent=2,
    ))


if __name__ == "__main__":
    main()
