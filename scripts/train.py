"""Default: validate contracts only. Full experiments require --execute-training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_baseline.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "artifacts/infrastructure/mlflow_state.json")
    parser.add_argument("--verify-audio", action="store_true", help="SHA256-check every raw file before proceeding")
    parser.add_argument("--execute-training", action="store_true", help="Explicit authorization to execute the configured full baseline or training experiment")
    parser.add_argument("--resume", type=Path, help="Resume an existing experiment directory with exactly the same recipe")
    args = parser.parse_args()
    from speaker_id.training.contracts import load_contract
    contract = load_contract(args.config.resolve(), ROOT, verify_audio=args.verify_audio or args.execute_training)
    if not args.execute_training:
        if args.resume:
            parser.error("--resume requires --execute-training")
        print(json.dumps({"status": "contract_validated_no_training_started", "signature": contract["signature"],
                          "model": contract["model"]["architecture"], **contract["summary"]}, indent=2))
        return
    from speaker_id.training.runner import execute
    print(json.dumps(execute(contract, ROOT, args.binding, resume_dir=args.resume), indent=2))


if __name__ == "__main__":
    main()
