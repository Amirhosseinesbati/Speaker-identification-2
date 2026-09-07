"""Validate P001 inputs; --execute performs a tracked build on the authorized server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/package/campp_s002f.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "artifacts/infrastructure/mlflow_state.json")
    parser.add_argument("--execute", action="store_true", help="Explicitly build the final tracked scorer/package on Vast 50079023; no encoder training")
    args = parser.parse_args()
    from speaker_id.packaging.frozen import execute_build, load_build_contract
    if args.execute:
        print(json.dumps(execute_build(ROOT, args.config, args.binding), indent=2))
    else:
        config, contract, inputs = load_build_contract(ROOT, args.config)
        print(json.dumps({"status": "build_inputs_validated_no_calibration_or_training", "package": config["experiment_code"],
                          "baseline_signature": contract["signature"], "input_files": len(inputs),
                          "selected_recipe": config["selection_recipe"], "encoder_updates": 0}, indent=2))


if __name__ == "__main__":
    main()
