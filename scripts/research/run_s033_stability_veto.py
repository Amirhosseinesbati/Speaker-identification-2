"""Validate S033 by default; --execute runs its tracked GPU similarity screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/s033_reference_resampling_veto.json"))
    parser.add_argument("--binding-state", type=Path,
                        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from speaker_id.research import s033_stability_experiment

    result = (
        s033_stability_experiment.execute(ROOT, args.config, args.binding_state)
        if args.execute else s033_stability_experiment.validate(ROOT, args.config)
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
