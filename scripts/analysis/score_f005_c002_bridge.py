"""Replay C002b and score F005 control with its fixed historic policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f005-dir", type=Path, required=True)
    parser.add_argument("--c002-run-dir", type=Path, required=True)
    parser.add_argument("--c002b-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--folds", type=Path,
                        default=ROOT / "data/processed/eda_v1/folds.csv")
    parser.add_argument("--label-map", type=Path,
                        default=ROOT / "data/processed/eda_v1/label_map.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from speaker_id.evaluation.scoring_bridge import (
        analyze_f005_control_under_c002b_policy,
        write_scoring_bridge,
    )

    report = analyze_f005_control_under_c002b_policy(
        args.f005_dir, args.c002_run_dir, args.c002b_dir,
        args.manifest, args.folds, args.label_map,
    )
    write_scoring_bridge(args.output_dir, report)
    result = report["f005_control_under_fixed_c002b_policy"]
    print(json.dumps({
        "status": report["status"],
        "output_dir": str(args.output_dir),
        "c002b_macro_f1": report["c002b_reproduction"]["metrics"]["macro_f1"],
        "f005_control_fixed_c002b_macro_f1": result["metrics"]["macro_f1"],
        "macro_f1_delta_vs_c002b": result["macro_f1_delta_vs_c002b"],
        "cache_only": report["cache_only"],
        "policy_refit": report["policy_refit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
