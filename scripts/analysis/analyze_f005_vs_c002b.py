"""Create a read-only paired F005-versus-C002b diagnosis from saved outputs."""
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
    parser.add_argument("--c002b-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--folds", type=Path,
                        default=ROOT / "data/processed/eda_v1/folds.csv")
    parser.add_argument("--roles", type=Path,
                        default=ROOT / "data/processed/eda_v1/calibration_roles.csv")
    parser.add_argument("--label-map", type=Path,
                        default=ROOT / "data/processed/eda_v1/label_map.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from speaker_id.evaluation.f005_paired_analysis import (
        analyze_f005_vs_c002b,
        write_f005_paired_analysis,
    )

    report, paired_rows, class_rows = analyze_f005_vs_c002b(
        args.f005_dir, args.c002b_dir, args.manifest, args.folds, args.roles, args.label_map,
    )
    write_f005_paired_analysis(args.output_dir, report, paired_rows, class_rows)
    primary = report["comparisons"]["c002b_to_selected_arm"]
    print(json.dumps({
        "status": report["status"], "output_dir": str(args.output_dir),
        "row_count": report["row_count"],
        "macro_f1_delta_vs_c002b": primary["macro_f1_delta"],
        "corrected": report["paired_transitions"]["c002b_wrong_to_f005_correct"],
        "regressed": report["paired_transitions"]["c002b_correct_to_f005_wrong"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
