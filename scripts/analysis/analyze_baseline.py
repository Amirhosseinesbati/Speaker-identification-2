"""Analyze completed OOF probabilities without loading an encoder or fitting anything."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--roles", type=Path, default=ROOT / "data/processed/eda_v1/calibration_roles.csv")
    parser.add_argument("--label-map", type=Path, default=ROOT / "data/processed/eda_v1/label_map.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from speaker_id.evaluation.error_analysis import analyze_saved_probabilities, write_analysis
    report, rows, classes = analyze_saved_probabilities(args.experiment_dir, args.manifest, args.roles, args.label_map)
    write_analysis(args.output_dir, report, rows, classes)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir),
                      "nonzero_known": report["nonzero_known"],
                      "diagnostic_oracle_macro_f1": report["ground_truth_knownness_oracle_with_current_known_argmax"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
