"""Freeze inner calibration roles and the 447-class metric contract; fit no model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.data.calibration import build_calibration_roles
from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions, validate_labels


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/eda_v1")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/eda")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    paths = {name: args.processed_dir / name for name in ("folds.csv", "audio_manifest.csv", "label_map.json")}
    folds, manifest = read_csv(paths["folds.csv"]), read_csv(paths["audio_manifest.csv"])
    labels = validate_labels(json.loads(paths["label_map.json"].read_text(encoding="utf-8"))["labels"])
    if len(folds) != len(manifest) or {r["audio_file"]: r["speaker_id"] for r in folds} != {r["audio_file"]: r["speaker_id"] for r in manifest}:
        raise ValueError("Fold rows must match the full manifest")
    roles, summary = build_calibration_roles(folds, args.seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    role_path = args.processed_dir / "calibration_roles.csv"
    with role_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(roles[0]))
        writer.writeheader()
        writer.writerows(roles)
    source_hashes = {name: digest(path) for name, path in paths.items()}
    code_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in
                   (ROOT / "src/speaker_id/data/calibration.py", ROOT / "src/speaker_id/evaluation/metrics.py", Path(__file__).resolve())}
    summary.update({"source_sha256": source_hashes, "code_sha256": code_hashes,
                    "calibration_roles_sha256": digest(role_path)})
    (args.report_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reference = [{"audio_file": r["audio_file"], "speaker_id": r["speaker_id"]} for r in manifest]
    all_unknown = score_predictions(reference, [{**r, "speaker_id": "unknown"} for r in reference], labels)
    hypothetical = score_predictions(reference, [{"audio_file": r["audio_file"], "speaker_id": r["speaker_id"] if truth(r["has_nonzero_signal"]) else "unknown"} for r in manifest], labels)
    compact = lambda score: {key: value for key, value in score.items() if key != "per_class"}
    contract = {"version": "metric_contract_v1", "class_count": len(labels), "unknown_index": 0,
                "source_sha256": source_hashes, "code_sha256": code_hashes,
                "primary": "Macro-F1: unweighted mean over the fixed 447 label-map entries, calculated once from all original pooled out-of-fold predictions",
                "row_alignment": "Match audio_file exactly; reject duplicate, missing, extra filenames and unknown label values",
                "zero_denominator": "Per-class precision, recall or F1 with zero denominator is zero; never drop a class",
                "probability_contract": "447 finite probabilities in [0,1], sum within 1e-6 of one; first-index argmax in fixed label-map order",
                "fold_reporting": "Individual fold scores are diagnostics; their mean is not substituted for pooled out-of-fold Macro-F1",
                "invalid_signal_policy": "All invalid or zero inputs remain scoreable; production fallback must be explicit and tested",
                "all_unknown_baseline": compact(all_unknown),
                "perfect_nonzero_with_zero_to_unknown_diagnostic": compact(hypothetical),
                "diagnostic_interpretation": "Analytical policy-specific scenario using true labels for all nonzero files; no model was fitted. This is neither a measured model performance nor a universal attainable ceiling",
                "zero_affected_known_classes": [r for r in hypothetical["per_class"] if r["speaker_id"] != "unknown" and r["false_negative"]],
                "status": "contract_tested_no_model_or_threshold_fitted"}
    (args.report_dir / "metric_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role_rows": len(roles), "folds": summary["folds"],
                      "all_unknown_macro_f1": all_unknown["macro_f1"],
                      "perfect_nonzero_zero_unknown_macro_f1": hypothetical["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
