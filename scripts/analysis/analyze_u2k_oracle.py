"""Audit the exact OOF gain from an unavailable U->K oracle.

This is deliberately a *post-hoc diagnostic*.  It reads only tabular OOF
predictions and metadata, then changes a prediction to ``unknown`` exactly
when the saved reference says that its true label is ``unknown`` and the
baseline predicted a known speaker.  No model, audio, embedding, threshold,
or calibration routine is opened or fitted.

The counterfactual is useful for sizing the open-set problem, but it is not a
valid score for selection, promotion, packaging, or a leaderboard submission:
at inference time the true known/unknown membership is unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from speaker_id.evaluation.metrics import score_predictions, validate_labels


ANALYSIS_SCHEMA_VERSION = 1
ERROR_MODES = ("correct", "unknown_to_known", "known_to_unknown", "known_to_other_known")


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path, name: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{name}: missing header")
        return list(reader)


def _index_rows(rows: Iterable[dict[str, str]], labels: set[str], name: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name}: each row must be an object")
        audio_file, speaker_id = row.get("audio_file"), row.get("speaker_id")
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError(f"{name}: audio_file must be a nonempty string")
        if speaker_id not in labels:
            raise ValueError(f"{name}: speaker_id outside fixed label map: {speaker_id!r}")
        if audio_file in indexed:
            raise ValueError(f"{name}: duplicate audio_file: {audio_file!r}")
        indexed[audio_file] = dict(row)
    if not indexed:
        raise ValueError(f"{name}: no rows")
    return indexed


def _optional_folds(path: Path | None, reference: dict[str, dict[str, str]],
                    labels: set[str]) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = _index_rows(_read_csv(path, "folds"), labels, "folds")
    if set(rows) != set(reference):
        raise ValueError("folds: audio_file coverage differs from reference")
    for name, fold in rows.items():
        stated = fold.get("speaker_id")
        if stated not in (None, "", reference[name]["speaker_id"]):
            raise ValueError("folds: speaker_id differs from reference")
    return rows


def _error_mode(actual: str, predicted: str) -> str:
    if actual == predicted:
        return "correct"
    if actual == "unknown":
        return "unknown_to_known"
    if predicted == "unknown":
        return "known_to_unknown"
    return "known_to_other_known"


def _as_finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _duration_band(value: object) -> str:
    seconds = _as_finite_float(value)
    if seconds is None or seconds < 0:
        return "unavailable"
    if seconds < 1:
        return "under_1s"
    if seconds < 3:
        return "1_to_3s"
    if seconds < 5:
        return "3_to_5s"
    if seconds < 30:
        return "5_to_30s"
    return "at_least_30s"


def _signal_status(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "unavailable"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return "nonzero_signal"
    if normalized in {"0", "false", "no"}:
        return "zero_signal"
    return "unavailable"


def _compact(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _per_class_by_label(metrics: dict) -> dict[str, dict]:
    return {row["speaker_id"]: row for row in metrics["per_class"]}


def _unknown_population_slices(records: list[dict], field: str) -> dict[str, dict]:
    """Summarize U->K rates within true-unknown records for a fixed slice."""
    buckets: dict[str, list[dict]] = {}
    for record in records:
        if record["true_label"] != "unknown":
            continue
        buckets.setdefault(str(record[field]), []).append(record)
    return {
        key: {
            "true_unknown_rows": len(rows),
            "baseline_unknown_to_known": sum(row["is_u2k"] for row in rows),
            "baseline_unknown_to_known_rate": (
                sum(row["is_u2k"] for row in rows) / len(rows) if rows else None
            ),
        }
        for key, rows in sorted(buckets.items())
    }


def _group_summary(records: list[dict]) -> dict:
    known_groups = [row["group_id"] for row in records if row["group_id"] is not None]
    if not known_groups:
        return {"available": False}
    unknown = [row for row in records if row["true_label"] == "unknown" and row["group_id"] is not None]
    u2k = [row for row in unknown if row["is_u2k"]]
    per_group = Counter(row["group_id"] for row in u2k)
    labels_by_group: dict[str, set[str]] = {}
    for row in records:
        if row["group_id"] is not None:
            labels_by_group.setdefault(row["group_id"], set()).add(row["true_label"])
    return {
        "available": True,
        "all_content_groups": len(set(known_groups)),
        "true_unknown_content_groups": len({row["group_id"] for row in unknown}),
        "u2k_affected_content_groups": len(per_group),
        "u2k_groups_with_multiple_files": sum(count > 1 for count in per_group.values()),
        "max_u2k_files_in_one_group": max(per_group.values(), default=0),
        "mixed_label_content_groups_in_reference": sum(len(values) > 1 for values in labels_by_group.values()),
        "u2k_files_per_affected_group": {
            str(count): sum(value == count for value in per_group.values())
            for count in sorted(set(per_group.values()))
        },
    }


def analyze_u2k_oracle(reference_path: Path, predictions_path: Path, label_map_path: Path,
                       folds_path: Path | None = None) -> tuple[dict, list[dict], list[dict]]:
    """Score a truth-knownness oracle that changes *only* baseline U->K errors.

    The returned row records are in reference CSV order.  They preserve enough
    metadata for post-hoc slices, while the primary score is still the exact
    fixed 447-class metric from :func:`score_predictions`.
    """
    reference_path, predictions_path, label_map_path = map(Path, (
        reference_path, predictions_path, label_map_path,
    ))
    folds_path = Path(folds_path) if folds_path is not None else None
    labels = validate_labels(_read_json(label_map_path).get("labels", []))
    label_set = set(labels)
    reference_rows = _read_csv(reference_path, "reference")
    reference = _index_rows(reference_rows, label_set, "reference")
    predictions = _index_rows(_read_csv(predictions_path, "predictions"), label_set, "predictions")
    if set(reference) != set(predictions):
        raise ValueError("predictions: audio_file coverage differs from reference")
    folds = _optional_folds(folds_path, reference, label_set)

    baseline_predictions, oracle_predictions, records = [], [], []
    for source in reference_rows:
        audio_file = source["audio_file"]
        actual = reference[audio_file]["speaker_id"]
        predicted = predictions[audio_file]["speaker_id"]
        is_u2k = actual == "unknown" and predicted != "unknown"
        oracle_prediction = "unknown" if is_u2k else predicted
        fold = folds.get(audio_file, {})
        outer_fold = fold.get("outer_fold", fold.get("fold", source.get("outer_fold", source.get("fold", "unavailable"))))
        group_id = fold.get("group_id", source.get("group_id")) or None
        baseline_predictions.append({"audio_file": audio_file, "speaker_id": predicted})
        oracle_predictions.append({"audio_file": audio_file, "speaker_id": oracle_prediction})
        records.append({
            "audio_file": audio_file,
            "true_label": actual,
            "baseline_prediction": predicted,
            "oracle_prediction": oracle_prediction,
            "baseline_error_mode": _error_mode(actual, predicted),
            "oracle_error_mode": _error_mode(actual, oracle_prediction),
            "is_u2k": is_u2k,
            "outer_fold": str(outer_fold) if outer_fold not in (None, "") else "unavailable",
            "group_id": group_id,
            "duration_band": _duration_band(source.get("duration_seconds")),
            "signal_status": _signal_status(source.get("has_nonzero_signal")),
        })

    baseline = score_predictions(reference_rows, baseline_predictions, labels)
    oracle = score_predictions(reference_rows, oracle_predictions, labels)
    changed = [row for row in records if row["is_u2k"]]
    if len(changed) != baseline["errors"]["unknown_to_known"]:
        raise AssertionError("U->K oracle rows do not equal the baseline U->K error count")
    if oracle["errors"]["unknown_to_known"] != 0:
        raise AssertionError("U->K oracle did not remove every U->K error")
    if any(row["oracle_prediction"] != "unknown" for row in changed):
        raise AssertionError("U->K oracle changed row is not unknown")
    if any(row["oracle_prediction"] != row["baseline_prediction"] for row in records if not row["is_u2k"]):
        raise AssertionError("U->K oracle changed a non-U->K prediction")
    expected_accuracy_delta = len(changed) / len(records)
    accuracy_delta = oracle["accuracy"] - baseline["accuracy"]
    if not math.isclose(accuracy_delta, expected_accuracy_delta, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("Oracle accuracy delta must equal corrected U->K rows / all rows")
    if oracle["macro_f1"] + 1e-15 < baseline["macro_f1"]:
        raise AssertionError("Correcting true U->K errors unexpectedly lowered macro-F1")

    baseline_classes = _per_class_by_label(baseline)
    oracle_classes = _per_class_by_label(oracle)
    label_counts = Counter(row["baseline_prediction"] for row in changed)
    all_prediction_counts = Counter(row["baseline_prediction"] for row in records)
    by_predicted_label = []
    for label in sorted(label_counts):
        before, after = baseline_classes[label], oracle_classes[label]
        by_predicted_label.append({
            "predicted_speaker_id": label,
            "u2k_rows_removed_as_false_positives": label_counts[label],
            "baseline_predicted_rows": all_prediction_counts[label],
            "baseline_false_positives": before["false_positive"],
            "baseline_f1": before["f1"],
            "oracle_f1": after["f1"],
            "f1_delta": after["f1"] - before["f1"],
        })
    per_class_delta = [
        {"speaker_id": label, "baseline_f1": baseline_classes[label]["f1"],
         "oracle_f1": oracle_classes[label]["f1"],
         "f1_delta": oracle_classes[label]["f1"] - baseline_classes[label]["f1"]}
        for label in labels
        if not math.isclose(oracle_classes[label]["f1"], baseline_classes[label]["f1"], rel_tol=0.0, abs_tol=1e-15)
    ]
    baseline_modes = Counter(row["baseline_error_mode"] for row in records)
    transition_counts = Counter(
        f"{row['baseline_error_mode']}->{row['oracle_error_mode']}" for row in records
    )
    input_hashes = {
        "reference": _sha256(reference_path),
        "predictions": _sha256(predictions_path),
        "label_map": _sha256(label_map_path),
        "folds": _sha256(folds_path) if folds_path is not None else None,
    }
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "kind": "posthoc_truth_knownness_unknown_to_known_oracle",
        "metadata_only": True,
        "audio_opened": False,
        "model_loaded": False,
        "embedding_loaded": False,
        "threshold_search_performed": False,
        "eligible_for_model_selection": False,
        "eligible_for_submission": False,
        "counterfactual_definition": (
            "For each fixed OOF prediction, change it to unknown if and only if the true reference label is unknown "
            "and the baseline predicted a known speaker; retain every other baseline prediction exactly."
        ),
        "inputs": input_hashes,
        "alignment": {
            "reference_rows": len(reference),
            "prediction_rows": len(predictions),
            "exact_audio_file_coverage": True,
            "label_count": len(labels),
            "fold_metadata_checked": folds_path is not None,
        },
        "baseline": _compact(baseline),
        "u2k_oracle": _compact(oracle),
        "delta": {
            "macro_f1": oracle["macro_f1"] - baseline["macro_f1"],
            "accuracy": accuracy_delta,
            "corrected_rows": len(changed),
            "expected_accuracy_delta_from_corrected_rows": expected_accuracy_delta,
        },
        "taxonomy": {
            "baseline_error_modes": {name: baseline_modes[name] for name in ERROR_MODES},
            "baseline_to_oracle_transitions": dict(sorted(transition_counts.items())),
            "u2k_by_predicted_known_label": {
                row["predicted_speaker_id"]: row["u2k_rows_removed_as_false_positives"]
                for row in by_predicted_label
            },
            "per_class_f1_changed_by_oracle": per_class_delta,
        },
        "true_unknown_population_slices": {
            "outer_fold": _unknown_population_slices(records, "outer_fold"),
            "duration_band": _unknown_population_slices(records, "duration_band"),
            "signal_status": _unknown_population_slices(records, "signal_status"),
        },
        "content_group_summary": _group_summary(records),
        "interpretation_limits": [
            "This oracle uses the outer truth label to decide rejection, so it is unavailable at inference time and invalid for selection, promotion, packaging, or a leaderboard claim.",
            "It measures only the error mass of U->K on this frozen OOF prediction set. A feasible gate must trade missed unknown rejections against new K->U errors and be fitted only on eligible disjoint calibration rows.",
            "The oracle preserves the baseline ranking and every known-file prediction. It does not demonstrate that identity representation, calibration, or a threshold can realize the gain.",
            "Subgroup rates are descriptive. They do not establish a causal duration, signal, speaker, or content-group mechanism.",
        ],
    }
    return report, records, by_predicted_label


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output_dir: Path, report: dict, records: list[dict], by_predicted_label: list[dict]) -> None:
    """Write compact, traceable post-hoc receipts; no audio, model, or embeddings."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "u2k_rows.csv", [
        "audio_file", "true_label", "baseline_prediction", "oracle_prediction",
        "baseline_error_mode", "oracle_error_mode", "is_u2k", "outer_fold", "group_id",
        "duration_band", "signal_status",
    ], records)
    _write_csv(output_dir / "u2k_by_predicted_label.csv", [
        "predicted_speaker_id", "u2k_rows_removed_as_false_positives", "baseline_predicted_rows",
        "baseline_false_positives", "baseline_f1", "oracle_f1", "f1_delta",
    ], by_predicted_label)
    baseline, oracle, delta = report["baseline"], report["u2k_oracle"], report["delta"]
    markdown = f"""# U→K oracle audit

Baseline frozen OOF Macro-F1: **{baseline['macro_f1']:.9f}**.  If and only if every
true-unknown file that C002b predicted as a known speaker is corrected with
unavailable ground truth, Macro-F1 becomes **{oracle['macro_f1']:.9f}**
({delta['macro_f1']:+.9f}); accuracy changes by **{delta['accuracy']:+.9f}**
from {delta['corrected_rows']} corrected files.

This is an **oracle counterfactual**, not an eligible experimental result. It
keeps every known prediction and the full fixed C002b ranking unchanged, then
uses the outer truth label to identify exactly the unknown files to reject. A
real post-processing method must instead learn its decision from disjoint
calibration data and show that it does not introduce offsetting K→U errors.

`summary.json` records exact input hashes, alignment checks, error transitions,
per-label false-positive concentration, and descriptive duration/signal/fold
slices. `u2k_rows.csv` and `u2k_by_predicted_label.csv` make the fixed
counterfactual traceable. No audio, model, embedding, calibration, or threshold
search was used.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="C002b reference/manifest CSV with audio_file,speaker_id and optional metadata")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="C002b/oof_predictions.csv")
    parser.add_argument("--label-map", type=Path, required=True,
                        help="Fixed 447-label JSON map")
    parser.add_argument("--folds", type=Path,
                        help="Optional folds CSV for group and outer-fold slices")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, records, by_label = analyze_u2k_oracle(
        args.reference, args.predictions, args.label_map, args.folds,
    )
    write_analysis(args.output_dir, report, records, by_label)
    print(json.dumps({
        "status": report["status"],
        "baseline_macro_f1": report["baseline"]["macro_f1"],
        "u2k_oracle_macro_f1": report["u2k_oracle"]["macro_f1"],
        "macro_f1_delta": report["delta"]["macro_f1"],
        "corrected_u2k_rows": report["delta"]["corrected_rows"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
