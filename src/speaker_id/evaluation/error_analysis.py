"""Read-only diagnosis of saved OOF probabilities; no model or threshold fitting.

Known-class ranks remove the recorded unknown logit from the ranking. A separate
counterfactual uses ground-truth known/unknown membership and therefore cannot
be used as a model result, leaderboard prediction, or selection criterion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions, validate_labels


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_known(rows: list[dict]) -> dict:
    """Ranks are defined only for known files with a nonzero saved signal."""
    if any(not row["known"] or not row["has_nonzero_signal"] or row["known_rank"] is None for row in rows):
        raise ValueError("Known ranking summaries require nonzero known examples")
    count = len(rows)
    result = {"files": count, "class_fold_pairs": len({(row["outer_fold"], row["speaker_id"]) for row in rows}),
              "baseline_correct": sum(row["correct"] for row in rows),
              "rejected_as_unknown": sum(row["predicted_speaker_id"] == "unknown" for row in rows),
              "true_top1_but_rejected": sum(row["known_rank"] == 1 and row["predicted_speaker_id"] == "unknown" for row in rows)}
    result["baseline_accuracy"] = result["baseline_correct"] / count if count else None
    for rank in (1, 2, 5):
        correct = sum(row["known_rank"] <= rank for row in rows)
        result[f"top{rank}_correct"] = correct
        result[f"top{rank}_accuracy"] = correct / count if count else None
    return result


def analyze_saved_probabilities(experiment_dir: Path, manifest_path: Path,
                                roles_path: Path, label_map_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Diagnose a completed experiment, preserving its original input identity."""
    experiment_dir = Path(experiment_dir)
    inputs = {"manifest": Path(manifest_path), "roles": Path(roles_path), "label_map": Path(label_map_path)}
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    resolved = json.loads((experiment_dir / "resolved_config.json").read_text(encoding="utf-8"))
    for name, digest in input_hashes.items():
        if resolved["input_hashes"][name] != digest:
            raise ValueError(f"Analysis {name} does not match this experiment's original input hash")
    labels = validate_labels(json.loads(inputs["label_map"].read_text(encoding="utf-8"))["labels"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    manifest = _csv(inputs["manifest"])
    sources = {row["audio_file"]: row for row in manifest}
    if not manifest or len(sources) != len(manifest):
        raise ValueError("Manifest must contain unique nonempty source rows")
    roles = _csv(inputs["roles"])
    fold_ids = sorted({int(row["outer_fold"]) for row in roles})
    rows, predictions, oracle_predictions, seen, probability_hashes = [], [], [], set(), {}
    for outer in fold_ids:
        selected = [row for row in roles if int(row["outer_fold"]) == outer]
        enrollment_files = Counter()
        enrollment_groups = defaultdict(set)
        expected_outer = {}
        for role in selected:
            filename = role["audio_file"]
            if filename not in sources or role["speaker_id"] != sources[filename]["speaker_id"]:
                raise ValueError("Calibration role source or label differs from manifest")
            if truth(role["outer_evaluation_included"]):
                if filename in expected_outer:
                    raise ValueError("Duplicate outer role")
                if truth(role["enrollment_allowed"]) or truth(role["calibration_query"]) or truth(role["encoder_fit_allowed"]):
                    raise ValueError("Outer evaluation role leaked into fit, gallery, or calibration")
                expected_outer[filename] = role
            if truth(role["enrollment_allowed"]):
                if truth(role["calibration_query"]) or role["speaker_id"] == "unknown":
                    raise ValueError("Invalid enrollment role")
                enrollment_files[role["speaker_id"]] += 1
                enrollment_groups[role["speaker_id"]].add(role["group_id"])
        path = experiment_dir / f"fold_{outer}" / "outer_probabilities.npz"
        probability_hashes[str(outer)] = _sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            probabilities = saved["probabilities"].copy()
            filenames = [str(value) for value in saved["audio_files"]]
            if [str(value) for value in saved["labels"]] != labels:
                raise ValueError("Saved probability label order differs from the fixed label map")
        if (probabilities.shape != (len(filenames), 447) or len(set(filenames)) != len(filenames)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
            raise ValueError("Malformed saved 447-class probability distributions")
        if set(filenames) != set(expected_outer) or seen.intersection(filenames):
            raise ValueError("Saved probabilities do not preserve disjoint outer-fold coverage")
        seen.update(filenames)
        # Stable tie handling matches first-index argmax in the recorded label map.
        ordered_known = np.argsort(-probabilities[:, 1:], axis=1, kind="stable") + 1
        for index, filename in enumerate(filenames):
            source = sources[filename]
            label = source["speaker_id"]
            if label not in label_to_index:
                raise ValueError("Source label is outside the fixed label map")
            known = label != "unknown"
            nonzero = truth(source["has_nonzero_signal"])
            prediction = labels[int(probabilities[index].argmax())]
            if nonzero and probabilities[index, 1:].sum() <= 0:
                raise ValueError("Known-class ranking information was lost in saved probabilities")
            top_known = labels[int(ordered_known[index, 0])] if nonzero else None
            rank = int(np.flatnonzero(ordered_known[index] == label_to_index[label])[0] + 1) if known and nonzero else None
            record = {"audio_file": filename, "outer_fold": outer, "speaker_id": label,
                      "predicted_speaker_id": prediction, "correct": prediction == label,
                      "known": known, "has_nonzero_signal": nonzero, "known_rank": rank,
                      "top_known_speaker_id": top_known, "duration_seconds": float(source["duration_seconds"]),
                      "mono_rms_dbfs": float(source["mono_rms_dbfs"]),
                      "enrollment_files": enrollment_files[label] if known else None,
                      "enrollment_groups": len(enrollment_groups[label]) if known else None}
            if known and not record["enrollment_files"]:
                raise ValueError("A known evaluation class has no permitted enrollment")
            rows.append(record)
            predictions.append({"audio_file": filename, "speaker_id": prediction})
            # This diagnostic explicitly uses ground truth to supply knownness.
            # All-zero audio keeps the original unknown fallback; arbitrary tied
            # known ranks from an all-zero distribution are never interpreted.
            oracle_predictions.append({"audio_file": filename, "speaker_id": top_known if known and nonzero else "unknown"})
    if seen != set(sources):
        raise ValueError("Saved outer probabilities must cover every original source exactly once")
    baseline = score_predictions(manifest, predictions, labels)
    oracle = score_predictions(manifest, oracle_predictions, labels)
    nonzero_known = [row for row in rows if row["known"] and row["has_nonzero_signal"]]
    per_class = []
    grouped = defaultdict(list)
    for row in rows:
        if row["known"]:
            grouped[(row["outer_fold"], row["speaker_id"])].append(row)
    for (outer, label), members in sorted(grouped.items()):
        eligible = [row for row in members if row["has_nonzero_signal"]]
        per_class.append({"outer_fold": outer, "speaker_id": label,
                          "enrollment_files": members[0]["enrollment_files"],
                          "enrollment_groups": members[0]["enrollment_groups"],
                          "all_evaluation_files": len(members), "zero_evaluation_files": len(members) - len(eligible),
                          **summarize_known(eligible)})
    support = {}
    for name in ("enrollment_files", "enrollment_groups"):
        support[name] = {}
        for value in sorted({row[name] for row in nonzero_known}):
            selected = [row for row in nonzero_known if row[name] == value]
            class_rows = [row for row in per_class if row[name] == value and row["files"]]
            support[name][str(value)] = {**summarize_known(selected),
                                        "mean_class_fold_top1_accuracy": float(np.mean([row["top1_accuracy"] for row in class_rows]))}
    slices = {
        "duration_under_5_seconds": lambda row: row["duration_seconds"] < 5,
        "duration_5_to_30_seconds": lambda row: 5 <= row["duration_seconds"] < 30,
        "duration_at_least_30_seconds": lambda row: row["duration_seconds"] >= 30,
        "rms_below_minus50_dbfs": lambda row: row["mono_rms_dbfs"] < -50,
        "rms_at_least_minus50_dbfs": lambda row: row["mono_rms_dbfs"] >= -50,
    }
    compact = lambda metrics: {key: value for key, value in metrics.items() if key != "per_class"}
    report = {"analysis_version": 1, "status": "complete", "analysis_kind": "observational_saved_probability_diagnosis",
              "training_or_threshold_search_performed": False, "experiment_dir": str(experiment_dir),
              "input_hashes": input_hashes, "probability_sha256": probability_hashes,
              "analysis_code_sha256": _sha256(Path(__file__)), "baseline": compact(baseline),
              "known_zero_signal_files_excluded_from_ranking": sum(row["known"] and not row["has_nonzero_signal"] for row in rows),
              "nonzero_known": summarize_known(nonzero_known),
              "folds": {str(outer): summarize_known([row for row in nonzero_known if row["outer_fold"] == outer]) for outer in fold_ids},
              "quality_slices_nonzero_known": {name: summarize_known([row for row in nonzero_known if condition(row)]) for name, condition in slices.items()},
              "enrollment_support_association": support,
              "ground_truth_knownness_oracle_with_current_known_argmax": compact(oracle),
              "interpretation_limits": [
                  "The oracle uses ground-truth known/unknown membership and is ineligible for model selection, submission, or performance claims.",
                  "The oracle retains zero-signal unknown fallback; it is a counterfactual for this fixed ranking, not an absolute upper bound for every rejection strategy.",
                  "Top-k accuracy is conditional on nonzero known files and is not the competition's 447-class Macro-F1.",
                  "Support and quality associations are descriptive, confounded by class identity and recording characteristics, and are not causal ablations.",
                  "Outer labels are used here only to diagnose an already completed run; future learned scoring and threshold choices must use eligible inner data.",
                  "Internal small-gallery validation cannot be directly equated to an unverified leaderboard number or a different enrollment protocol.",
              ]}
    return report, rows, per_class


def write_analysis(output_dir: Path, report: dict, rows: list[dict], per_class: list[dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, data in (("file_ranks.csv", rows), ("class_enrollment_support.csv", per_class)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    known = report["nonzero_known"]
    oracle = report["ground_truth_knownness_oracle_with_current_known_argmax"]
    text = f"""# Saved-probability diagnosis

Baseline OOF Macro-F1: **{report['baseline']['macro_f1']:.6f}** on {report['baseline']['row_count']} files / 447 labels.

Among {known['files']} nonzero known files, threshold-independent top-1 / top-2 / top-5 accuracy is **{known['top1_accuracy']:.4%} / {known['top2_accuracy']:.4%} / {known['top5_accuracy']:.4%}**. Of {known['rejected_as_unknown']} rejected nonzero known files, {known['true_top1_but_rejected']} have the true identity ranked first. {report['known_zero_signal_files_excluded_from_ranking']} known zero-signal files have no interpretable stored identity ranking.

The ground-truth-knownness counterfactual gives **{oracle['macro_f1']:.6f} Macro-F1** with current known argmax and zero-signal fallback. It uses unavailable ground truth, is not an eligible model result, and is not an absolute upper bound for all scoring changes.

`summary.json` contains fold, duration, signal-level and enrollment-support summaries. `file_ranks.csv` and `class_enrollment_support.csv` provide traceable observations. Support associations are not causal: class identity and recording quality differ between groups. No model training or threshold search was performed.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")
