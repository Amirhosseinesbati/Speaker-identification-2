"""Read-only paired diagnosis of completed F005 against its pinned C002b source.

The module intentionally consumes only saved predictions, score metadata, and
probabilities.  It never loads an encoder, audio file, checkpoint, or a
calibration routine.  The file-level output is observational evidence for the
next experiment; it is not a submission or a selection metric.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import (
    predictions_from_probabilities,
    score_predictions,
    validate_labels,
)
from speaker_id.models.campp import file_sha256


ANALYSIS_VERSION = 1
F005_COMPARATORS = ("frozen_same_protocol", "fresh_control", "selected_arm")
PAIR_COMPARISONS = (
    ("c002b_to_frozen_same_protocol", "c002b", "frozen_same_protocol"),
    ("frozen_same_protocol_to_fresh_control", "frozen_same_protocol", "fresh_control"),
    ("c002b_to_selected_arm", "c002b", "selected_arm"),
)


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: list[dict], labels: set[str], name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name}: rows must be objects")
        audio_file = row.get("audio_file")
        speaker_id = row.get("speaker_id")
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError(f"{name}: audio_file must be a nonempty string")
        if speaker_id not in labels:
            raise ValueError(f"{name}: speaker_id is outside the fixed label map")
        if audio_file in indexed:
            raise ValueError(f"{name}: duplicate audio_file {audio_file}")
        indexed[audio_file] = row
    if not indexed:
        raise ValueError(f"{name}: no rows")
    return indexed


def _prediction_index(rows: object, labels: set[str], name: str, expected: set[str]) -> dict[str, str]:
    if not isinstance(rows, list):
        raise ValueError(f"{name}: predictions must be a list")
    indexed = _index_rows(rows, labels, name)
    if set(indexed) != expected:
        raise ValueError(f"{name}: predictions do not cover the expected outer files")
    return {audio_file: row["speaker_id"] for audio_file, row in indexed.items()}


def _float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: expected a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number")
    return result


def _compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _assert_metrics_match(recorded: object, computed: dict, name: str) -> None:
    if not isinstance(recorded, dict):
        raise ValueError(f"{name}: missing recorded metrics")
    expected = _compact_metrics(computed)
    for key, value in expected.items():
        if key == "errors":
            if recorded.get(key) != value:
                raise ValueError(f"{name}: recorded error counts differ from predictions")
        elif isinstance(value, float):
            if not np.isclose(recorded.get(key), value, rtol=0.0, atol=1e-12):
                raise ValueError(f"{name}: recorded {key} differs from predictions")
        elif recorded.get(key) != value:
            raise ValueError(f"{name}: recorded {key} differs from predictions")


def _error_mode(actual: str, predicted: str) -> str:
    if predicted == actual:
        return "correct"
    if actual == "unknown":
        return "unknown_to_known"
    if predicted == "unknown":
        return "known_to_unknown"
    return "known_to_other_known"


def _duration_band(seconds: float) -> str:
    if seconds < 3.0:
        return "under_3s"
    if seconds < 5.0:
        return "3_to_5s"
    if seconds < 8.0:
        return "5_to_8s"
    if seconds < 30.0:
        return "8_to_30s"
    return "at_least_30s"


def _transition(before_correct: bool, after_correct: bool, before: str, after: str) -> str:
    if before_correct and after_correct:
        return "both_correct"
    if before_correct:
        return "regressed"
    if after_correct:
        return "corrected"
    return "both_wrong_same_prediction" if before == after else "both_wrong_changed_prediction"


def _load_inputs(manifest_path: Path, folds_path: Path, roles_path: Path,
                 label_map_path: Path) -> tuple[list[str], dict[str, dict], dict[str, dict], dict[int, dict[str, dict]], dict]:
    labels = validate_labels(_read_json(label_map_path)["labels"])
    label_set = set(labels)
    manifest = _index_rows(_read_csv(manifest_path), label_set, "manifest")
    folds = _index_rows(_read_csv(folds_path), label_set, "folds")
    if set(manifest) != set(folds):
        raise ValueError("Manifest and folds do not cover the same files")
    for audio_file, source in manifest.items():
        fold = folds[audio_file]
        if fold.get("speaker_id") != source.get("speaker_id"):
            raise ValueError("Manifest and folds disagree on label")
        if not isinstance(fold.get("group_id"), str) or not fold["group_id"]:
            raise ValueError("Folds must contain a nonempty group_id")
        # Older, otherwise valid manifests did not duplicate group_id.  When
        # it is present it must agree; folds remain the canonical split table.
        if source.get("group_id") not in (None, "", fold["group_id"]):
            raise ValueError("Manifest and folds disagree on content group")
    roles_by_outer: dict[int, dict[str, dict]] = {}
    for row in _read_csv(roles_path):
        if not isinstance(row, dict) or "outer_fold" not in row:
            raise ValueError("Roles need outer_fold")
        try:
            outer = int(row["outer_fold"])
        except (TypeError, ValueError) as error:
            raise ValueError("Roles outer_fold must be an integer") from error
        item = _index_rows([row], label_set, "roles")
        audio_file = next(iter(item))
        bucket = roles_by_outer.setdefault(outer, {})
        if audio_file in bucket:
            raise ValueError("Roles duplicate an audio_file within an outer fold")
        bucket[audio_file] = row
    if not roles_by_outer or any(set(rows) != set(manifest) for rows in roles_by_outer.values()):
        raise ValueError("Each outer fold must have one role for every manifest file")
    for outer, rows in roles_by_outer.items():
        for audio_file, role in rows.items():
            source, fold = manifest[audio_file], folds[audio_file]
            if role.get("speaker_id") != source.get("speaker_id") or role.get("group_id") != fold.get("group_id"):
                raise ValueError("Roles disagree with manifest or folds")
            is_outer = int(fold["fold"]) == outer
            if truth(role.get("outer_evaluation_included")) != is_outer:
                raise ValueError("Roles outer-evaluation assignment differs from folds")
    input_hashes = {
        "manifest": _sha256(manifest_path), "folds": _sha256(folds_path),
        "roles": _sha256(roles_path), "label_map": _sha256(label_map_path),
    }
    return labels, manifest, folds, roles_by_outer, input_hashes


def _load_f005_fold(f005_dir: Path, outer: int, labels: list[str]) -> dict:
    path = f005_dir / "evaluation" / f"fold_{outer}.json"
    value = _read_json(path)
    if value.get("outer_fold") != outer:
        raise ValueError("F005 evaluation belongs to a different outer fold")
    references = _index_rows(value.get("outer_reference", []), set(labels), "F005 outer_reference")
    for audio_file, row in references.items():
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("F005 outer_reference has no content group")
        _float(row.get("duration_seconds"), f"F005 duration {audio_file}")
        try:
            truth(row.get("has_nonzero_signal"))
        except Exception as error:
            raise ValueError("F005 outer_reference has invalid has_nonzero_signal") from error
    comparators = value.get("comparators")
    if not isinstance(comparators, dict) or set(comparators) != set(F005_COMPARATORS):
        raise ValueError("F005 comparators differ from the saved paired protocol")
    expected = set(references)
    predictions, known_top1, metrics = {}, {}, {}
    reference_rows = [{"audio_file": name, "speaker_id": row["speaker_id"]} for name, row in references.items()]
    for comparator in F005_COMPARATORS:
        item = comparators[comparator]
        if not isinstance(item, dict):
            raise ValueError("F005 comparator must be an object")
        predictions[comparator] = _prediction_index(
            item.get("predictions"), set(labels), f"F005 {comparator}", expected
        )
        known_top1[comparator] = _prediction_index(
            item.get("known_top1_predictions"), set(labels), f"F005 {comparator} known top1", expected
        )
        if any(label == "unknown" and truth(references[name]["has_nonzero_signal"])
               for name, label in known_top1[comparator].items()):
            raise ValueError("F005 nonzero known-top1 predictions may not contain unknown")
        calculated = score_predictions(
            reference_rows,
            [{"audio_file": name, "speaker_id": predictions[comparator][name]} for name in references],
            labels,
        )
        _assert_metrics_match(item.get("metrics"), calculated, f"F005 {comparator}")
        metrics[comparator] = calculated
    selected_arm = value.get("selected_arm")
    if not isinstance(selected_arm, str) or not selected_arm:
        raise ValueError("F005 fold does not record its selected arm")
    selected_matches_fresh = predictions["selected_arm"] == predictions["fresh_control"]
    if selected_arm == "control" and not selected_matches_fresh:
        raise ValueError("F005 selected control differs from its fresh control predictions")
    return {
        "references": references, "predictions": predictions, "known_top1": known_top1,
        "metrics": metrics, "selected_arm": selected_arm,
        "selected_matches_fresh": selected_matches_fresh, "sha256": _sha256(path),
    }


def _load_c002b_probabilities(c002b_dir: Path, outer: int, labels: list[str], expected: set[str]) -> tuple[dict[str, str], dict[str, np.ndarray], str]:
    path = c002b_dir / f"fold_{outer}" / "outer_probabilities.npz"
    with np.load(path, allow_pickle=False) as saved:
        required = {"probabilities", "audio_files", "labels"}
        if set(saved.files) != required:
            raise ValueError("C002b saved probabilities have unexpected fields")
        probabilities = saved["probabilities"].copy()
        audio_files = [str(value) for value in saved["audio_files"].tolist()]
        saved_labels = [str(value) for value in saved["labels"].tolist()]
    if saved_labels != labels:
        raise ValueError("C002b probability label order differs from the fixed label map")
    if (probabilities.shape != (len(audio_files), len(labels)) or len(set(audio_files)) != len(audio_files)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)):
        raise ValueError("C002b saved probabilities are malformed")
    if set(audio_files) != expected:
        raise ValueError("C002b probability rows do not match F005 outer coverage")
    predictions = predictions_from_probabilities(audio_files, probabilities, labels)
    prediction_map = {row["audio_file"]: row["speaker_id"] for row in predictions}
    ordered_known = np.argsort(-probabilities[:, 1:], axis=1, kind="stable")
    # The full rank for the truth is filled once its label is known.  Keep the
    # saved known ordering only in memory; it is never written to the report.
    rank_orders = {name: ordered_known[position] for position, name in enumerate(audio_files)}
    return prediction_map, rank_orders, _sha256(path)


def _rank_for_truth(rank_orders: dict[str, np.ndarray], audio_file: str, speaker_id: str,
                    labels: list[str], *, known: bool, nonzero: bool) -> int | None:
    if not known or not nonzero:
        return None
    index = labels.index(speaker_id) - 1
    if index < 0:
        raise ValueError("Known rank requested for unknown label")
    return int(np.flatnonzero(rank_orders[audio_file] == index)[0] + 1)


def _array_from_descriptor(structure: object, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(structure, dict) or set(structure) != {"__ndarray__", "dtype", "shape"}:
        raise ValueError("F005 pretruth bundle array descriptor is malformed")
    key = structure["__ndarray__"]
    if key not in arrays:
        raise ValueError("F005 pretruth bundle array is missing")
    value = np.asarray(arrays[key])
    if value.dtype.str != structure["dtype"] or list(value.shape) != structure["shape"]:
        raise ValueError("F005 pretruth bundle array metadata differs from bytes")
    return value


def _load_selected_ranks(f005_dir: Path, outer: int, selected_arm: str,
                         expected_files: list[str], labels: list[str]) -> dict[str, tuple[np.ndarray | None, str | None]] | None:
    # One pretruth bundle stores every comparator for an outer fold.  Arm
    # checkpoints/caches sit below arm directories, but this sealed scoring
    # receipt deliberately sits at the fold root.
    base = f005_dir / "full_scoring" / f"fold_{outer}"
    metadata_path, arrays_path = base / "pretruth_bundle.json", base / "pretruth_bundle.npz"
    if not metadata_path.is_file() and not arrays_path.is_file():
        return None
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise ValueError("F005 selected pretruth bundle is incomplete")
    metadata = _read_json(metadata_path)
    if metadata.get("schema_version") != 1 or metadata.get("arrays_file_sha256") != _sha256(arrays_path):
        raise ValueError("F005 selected pretruth bundle hash differs")
    structure = metadata.get("structure")
    if (not isinstance(structure, dict) or structure.get("outer_fold") != outer
            or structure.get("selected_arm") != selected_arm):
        raise ValueError("F005 selected pretruth bundle belongs to another fold")
    if structure.get("outer_files") != expected_files:
        raise ValueError("F005 selected pretruth bundle order differs from saved evaluation")
    bundles = structure.get("score_bundles")
    if not isinstance(bundles, dict) or "selected_arm" not in bundles:
        raise ValueError("F005 selected pretruth bundle lacks selected-arm scores")
    bundle = bundles["selected_arm"]
    if not isinstance(bundle, dict) or bundle.get("known_labels") != labels[1:]:
        raise ValueError("F005 selected pretruth label order differs from fixed labels")
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    scores = _array_from_descriptor(bundle.get("outer_known_scores"), arrays)
    valid = _array_from_descriptor(bundle.get("outer_valid"), arrays)
    if (scores.shape != (len(expected_files), len(labels) - 1) or scores.dtype != np.float32
            or valid.shape != (len(expected_files),) or valid.dtype != np.bool_
            or not np.isfinite(scores).all()):
        raise ValueError("F005 selected pretruth score geometry differs")
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        audio_file: (order[position].copy(), labels[int(order[position, 0] + 1)])
        if bool(valid[position]) else (None, None)
        for position, audio_file in enumerate(expected_files)
    }


def _enrollment_support(roles_by_outer: dict[int, dict[str, dict]], outer: int) -> dict[str, tuple[int, int]]:
    counts: Counter = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in roles_by_outer[outer].values():
        if truth(row.get("enrollment_allowed")):
            label = row["speaker_id"]
            if label == "unknown":
                raise ValueError("Unknown file enters enrollment support")
            counts[label] += 1
            groups[label].add(row["group_id"])
    return {label: (counts[label], len(groups[label])) for label in counts}


def _pair_counts(rows: list[dict], before: str, after: str) -> dict:
    counts = Counter()
    modes = Counter()
    for row in rows:
        before_correct, after_correct = row[f"{before}_correct"], row[f"{after}_correct"]
        counts[_transition(before_correct, after_correct, row[f"{before}_prediction"], row[f"{after}_prediction"])] += 1
        modes[f"{row[f'{before}_error_mode']}->{row[f'{after}_error_mode']}"] += 1
    return {
        "row_count": len(rows), "corrected": counts["corrected"], "regressed": counts["regressed"],
        "both_correct": counts["both_correct"],
        "both_wrong_same_prediction": counts["both_wrong_same_prediction"],
        "both_wrong_changed_prediction": counts["both_wrong_changed_prediction"],
        "error_mode_transitions": dict(sorted(modes.items())),
    }


def _pair_summary(rows: list[dict], before: str, after: str, metrics: dict[str, dict]) -> dict:
    before_metrics, after_metrics = _compact_metrics(metrics[before]), _compact_metrics(metrics[after])
    slices = {}
    for band in ("under_3s", "3_to_5s", "5_to_8s", "8_to_30s", "at_least_30s"):
        slices[band] = _pair_counts([row for row in rows if row["duration_band"] == band], before, after)
    return {
        "before": before, "after": after,
        "before_metrics": before_metrics, "after_metrics": after_metrics,
        "macro_f1_delta": after_metrics["macro_f1"] - before_metrics["macro_f1"],
        "accuracy_delta": after_metrics["accuracy"] - before_metrics["accuracy"],
        "transitions": _pair_counts(rows, before, after), "duration_slices": slices,
    }


def _rank_summary(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["speaker_id"] != "unknown" and row["has_nonzero_signal"]]
    c002b_rank1 = sum(row["c002b_known_rank"] == 1 for row in eligible)
    selected_top1 = sum(row["selected_arm_known_top1"] == row["speaker_id"] for row in eligible)
    selected_rank_available = all(row["f005_selected_known_rank"] is not None for row in eligible)
    result = {
        "nonzero_known_files": len(eligible),
        "c002b_rank1": c002b_rank1,
        "c002b_rank1_accuracy": c002b_rank1 / len(eligible) if eligible else None,
        "f005_selected_known_top1": selected_top1,
        "f005_selected_known_top1_accuracy": selected_top1 / len(eligible) if eligible else None,
        "f005_full_rank_available": selected_rank_available,
    }
    if selected_rank_available:
        selected_rank1 = sum(row["f005_selected_known_rank"] == 1 for row in eligible)
        result.update({
            "f005_selected_rank1": selected_rank1,
            "f005_selected_rank1_accuracy": selected_rank1 / len(eligible) if eligible else None,
            "rank_improved": sum(row["f005_selected_known_rank"] < row["c002b_known_rank"] for row in eligible),
            "rank_worsened": sum(row["f005_selected_known_rank"] > row["c002b_known_rank"] for row in eligible),
            "rank_unchanged": sum(row["f005_selected_known_rank"] == row["c002b_known_rank"] for row in eligible),
        })
    return result


def _class_rows(rows: list[dict], labels: list[str], metrics: dict[str, dict]) -> list[dict]:
    before = {row["speaker_id"]: row for row in metrics["c002b"]["per_class"]}
    after = {row["speaker_id"]: row for row in metrics["selected_arm"]["per_class"]}
    output = []
    for label in labels:
        matching = [row for row in rows if row["speaker_id"] == label]
        output.append({
            "speaker_id": label, "support": before[label]["support"],
            "c002b_f1": before[label]["f1"], "f005_selected_f1": after[label]["f1"],
            "f1_delta": after[label]["f1"] - before[label]["f1"],
            "c002b_correct": sum(row["c002b_correct"] for row in matching),
            "f005_selected_correct": sum(row["selected_arm_correct"] for row in matching),
            "corrected": sum(not row["c002b_correct"] and row["selected_arm_correct"] for row in matching),
            "regressed": sum(row["c002b_correct"] and not row["selected_arm_correct"] for row in matching),
        })
    return output


def _verify_source_binding(f005_dir: Path, c002b_dir: Path, outer_folds: list[int]) -> dict:
    path = f005_dir / "source_verification.json"
    if not path.is_file():
        return {"available": False, "verified": False}
    source = _read_json(path)
    artifacts = source.get("c002b_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("F005 source verification lacks C002b artifact identities")
    required = ["C002b/oof_predictions.csv"] + [f"C002b/fold_{outer}/outer_probabilities.npz" for outer in outer_folds]
    for relative in required:
        expected = artifacts.get(relative)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("F005 source verification lacks a required C002b hash")
        actual = file_sha256(c002b_dir / relative.removeprefix("C002b/"))
        if actual != expected:
            raise ValueError("C002b source bytes differ from F005's pinned receipt")
    return {"available": True, "verified": True, "source_verification_sha256": _sha256(path)}


def analyze_f005_vs_c002b(f005_dir: Path, c002b_dir: Path, manifest_path: Path,
                           folds_path: Path, roles_path: Path, label_map_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Compare a completed F005 output against its immutable C002b predictions.

    This is a read-only post-run analysis.  It validates every saved prediction
    against the fixed input tables and returns aggregate evidence plus file and
    class transition tables.  It performs no threshold search or model work.
    """
    f005_dir, c002b_dir = Path(f005_dir), Path(c002b_dir)
    labels, manifest, folds, roles_by_outer, input_hashes = _load_inputs(
        Path(manifest_path), Path(folds_path), Path(roles_path), Path(label_map_path)
    )
    outer_folds = sorted({int(row["fold"]) for row in folds.values()})
    if set(outer_folds) != set(roles_by_outer):
        raise ValueError("Folds and roles disagree on outer-fold coverage")
    binding = _verify_source_binding(f005_dir, c002b_dir, outer_folds)
    c002b_csv = _prediction_index(_read_csv(c002b_dir / "oof_predictions.csv"), set(labels), "C002b OOF CSV", set(manifest))
    all_rows: list[dict] = []
    metrics_inputs: dict[str, list[dict]] = {"c002b": [], **{name: [] for name in F005_COMPARATORS}}
    f005_hashes, c002b_hashes = {}, {"oof_predictions.csv": _sha256(c002b_dir / "oof_predictions.csv")}
    selected_matches_fresh, c002b_argmax_matches_csv = [], []
    for outer in outer_folds:
        expected = {name for name, row in folds.items() if int(row["fold"]) == outer}
        if not expected:
            raise ValueError("An outer fold has no files")
        f005 = _load_f005_fold(f005_dir, outer, labels)
        if set(f005["references"]) != expected:
            raise ValueError("F005 outer evaluation differs from original folds")
        f005_hashes[str(outer)] = f005["sha256"]
        c002b_predictions, c002b_order, c002b_hash = _load_c002b_probabilities(c002b_dir, outer, labels, expected)
        c002b_hashes[f"fold_{outer}/outer_probabilities.npz"] = c002b_hash
        if any(c002b_predictions[name] != c002b_csv[name] for name in expected):
            raise ValueError("C002b OOF CSV differs from its saved probability argmax")
        c002b_argmax_matches_csv.append(True)
        support = _enrollment_support(roles_by_outer, outer)
        ordered_names = list(f005["references"])
        selected_ranks = _load_selected_ranks(f005_dir, outer, f005["selected_arm"], ordered_names, labels)
        for audio_file in ordered_names:
            source, fold, reference = manifest[audio_file], folds[audio_file], f005["references"][audio_file]
            if (reference["speaker_id"] != source["speaker_id"] or reference["group_id"] != fold["group_id"]
                    or reference["group_id"] != fold["group_id"]):
                raise ValueError("F005 outer reference differs from source identity")
            duration = _float(reference["duration_seconds"], "F005 duration")
            has_nonzero_signal = truth(reference["has_nonzero_signal"])
            actual = source["speaker_id"]
            known = actual != "unknown"
            row = {
                "audio_file": audio_file, "outer_fold": outer, "speaker_id": actual,
                "group_id": fold["group_id"], "duration_seconds": duration,
                "duration_band": _duration_band(duration), "has_nonzero_signal": has_nonzero_signal,
                "enrollment_files": support.get(actual, (None, None))[0] if known else None,
                "enrollment_groups": support.get(actual, (None, None))[1] if known else None,
                "c002b_prediction": c002b_predictions[audio_file],
                "c002b_known_rank": _rank_for_truth(c002b_order, audio_file, actual, labels, known=known, nonzero=has_nonzero_signal),
            }
            for comparator in F005_COMPARATORS:
                prediction = f005["predictions"][comparator][audio_file]
                row[f"{comparator}_prediction"] = prediction
                row[f"{comparator}_known_top1"] = f005["known_top1"][comparator][audio_file]
            # Keep the published table self-explanatory while the internal
            # comparison key remains ``selected_arm`` for all F005 variants.
            row["f005_selected_prediction"] = row["selected_arm_prediction"]
            row["f005_selected_known_top1"] = row["selected_arm_known_top1"]
            if selected_ranks is None:
                row["f005_selected_known_rank"] = None
            else:
                rank_order, top1 = selected_ranks[audio_file]
                row["f005_selected_known_rank"] = (
                    _rank_for_truth({audio_file: rank_order}, audio_file, actual, labels,
                                    known=known, nonzero=has_nonzero_signal)
                    if rank_order is not None else None
                )
                if top1 is not None and top1 != row["selected_arm_known_top1"]:
                    raise ValueError("F005 selected pretruth rank cache differs from saved known top1")
            for prefix in ("c002b", *F005_COMPARATORS):
                row[f"{prefix}_correct"] = row[f"{prefix}_prediction"] == actual
                row[f"{prefix}_error_mode"] = _error_mode(actual, row[f"{prefix}_prediction"])
                metrics_inputs[prefix].append({"audio_file": audio_file, "speaker_id": row[f"{prefix}_prediction"]})
            row["f005_selected_correct"] = row["selected_arm_correct"]
            row["f005_error_mode"] = row["selected_arm_error_mode"]
            all_rows.append(row)
        selected_matches_fresh.append(f005["selected_matches_fresh"])
    if {row["audio_file"] for row in all_rows} != set(manifest) or len(all_rows) != len(manifest):
        raise ValueError("Paired analysis did not preserve complete unique OOF coverage")
    reference_rows = [{"audio_file": name, "speaker_id": row["speaker_id"]} for name, row in manifest.items()]
    metrics = {name: score_predictions(reference_rows, predictions, labels) for name, predictions in metrics_inputs.items()}
    primary = _pair_counts(all_rows, "c002b", "selected_arm")
    paired_transitions = {
        "c002b_correct_to_f005_wrong": primary["regressed"],
        "c002b_wrong_to_f005_correct": primary["corrected"],
        "both_correct": primary["both_correct"],
        "both_wrong": primary["both_wrong_same_prediction"] + primary["both_wrong_changed_prediction"],
        **primary,
    }
    report = {
        "analysis_version": ANALYSIS_VERSION, "status": "complete",
        "analysis_kind": "f005_vs_c002b_paired_saved_prediction_analysis",
        "training_or_threshold_search_performed": False, "policy_or_threshold_changed": False,
        "input_hashes": input_hashes, "f005_evaluation_sha256": f005_hashes,
        "c002b_artifact_sha256": c002b_hashes, "source_hash_binding": binding,
        "analysis_code_sha256": _sha256(Path(__file__)),
        "row_count": len(all_rows), "class_count": len(labels),
        "integrity": {
            "full_oof_coverage": True,
            "row_count": len(all_rows), "class_count": len(labels), "outer_folds": outer_folds,
            "f005_selected_matches_fresh_control": all(selected_matches_fresh),
            "c002b_csv_matches_probability_argmax": all(c002b_argmax_matches_csv),
        },
        "metrics": {name: _compact_metrics(value) for name, value in metrics.items()},
        "paired_transitions": paired_transitions,
        "comparisons": {name: _pair_summary(all_rows, before, after, metrics)
                        for name, before, after in PAIR_COMPARISONS},
        "known_rank_diagnosis": _rank_summary(all_rows),
        "interpretation_limits": [
            "This report diagnoses already materialized outer predictions; it is not eligible for threshold, policy, or model selection.",
            "Saved top-1 and rank evidence describes representation and scoring jointly; it does not identify a causal layer or loss term by itself.",
            "No raw audio, embedding, model weights, optimizer state, or credentials are written by this analysis.",
        ],
    }
    return report, all_rows, _class_rows(all_rows, labels, metrics)


def write_f005_paired_analysis(output_dir: Path, report: dict, paired_rows: list[dict], class_rows: list[dict]) -> None:
    """Write metadata-only analysis artifacts suitable for post-run review."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename, rows in (("paired_file_transitions.csv", paired_rows), ("class_comparison.csv", class_rows)):
        if not rows:
            raise ValueError("Cannot write an empty paired-analysis table")
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    primary = report["comparisons"]["c002b_to_selected_arm"]
    transitions = report["paired_transitions"]
    rank = report["known_rank_diagnosis"]
    text = f"""# F005 paired diagnosis against C002b

This is a saved-prediction, read-only analysis. It trained no model and fitted no threshold.

The selected F005 output changed Macro-F1 by **{primary['macro_f1_delta']:+.6f}** and accuracy by **{primary['accuracy_delta']:+.6f}** versus C002b. It corrected {transitions['c002b_wrong_to_f005_correct']} files and regressed {transitions['c002b_correct_to_f005_wrong']} files.

Among {rank['nonzero_known_files']} nonzero known files, C002b's true-known rank-1 accuracy was **{rank['c002b_rank1_accuracy']:.4%}** and F005 selected known-top1 accuracy was **{rank['f005_selected_known_top1_accuracy']:.4%}**. Full F005 rank evidence available: `{rank['f005_full_rank_available']}`.

`summary.json` contains protocol, error-mode, duration and rank summaries. The CSV files contain prediction-level transitions and per-class F1 deltas. They contain no raw audio, embeddings, model weights, optimizer state, or credentials.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")
