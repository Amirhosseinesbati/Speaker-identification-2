"""Replay the C002b scorer and apply its sealed policy to F005 control caches.

This is a *diagnostic*, cache-only bridge.  It never opens audio, loads a
Torch model, extracts an embedding, or fits a threshold/alpha.  Its first
control must reproduce C002b's saved probabilities exactly.  Only then does
it apply each fold's already sealed C002b alpha and open-set gate to the
corresponding full F005-control embedding cache.

The resulting F005 number is not eligible for model selection: the historic
C002b calibration rows include encoder-fit rows, whereas the adapted encoder
was fitted on them.  It isolates scorer and representation effects for the
next independently calibrated experiment.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import (
    predictions_from_probabilities,
    score_predictions,
    validate_labels,
)
from speaker_id.models.campp import file_sha256
from speaker_id.training.candidate_fusion import ALPHAS, scores_for_alpha
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.reference_scoring import reference_probabilities


BRIDGE_SCHEMA_VERSION = 1
PROBABILITY_TEMPERATURE = 0.05


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Bridge requires a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Bridge JSON must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Bridge requires a regular CSV file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_string(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _rooted_regular(path: Path, root: Path) -> Path:
    path, root = Path(path), Path(root)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Bridge cache entry is not a regular file: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Bridge cache entry escapes its cache root") from error
    return path


def _index_rows(rows: list[dict], labels: set[str], name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        audio_file, speaker_id = row.get("audio_file"), row.get("speaker_id")
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError(f"{name} has a missing audio_file")
        if speaker_id not in labels:
            raise ValueError(f"{name} has a label outside the label map")
        if audio_file in indexed:
            raise ValueError(f"{name} has duplicate audio_file: {audio_file}")
        indexed[audio_file] = row
    if not indexed:
        raise ValueError(f"{name} is empty")
    return indexed


def _load_inputs(manifest_path: Path, folds_path: Path,
                 label_map_path: Path) -> tuple[list[str], list[dict], list[dict], dict[str, str]]:
    labels = validate_labels(_read_json(label_map_path).get("labels"))
    label_set = set(labels)
    manifest = _read_csv(manifest_path)
    folds = _read_csv(folds_path)
    manifest_by_name = _index_rows(manifest, label_set, "manifest")
    folds_by_name = _index_rows(folds, label_set, "folds")
    if set(manifest_by_name) != set(folds_by_name):
        raise ValueError("Manifest and folds do not cover the same files")
    for row in manifest:
        audio_file = row["audio_file"]
        fold = folds_by_name[audio_file]
        if fold.get("speaker_id") != row.get("speaker_id"):
            raise ValueError("Manifest and folds disagree on speaker_id")
        if not isinstance(fold.get("group_id"), str) or not fold["group_id"]:
            raise ValueError("Folds require nonempty group_id")
        if row.get("group_id") not in (None, "", fold["group_id"]):
            raise ValueError("Manifest and folds disagree on group_id")
    fold_ids = sorted({int(row["fold"]) for row in folds})
    if fold_ids != [0, 1]:
        raise ValueError("C002b bridge requires exactly outer folds 0 and 1")
    return labels, manifest, folds, {
        "manifest": _file_sha256(manifest_path),
        "folds": _file_sha256(folds_path),
        "label_map": _file_sha256(label_map_path),
    }


def _fold_item(mapping: object, outer: int, name: str) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a fold-indexed object")
    keys = {str(outer), outer}
    present = [key for key in keys if key in mapping]
    if len(present) != 1 or not isinstance(mapping[present[0]], dict):
        raise ValueError(f"{name} lacks exactly one object for fold {outer}")
    return mapping[present[0]]


def _load_historical_policy(c002_run_dir: Path, c002b_dir: Path, outer: int) -> dict:
    """Return only the alpha/gate actually sealed for C002b identity."""
    frozen = _read_json(Path(c002_run_dir) / "frozen_inner_choices.json")
    fit = _fold_item(frozen.get("inner_fits"), outer, "C002b inner_fits")
    identity = fit.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("selected"), dict):
        raise ValueError("C002b lacks the sealed identity frontend selection")
    selected = identity["selected"]
    choice = _fold_item(frozen.get("selected"), outer, "C002b selected frontends")
    if choice.get("frontend") not in {"identity", "gain"}:
        raise ValueError("C002b selected frontend is invalid")
    # C002b is deliberately the identity recipe, even if C002d selected gain.
    alpha = _finite_float(selected.get("advanced_weight"), "C002b sealed alpha")
    if alpha not in ALPHAS:
        raise ValueError("C002b sealed alpha is outside the fixed grid")
    calibration_file = Path(c002b_dir) / f"fold_{outer}" / "calibration.json"
    calibration_doc = _read_json(calibration_file)
    calibration = calibration_doc.get("selected")
    required = {"unknown_weight", "margin_weight", "threshold", "inner_macro_f1_447"}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError("C002b calibration schema changed")
    for key in required:
        _finite_float(calibration[key], "C002b calibration/" + key)
    embedded = selected.get("calibration")
    if embedded != calibration:
        raise ValueError("C002b alpha selection and fold calibration disagree")
    return {
        "advanced_weight": alpha,
        "calibration": calibration,
        "frozen_choices_sha256": _file_sha256(Path(c002_run_dir) / "frozen_inner_choices.json"),
        "calibration_sha256": _file_sha256(calibration_file),
    }


def _load_c002_identity_cache(c002_run_dir: Path, manifest: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Read C002's already verified frozen identity cache; no audio is opened."""
    from speaker_id.training.gain_suite import verify_gain_cache

    c002_run_dir = Path(c002_run_dir)
    identity_path = c002_run_dir / "identity_cache_identity.json"
    receipt_path = c002_run_dir / "identity_cache_manifest.json"
    identity, receipt = _read_json(identity_path), _read_json(receipt_path)
    vectors, valid = verify_gain_cache(
        c002_run_dir / "identity_embedding_cache", identity, manifest, receipt,
    )
    if set(vectors) != {"public", "advanced"} or vectors["public"].shape != (len(manifest), 512) or vectors["advanced"].shape != (len(manifest), 192):
        raise ValueError("C002 identity cache dimensions changed")
    if valid.shape != (len(manifest),) or valid.dtype != np.bool_:
        raise ValueError("C002 identity cache validity changed")
    return vectors, valid, {
        "identity_cache_identity_sha256": _file_sha256(identity_path),
        "identity_cache_manifest_sha256": _file_sha256(receipt_path),
        "identity_signature": identity.get("signature"),
        "cache_files_verified": len(manifest),
    }


def _load_f005_control_cache(f005_dir: Path, outer: int, manifest: list[dict],
                             expected_valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load and authenticate one full 192-d F005-control cache exactly once."""
    base = Path(f005_dir) / "full_scoring" / f"fold_{outer}" / "control"
    cache_dir = base / "embedding_cache"
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError("F005 control embedding cache directory is unavailable")
    identity_path = base / "full_scoring_cache_identity.json"
    receipt_path = base / "full_scoring_cache_receipt.json"
    identity, receipt = _read_json(identity_path), _read_json(receipt_path)
    identity_body = {key: value for key, value in identity.items() if key != "signature"}
    required_identity = {
        "schema_version", "experiment_signature", "outer_fold", "arm_id", "scope",
        "indices_sha256", "checkpoint_sha256", "checkpoint_metadata_sha256",
        "embedding_dimension", "inference", "server_only", "mlflow_upload_allowed",
    }
    if (set(identity_body) != required_identity or identity.get("signature") != _canonical_sha(identity_body)
            or identity.get("schema_version") != 1 or identity.get("outer_fold") != outer
            or identity.get("arm_id") != "control" or identity.get("scope") != "full_scoring"
            or identity.get("embedding_dimension") != 192 or identity.get("server_only") is not True
            or identity.get("mlflow_upload_allowed") is not False):
        raise ValueError("F005 control cache identity changed")
    indices = np.arange(len(manifest), dtype=np.int64)
    expected_indices_sha = hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes()).hexdigest()
    if identity.get("indices_sha256") != expected_indices_sha:
        raise ValueError("F005 control cache row order differs from manifest")
    for key in ("experiment_signature", "checkpoint_sha256", "checkpoint_metadata_sha256"):
        _sha256_string(identity.get(key), "F005 control " + key)
    expected_receipt_fields = {
        "schema_version", "identity", "file_count", "files", "completed", "server_only",
        "embedding_artifacts_mlflow_uploaded", "receipt_sha256",
    }
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (set(receipt) != expected_receipt_fields or receipt_body.get("identity") != identity
            or receipt.get("receipt_sha256") != _canonical_sha(receipt_body)
            or receipt.get("schema_version") != 1 or receipt.get("file_count") != len(manifest)
            or receipt.get("completed") is not True or receipt.get("server_only") is not True
            or receipt.get("embedding_artifacts_mlflow_uploaded") is not False
            or not isinstance(receipt.get("files"), list) or len(receipt["files"]) != len(manifest)):
        raise ValueError("F005 control cache receipt changed")
    vectors: list[np.ndarray] = []
    required_record = {"audio_file", "audio_sha256", "cache_file", "cache_sha256", "bytes", "valid"}
    required_npz = {"embedding", "valid", "signature", "audio_file", "audio_sha256"}
    for index, (source, record) in enumerate(zip(manifest, receipt["files"], strict=True)):
        if not isinstance(record, dict) or set(record) != required_record:
            raise ValueError("F005 control cache record schema changed")
        name = source["audio_file"]
        cache_name = record.get("cache_file")
        if (record.get("audio_file") != name or record.get("audio_sha256") != source.get("input_sha256")
                or not isinstance(cache_name, str) or Path(cache_name).name != cache_name
                or cache_name != Path(name).stem + ".npz" or type(record.get("bytes")) is not int
                or record["bytes"] <= 0 or type(record.get("valid")) is not bool):
            raise ValueError("F005 control cache record identity changed")
        _sha256_string(record.get("cache_sha256"), "F005 control cache_sha256")
        path = _rooted_regular(cache_dir / cache_name, cache_dir)
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["cache_sha256"]:
            raise ValueError("F005 control cache bytes differ from the sealed receipt")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != required_npz:
                raise ValueError("F005 control cache NPZ schema changed")
            vector = saved["embedding"].copy()
            cache_valid = bool(saved["valid"])
            if (str(saved["signature"]) != identity["signature"] or str(saved["audio_file"]) != name
                    or str(saved["audio_sha256"]) != source["input_sha256"]):
                raise ValueError("F005 control cache NPZ binding changed")
        expected = bool(expected_valid[index])
        if (cache_valid != expected or record["valid"] != expected
                or vector.shape != (192,) or vector.dtype != np.float32 or not np.isfinite(vector).all()
                or (not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5) if cache_valid else np.any(vector))):
            raise ValueError("F005 control embedding validity or geometry changed")
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32), {
        "cache_identity_sha256": _file_sha256(identity_path),
        "cache_receipt_sha256": _file_sha256(receipt_path),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "cache_files_verified": len(vectors),
    }


def _score_with_fixed_policy(public: np.ndarray, advanced: np.ndarray, valid: np.ndarray,
                             manifest: list[dict], folds: list[dict], labels: list[str],
                             outer: int, policy: dict) -> tuple[np.ndarray, list[str], dict]:
    """Score one fold using the C002b alpha/gate without any calibration fit."""
    classes = len(labels) - 1
    endpoints = {
        name: crossfit_scores(values, valid, manifest, folds, outer, "max_reference", classes)
        for name, values in (("public", public), ("advanced", advanced))
    }
    if any(item["known_labels"] != labels[1:] for item in endpoints.values()):
        raise ValueError("C002b bridge score columns differ from fixed label map")
    score = scores_for_alpha(
        public, advanced, valid, manifest, folds, outer, policy["advanced_weight"], endpoints,
        classes=classes,
    )
    probabilities = reference_probabilities(
        score["outer_known_scores"], score["outer_unknown_similarity"],
        policy["calibration"], score["outer_valid"], PROBABILITY_TEMPERATURE,
    )
    names = [manifest[int(index)]["audio_file"] for index in score["outer_indices"]]
    if probabilities.shape != (len(names), len(labels)) or len(set(names)) != len(names):
        raise ValueError("C002b bridge probability rows are malformed")
    return probabilities, names, score


def _load_historical_probabilities(c002b_dir: Path, outer: int,
                                   labels: list[str]) -> tuple[np.ndarray, list[str], dict]:
    path = Path(c002b_dir) / f"fold_{outer}" / "outer_probabilities.npz"
    _rooted_regular(path, Path(c002b_dir))
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"probabilities", "audio_files", "labels"}:
            raise ValueError("C002b probability artifact schema changed")
        probabilities = saved["probabilities"].copy()
        names = [str(value) for value in saved["audio_files"].tolist()]
        saved_labels = [str(value) for value in saved["labels"].tolist()]
    if (saved_labels != labels or probabilities.shape != (len(names), len(labels))
            or len(set(names)) != len(names) or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("C002b historical probability artifact is malformed")
    return probabilities, names, {"outer_probabilities_sha256": _file_sha256(path)}


def _prediction_map(names: list[str], probabilities: np.ndarray, labels: list[str]) -> dict[str, str]:
    return {
        row["audio_file"]: row["speaker_id"]
        for row in predictions_from_probabilities(names, probabilities, labels)
    }


def _compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _transition_report(manifest: list[dict], before: dict[str, str], after: dict[str, str]) -> dict:
    categories, modes = Counter(), Counter()
    for source in manifest:
        name, actual = source["audio_file"], source["speaker_id"]
        old, new = before[name], after[name]
        old_correct, new_correct = old == actual, new == actual
        if old_correct and not new_correct:
            categories["regressed"] += 1
        elif not old_correct and new_correct:
            categories["corrected"] += 1
        elif old_correct:
            categories["both_correct"] += 1
        elif old == new:
            categories["both_wrong_same_prediction"] += 1
        else:
            categories["both_wrong_changed_prediction"] += 1
        def mode(predicted: str) -> str:
            if predicted == actual:
                return "correct"
            if actual == "unknown":
                return "unknown_to_known"
            if predicted == "unknown":
                return "known_to_unknown"
            return "known_to_other_known"
        modes[mode(old) + "->" + mode(new)] += 1
    return {"row_count": len(manifest), **{key: categories[key] for key in (
        "corrected", "regressed", "both_correct", "both_wrong_same_prediction", "both_wrong_changed_prediction",
    )}, "error_mode_transitions": dict(sorted(modes.items()))}


def _historical_csv_predictions(c002b_dir: Path, labels: list[str], expected: set[str]) -> dict[str, str]:
    indexed = _index_rows(_read_csv(Path(c002b_dir) / "oof_predictions.csv"), set(labels), "C002b OOF predictions")
    if set(indexed) != expected:
        raise ValueError("C002b OOF predictions do not cover the manifest")
    return {name: row["speaker_id"] for name, row in indexed.items()}


def analyze_f005_control_under_c002b_policy(
        f005_dir: Path, c002_run_dir: Path, c002b_dir: Path,
        manifest_path: Path, folds_path: Path, label_map_path: Path,
) -> dict:
    """Build the cache-only bridge report and fail closed on any identity mismatch."""
    f005_dir, c002_run_dir, c002b_dir = map(Path, (f005_dir, c002_run_dir, c002b_dir))
    labels, manifest, folds, input_hashes = _load_inputs(manifest_path, folds_path, label_map_path)
    folds_by_name = {row["audio_file"]: row for row in folds}
    public_vectors, frozen_valid, c002_cache = _load_c002_identity_cache(c002_run_dir, manifest)
    historic_predictions = _historical_csv_predictions(c002b_dir, labels, {row["audio_file"] for row in manifest})
    reproduced_predictions: dict[str, str] = {}
    control_predictions: dict[str, str] = {}
    fold_reports: list[dict] = []
    for outer in (0, 1):
        policy = _load_historical_policy(c002_run_dir, c002b_dir, outer)
        historical_probabilities, historical_names, historical_hash = _load_historical_probabilities(c002b_dir, outer, labels)
        frozen_probabilities, frozen_names, frozen_score = _score_with_fixed_policy(
            public_vectors["public"], public_vectors["advanced"], frozen_valid,
            manifest, folds, labels, outer, policy,
        )
        if frozen_names != historical_names or not np.array_equal(frozen_probabilities, historical_probabilities):
            maximum = (float(np.max(np.abs(frozen_probabilities - historical_probabilities)))
                       if frozen_probabilities.shape == historical_probabilities.shape else None)
            raise ValueError(f"C002b exact probability replay failed for fold {outer}; max_abs_diff={maximum}")
        frozen_predictions = _prediction_map(frozen_names, frozen_probabilities, labels)
        if any(frozen_predictions[name] != historic_predictions[name] for name in frozen_names):
            raise ValueError("C002b probability argmax differs from historical OOF predictions")
        control_vectors, control_cache = _load_f005_control_cache(f005_dir, outer, manifest, frozen_valid)
        control_probabilities, control_names, control_score = _score_with_fixed_policy(
            public_vectors["public"], control_vectors, frozen_valid,
            manifest, folds, labels, outer, policy,
        )
        if control_names != historical_names:
            raise ValueError("F005 control bridge outer row order differs from C002b")
        reproduced_predictions.update(frozen_predictions)
        control_predictions.update(_prediction_map(control_names, control_probabilities, labels))
        fold_sources = [
            row for row in manifest
            if int(folds_by_name[row["audio_file"]]["fold"]) == outer
        ]
        historical_metrics = _compact_metrics(score_predictions(
            fold_sources,
            [{"audio_file": name, "speaker_id": historic_predictions[name]} for name in historical_names], labels,
        ))
        control_metrics = _compact_metrics(score_predictions(
            fold_sources,
            [{"audio_file": name, "speaker_id": control_predictions[name]} for name in control_names], labels,
        ))
        fold_reports.append({
            "outer_fold": outer,
            "policy": policy,
            "historical_probability_artifact": historical_hash,
            "historical_replay": {
                "exact_probability_arrays": True,
                "exact_prediction_argmax": True,
                "rows": len(frozen_names),
                "metrics": historical_metrics,
            },
            "f005_control_under_fixed_c002b_policy": {
                "rows": len(control_names), "metrics": control_metrics,
                "macro_f1_delta_vs_c002b": control_metrics["macro_f1"] - historical_metrics["macro_f1"],
                "accuracy_delta_vs_c002b": control_metrics["accuracy"] - historical_metrics["accuracy"],
                "score_receipt": {
                    "outer_indices_count": len(control_score["outer_indices"]),
                    "reference_counts": {
                        key: int(value) if isinstance(value, (int, np.integer)) else value
                        for key, value in control_score["reference_counts"].items()
                        if key in {"unknown_files", "unknown_groups"}
                    },
                },
            },
            "f005_control_cache": control_cache,
            "frozen_score_rows": len(frozen_score["outer_indices"]),
        })
    if set(reproduced_predictions) != {row["audio_file"] for row in manifest} or reproduced_predictions != historic_predictions:
        raise ValueError("C002b replay does not cover or equal the historical OOF predictions")
    historical_oof = _compact_metrics(score_predictions(
        manifest, [{"audio_file": row["audio_file"], "speaker_id": historic_predictions[row["audio_file"]]} for row in manifest], labels,
    ))
    control_oof = _compact_metrics(score_predictions(
        manifest, [{"audio_file": row["audio_file"], "speaker_id": control_predictions[row["audio_file"]]} for row in manifest], labels,
    ))
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "status": "complete",
        "kind": "cache_only_fixed_historical_c002b_policy_bridge",
        "cache_only": True,
        "audio_opened": False,
        "model_loaded": False,
        "embedding_extraction": False,
        "encoder_updates": 0,
        "policy_refit": False,
        "c002b_reproduction": {
            "exact_probability_arrays": True,
            "exact_oof_predictions": True,
            "metrics": historical_oof,
        },
        "f005_control_under_fixed_c002b_policy": {
            "metrics": control_oof,
            "macro_f1_delta_vs_c002b": control_oof["macro_f1"] - historical_oof["macro_f1"],
            "accuracy_delta_vs_c002b": control_oof["accuracy"] - historical_oof["accuracy"],
            "paired_transitions_vs_c002b": _transition_report(manifest, historic_predictions, control_predictions),
            "not_a_selectable_or_deployment_policy": True,
            "reason": "C002b calibration includes F005 encoder-fit rows; this fixed historical policy is diagnostic only.",
        },
        "folds": fold_reports,
        "inputs": input_hashes,
        "c002_identity_cache": c002_cache,
    }


def write_scoring_bridge(output_dir: Path, report: dict) -> None:
    """Write only compact diagnostic receipts; embeddings and predictions stay server-side."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Bridge output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    baseline = report["c002b_reproduction"]["metrics"]
    control = report["f005_control_under_fixed_c002b_policy"]["metrics"]
    delta = report["f005_control_under_fixed_c002b_policy"]["macro_f1_delta_vs_c002b"]
    markdown = (
        "# C002b / F005 scoring bridge\n\n"
        "This cache-only diagnostic exactly replayed the historical C002b probabilities before applying the sealed C002b policy to F005-control embeddings. "
        "It did not open audio, load a model, extract embeddings, or refit a policy.\n\n"
        f"- C002b replay Macro-F1: **{baseline['macro_f1']:.9f}**\n"
        f"- F005 control under fixed C002b policy: **{control['macro_f1']:.9f}**\n"
        f"- Delta: **{delta:+.9f}**\n\n"
        "The control result is diagnostic only because the historic C002b calibration includes rows used for F005 encoder fitting. "
        "It must not select, promote, or package a model.\n"
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
