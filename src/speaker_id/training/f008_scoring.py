"""Role-safe, pretruth scoring primitives for F008 open-set adaptation.

F008 changes only the trainable 192-dimensional CAM++ endpoint.  This module
therefore keeps the public 512-dimensional endpoint frozen, combines the two
with the established sqrt-weighted same-reference fusion, and calibrates every
decision solely on the original ``calibration_query`` roles.

The module deliberately does *not* extract embeddings, load a checkpoint,
open audio, create an MLflow run, or train a model.  It is the CPU-only bridge
between an authenticated embedding cache and an outer-fold evaluation.  In
particular, it writes immutable pretruth seals before an API that accepts
outer speaker labels can be called.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training.candidate_fusion import (
    ALPHAS as FUSION_ALPHAS,
    MARGIN_WEIGHTS as FUSION_MARGIN_WEIGHTS,
    TIE_ORDER as FUSION_ALPHA_TIE_ORDER,
    UNKNOWN_WEIGHTS as FUSION_UNKNOWN_WEIGHTS,
    weighted_encoder_pair,
)
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import (
    calibrate_gate,
    gate_scores,
    reference_probabilities,
)
from speaker_id.training.scoring import macro_f1_indices


PUBLIC_DIMENSION = 512
ADVANCED_DIMENSION = 192
FROZEN_COMPARATOR = "frozen_same_protocol"
F005_COMPARATOR = "f005_source_control"
COMPARATORS = (FROZEN_COMPARATOR, F005_COMPARATOR, "selected_arm")
SOURCE_BINDING_SCHEMA = "f008-authenticated-f005-control-embeddings-v1"
POLICY_SCHEMA = "f008-role-safe-pretruth-policy-v1"
ALL_POLICY_SCHEMA = "f008-all-fold-pretruth-reloads-v1"
EVALUATION_SCHEMA = "f008-one-shot-outer-evaluation-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    _require(type(value) in (int, float) and not isinstance(value, bool),
             f"F008 {name} must be a finite number")
    result = float(value)
    _require(math.isfinite(result) and (result >= 0.0 if nonnegative else True),
             f"F008 {name} must be a finite{' nonnegative' if nonnegative else ''} number")
    return result


def _array_receipt(value: object) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
    }


def _write_new_json(path: Path, value: dict[str, object]) -> bytes:
    """Atomically create an immutable receipt and refuse replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F008 refuses to replace sealed evidence: {path}")
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"F008 refuses to replace sealed evidence: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _read_regular_json(path: Path) -> tuple[dict[str, object], bytes]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             "F008 seal must be a regular JSON file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F008 seal must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), "F008 seal must contain a JSON object")
    return value, payload


def normalize_scoring_spec(spec: Mapping[str, object]) -> dict[str, object]:
    """Validate the small, serializable F008 scoring surface.

    The experiment config remains the source of the actual values.  Keeping a
    normalized copy here makes this pure scorer reusable with synthetic
    contracts and lets its seal bind every selection-relevant setting.
    """
    required = {
        "arm_ids", "control_arm_id", "arm_tie_order", "alphas",
        "alpha_tie_order", "unknown_weights", "margin_weights",
        "threshold_candidates", "probability_temperature",
        "maximum_known_preservation_decline_vs_control", "class_count",
    }
    _require(isinstance(spec, Mapping) and set(spec) == required,
             "F008 scoring spec fields changed")
    arm_ids = spec["arm_ids"]
    tie_order = spec["arm_tie_order"]
    _require(
        isinstance(arm_ids, list) and len(arm_ids) >= 1
        and all(isinstance(value, str) and value for value in arm_ids)
        and len(set(arm_ids)) == len(arm_ids)
        and isinstance(spec["control_arm_id"], str)
        and spec["control_arm_id"] in arm_ids
        and isinstance(tie_order, list) and tie_order == arm_ids
        and set(tie_order) == set(arm_ids),
        "F008 arm identity or fixed tie order is invalid",
    )
    _require(
        spec["alphas"] == list(FUSION_ALPHAS)
        and spec["alpha_tie_order"] == list(FUSION_ALPHA_TIE_ORDER)
        and spec["unknown_weights"] == list(FUSION_UNKNOWN_WEIGHTS)
        and spec["margin_weights"] == list(FUSION_MARGIN_WEIGHTS),
        "F008 must use the established fusion and gate grids",
    )
    _require(
        type(spec["threshold_candidates"]) is int
        and spec["threshold_candidates"] >= 2
        and type(spec["class_count"]) is int and spec["class_count"] >= 3,
        "F008 scoring cardinality is invalid",
    )
    temperature = _finite(spec["probability_temperature"], "probability temperature")
    decline = _finite(
        spec["maximum_known_preservation_decline_vs_control"],
        "maximum known-preservation decline", nonnegative=True,
    )
    _require(temperature > 0.0, "F008 probability temperature must be positive")
    return {
        "arm_ids": list(arm_ids),
        "control_arm_id": spec["control_arm_id"],
        "arm_tie_order": list(tie_order),
        "alphas": list(spec["alphas"]),
        "alpha_tie_order": list(spec["alpha_tie_order"]),
        "unknown_weights": list(spec["unknown_weights"]),
        "margin_weights": list(spec["margin_weights"]),
        "threshold_candidates": int(spec["threshold_candidates"]),
        "probability_temperature": temperature,
        "maximum_known_preservation_decline_vs_control": decline,
        "class_count": int(spec["class_count"]),
    }


def scoring_spec_from_f008_config(config: Mapping[str, object]) -> dict[str, object]:
    """Project the existing F008 config into the pure scorer's fixed surface."""
    _require(isinstance(config, Mapping), "F008 config must be a mapping")
    arms, selection, scoring = config.get("arms"), config.get("selection"), config.get("scoring")
    _require(
        isinstance(arms, list) and isinstance(selection, Mapping) and isinstance(scoring, Mapping)
        and type(config.get("evaluation_classes")) is int,
        "F008 config lacks scoring fields",
    )
    arm_ids = [row.get("id") for row in arms if isinstance(row, Mapping)]
    _require(len(arm_ids) == len(arms), "F008 arms must be mappings")
    return normalize_scoring_spec({
        "arm_ids": arm_ids,
        "control_arm_id": "control_f005",
        "arm_tie_order": selection.get("tie_order"),
        "alphas": scoring.get("alphas"),
        "alpha_tie_order": scoring.get("alpha_tie_order"),
        "unknown_weights": scoring.get("unknown_weights"),
        "margin_weights": scoring.get("margin_weights"),
        "threshold_candidates": scoring.get("threshold_candidates"),
        "probability_temperature": scoring.get("probability_temperature"),
        "maximum_known_preservation_decline_vs_control": selection.get(
            "maximum_known_preservation_decline_vs_control"
        ),
        "class_count": config["evaluation_classes"],
    })


def _validate_contract(contract: Mapping[str, object], spec: Mapping[str, object]) -> None:
    _require(isinstance(contract, Mapping) and _is_sha256(contract.get("signature")),
             "F008 scoring needs a signed contract")
    labels = contract.get("labels")
    _require(
        isinstance(labels, list) and len(labels) == spec["class_count"]
        and labels[0] == "unknown" and len(labels) == len(set(labels))
        and all(isinstance(label, str) and label for label in labels),
        "F008 label map differs from the full evaluation class map",
    )
    _require(
        isinstance(contract.get("manifest"), list)
        and isinstance(contract.get("folds"), list)
        and isinstance(contract.get("roles"), list)
        and isinstance(contract.get("config"), Mapping)
        and isinstance(contract["config"].get("fold_ids"), list),
        "F008 contract lacks manifest, role, or fold data",
    )


def _validate_embeddings(embeddings: object, valid: object, row_count: int,
                         dimension: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    values, mask = np.asarray(embeddings), np.asarray(valid)
    _require(
        values.shape == (row_count, dimension) and values.dtype == np.float32
        and mask.shape == (row_count,) and mask.dtype == np.bool_
        and np.isfinite(values).all() and not np.any(values[~mask]) and np.any(mask),
        f"F008 {name} embeddings must be aligned finite float32 vectors with zero invalid rows",
    )
    _require(
        np.allclose(np.linalg.norm(values[mask], axis=1), 1.0, atol=1e-5),
        f"F008 {name} valid embeddings must be unit vectors",
    )
    return values, mask


def _roles_for_outer(contract: Mapping[str, object], outer: int) -> tuple[dict[str, int], list[dict[str, object]]]:
    config = contract["config"]
    _require(type(outer) is int and outer in config["fold_ids"],
             "F008 outer fold is invalid")
    manifest = contract["manifest"]
    names = [row.get("audio_file") for row in manifest]
    _require(
        all(isinstance(name, str) and name for name in names) and len(names) == len(set(names)),
        "F008 manifest filenames must be unique and nonempty",
    )
    roles = [row for row in contract["roles"] if int(row.get("outer_fold", -1)) == outer]
    _require(
        len(roles) == len(manifest)
        and {row.get("audio_file") for row in roles} == set(names),
        "F008 outer role table does not cover the manifest exactly",
    )
    return {name: index for index, name in enumerate(names)}, roles


def _expected_calibration_indices(contract: Mapping[str, object], outer: int) -> np.ndarray:
    positions, roles = _roles_for_outer(contract, outer)
    indices = np.asarray([
        positions[row["audio_file"]] for row in roles
        if truth(row.get("calibration_query"))
    ], dtype=np.int64)
    _require(len(indices) and len(set(indices.tolist())) == len(indices),
             "F008 has no unique calibration-role queries")
    return indices


def _expected_outer_indices(contract: Mapping[str, object], outer: int) -> np.ndarray:
    positions, roles = _roles_for_outer(contract, outer)
    indices = np.asarray([
        positions[row["audio_file"]] for row in roles
        if truth(row.get("outer_evaluation_included"))
    ], dtype=np.int64)
    _require(len(indices) and len(set(indices.tolist())) == len(indices),
             "F008 has no unique outer-evaluation rows")
    return indices


class _OuterTruthForbidden(dict):
    """A manifest/role row that raises if pretruth code reads a speaker label."""

    def __getitem__(self, key):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before F008 policies were sealed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before F008 policies were sealed")
        return super().get(key, default)


def _guard_outer_truth(contract: Mapping[str, object], outer: int) -> dict[str, object]:
    outer_indices = set(_expected_outer_indices(contract, outer).tolist())
    manifest = [
        _OuterTruthForbidden(row) if index in outer_indices else row
        for index, row in enumerate(contract["manifest"])
    ]
    guarded_roles = [
        _OuterTruthForbidden(row)
        if int(row.get("outer_fold", -1)) == outer and truth(row.get("outer_evaluation_included"))
        else row
        for row in contract["roles"]
    ]
    guarded = dict(contract)
    guarded["manifest"] = manifest
    guarded["roles"] = guarded_roles
    return guarded


def _outer_public_metadata(contract: Mapping[str, object], outer: int,
                           indices: np.ndarray) -> list[dict[str, object]]:
    manifest = contract["manifest"]
    folds = {row.get("audio_file"): row for row in contract["folds"]}
    _require(len(folds) == len(contract["folds"]), "F008 fold filenames are duplicated")
    result: list[dict[str, object]] = []
    for index in indices:
        row = manifest[int(index)]
        name = row.get("audio_file")
        fold = folds.get(name)
        _require(isinstance(name, str) and isinstance(fold, Mapping),
                 "F008 outer public row has no aligned fold")
        duration = _finite(row.get("duration_seconds"), "outer duration", nonnegative=True)
        group_id = fold.get("group_id")
        _require(isinstance(group_id, str) and group_id,
                 "F008 outer public row has no content group")
        result.append({
            "audio_file": name,
            "group_id": group_id,
            "duration_seconds": duration,
            "has_nonzero_signal": truth(row.get("has_nonzero_signal")),
        })
    return result


def _source_summary(source_receipt: Mapping[str, object], outer: int) -> dict[str, str]:
    """Read the compact, authenticated F005-source receipt without opening a cache."""
    _require(
        isinstance(source_receipt, Mapping)
        and source_receipt.get("schema_version") == "f007-f005-source-receipt-v1"
        and isinstance(source_receipt.get("experiment_state"), Mapping)
        and isinstance(source_receipt.get("fold_ids"), list)
        and isinstance(source_receipt.get("folds"), list),
        "F008 requires the authenticated F005 source receipt",
    )
    state = source_receipt["experiment_state"]
    signature = state.get("experiment_signature")
    _require(_is_sha256(signature) and outer in source_receipt["fold_ids"],
             "F008 F005 source receipt has an invalid experiment/fold binding")
    matches = [row for row in source_receipt["folds"] if row.get("outer_fold") == outer]
    _require(len(matches) == 1 and isinstance(matches[0].get("full_scoring_control"), Mapping),
             "F008 F005 source receipt lacks this fold's control cache")
    control = matches[0]["full_scoring_control"]
    identity, receipt = control.get("identity"), control.get("receipt")
    _require(
        isinstance(identity, Mapping) and isinstance(receipt, Mapping)
        and _is_sha256(identity.get("sha256")) and _is_sha256(receipt.get("sha256")),
        "F008 F005 source control-cache receipt is malformed",
    )
    return {
        "source_f005_receipt_sha256": _sha(dict(source_receipt)),
        "source_f005_experiment_signature": signature,
        "source_control_cache_identity_sha256": identity["sha256"],
        "source_control_cache_receipt_sha256": receipt["sha256"],
    }


def bind_authenticated_f005_control_embeddings(
        source_f005_receipt: Mapping[str, object], outer: int, *,
        embeddings: object, valid: object,
) -> dict[str, object]:
    """Bind a loaded F005-control array to its already authenticated source receipt.

    Loading and byte-authenticating the cache stays in the extraction layer.
    This small receipt binds that resulting aligned array to the F005 source so
    the scorer cannot silently compare an arbitrary 192-D control instead.
    """
    summary = _source_summary(source_f005_receipt, outer)
    values, mask = _validate_embeddings(
        embeddings, valid, len(np.asarray(embeddings)), ADVANCED_DIMENSION,
        "authenticated F005 control",
    )
    body = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "outer_fold": outer,
        **summary,
        "advanced_embeddings": _array_receipt(values),
        "valid": _array_receipt(mask),
    }
    return {**body, "binding_sha256": _sha(body)}


def _verify_f005_control_binding(binding: Mapping[str, object],
                                 source_f005_receipt: Mapping[str, object], outer: int,
                                 embeddings: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    required = {
        "schema_version", "outer_fold", "source_f005_receipt_sha256",
        "source_f005_experiment_signature", "source_control_cache_identity_sha256",
        "source_control_cache_receipt_sha256", "advanced_embeddings", "valid",
        "binding_sha256",
    }
    _require(isinstance(binding, Mapping) and set(binding) == required,
             "F008 F005-control binding schema changed")
    body = {key: binding[key] for key in binding if key != "binding_sha256"}
    summary = _source_summary(source_f005_receipt, outer)
    _require(
        binding.get("schema_version") == SOURCE_BINDING_SCHEMA
        and binding.get("outer_fold") == outer
        and binding.get("binding_sha256") == _sha(body)
        and all(binding.get(key) == value for key, value in summary.items())
        and binding.get("advanced_embeddings") == _array_receipt(embeddings)
        and binding.get("valid") == _array_receipt(valid),
        "F008 F005-control array is not bound to its authenticated source",
    )
    return {**summary, "control_binding_sha256": binding["binding_sha256"]}


def _validate_score(score: Mapping[str, object], contract: Mapping[str, object], outer: int,
                    expected_calibration: np.ndarray, expected_outer: np.ndarray,
                    valid: np.ndarray) -> None:
    required = {
        "calibration_indices", "outer_indices", "known_labels", "inner_known_scores",
        "outer_known_scores", "inner_unknown_similarity", "outer_unknown_similarity",
        "outer_valid", "reference_counts", "provenance",
    }
    _require(isinstance(score, Mapping) and required.issubset(score),
             "F008 heldout scorer output is incomplete")
    labels = contract["labels"]
    calibration_indices = np.asarray(score["calibration_indices"])
    outer_indices = np.asarray(score["outer_indices"])
    inner = np.asarray(score["inner_known_scores"])
    outer_scores = np.asarray(score["outer_known_scores"])
    inner_unknown = np.asarray(score["inner_unknown_similarity"])
    outer_unknown = np.asarray(score["outer_unknown_similarity"])
    outer_valid = np.asarray(score["outer_valid"])
    _require(
        np.array_equal(calibration_indices, expected_calibration)
        and np.array_equal(outer_indices, expected_outer)
        and score["known_labels"] == labels[1:]
        and inner.shape == (len(expected_calibration), len(labels) - 1)
        and outer_scores.shape == (len(expected_outer), len(labels) - 1)
        and inner.dtype == np.float32 and outer_scores.dtype == np.float32
        and inner_unknown.shape == (len(expected_calibration),)
        and outer_unknown.shape == (len(expected_outer),)
        and np.isfinite(inner).all() and np.isfinite(outer_scores).all()
        and np.isfinite(inner_unknown).all() and np.isfinite(outer_unknown).all()
        and outer_valid.dtype == np.bool_ and np.array_equal(outer_valid, valid[expected_outer])
        and isinstance(score["provenance"], Mapping)
        and score["provenance"].get("protocol")
        == "original_heldout_queries_expanded_gallery_v1"
        and score["provenance"].get("outer_fold") == outer,
        "F008 heldout score rows, columns, or role-safe provenance changed",
    )


def _score_evidence(score: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "calibration_indices", "outer_indices", "inner_known_scores",
        "outer_known_scores", "inner_unknown_similarity", "outer_unknown_similarity",
        "outer_valid",
    )
    evidence = {key: _array_receipt(score[key]) for key in keys}
    evidence["known_labels_sha256"] = _sha(score["known_labels"])
    evidence["protocol"] = score["provenance"].get("protocol")
    evidence["outer_fold"] = score["provenance"].get("outer_fold")
    return {**evidence, "receipt_sha256": _sha(evidence)}


def _score_candidates(contract: Mapping[str, object], outer: int, *,
                      public_embeddings: np.ndarray, advanced_embeddings: np.ndarray,
                      valid: np.ndarray, scorer: Callable[..., dict[str, object]]) -> dict[float, dict[str, object]]:
    expected_calibration = _expected_calibration_indices(contract, outer)
    expected_outer = _expected_outer_indices(contract, outer)
    candidates: dict[float, dict[str, object]] = {}
    for alpha in FUSION_ALPHAS:
        if alpha == 0.0:
            values = public_embeddings
        elif alpha == 1.0:
            values = advanced_embeddings
        else:
            values = weighted_encoder_pair(public_embeddings, advanced_embeddings, valid, alpha)
        score = scorer(contract, values, valid, outer)
        _validate_score(score, contract, outer, expected_calibration, expected_outer, valid)
        candidates[float(alpha)] = dict(score)
    anchor = candidates[float(FUSION_ALPHAS[0])]
    for score in candidates.values():
        _require(
            np.array_equal(score["calibration_indices"], anchor["calibration_indices"])
            and np.array_equal(score["outer_indices"], anchor["outer_indices"])
            and np.array_equal(score["outer_valid"], anchor["outer_valid"])
            and score["known_labels"] == anchor["known_labels"],
            "F008 alpha candidates use different role-safe rows",
        )
    return candidates


def _calibration_truth(contract: Mapping[str, object], outer: int,
                       score: Mapping[str, object]) -> np.ndarray:
    indices = np.asarray(score["calibration_indices"])
    expected = _expected_calibration_indices(contract, outer)
    _require(np.array_equal(indices, expected),
             "F008 calibration score rows differ from immutable roles")
    labels = contract["labels"]
    label_index = {label: index for index, label in enumerate(labels)}
    positions, roles_list = _roles_for_outer(contract, outer)
    roles = {row["audio_file"]: row for row in roles_list}
    actual: list[int] = []
    for row_index in indices:
        row = contract["manifest"][int(row_index)]
        role = roles.get(row["audio_file"])
        _require(isinstance(role, Mapping) and truth(role.get("calibration_query")),
                 "F008 calibration score includes a non-query row")
        label = row.get("speaker_id")
        _require(label in label_index, "F008 calibration label is absent from the fixed map")
        actual.append(label_index[label])
    result = np.asarray(actual, dtype=np.int64)
    _require(
        len(result) == len(indices) and np.any(result == 0) and np.any(result > 0)
        and np.all(result < len(labels)),
        "F008 calibration needs independent known and unknown query roles",
    )
    return result


def _known_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    known = np.asarray(actual) > 0
    _require(np.any(known), "F008 known-preservation needs known calibration queries")
    actual_known = np.asarray(actual)[known].astype(np.int64, copy=False)
    predicted_known = np.asarray(predicted)[known].astype(np.int64, copy=False)
    observed = sorted(set(actual_known.tolist()))
    support = Counter(actual_known.tolist())
    assigned = Counter(predicted_known.tolist())
    correct = Counter(
        int(target) for target, guess in zip(actual_known, predicted_known, strict=True)
        if target == guess
    )
    f1 = [
        2.0 * correct[label] / (support[label] + assigned[label])
        if support[label] + assigned[label] else 0.0
        for label in observed
    ]
    return {
        "known_query_rows": int(len(actual_known)),
        "known_query_observed_labels": int(len(observed)),
        "known_query_macro_f1_over_observed_labels": float(np.mean(f1)),
        "known_query_top1_accuracy": float(np.mean(actual_known == predicted_known)),
        "known_query_rejected": int(np.sum(predicted_known == 0)),
    }


def _annotate_curve(score: Mapping[str, object], calibration_truth: np.ndarray,
                    curve: list[dict[str, float]], classes: int) -> list[dict[str, object]]:
    scores = np.asarray(score["inner_known_scores"])
    unknown = np.asarray(score["inner_unknown_similarity"])
    top_known = scores.argmax(axis=1).astype(np.int64) + 1
    annotated: list[dict[str, object]] = []
    for row in curve:
        gate = gate_scores(scores, unknown, row["unknown_weight"], row["margin_weight"])
        prediction = np.where(gate > row["threshold"], top_known, 0).astype(np.int64)
        full_f1 = macro_f1_indices(calibration_truth, prediction, classes)
        _require(
            math.isclose(full_f1, float(row["inner_macro_f1_447"]), rel_tol=0.0, abs_tol=1e-12),
            "F008 gate primitive and full-label Macro-F1 disagree",
        )
        annotated.append({
            "unknown_weight": float(row["unknown_weight"]),
            "margin_weight": float(row["margin_weight"]),
            "threshold": float(row["threshold"]),
            # Keep the established key for downstream reports, while making
            # explicit that it always spans the *entire* fixed label map.
            "inner_macro_f1_447": full_f1,
            "inner_macro_f1_full_label_map": full_f1,
            "known_metrics": _known_metrics(calibration_truth, prediction),
            "inner_prediction_sha256": _array_receipt(prediction)["sha256"],
        })
    return annotated


def _curve_key(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["inner_macro_f1_full_label_map"]),
        -float(row["unknown_weight"]),
        -float(row["margin_weight"]),
        float(row["threshold"]),
    )


def _fit_policy(candidates: Mapping[float, Mapping[str, object]],
                calibration_truth: np.ndarray, spec: Mapping[str, object], *,
                known_floor: float | None) -> tuple[dict[str, object], dict[float, dict[str, object]], dict[str, object]]:
    """Jointly select fusion/gate, optionally preserving known-class quality."""
    _require(set(candidates) == {float(value) for value in spec["alphas"]},
             "F008 alpha candidates differ from the sealed grid")
    curves: dict[float, dict[str, object]] = {}
    per_alpha: dict[float, dict[str, object]] = {}
    for alpha in spec["alphas"]:
        score = candidates[float(alpha)]
        _selected, raw_curve = calibrate_gate(
            np.asarray(score["inner_known_scores"]), calibration_truth,
            np.asarray(score["inner_unknown_similarity"]),
            list(spec["unknown_weights"]), list(spec["margin_weights"]),
            int(spec["threshold_candidates"]), classes=int(spec["class_count"]),
        )
        annotated = _annotate_curve(score, calibration_truth, raw_curve, int(spec["class_count"]))
        eligible = [
            row for row in annotated if known_floor is None
            or float(row["known_metrics"]["known_query_macro_f1_over_observed_labels"])
            >= known_floor - 1e-12
        ]
        curves[float(alpha)] = {
            "advanced_weight": float(alpha), "curve": annotated,
            "eligible_curve_count": len(eligible),
        }
        if eligible:
            per_alpha[float(alpha)] = max(eligible, key=_curve_key)
    _require(per_alpha, "F008 arm has no calibration policy that preserves known queries")
    alpha_order = [float(value) for value in spec["alpha_tie_order"]]
    available = [alpha for alpha in alpha_order if alpha in per_alpha]
    selected_alpha = max(
        available,
        key=lambda alpha: float(per_alpha[alpha]["inner_macro_f1_full_label_map"]),
    )
    selected = per_alpha[selected_alpha]
    chosen_score = candidates[selected_alpha]
    policy = {
        "advanced_weight": selected_alpha,
        "calibration": {
            key: selected[key] for key in (
                "unknown_weight", "margin_weight", "threshold", "inner_macro_f1_447",
            )
        },
        "inner_metrics": {
            "inner_macro_f1_full_label_map": selected["inner_macro_f1_full_label_map"],
            **selected["known_metrics"],
            "inner_prediction_sha256": selected["inner_prediction_sha256"],
        },
        "known_preservation": {
            "constraint_applied": known_floor is not None,
            "minimum_known_macro_f1": known_floor,
            "passed": True,
        },
        "alpha_tie_order": list(spec["alpha_tie_order"]),
        "selection_scope": "original_group_disjoint_known_and_unknown_calibration_query_rows_only",
        "calibration_indices_sha256": _array_receipt(chosen_score["calibration_indices"])["sha256"],
        "inner_truth_sha256": _array_receipt(calibration_truth)["sha256"],
        "candidate_curves_sha256": _sha({str(alpha): curves[alpha] for alpha in curves}),
        "score_evidence": _score_evidence(chosen_score),
    }
    return policy, curves, dict(chosen_score)


def select_arm_from_policies(arm_policies: Mapping[str, Mapping[str, object]],
                             scoring_spec: Mapping[str, object]) -> dict[str, object]:
    """Select one F008 arm using full-label Macro-F1 and a known-quality floor.

    The control arm establishes the observed-known F1 floor.  All candidates
    have already fitted their gates on the same calibration roles; this
    function never sees an outer score or label.
    """
    spec = normalize_scoring_spec(scoring_spec)
    _require(set(arm_policies) == set(spec["arm_ids"]),
             "F008 arm policy inventory differs from the fixed experiment arms")
    control = arm_policies[spec["control_arm_id"]]
    try:
        control_known = float(
            control["inner_metrics"]["known_query_macro_f1_over_observed_labels"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("F008 control policy lacks known-preservation metrics") from error
    _require(math.isfinite(control_known), "F008 control known metric is nonfinite")
    floor = control_known - float(spec["maximum_known_preservation_decline_vs_control"])
    eligible: list[str] = []
    rejected: list[str] = []
    for arm in spec["arm_tie_order"]:
        policy = arm_policies[arm]
        try:
            full_f1 = float(policy["inner_metrics"]["inner_macro_f1_full_label_map"])
            known_f1 = float(policy["inner_metrics"]["known_query_macro_f1_over_observed_labels"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"F008 {arm} policy metrics are malformed") from error
        _require(math.isfinite(full_f1) and math.isfinite(known_f1),
                 "F008 arm policy metric is nonfinite")
        if known_f1 >= floor - 1e-12:
            eligible.append(arm)
        else:
            rejected.append(arm)
    _require(spec["control_arm_id"] in eligible,
             "F008 control must always satisfy its own known-preservation floor")
    selected = max(
        eligible,
        key=lambda arm: float(arm_policies[arm]["inner_metrics"]["inner_macro_f1_full_label_map"]),
    )
    return {
        "selected_arm": selected,
        "primary_metric": "inner_macro_f1_full_label_map",
        "known_preservation_metric": "known_query_macro_f1_over_observed_labels",
        "control_known_macro_f1": control_known,
        "minimum_known_macro_f1": floor,
        "maximum_known_preservation_decline_vs_control": spec[
            "maximum_known_preservation_decline_vs_control"
        ],
        "eligible_arms": eligible,
        "rejected_by_known_preservation": rejected,
        "tie_order": list(spec["arm_tie_order"]),
        "outer_truth_read": False,
    }


def _validate_policy(policy: Mapping[str, object], *, outer: int,
                     spec: Mapping[str, object]) -> None:
    required = {
        "advanced_weight", "calibration", "inner_metrics", "known_preservation",
        "alpha_tie_order", "selection_scope", "calibration_indices_sha256",
        "inner_truth_sha256", "candidate_curves_sha256", "score_evidence",
    }
    _require(isinstance(policy, Mapping) and set(policy) == required,
             "F008 sealed policy schema changed")
    calibration = policy["calibration"]
    metrics = policy["inner_metrics"]
    preservation = policy["known_preservation"]
    evidence = policy["score_evidence"]
    _require(
        policy["advanced_weight"] in spec["alphas"]
        and policy["alpha_tie_order"] == spec["alpha_tie_order"]
        and policy["selection_scope"]
        == "original_group_disjoint_known_and_unknown_calibration_query_rows_only"
        and isinstance(calibration, Mapping)
        and set(calibration) == {
            "unknown_weight", "margin_weight", "threshold", "inner_macro_f1_447",
        }
        and calibration["unknown_weight"] in spec["unknown_weights"]
        and calibration["margin_weight"] in spec["margin_weights"]
        and all(math.isfinite(float(calibration[key])) for key in ("threshold", "inner_macro_f1_447"))
        and isinstance(metrics, Mapping)
        and isinstance(preservation, Mapping)
        and isinstance(evidence, Mapping)
        and evidence.get("protocol") == "original_heldout_queries_expanded_gallery_v1"
        and evidence.get("outer_fold") == outer
        and evidence.get("receipt_sha256") == _sha({
            key: value for key, value in evidence.items() if key != "receipt_sha256"
        })
        and all(_is_sha256(policy[key]) for key in (
            "calibration_indices_sha256", "inner_truth_sha256", "candidate_curves_sha256",
        )),
        "F008 sealed policy identity changed",
    )
    for key in (
        "inner_macro_f1_full_label_map", "known_query_macro_f1_over_observed_labels",
        "known_query_top1_accuracy",
    ):
        _require(key in metrics and math.isfinite(float(metrics[key]))
                 and 0.0 <= float(metrics[key]) <= 1.0,
                 f"F008 sealed policy {key} is invalid")
    _require(
        isinstance(preservation.get("constraint_applied"), bool)
        and isinstance(preservation.get("passed"), bool)
        and (preservation.get("minimum_known_macro_f1") is None
             or math.isfinite(float(preservation["minimum_known_macro_f1"]))),
        "F008 sealed known-preservation record is invalid",
    )


def prepare_and_seal_pretruth(
        contract: Mapping[str, object], outer: int, *,
        public_embeddings: object, frozen_advanced_embeddings: object,
        f005_control_embeddings: object, f005_source_receipt: Mapping[str, object],
        f005_control_binding: Mapping[str, object],
        f008_advanced_embeddings_by_arm: Mapping[str, object], valid: object,
        scoring_spec: Mapping[str, object], policy_seal_path: Path,
        heldout_scorer: Callable[..., dict[str, object]] = heldout_reference_scores,
        require_control_arm_same_as_source: bool = True,
) -> dict[str, object]:
    """Fit all inner policies and create one immutable outer-pretruth seal.

    All paths through this function replace the outer rows' ``speaker_id`` with
    a raising sentinel before the heldout scorer is reached.  Thus any outer
    truth access fails before a policy can be written.
    """
    spec = normalize_scoring_spec(scoring_spec)
    _validate_contract(contract, spec)
    row_count = len(contract["manifest"])
    public, mask = _validate_embeddings(public_embeddings, valid, row_count, PUBLIC_DIMENSION, "public")
    frozen, frozen_mask = _validate_embeddings(
        frozen_advanced_embeddings, valid, row_count, ADVANCED_DIMENSION, "frozen advanced",
    )
    source_control, source_mask = _validate_embeddings(
        f005_control_embeddings, valid, row_count, ADVANCED_DIMENSION, "F005 source control",
    )
    _require(np.array_equal(mask, frozen_mask) and np.array_equal(mask, source_mask),
             "F008 public/frozen/F005-control validity masks differ")
    source_summary = _verify_f005_control_binding(
        f005_control_binding, f005_source_receipt, outer, source_control, mask,
    )
    _require(
        isinstance(f008_advanced_embeddings_by_arm, Mapping)
        and set(f008_advanced_embeddings_by_arm) == set(spec["arm_ids"]),
        "F008 supplied-arm inventory differs from the fixed tie order",
    )
    arms: dict[str, np.ndarray] = {}
    for arm in spec["arm_ids"]:
        values, arm_mask = _validate_embeddings(
            f008_advanced_embeddings_by_arm[arm], valid, row_count, ADVANCED_DIMENSION,
            f"{arm} advanced",
        )
        _require(np.array_equal(mask, arm_mask),
                 "F008 supplied-arm validity mask differs from frozen public inputs")
        arms[arm] = values
    if require_control_arm_same_as_source:
        _require(
            np.array_equal(arms[spec["control_arm_id"]], source_control),
            "F008 control arm must exactly equal the authenticated F005 source control",
        )

    guarded = _guard_outer_truth(contract, outer)
    frozen_candidates = _score_candidates(
        guarded, outer, public_embeddings=public, advanced_embeddings=frozen,
        valid=mask, scorer=heldout_scorer,
    )
    source_candidates = _score_candidates(
        guarded, outer, public_embeddings=public, advanced_embeddings=source_control,
        valid=mask, scorer=heldout_scorer,
    )
    arm_candidates = {
        arm: _score_candidates(
            guarded, outer, public_embeddings=public, advanced_embeddings=values,
            valid=mask, scorer=heldout_scorer,
        ) for arm, values in arms.items()
    }
    calibration_truth = _calibration_truth(guarded, outer, frozen_candidates[0.0])
    anchor_indices = frozen_candidates[0.0]["calibration_indices"]
    for bundle in [source_candidates, *arm_candidates.values()]:
        _require(
            np.array_equal(bundle[0.0]["calibration_indices"], anchor_indices),
            "F008 comparators do not share the calibration-role rows",
        )

    frozen_policy, frozen_curves, frozen_score = _fit_policy(
        frozen_candidates, calibration_truth, spec, known_floor=None,
    )
    source_policy, source_curves, source_score = _fit_policy(
        source_candidates, calibration_truth, spec, known_floor=None,
    )
    control_id = spec["control_arm_id"]
    control_policy, control_curves, control_score = _fit_policy(
        arm_candidates[control_id], calibration_truth, spec, known_floor=None,
    )
    control_known = float(
        control_policy["inner_metrics"]["known_query_macro_f1_over_observed_labels"]
    )
    known_floor = control_known - float(spec["maximum_known_preservation_decline_vs_control"])
    arm_policies: dict[str, dict[str, object]] = {control_id: control_policy}
    arm_curves: dict[str, dict[float, dict[str, object]]] = {control_id: control_curves}
    arm_scores: dict[str, dict[str, object]] = {control_id: control_score}
    for arm in spec["arm_ids"]:
        if arm == control_id:
            continue
        policy, curves, score = _fit_policy(
            arm_candidates[arm], calibration_truth, spec, known_floor=known_floor,
        )
        arm_policies[arm], arm_curves[arm], arm_scores[arm] = policy, curves, score
    selection = select_arm_from_policies(arm_policies, spec)
    selected_arm = selection["selected_arm"]
    selected_score = arm_scores[selected_arm]
    expected_outer = _expected_outer_indices(contract, outer)
    _require(np.array_equal(selected_score["outer_indices"], expected_outer),
             "F008 selected scorer outer rows differ from immutable roles")
    outer_metadata = _outer_public_metadata(contract, outer, expected_outer)

    body = {
        "schema_version": POLICY_SCHEMA,
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "scoring_spec": spec,
        "scoring_spec_sha256": _sha(spec),
        "f005_source": source_summary,
        "labels_sha256": _sha(contract["labels"]),
        "label_count": len(contract["labels"]),
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "comparator_policies": {
            FROZEN_COMPARATOR: frozen_policy,
            F005_COMPARATOR: source_policy,
        },
        "arm_policies": arm_policies,
        "arm_selection": selection,
        "selected_arm": selected_arm,
        "outer_truth_read": False,
        "outer_truth_accepted_by_pretruth_api": False,
        "all_fold_seals_required_before_outer_truth": True,
    }
    seal = {**body, "seal_sha256": _sha(body)}
    _write_new_json(Path(policy_seal_path), seal)
    reloaded = reload_pretruth_seal(
        Path(policy_seal_path), contract, outer, scoring_spec=spec,
        f005_source_receipt=f005_source_receipt,
        f005_control_binding=f005_control_binding,
    )
    return {
        "kind": "f008_pretruth_scoring_bundle",
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "selected_arm": selected_arm,
        "labels_sha256": _sha(contract["labels"]),
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "outer_files": [row["audio_file"] for row in outer_metadata],
        "score_bundles": {
            FROZEN_COMPARATOR: frozen_score,
            F005_COMPARATOR: source_score,
            "selected_arm": selected_score,
        },
        "arm_score_bundles": arm_scores,
        "candidate_curves": {
            FROZEN_COMPARATOR: frozen_curves,
            F005_COMPARATOR: source_curves,
            **arm_curves,
        },
        "policy_reload": reloaded,
        "outer_truth_read": False,
    }


def reload_pretruth_seal(path: Path, contract: Mapping[str, object], outer: int, *,
                         scoring_spec: Mapping[str, object],
                         f005_source_receipt: Mapping[str, object] | None = None,
                         f005_control_binding: Mapping[str, object] | None = None) -> dict[str, object]:
    """Reload and authenticate one immutable pretruth seal without outer labels."""
    spec = normalize_scoring_spec(scoring_spec)
    _validate_contract(contract, spec)
    seal, payload = _read_regular_json(Path(path))
    required = {
        "schema_version", "experiment_signature", "outer_fold", "scoring_spec",
        "scoring_spec_sha256", "f005_source", "labels_sha256", "label_count",
        "outer_public_metadata", "outer_public_metadata_sha256", "comparator_policies",
        "arm_policies", "arm_selection", "selected_arm", "outer_truth_read",
        "outer_truth_accepted_by_pretruth_api", "all_fold_seals_required_before_outer_truth",
        "seal_sha256",
    }
    _require(set(seal) == required, "F008 pretruth seal schema changed")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    expected_outer = _expected_outer_indices(contract, outer)
    expected_metadata = _outer_public_metadata(contract, outer, expected_outer)
    _require(
        seal["schema_version"] == POLICY_SCHEMA
        and seal["experiment_signature"] == contract["signature"]
        and seal["outer_fold"] == outer
        and seal["scoring_spec"] == spec
        and seal["scoring_spec_sha256"] == _sha(spec)
        and seal["labels_sha256"] == _sha(contract["labels"])
        and seal["label_count"] == len(contract["labels"])
        and seal["outer_public_metadata"] == expected_metadata
        and seal["outer_public_metadata_sha256"] == _sha(expected_metadata)
        and seal["outer_truth_read"] is False
        and seal["outer_truth_accepted_by_pretruth_api"] is False
        and seal["all_fold_seals_required_before_outer_truth"] is True
        and seal["seal_sha256"] == _sha(body)
        and isinstance(seal["comparator_policies"], Mapping)
        and set(seal["comparator_policies"]) == {FROZEN_COMPARATOR, F005_COMPARATOR}
        and isinstance(seal["arm_policies"], Mapping)
        and set(seal["arm_policies"]) == set(spec["arm_ids"]),
        "F008 pretruth seal identity, role boundary, or policy inventory changed",
    )
    for policy in seal["comparator_policies"].values():
        _validate_policy(policy, outer=outer, spec=spec)
    for policy in seal["arm_policies"].values():
        _validate_policy(policy, outer=outer, spec=spec)
    expected_selection = select_arm_from_policies(seal["arm_policies"], spec)
    _require(
        seal["arm_selection"] == expected_selection
        and seal["selected_arm"] == expected_selection["selected_arm"],
        "F008 selected arm is not the recomputed constrained inner winner",
    )
    if f005_source_receipt is not None:
        summary = _source_summary(f005_source_receipt, outer)
        _require(isinstance(seal["f005_source"], Mapping),
                 "F008 sealed F005 source binding is malformed")
        _require(seal["f005_source"] == {
            **summary,
            "control_binding_sha256": seal["f005_source"].get("control_binding_sha256"),
        }, "F008 seal names a different authenticated F005 source")
    source = seal["f005_source"]
    _require(
        isinstance(source, Mapping)
        and set(source) == {
            "source_f005_receipt_sha256", "source_f005_experiment_signature",
            "source_control_cache_identity_sha256", "source_control_cache_receipt_sha256",
            "control_binding_sha256",
        }
        and all(_is_sha256(source[key]) for key in source),
        "F008 sealed F005 source binding is malformed",
    )
    if f005_control_binding is not None:
        _require(
            f005_control_binding.get("binding_sha256") == source["control_binding_sha256"],
            "F008 pretruth seal names another F005-control array binding",
        )
    return {
        "kind": "f008_pretruth_policy_disk_reload",
        "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "seal_sha256": seal["seal_sha256"],
        "disk_reloaded": True,
        "seal": seal,
    }


def reload_all_pretruth_seals(
        contract: Mapping[str, object], *, scoring_spec: Mapping[str, object],
        policy_paths_by_outer: Mapping[int, Path],
        f005_source_receipt: Mapping[str, object],
        f005_control_bindings_by_outer: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Reload every fold's immutable policy seal before accepting any outer truth."""
    spec = normalize_scoring_spec(scoring_spec)
    _validate_contract(contract, spec)
    folds = list(contract["config"]["fold_ids"])
    _require(
        isinstance(policy_paths_by_outer, Mapping)
        and isinstance(f005_control_bindings_by_outer, Mapping)
        and set(policy_paths_by_outer) == set(folds)
        and set(f005_control_bindings_by_outer) == set(folds),
        "F008 requires all fold seals and F005-control bindings before outer truth",
    )
    reloads = {
        outer: reload_pretruth_seal(
            Path(policy_paths_by_outer[outer]), contract, outer, scoring_spec=spec,
            f005_source_receipt=f005_source_receipt,
            f005_control_binding=f005_control_bindings_by_outer[outer],
        ) for outer in folds
    }
    return {
        "kind": ALL_POLICY_SCHEMA,
        "experiment_signature": contract["signature"],
        "scoring_spec_sha256": _sha(spec),
        "fold_ids": folds,
        "policy_reloads": reloads,
        "all_folds_sealed": True,
    }


def _verify_all_pretruth_reloads(all_reloads: Mapping[str, object],
                                 contract: Mapping[str, object],
                                 spec: Mapping[str, object]) -> dict[int, dict[str, object]]:
    folds = list(contract["config"]["fold_ids"])
    _require(
        isinstance(all_reloads, Mapping) and all_reloads.get("kind") == ALL_POLICY_SCHEMA
        and all_reloads.get("experiment_signature") == contract["signature"]
        and all_reloads.get("scoring_spec_sha256") == _sha(spec)
        and all_reloads.get("fold_ids") == folds and all_reloads.get("all_folds_sealed") is True
        and isinstance(all_reloads.get("policy_reloads"), Mapping)
        and set(all_reloads["policy_reloads"]) == set(folds),
        "F008 outer truth is forbidden until every pretruth seal is reloaded",
    )
    refreshed: dict[int, dict[str, object]] = {}
    for outer in folds:
        candidate = all_reloads["policy_reloads"][outer]
        _require(isinstance(candidate, Mapping) and candidate.get("disk_reloaded") is True,
                 "F008 all-fold reload lacks a disk receipt")
        current = reload_pretruth_seal(
            Path(candidate["path"]), contract, outer, scoring_spec=spec,
        )
        _require(
            current["file_sha256"] == candidate.get("file_sha256")
            and current["seal_sha256"] == candidate.get("seal_sha256")
            and current["seal"] == candidate.get("seal"),
            "F008 policy seal changed after its all-fold reload",
        )
        refreshed[outer] = current
    return refreshed


def _verify_pretruth_bundle(pretruth: Mapping[str, object], all_reloads: Mapping[str, object],
                            contract: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    reloaded = _verify_all_pretruth_reloads(all_reloads, contract, spec)
    _require(
        isinstance(pretruth, Mapping) and pretruth.get("kind") == "f008_pretruth_scoring_bundle"
        and pretruth.get("experiment_signature") == contract["signature"]
        and pretruth.get("outer_truth_read") is False
        and set(pretruth.get("score_bundles", {})) == set(COMPARATORS)
        and isinstance(pretruth.get("arm_score_bundles"), Mapping),
        "F008 pretruth score bundle is malformed",
    )
    outer = pretruth.get("outer_fold")
    _require(type(outer) is int and outer in reloaded,
             "F008 pretruth bundle refers to an unsealed outer fold")
    policy = reloaded[outer]
    supplied = pretruth.get("policy_reload")
    _require(
        isinstance(supplied, Mapping)
        and supplied.get("file_sha256") == policy["file_sha256"]
        and supplied.get("seal_sha256") == policy["seal_sha256"]
        and supplied.get("seal") == policy["seal"],
        "F008 pretruth bundle and disk-reloaded policy seal differ",
    )
    seal = policy["seal"]
    _require(
        pretruth.get("selected_arm") == seal["selected_arm"]
        and pretruth.get("labels_sha256") == seal["labels_sha256"]
        and pretruth.get("outer_public_metadata") == seal["outer_public_metadata"]
        and pretruth.get("outer_public_metadata_sha256") == seal["outer_public_metadata_sha256"],
        "F008 pretruth public identity differs from its seal",
    )
    expected_policies = {
        FROZEN_COMPARATOR: seal["comparator_policies"][FROZEN_COMPARATOR],
        F005_COMPARATOR: seal["comparator_policies"][F005_COMPARATOR],
        "selected_arm": seal["arm_policies"][seal["selected_arm"]],
    }
    for comparator in COMPARATORS:
        _require(
            _score_evidence(pretruth["score_bundles"][comparator])
            == expected_policies[comparator]["score_evidence"],
            f"F008 {comparator} score evidence changed after pretruth sealing",
        )
    return seal


def outer_predictions_from_pretruth(
        contract: Mapping[str, object], pretruth: Mapping[str, object],
        all_reloads: Mapping[str, object], *, scoring_spec: Mapping[str, object],
        labels: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Return sealed predictions without accepting outer speaker labels."""
    spec = normalize_scoring_spec(scoring_spec)
    _validate_contract(contract, spec)
    seal = _verify_pretruth_bundle(pretruth, all_reloads, contract, spec)
    _require(isinstance(labels, list) and _sha(labels) == seal["labels_sha256"],
             "F008 output labels differ from the sealed label map")
    names = [row["audio_file"] for row in seal["outer_public_metadata"]]
    outputs: dict[str, list[dict[str, str]]] = {}
    policies = {
        FROZEN_COMPARATOR: seal["comparator_policies"][FROZEN_COMPARATOR],
        F005_COMPARATOR: seal["comparator_policies"][F005_COMPARATOR],
        "selected_arm": seal["arm_policies"][seal["selected_arm"]],
    }
    for comparator in COMPARATORS:
        score = pretruth["score_bundles"][comparator]
        probabilities = reference_probabilities(
            np.asarray(score["outer_known_scores"]),
            np.asarray(score["outer_unknown_similarity"]),
            policies[comparator]["calibration"], np.asarray(score["outer_valid"]),
            float(spec["probability_temperature"]),
        )
        _require(probabilities.shape == (len(names), len(labels)),
                 "F008 sealed probabilities do not align with outer public rows")
        outputs[comparator] = [
            {"audio_file": name, "speaker_id": labels[int(index)]}
            for name, index in zip(names, probabilities.argmax(axis=1), strict=True)
        ]
    return outputs


def evaluate_outer_once(
        contract: Mapping[str, object], pretruth: Mapping[str, object],
        all_reloads: Mapping[str, object], outer_truth_rows: list[dict[str, object]],
        labels: list[str], *, scoring_spec: Mapping[str, object], evaluation_path: Path,
) -> dict[str, object]:
    """Consume outer truth once, only after all disk-reloaded pretruth seals.

    Authentication happens before any caller-provided truth row is inspected.
    """
    predictions = outer_predictions_from_pretruth(
        contract, pretruth, all_reloads, scoring_spec=scoring_spec, labels=labels,
    )
    seal = pretruth["policy_reload"]["seal"]
    _require(isinstance(outer_truth_rows, list), "F008 outer truth must be an ordered list")
    observed = []
    for row in outer_truth_rows:
        _require(isinstance(row, Mapping), "F008 outer truth row must be a mapping")
        duration = _finite(row.get("duration_seconds"), "outer duration", nonnegative=True)
        group_id = row.get("group_id")
        _require(isinstance(row.get("audio_file"), str) and isinstance(group_id, str) and group_id,
                 "F008 outer truth public metadata are incomplete")
        observed.append({
            "audio_file": row["audio_file"], "group_id": group_id,
            "duration_seconds": duration,
            "has_nonzero_signal": truth(row.get("has_nonzero_signal")),
        })
    _require(observed == seal["outer_public_metadata"],
             "F008 outer truth public rows differ from the pretruth seal")
    metrics = {
        comparator: score_predictions(outer_truth_rows, rows, labels)
        for comparator, rows in predictions.items()
    }
    body = {
        "schema_version": EVALUATION_SCHEMA,
        "experiment_signature": contract["signature"],
        "outer_fold": pretruth["outer_fold"],
        "selected_arm": pretruth["selected_arm"],
        "policy_file_sha256": pretruth["policy_reload"]["file_sha256"],
        "policy_seal_sha256": pretruth["policy_reload"]["seal_sha256"],
        "outer_reference": outer_truth_rows,
        "predictions": predictions,
        "metrics": metrics,
        "one_shot_outer_evaluation": True,
        "outer_truth_first_access_stage": "after_all_disk_reloaded_pretruth_seals",
    }
    receipt = {**body, "evaluation_sha256": _sha(body)}
    _write_new_json(Path(evaluation_path), receipt)
    return receipt
