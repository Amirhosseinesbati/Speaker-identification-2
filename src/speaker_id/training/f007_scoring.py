"""Role-safe selection, calibration and outer scoring for F007.

F007 is deliberately a sibling of F005.  It selects exactly its three arms on
original group-disjoint known queries, seals the choice, then exposes unknown
queries only to the post-selection gate.  Outer speaker labels are accepted
only after policy seals for both folds have been reloaded from disk.

This module is CPU-only: it neither imports Torch nor opens audio, model state,
MLflow, or a historic C002b fixed scoring policy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities


ARM_IDS = ("control_f005", "l2sp_001", "l2sp_01")
COMPARATORS = ("frozen_same_protocol", "reused_control", "selected_arm")
SELECTION_SCHEMA = "f007-known-arm-selection-v1"
POLICY_SCHEMA = "f007-role-safe-policy-v1"
ALL_POLICY_SCHEMA = "f007-all-fold-policy-reloads-v1"
EVALUATION_SCHEMA = "f007-one-shot-outer-evaluation-v1"
ADVANCED_DIMENSION = 192
PUBLIC_DIMENSION = 512


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


def _array_receipt(value: object) -> dict:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
    }


def _write_new_json(path: Path, value: dict) -> bytes:
    """Atomically create immutable evidence; seals are never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F007 refuses to replace sealed evidence: {path}")
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
            raise FileExistsError(f"F007 refuses to replace sealed evidence: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _read_regular_json(path: Path) -> tuple[dict, bytes]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "F007 seal must be a regular JSON file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F007 seal must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), "F007 seal must contain a JSON object")
    return value, payload


def _validate_f007_config(contract: dict) -> None:
    """Validate the fixed F007 selection/scoring surface, not runtime state."""
    _require(isinstance(contract, dict), "F007 contract must be a dictionary")
    config = contract.get("config")
    _require(
        isinstance(config, dict)
        and isinstance(contract.get("signature"), str)
        and len(contract["signature"]) == 64
        and config.get("experiment_code") == "F007"
        and tuple(config.get("fold_ids", ())) == (0, 1)
        and tuple(arm.get("id") for arm in config.get("arms", ())) == ARM_IDS,
        "F007 needs an authenticated config with exactly two folds and three ordered arms",
    )
    selection, scoring = config.get("selection"), config.get("scoring")
    _require(
        isinstance(selection, dict)
        and selection.get("known_query_scope")
        == "original_group_disjoint_calibration_query_rows_only"
        and selection.get("metric")
        == "known_query_macro_f1_over_observed_labels_then_top1_accuracy"
        and selection.get("arm_tie_order") == list(ARM_IDS)
        and selection.get("unknown_calibration_hidden_until_arm_sealed") is True
        and selection.get("outer_labels_forbidden_until_all_policies_sealed") is True
        and selection.get("refit_after_selection") is False,
        "F007 known-only selection contract changed",
    )
    _require(
        isinstance(scoring, dict)
        and scoring.get("protocol") == "c002b_family_disjoint_roles_v1"
        and scoring.get("reference_method") == "max_reference"
        and scoring.get("fusion") == "same_reference_sqrt_weighted_encoder_concatenation"
        and scoring.get("alphas") == [0.0, 0.25, 0.5, 0.75, 1.0]
        and scoring.get("alpha_tie_order") == [0.0, 1.0, 0.25, 0.5, 0.75]
        and scoring.get("unknown_weights") == [0.0, 0.25, 0.5, 0.75, 1.0]
        and scoring.get("margin_weights") == [0.0, 0.5]
        and scoring.get("threshold_candidates") == 201
        and type(scoring.get("threshold_candidates")) is int
        and scoring.get("probability_temperature") == 0.05
        and type(scoring.get("probability_temperature")) is float
        and scoring.get("historical_c002b_fixed_policy") == "diagnostic_only_never_selectable",
        "F007 role-safe scoring contract changed",
    )
    labels = contract.get("labels")
    _require(
        isinstance(labels, list)
        and len(labels) >= 3
        and labels[0] == "unknown"
        and len(labels) == len(set(labels))
        and all(isinstance(label, str) and label for label in labels),
        "F007 label map is malformed",
    )


def _validate_embeddings(embeddings, valid, row_count: int, dimension: int,
                         name: str) -> tuple[np.ndarray, np.ndarray]:
    values, mask = np.asarray(embeddings), np.asarray(valid)
    _require(
        values.shape == (row_count, dimension)
        and values.dtype == np.float32
        and mask.shape == (row_count,)
        and mask.dtype == np.bool_
        and np.isfinite(values).all()
        and not np.any(values[~mask]),
        f"F007 {name} embeddings must be aligned finite float32 vectors with zero invalid rows",
    )
    _require(
        np.allclose(np.linalg.norm(values[mask], axis=1), 1.0, atol=1e-5),
        f"F007 {name} valid embeddings must be unit vectors",
    )
    return values, mask


def _weighted_encoder_pair(public, advanced, valid, alpha: float) -> np.ndarray:
    """The fixed F007 sqrt-weighted same-reference fusion.

    This is kept local to avoid importing the historical S008 experiment module
    and its unrelated runtime dependencies.  Its arithmetic is intentionally
    identical to the existing pure ``candidate_fusion.weighted_encoder_pair``.
    """
    public, advanced, mask = np.asarray(public), np.asarray(advanced), np.asarray(valid)
    _require(
        alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
        and public.ndim == 2 and public.shape[1] == PUBLIC_DIMENSION
        and advanced.shape == (len(public), ADVANCED_DIMENSION)
        and mask.shape == (len(public),) and mask.dtype == np.bool_,
        "F007 fusion inputs are misaligned",
    )
    views = []
    for values in (public, advanced):
        _require(
            values.dtype == np.float32 and np.isfinite(values).all() and not np.any(values[~mask])
            and np.allclose(np.linalg.norm(values[mask], axis=1), 1.0, atol=1e-5),
            "F007 fusion sources must be finite unit vectors with zero invalid rows",
        )
        normalized = np.zeros_like(values)
        norms = np.linalg.norm(values[mask], axis=1)
        normalized[mask] = values[mask] / norms[:, None]
        views.append(normalized)
    return np.concatenate((
        np.float32(np.sqrt(np.float32(1.0 - alpha))) * views[0],
        np.float32(np.sqrt(np.float32(alpha))) * views[1],
    ), axis=1).astype(np.float32)


def _topology(contract: dict, outer: int, valid: np.ndarray) -> dict:
    """Validate public roles/group topology without reading any outer label."""
    _validate_f007_config(contract)
    config, manifest, folds = contract["config"], contract["manifest"], contract["folds"]
    _require(type(outer) is int and outer in config["fold_ids"], "F007 outer fold is invalid")
    _require(
        isinstance(manifest, list) and isinstance(folds, list) and isinstance(contract.get("roles"), list),
        "F007 contract lacks role tables",
    )
    names = [row.get("audio_file") for row in manifest]
    _require(
        all(isinstance(name, str) and name for name in names) and len(names) == len(set(names)),
        "F007 manifest filenames must be unique and nonempty",
    )
    positions = {name: index for index, name in enumerate(names)}
    split = {row.get("audio_file"): row for row in folds}
    roles_list = [row for row in contract["roles"] if int(row.get("outer_fold", -1)) == outer]
    roles = {row.get("audio_file"): row for row in roles_list}
    _require(
        len(split) == len(folds) and len(roles) == len(roles_list)
        and set(split) == set(names) and set(roles) == set(names),
        "F007 manifest/fold/role rows are not aligned",
    )
    groups, assigned, eligible, fit_groups = [], [], [], set()
    group_assignments: dict[str, set[tuple]] = defaultdict(set)
    for index, name in enumerate(names):
        fold, role = split[name], roles[name]
        group = fold.get("group_id")
        try:
            assigned_fold = int(fold.get("fold"))
        except (TypeError, ValueError) as error:
            raise ValueError("F007 fold assignment must be numeric") from error
        _require(
            isinstance(group, str) and bool(group) and role.get("group_id") == group,
            "F007 role and fold content groups differ",
        )
        fitting, enrolling, querying, evaluating = (
            truth(role.get(key)) for key in (
                "encoder_fit_allowed", "enrollment_allowed", "calibration_query",
                "outer_evaluation_included",
            )
        )
        is_eligible = truth(fold.get("train_eligible"))
        _require(
            evaluating == (assigned_fold == outer)
            and not (evaluating and (fitting or enrolling or querying))
            and not (querying and (fitting or enrolling))
            and not ((fitting or enrolling or querying) and (not is_eligible or not valid[index])),
            "F007 role leakage, invalid support, or outer assignment drift",
        )
        group_assignments[group].add((assigned_fold, fitting, enrolling, querying, evaluating))
        if fitting:
            fit_groups.add(group)
        groups.append(group); assigned.append(assigned_fold); eligible.append(is_eligible)
    _require(
        not any(len(value) != 1 for value in group_assignments.values()),
        "F007 content group crosses fit/query/outer roles",
    )
    return {
        "positions": positions, "split": split, "roles_list": roles_list,
        "roles": roles, "groups": np.asarray(groups, dtype=object),
        "assigned": np.asarray(assigned, dtype=np.int64),
        "eligible": np.asarray(eligible, dtype=np.bool_), "fit_groups": fit_groups,
    }


def _expected_known_query_indices(contract: dict, outer: int, topology: dict) -> np.ndarray:
    known = set(contract["labels"][1:])
    result = np.asarray([
        topology["positions"][row["audio_file"]]
        for row in topology["roles_list"]
        if truth(row.get("calibration_query"))
        and contract["manifest"][topology["positions"][row["audio_file"]]].get("speaker_id") in known
    ], dtype=np.int64)
    _require(len(result) > 0 and len(set(result.tolist())) == len(result),
             "F007 has no unique known calibration queries")
    return result


def known_selection_scores(contract: dict, embeddings, valid, outer: int) -> dict:
    """Create known-only score evidence, excluding each complete query group.

    Unknown query indices/cohorts/similarities are not materialized in this
    phase.  Unknown-labeled references are merely filtered out of the known
    gallery, exactly before scoring any candidate arm.
    """
    row_count = len(contract.get("manifest", ()))
    values, mask = _validate_embeddings(embeddings, valid, row_count, ADVANCED_DIMENSION, "advanced")
    topology = _topology(contract, outer, mask)
    manifest, labels = contract["manifest"], contract["labels"]
    groups, assigned, eligible = topology["groups"], topology["assigned"], topology["eligible"]
    all_references = np.flatnonzero((assigned != outer) & eligible & mask)
    query_indices = _expected_known_query_indices(contract, outer, topology)
    _require(
        len(all_references) > 0 and set(query_indices).issubset(set(all_references)),
        "F007 known selection has no permitted known calibration/references",
    )
    _require(
        not (set(groups[query_indices]) & topology["fit_groups"])
        and not (set(groups[all_references]) & set(groups[assigned == outer])),
        "F007 query/fit/outer groups are not disjoint",
    )
    all_reference_labels = np.asarray([manifest[int(i)]["speaker_id"] for i in all_references], dtype=object)
    for group in set(groups[all_references]):
        _require(
            len(set(all_reference_labels[groups[all_references] == group])) == 1,
            "F007 known selection found a conflicting-label reference group",
        )
    reference_indices = all_references[np.isin(all_reference_labels, labels[1:])]
    _require(len(reference_indices) > 0, "F007 known selection has no known references")
    reference_labels = np.asarray([manifest[int(i)]["speaker_id"] for i in reference_indices], dtype=object)
    similarities = values[query_indices] @ values[reference_indices].T
    same_group = groups[query_indices, None] == groups[reference_indices][None, :]
    similarities[same_group] = -np.inf
    scores = np.empty((len(query_indices), len(labels) - 1), dtype=np.float32)
    support = np.empty_like(scores, dtype=np.int64)
    for target, label in enumerate(labels[1:]):
        columns = reference_labels == label
        _require(np.any(columns), "F007 known selection lost a reference class")
        scores[:, target] = similarities[:, columns].max(axis=1)
        support[:, target] = np.sum(~same_group[:, columns], axis=1)
    _require(np.isfinite(scores).all() and np.all(support >= 1),
             "F007 group exclusion left an empty known reference class")
    np.clip(scores, -1.0, 1.0, out=scores)
    return {
        "known_calibration_indices": query_indices, "known_scores": scores,
        "known_labels": list(labels[1:]), "reference_support": support,
        "provenance": {
            "protocol": "f007_preselection_known_only_group_disjoint_gallery_v1",
            "authoritative_postselection_scorer": "speaker_id.training.heldout_references.heldout_reference_scores",
            "reference_scope": "all_eligible_outer_training_known_references",
            "whole_query_group_excluded": True,
            "unknown_calibration_indices_materialized": False,
            "unknown_reference_cohort_materialized": False,
            "unknown_similarity_computed": False,
            "outer_labels_read": False, "outer_fold": outer,
        },
    }


def _known_query_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    observed = sorted(set(int(value) for value in actual))
    support, assigned = Counter(map(int, actual)), Counter(map(int, predicted))
    correct = Counter(int(a) for a, p in zip(actual, predicted, strict=True) if a == p)
    f1 = [2.0 * correct[label] / (support[label] + assigned[label])
          if support[label] + assigned[label] else 0.0 for label in observed]
    return {
        "row_count": int(len(actual)), "observed_known_labels": int(len(observed)),
        "macro_f1_observed_known_labels": float(np.mean(f1)),
        "top1_accuracy": float(np.mean(actual == predicted)),
        "errors": int(np.sum(actual != predicted)),
    }


def _known_score_evidence(score: dict) -> dict:
    required = {"known_calibration_indices", "known_scores", "known_labels", "reference_support", "provenance"}
    _require(set(score) == required and isinstance(score["provenance"], dict),
             "F007 known score evidence schema changed")
    evidence = {
        "known_calibration_indices": _array_receipt(score["known_calibration_indices"]),
        "known_scores": _array_receipt(score["known_scores"]),
        "known_labels_sha256": _sha(score["known_labels"]),
        "reference_support": _array_receipt(score["reference_support"]),
        "protocol": score["provenance"].get("protocol"),
        "outer_fold": score["provenance"].get("outer_fold"),
    }
    return {**evidence, "receipt_sha256": _sha(evidence)}


def select_known_arm(contract: dict, outer: int, known_scores_by_arm: dict[str, dict]) -> dict:
    """Choose among exactly F007's control and two L2-SP tails.

    The selection input can only contain known-query score matrices.  It never
    receives an unknown score or an outer label.
    """
    _validate_f007_config(contract)
    _require(
        isinstance(known_scores_by_arm, dict) and set(known_scores_by_arm) == set(ARM_IDS),
        "F007 selection requires all three arms in fixed order",
    )
    topology = _topology(
        contract, outer, np.ones(len(contract["manifest"]), dtype=np.bool_),
    )
    expected_indices = _expected_known_query_indices(contract, outer, topology)
    labels, manifest = contract["labels"], contract["manifest"]
    anchor = known_scores_by_arm[ARM_IDS[0]]
    anchor_indices = np.asarray(anchor.get("known_calibration_indices"))
    _require(
        anchor_indices.dtype.kind in "iu" and anchor_indices.ndim == 1
        and np.array_equal(anchor_indices, expected_indices),
        "F007 known selection indices differ from immutable known-query roles",
    )
    for arm_id in ARM_IDS:
        score = known_scores_by_arm[arm_id]
        matrix, support = np.asarray(score.get("known_scores")), np.asarray(score.get("reference_support"))
        provenance = score.get("provenance", {})
        _require(
            isinstance(score, dict) and score.get("known_labels") == labels[1:]
            and np.array_equal(np.asarray(score.get("known_calibration_indices")), anchor_indices)
            and matrix.shape == (len(anchor_indices), len(labels) - 1)
            and matrix.dtype == np.float32 and np.isfinite(matrix).all()
            and support.shape == matrix.shape and support.dtype.kind in "iu" and np.all(support >= 1)
            and provenance.get("protocol") == "f007_preselection_known_only_group_disjoint_gallery_v1"
            and provenance.get("whole_query_group_excluded") is True
            and provenance.get("unknown_calibration_indices_materialized") is False
            and provenance.get("unknown_reference_cohort_materialized") is False
            and provenance.get("unknown_similarity_computed") is False
            and provenance.get("outer_labels_read") is False
            and provenance.get("outer_fold") == outer,
            f"F007 arm {arm_id} is not valid known-only evidence",
        )
    label_index = {label: index for index, label in enumerate(labels)}
    actual = np.asarray(
        [label_index[manifest[int(index)]["speaker_id"]] for index in anchor_indices],
        dtype=np.int64,
    )
    _require(np.all(actual > 0), "F007 selection saw a non-known calibration row")
    metrics = {
        arm_id: _known_query_metrics(
            actual,
            np.asarray(known_scores_by_arm[arm_id]["known_scores"]).argmax(axis=1).astype(np.int64) + 1,
        )
        for arm_id in ARM_IDS
    }
    selected = max(
        ARM_IDS,
        key=lambda arm_id: (
            metrics[arm_id]["macro_f1_observed_known_labels"],
            metrics[arm_id]["top1_accuracy"], -ARM_IDS.index(arm_id),
        ),
    )
    observed = {labels[int(index)] for index in actual}
    body = {
        "schema_version": SELECTION_SCHEMA,
        "experiment_signature": contract["signature"], "outer_fold": outer,
        "selected_arm": selected,
        "scientific_conclusion": (
            "control_retained_on_known_calibration" if selected == ARM_IDS[0]
            else "l2sp_candidate_selected_on_known_calibration"
        ),
        "arm_metrics": {arm_id: metrics[arm_id] for arm_id in ARM_IDS},
        "known_score_evidence": {
            arm_id: _known_score_evidence(known_scores_by_arm[arm_id]) for arm_id in ARM_IDS
        },
        "selection_order": [
            "macro_f1_observed_known_labels", "top1_accuracy", "fixed_arm_tie_order",
        ],
        "known_calibration_indices_sha256": _array_receipt(anchor_indices)["sha256"],
        "known_query_rows": int(len(anchor_indices)),
        "observed_known_labels": int(len(observed)),
        "absent_known_labels": sorted(set(labels[1:]) - observed),
        "unknown_calibration_indices_materialized": False,
        "unknown_reference_cohort_materialized": False,
        "unknown_similarity_computed": False,
        "outer_rows_or_labels_read": False,
        "refit_after_selection": False,
    }
    return {**body, "seal_sha256": _sha(body)}


def reload_selection_seal(path: Path, contract: dict, outer: int) -> dict:
    """Reload and authenticate a known-only F007 arm selection."""
    _validate_f007_config(contract)
    seal, payload = _read_regular_json(path)
    required = {
        "schema_version", "experiment_signature", "outer_fold", "selected_arm",
        "scientific_conclusion", "arm_metrics", "known_score_evidence", "selection_order",
        "known_calibration_indices_sha256", "known_query_rows", "observed_known_labels",
        "absent_known_labels", "unknown_calibration_indices_materialized",
        "unknown_reference_cohort_materialized", "unknown_similarity_computed",
        "outer_rows_or_labels_read", "refit_after_selection", "seal_sha256",
    }
    _require(set(seal) == required, "F007 selection seal schema changed")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    topology = _topology(
        contract, outer, np.ones(len(contract["manifest"]), dtype=np.bool_),
    )
    indices = _expected_known_query_indices(contract, outer, topology)
    labels, manifest = contract["labels"], contract["manifest"]
    actual = [labels.index(manifest[int(index)]["speaker_id"]) for index in indices]
    _require(all(index > 0 for index in actual), "F007 selection role labels are invalid")
    observed = {labels[index] for index in actual}
    metrics, evidence = seal.get("arm_metrics"), seal.get("known_score_evidence")
    _require(
        seal.get("schema_version") == SELECTION_SCHEMA
        and seal.get("experiment_signature") == contract["signature"]
        and seal.get("outer_fold") == outer and seal.get("selected_arm") in ARM_IDS
        and seal.get("selection_order") == [
            "macro_f1_observed_known_labels", "top1_accuracy", "fixed_arm_tie_order",
        ]
        and isinstance(metrics, dict) and set(metrics) == set(ARM_IDS)
        and isinstance(evidence, dict) and set(evidence) == set(ARM_IDS)
        and seal.get("known_calibration_indices_sha256") == _array_receipt(indices)["sha256"]
        and seal.get("known_query_rows") == len(indices)
        and seal.get("observed_known_labels") == len(observed)
        and seal.get("absent_known_labels") == sorted(set(labels[1:]) - observed)
        and seal.get("unknown_calibration_indices_materialized") is False
        and seal.get("unknown_reference_cohort_materialized") is False
        and seal.get("unknown_similarity_computed") is False
        and seal.get("outer_rows_or_labels_read") is False
        and seal.get("refit_after_selection") is False
        and seal.get("seal_sha256") == _sha(body),
        "F007 selection seal identity or anti-leak evidence is invalid",
    )
    for arm_id in ARM_IDS:
        row, receipt = metrics[arm_id], evidence[arm_id]
        _require(
            isinstance(row, dict)
            and set(row) == {
                "row_count", "observed_known_labels", "macro_f1_observed_known_labels",
                "top1_accuracy", "errors",
            }
            and row["row_count"] == len(indices)
            and row["observed_known_labels"] == len(observed)
            and all(
                isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
                for key in ("macro_f1_observed_known_labels", "top1_accuracy")
            )
            and 0.0 <= float(row["macro_f1_observed_known_labels"]) <= 1.0
            and 0.0 <= float(row["top1_accuracy"]) <= 1.0
            and isinstance(receipt, dict)
            and receipt.get("protocol") == "f007_preselection_known_only_group_disjoint_gallery_v1"
            and receipt.get("outer_fold") == outer
            and _is_sha256(receipt.get("receipt_sha256"))
            and receipt["receipt_sha256"] == _sha({
                key: value for key, value in receipt.items() if key != "receipt_sha256"
            }),
            f"F007 selection receipt for {arm_id} is invalid",
        )
    expected = max(
        ARM_IDS,
        key=lambda arm_id: (
            metrics[arm_id]["macro_f1_observed_known_labels"],
            metrics[arm_id]["top1_accuracy"], -ARM_IDS.index(arm_id),
        ),
    )
    _require(
        seal["selected_arm"] == expected
        and seal["scientific_conclusion"] == (
            "control_retained_on_known_calibration" if expected == ARM_IDS[0]
            else "l2sp_candidate_selected_on_known_calibration"
        ),
        "F007 selection seal does not contain the deterministic known-query winner",
    )
    return {
        "kind": "f007_selection_disk_reload", "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "seal_sha256": seal["seal_sha256"], "disk_reloaded": True, "seal": seal,
    }


def select_and_seal_known_arm(contract: dict, outer: int, *, embeddings_by_arm: dict,
                              valid, selection_seal_path: Path,
                              known_scorer=known_selection_scores) -> dict:
    """Score all preregistered arms on known rows and persist one choice."""
    _validate_f007_config(contract)
    _require(
        isinstance(embeddings_by_arm, dict) and set(embeddings_by_arm) == set(ARM_IDS),
        "F007 known selection needs exactly control_f005, l2sp_001 and l2sp_01",
    )
    scores = {
        arm_id: known_scorer(contract, embeddings_by_arm[arm_id], valid, outer)
        for arm_id in ARM_IDS
    }
    seal, path = select_known_arm(contract, outer, scores), Path(selection_seal_path)
    if path.exists() or path.is_symlink():
        reloaded = reload_selection_seal(path, contract, outer)
        _require(reloaded["seal"] == seal,
                 "F007 existing selection seal differs from authenticated evidence")
        return reloaded
    _write_new_json(path, seal)
    reloaded = reload_selection_seal(path, contract, outer)
    _require(reloaded["seal"] == seal, "F007 selection seal failed disk round-trip")
    return reloaded


def _verify_selection_reload(reload: dict, contract: dict, outer: int) -> dict:
    _require(
        isinstance(reload, dict) and reload.get("kind") == "f007_selection_disk_reload"
        and reload.get("disk_reloaded") is True,
        "F007 open-set scoring requires a disk-reloaded known-arm seal",
    )
    current = reload_selection_seal(Path(reload["path"]), contract, outer)
    _require(
        current["file_sha256"] == reload.get("file_sha256")
        and current["seal_sha256"] == reload.get("seal_sha256")
        and current["seal"] == reload.get("seal"),
        "F007 selection seal changed after its caller reloaded it",
    )
    return current


class _OuterTruthForbidden(dict):
    """A row that raises if pre-truth code touches its class label."""

    def __getitem__(self, key):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before both F007 policies were sealed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before both F007 policies were sealed")
        return super().get(key, default)


def _expected_outer_indices(contract: dict, outer: int) -> np.ndarray:
    positions = {row["audio_file"]: index for index, row in enumerate(contract["manifest"])}
    _require(len(positions) == len(contract["manifest"]), "F007 manifest names are not unique")
    result = np.asarray([
        positions[row["audio_file"]]
        for row in contract["roles"]
        if int(row.get("outer_fold", -1)) == outer and truth(row.get("outer_evaluation_included"))
    ], dtype=np.int64)
    _require(len(result) > 0 and len(set(result.tolist())) == len(result),
             "F007 outer role rows are empty or duplicated")
    return result


def _guard_outer_truth(contract: dict, outer: int) -> dict:
    outer_names = {
        contract["manifest"][int(index)]["audio_file"]
        for index in _expected_outer_indices(contract, outer)
    }
    guarded = dict(contract)
    guarded["manifest"] = [
        _OuterTruthForbidden(row) if row["audio_file"] in outer_names else row
        for row in contract["manifest"]
    ]
    guarded["roles"] = [
        _OuterTruthForbidden(row)
        if int(row.get("outer_fold", -1)) == outer and truth(row.get("outer_evaluation_included"))
        else row
        for row in contract["roles"]
    ]
    return guarded


def _normalized_public_metadata(row: dict, group_id: object) -> dict:
    try:
        duration = float(row.get("duration_seconds"))
    except (TypeError, ValueError) as error:
        raise ValueError("F007 duration metadata must be numeric") from error
    _require(
        math.isfinite(duration) and duration >= 0
        and isinstance(row.get("audio_file"), str) and bool(row["audio_file"])
        and isinstance(group_id, str) and bool(group_id)
        and "has_nonzero_signal" in row,
        "F007 outer public metadata are incomplete",
    )
    return {
        "audio_file": row["audio_file"], "group_id": group_id,
        "duration_seconds": duration, "has_nonzero_signal": truth(row["has_nonzero_signal"]),
    }


def _outer_public_metadata(contract: dict, outer: int, indices: np.ndarray) -> list[dict]:
    split = {row["audio_file"]: row for row in contract["folds"]}
    rows = []
    for value in indices:
        row, fold = contract["manifest"][int(value)], split.get(contract["manifest"][int(value)]["audio_file"])
        _require(isinstance(fold, dict) and int(fold.get("fold", -1)) == outer,
                 "F007 outer scorer returned a row outside the requested fold")
        rows.append(_normalized_public_metadata(row, fold.get("group_id")))
    return rows


def _score_evidence(score: dict) -> dict:
    required = {
        "calibration_indices", "outer_indices", "known_labels", "inner_known_scores",
        "outer_known_scores", "inner_unknown_similarity", "outer_unknown_similarity",
        "outer_valid", "reference_counts", "provenance",
    }
    _require(set(score) == required and isinstance(score["provenance"], dict),
             "F007 heldout scorer schema changed")
    evidence = {
        key: _array_receipt(score[key]) for key in (
            "calibration_indices", "outer_indices", "inner_known_scores",
            "outer_known_scores", "inner_unknown_similarity", "outer_unknown_similarity",
            "outer_valid",
        )
    }
    evidence["known_labels_sha256"] = _sha(score["known_labels"])
    evidence["protocol"] = score["provenance"].get("protocol")
    evidence["outer_fold"] = score["provenance"].get("outer_fold")
    return {**evidence, "receipt_sha256": _sha(evidence)}


def _score_candidates(contract: dict, outer: int, *, public_embeddings,
                      advanced_embeddings, valid, scorer) -> dict[float, dict]:
    alphas = tuple(contract["config"]["scoring"]["alphas"])
    candidates = {}
    for alpha in alphas:
        if alpha == 0.0:
            values = public_embeddings
        elif alpha == 1.0:
            values = advanced_embeddings
        else:
            values = _weighted_encoder_pair(public_embeddings, advanced_embeddings, valid, alpha)
        candidates[alpha] = scorer(contract, values, valid, outer)
    anchor = candidates[alphas[0]]
    for score in candidates.values():
        _require(
            score.get("known_labels") == anchor.get("known_labels")
            and np.array_equal(score.get("calibration_indices"), anchor.get("calibration_indices"))
            and np.array_equal(score.get("outer_indices"), anchor.get("outer_indices"))
            and np.array_equal(score.get("outer_valid"), anchor.get("outer_valid")),
            "F007 alpha candidates use different role-safe rows",
        )
        _score_evidence(score)
    return candidates


def _calibration_truth(contract: dict, outer: int, score: dict) -> np.ndarray:
    indices = np.asarray(score["calibration_indices"])
    labels, manifest = contract["labels"], contract["manifest"]
    index = {label: position for position, label in enumerate(labels)}
    roles = {
        row["audio_file"]: row for row in contract["roles"]
        if int(row.get("outer_fold", -1)) == outer
    }
    actual = []
    for row_index in indices:
        row = manifest[int(row_index)]
        _require(
            isinstance(roles.get(row["audio_file"]), dict)
            and truth(roles[row["audio_file"]].get("calibration_query")),
            "F007 policy calibration score includes a non-query row",
        )
        actual.append(index[row["speaker_id"]])
    result = np.asarray(actual, dtype=np.int64)
    _require(
        len(result) == len(indices) and np.any(result == 0) and np.any(result > 0),
        "F007 open-set calibration needs independent known and unknown query rows",
    )
    return result


def _select_alpha_and_gate(contract: dict, candidates: dict[float, dict],
                           calibration_truth: np.ndarray) -> tuple[dict, dict, dict]:
    scoring = contract["config"]["scoring"]
    alphas = tuple(scoring["alphas"])
    _require(tuple(candidates) == alphas, "F007 alpha candidate grid changed")
    curves = {}
    for alpha in alphas:
        score = candidates[alpha]
        calibration, curve = calibrate_gate(
            np.asarray(score["inner_known_scores"]), calibration_truth,
            np.asarray(score["inner_unknown_similarity"]),
            list(scoring["unknown_weights"]), list(scoring["margin_weights"]),
            int(scoring["threshold_candidates"]), classes=len(contract["labels"]),
        )
        curves[str(alpha)] = {"advanced_weight": float(alpha), "selected": calibration, "curve": curve}
    tie_order = tuple(scoring["alpha_tie_order"])
    selected = max(tie_order, key=lambda alpha: curves[str(alpha)]["selected"]["inner_macro_f1_447"])
    policy = {
        "advanced_weight": float(selected), "calibration": curves[str(selected)]["selected"],
        "alpha_tie_order": list(tie_order),
        "selection_scope": "independent_group_disjoint_known_and_unknown_calibration_queries_only",
        "calibration_indices_sha256": _array_receipt(candidates[selected]["calibration_indices"])["sha256"],
        "inner_truth_sha256": _array_receipt(calibration_truth)["sha256"],
        "candidate_curves_sha256": _sha(curves),
        "score_evidence": _score_evidence(candidates[selected]),
    }
    return policy, candidates[selected], curves


def _validate_policy_entry(entry: dict, comparator: str, contract: dict, outer: int) -> None:
    required = {
        "comparator", "advanced_weight", "calibration", "alpha_tie_order",
        "selection_scope", "calibration_indices_sha256", "inner_truth_sha256",
        "candidate_curves_sha256", "score_evidence",
    }
    _require(isinstance(entry, dict) and set(entry) == required,
             f"F007 {comparator} policy schema changed")
    scoring, calibration = contract["config"]["scoring"], entry["calibration"]
    evidence = entry["score_evidence"]
    _require(
        entry["comparator"] == comparator and entry["advanced_weight"] in scoring["alphas"]
        and entry["alpha_tie_order"] == scoring["alpha_tie_order"]
        and entry["selection_scope"] == "independent_group_disjoint_known_and_unknown_calibration_queries_only"
        and isinstance(calibration, dict)
        and set(calibration) == {"unknown_weight", "margin_weight", "threshold", "inner_macro_f1_447"}
        and calibration["unknown_weight"] in scoring["unknown_weights"]
        and calibration["margin_weight"] in scoring["margin_weights"]
        and all(math.isfinite(float(calibration[key])) for key in ("threshold", "inner_macro_f1_447"))
        and 0.0 <= float(calibration["inner_macro_f1_447"]) <= 1.0
        and all(_is_sha256(entry[key]) for key in (
            "calibration_indices_sha256", "inner_truth_sha256", "candidate_curves_sha256"
        ))
        and isinstance(evidence, dict)
        and evidence.get("protocol") == "original_heldout_queries_expanded_gallery_v1"
        and evidence.get("outer_fold") == outer and _is_sha256(evidence.get("receipt_sha256"))
        and evidence["receipt_sha256"] == _sha({
            key: value for key, value in evidence.items() if key != "receipt_sha256"
        }),
        f"F007 {comparator} policy identity changed",
    )


def reload_policy_seal(path: Path, contract: dict, outer: int, *,
                       selection_reload: dict | None = None) -> dict:
    """Reload one post-selection policy seal without opening outer truth."""
    _validate_f007_config(contract)
    seal, payload = _read_regular_json(path)
    required = {
        "schema_version", "experiment_signature", "outer_fold", "selected_arm",
        "selection_seal_sha256", "selection_file_sha256", "scoring_protocol",
        "comparators", "policies", "labels_sha256", "label_count",
        "outer_public_metadata", "outer_public_metadata_sha256",
        "unknown_scoring_started_after_selection_seal_reload", "outer_truth_read",
        "outer_truth_accepted_by_pretruth_api", "policy_selection_complete_before_outer_truth",
        "all_fold_seals_required_before_outer_truth", "seal_sha256",
    }
    _require(set(seal) == required, "F007 policy seal schema changed")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    indices = _expected_outer_indices(contract, outer)
    metadata = _outer_public_metadata(contract, outer, indices)
    _require(
        seal.get("schema_version") == POLICY_SCHEMA
        and seal.get("experiment_signature") == contract["signature"]
        and seal.get("outer_fold") == outer and seal.get("selected_arm") in ARM_IDS
        and _is_sha256(seal.get("selection_seal_sha256"))
        and _is_sha256(seal.get("selection_file_sha256"))
        and seal.get("scoring_protocol") == "c002b_family_disjoint_roles_v1"
        and seal.get("comparators") == list(COMPARATORS)
        and isinstance(seal.get("policies"), dict) and set(seal["policies"]) == set(COMPARATORS)
        and seal.get("labels_sha256") == _sha(contract["labels"])
        and seal.get("label_count") == len(contract["labels"])
        and seal.get("outer_public_metadata") == metadata
        and seal.get("outer_public_metadata_sha256") == _sha(metadata)
        and seal.get("unknown_scoring_started_after_selection_seal_reload") is True
        and seal.get("outer_truth_read") is False
        and seal.get("outer_truth_accepted_by_pretruth_api") is False
        and seal.get("policy_selection_complete_before_outer_truth") is True
        and seal.get("all_fold_seals_required_before_outer_truth") is True
        and seal.get("seal_sha256") == _sha(body),
        "F007 policy seal identity, roles, or anti-leak claims are invalid",
    )
    for comparator in COMPARATORS:
        _validate_policy_entry(seal["policies"][comparator], comparator, contract, outer)
    if selection_reload is not None:
        selection = _verify_selection_reload(selection_reload, contract, outer)
        _require(
            seal["selection_seal_sha256"] == selection["seal_sha256"]
            and seal["selection_file_sha256"] == selection["file_sha256"]
            and seal["selected_arm"] == selection["seal"]["selected_arm"],
            "F007 policy seal refers to another known-arm selection",
        )
    return {
        "kind": "f007_policy_disk_reload", "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "seal_sha256": seal["seal_sha256"], "disk_reloaded": True, "seal": seal,
    }


def prepare_and_seal_policy(
        contract: dict, outer: int, *, public_embeddings, frozen_advanced_embeddings,
        reused_control_embeddings, selected_arm_embeddings, valid,
        selection_reload: dict, policy_seal_path: Path,
        heldout_scorer=heldout_reference_scores) -> dict:
    """Fit and seal control/selected role-safe policies after arm selection.

    The API accepts the immutable frozen comparator, reused F005 control and
    already selected arm; it has no parameter for rejected tail embeddings.
    """
    _validate_f007_config(contract)
    selection = _verify_selection_reload(selection_reload, contract, outer)
    row_count = len(contract["manifest"])
    public, mask = _validate_embeddings(public_embeddings, valid, row_count, PUBLIC_DIMENSION, "public")
    frozen, frozen_mask = _validate_embeddings(
        frozen_advanced_embeddings, valid, row_count, ADVANCED_DIMENSION, "frozen comparator",
    )
    control, control_mask = _validate_embeddings(
        reused_control_embeddings, valid, row_count, ADVANCED_DIMENSION, "reused control",
    )
    selected, selected_mask = _validate_embeddings(
        selected_arm_embeddings, valid, row_count, ADVANCED_DIMENSION, "selected arm",
    )
    _require(
        np.array_equal(mask, frozen_mask) and np.array_equal(mask, control_mask)
        and np.array_equal(mask, selected_mask),
        "F007 public/frozen/control/selected validity masks differ",
    )
    guarded = _guard_outer_truth(contract, outer)
    advanced = {
        "frozen_same_protocol": frozen,
        "reused_control": control,
        "selected_arm": selected,
    }
    policies, score_bundles, curves_by_comparator = {}, {}, {}
    for comparator in COMPARATORS:
        candidates = _score_candidates(
            guarded, outer, public_embeddings=public,
            advanced_embeddings=advanced[comparator], valid=mask, scorer=heldout_scorer,
        )
        calibration_truth = _calibration_truth(guarded, outer, candidates[0.0])
        policy, chosen, curves = _select_alpha_and_gate(guarded, candidates, calibration_truth)
        policies[comparator] = {"comparator": comparator, **policy}
        score_bundles[comparator], curves_by_comparator[comparator] = chosen, curves
    anchor = score_bundles[COMPARATORS[0]]
    _require(
        all(
            np.array_equal(score["calibration_indices"], anchor["calibration_indices"])
            and np.array_equal(score["outer_indices"], anchor["outer_indices"])
            and np.array_equal(score["outer_valid"], anchor["outer_valid"])
            for score in score_bundles.values()
        ),
        "F007 control/selected role-safe score rows are misaligned",
    )
    outer_indices = np.asarray(anchor["outer_indices"], dtype=np.int64)
    _require(np.array_equal(outer_indices, _expected_outer_indices(contract, outer)),
             "F007 heldout scorer outer order differs from immutable roles")
    outer_metadata = _outer_public_metadata(contract, outer, outer_indices)
    body = {
        "schema_version": POLICY_SCHEMA,
        "experiment_signature": contract["signature"], "outer_fold": outer,
        "selected_arm": selection["seal"]["selected_arm"],
        "selection_seal_sha256": selection["seal_sha256"],
        "selection_file_sha256": selection["file_sha256"],
        "scoring_protocol": "c002b_family_disjoint_roles_v1",
        "comparators": list(COMPARATORS), "policies": policies,
        "labels_sha256": _sha(contract["labels"]), "label_count": len(contract["labels"]),
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "unknown_scoring_started_after_selection_seal_reload": True,
        "outer_truth_read": False, "outer_truth_accepted_by_pretruth_api": False,
        "policy_selection_complete_before_outer_truth": True,
        "all_fold_seals_required_before_outer_truth": True,
    }
    seal = {**body, "seal_sha256": _sha(body)}
    _write_new_json(policy_seal_path, seal)
    reloaded = reload_policy_seal(policy_seal_path, contract, outer, selection_reload=selection)
    return {
        "kind": "f007_pretruth_scoring_bundle", "experiment_signature": contract["signature"],
        "outer_fold": outer, "selected_arm": selection["seal"]["selected_arm"],
        "labels_sha256": _sha(contract["labels"]),
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "outer_files": [row["audio_file"] for row in outer_metadata],
        "outer_indices_sha256": _array_receipt(outer_indices)["sha256"],
        "score_bundles": score_bundles, "policy_reload": reloaded,
        "candidate_curves": curves_by_comparator, "outer_truth_read": False,
    }


def reload_all_policy_seals(contract: dict, *, policy_paths_by_outer: dict,
                            selection_reloads_by_outer: dict) -> dict:
    """Reload policy seals for both outer folds, the gate before outer truth."""
    _validate_f007_config(contract)
    folds = tuple(contract["config"]["fold_ids"])
    _require(
        isinstance(policy_paths_by_outer, dict) and isinstance(selection_reloads_by_outer, dict)
        and set(policy_paths_by_outer) == set(folds) and set(selection_reloads_by_outer) == set(folds),
        "F007 needs policy and selection seals for every outer fold before truth",
    )
    policy_reloads = {
        outer: reload_policy_seal(
            Path(policy_paths_by_outer[outer]), contract, outer,
            selection_reload=selection_reloads_by_outer[outer],
        ) for outer in folds
    }
    return {
        "kind": ALL_POLICY_SCHEMA, "experiment_signature": contract["signature"],
        "fold_ids": list(folds), "policy_reloads": policy_reloads,
        "selection_reloads": selection_reloads_by_outer, "all_folds_sealed": True,
    }


def _verify_all_policy_reloads(all_reloads: dict, contract: dict) -> dict:
    _validate_f007_config(contract)
    folds = tuple(contract["config"]["fold_ids"])
    _require(
        isinstance(all_reloads, dict) and all_reloads.get("kind") == ALL_POLICY_SCHEMA
        and all_reloads.get("experiment_signature") == contract["signature"]
        and all_reloads.get("fold_ids") == list(folds) and all_reloads.get("all_folds_sealed") is True
        and isinstance(all_reloads.get("policy_reloads"), dict)
        and isinstance(all_reloads.get("selection_reloads"), dict)
        and set(all_reloads["policy_reloads"]) == set(folds)
        and set(all_reloads["selection_reloads"]) == set(folds),
        "F007 outer truth is forbidden until both policy seals are present",
    )
    refreshed = {}
    for outer in folds:
        candidate = all_reloads["policy_reloads"][outer]
        current = reload_policy_seal(
            Path(candidate["path"]), contract, outer,
            selection_reload=all_reloads["selection_reloads"][outer],
        )
        _require(
            current["file_sha256"] == candidate.get("file_sha256")
            and current["seal_sha256"] == candidate.get("seal_sha256")
            and current["seal"] == candidate.get("seal"),
            "F007 policy seal changed after the all-fold reload",
        )
        refreshed[outer] = current
    return refreshed


def _verify_pretruth(pretruth: dict, all_policy_reloads: dict, contract: dict) -> dict:
    _require(
        isinstance(pretruth, dict) and pretruth.get("kind") == "f007_pretruth_scoring_bundle"
        and pretruth.get("experiment_signature") == contract["signature"]
        and pretruth.get("outer_truth_read") is False
        and set(pretruth.get("score_bundles", {})) == set(COMPARATORS),
        "F007 pretruth score bundle is malformed",
    )
    outer = pretruth.get("outer_fold")
    _require(type(outer) is int and outer in all_policy_reloads,
             "F007 pretruth bundle references an unsealed outer fold")
    policy, supplied = all_policy_reloads[outer], pretruth.get("policy_reload")
    _require(
        isinstance(supplied, dict) and supplied.get("file_sha256") == policy["file_sha256"]
        and supplied.get("seal_sha256") == policy["seal_sha256"]
        and supplied.get("seal") == policy["seal"],
        "F007 pretruth bundle and all-fold policy reload differ",
    )
    seal = policy["seal"]
    _require(
        pretruth.get("selected_arm") == seal["selected_arm"]
        and pretruth.get("labels_sha256") == seal["labels_sha256"]
        and pretruth.get("outer_public_metadata") == seal["outer_public_metadata"]
        and pretruth.get("outer_public_metadata_sha256") == seal["outer_public_metadata_sha256"],
        "F007 pretruth identity differs from its policy seal",
    )
    for comparator in COMPARATORS:
        _require(
            _score_evidence(pretruth["score_bundles"][comparator])
            == seal["policies"][comparator]["score_evidence"],
            f"F007 {comparator} score evidence changed after policy sealing",
        )
    return seal


def evaluate_outer_once(
        contract: dict, pretruth: dict, all_policy_reloads: dict,
        outer_truth_rows: list[dict], labels: list[str], *, evaluation_path: Path) -> dict:
    """Consume an outer fold once, only after both disk-reloaded policy seals.

    The authentication block deliberately precedes even type/metadata access to
    ``outer_truth_rows``.  This makes a missing fold seal fail before an outer
    speaker label can be read by an accidental caller.
    """
    all_policies = _verify_all_policy_reloads(all_policy_reloads, contract)
    seal = _verify_pretruth(pretruth, all_policies, contract)
    _require(
        isinstance(labels, list) and _sha(labels) == seal["labels_sha256"],
        "F007 outer labels differ from the sealed label map",
    )
    temperature = contract["config"]["scoring"]["probability_temperature"]
    _require(temperature == 0.05 and type(temperature) is float,
             "F007 probability temperature changed")

    # Verify all public fields before score_predictions sees true speaker_id.
    _require(isinstance(outer_truth_rows, list), "F007 outer truth must be an ordered list")
    observed_public = [
        _normalized_public_metadata(row, row.get("group_id")) for row in outer_truth_rows
    ]
    _require(observed_public == seal["outer_public_metadata"],
             "F007 outer truth public rows differ from sealed role metadata")
    outputs = {}
    for comparator in COMPARATORS:
        score, calibration = (
            pretruth["score_bundles"][comparator],
            seal["policies"][comparator]["calibration"],
        )
        probabilities = reference_probabilities(
            np.asarray(score["outer_known_scores"]),
            np.asarray(score["outer_unknown_similarity"]), calibration,
            np.asarray(score["outer_valid"]), temperature,
        )
        _require(
            len(probabilities) == len(outer_truth_rows) and probabilities.shape[1] == len(labels),
            "F007 sealed score dimensions differ from outer labels",
        )
        predictions = [
            {"audio_file": row["audio_file"], "speaker_id": labels[int(index)]}
            for row, index in zip(outer_truth_rows, probabilities.argmax(axis=1), strict=True)
        ]
        metrics = score_predictions(outer_truth_rows, predictions, labels)
        outputs[comparator] = {
            "predictions": predictions,
            "metrics": {key: metrics[key] for key in (
                "row_count", "class_count", "macro_f1", "accuracy", "errors",
            )},
            "prediction_sha256": _sha(predictions),
        }
    body = {
        "schema_version": EVALUATION_SCHEMA,
        "experiment_signature": contract["signature"],
        "outer_fold": pretruth["outer_fold"], "policy_seal_sha256": seal["seal_sha256"],
        "all_policy_seal_sha256": _sha({
            str(fold): all_policies[fold]["seal_sha256"]
            for fold in contract["config"]["fold_ids"]
        }),
        "selected_arm": seal["selected_arm"],
        "outer_files": [row["audio_file"] for row in observed_public],
        "comparators": outputs,
    }
    receipt = {**body, "evaluation_sha256": _sha(body)}
    _write_new_json(evaluation_path, receipt)
    return receipt
