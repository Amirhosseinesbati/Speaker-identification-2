"""Leak-resistant CPU scoring and promotion primitives for F005.

The public functions deliberately split the experiment into three phases:

1. reload the known-query arm-selection seal from disk;
2. fit and seal all inner open-set policies without accepting outer truth;
3. evaluate each outer fold once, then aggregate the already evaluated folds.

No function in this module imports Torch, opens MLflow, reads audio, or writes
model state.  The only scorer used for adapted embeddings is the original-role
``heldout_reference_scores`` protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.group_bootstrap import paired_whole_group_cluster_bootstrap
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training.candidate_fusion import (
    ALPHAS,
    MARGIN_WEIGHTS,
    TIE_ORDER,
    UNKNOWN_WEIGHTS,
    select_inner_alpha,
    weighted_encoder_pair,
)
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import reference_probabilities


COMPARATORS = ("frozen_same_protocol", "fresh_control", "selected_arm")
ARM_IDS = ("control", "treatment_mse0", "treatment_mse01", "treatment_mse05")
POLICY_SCHEMA = "f005-heldout-inner-policies-v1"
EVALUATION_SCHEMA = "f005-one-shot-outer-evaluation-v1"


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
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_receipt(value: object) -> dict:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _write_new_json(path: Path, value: dict) -> bytes:
    """Create one immutable JSON receipt and refuse replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F005 refuses to replace sealed evidence: {path}")
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"F005 refuses to replace sealed evidence: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _read_regular_json(path: Path) -> tuple[dict, bytes]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "F005 seal must be a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F005 seal must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), "F005 seal must contain a JSON object")
    return value, payload


def _validate_scoring_config(config: dict) -> None:
    scoring = config.get("scoring", {})
    _require(
        scoring.get("protocol") == "heldout_reference_scores_original_disjoint_roles"
        and scoring.get("reference_method") == "max_reference"
        and scoring.get("alphas") == list(ALPHAS)
        and scoring.get("alpha_tie_order") == list(TIE_ORDER)
        and scoring.get("unknown_weights") == list(UNKNOWN_WEIGHTS)
        and scoring.get("margin_weights") == list(MARGIN_WEIGHTS)
        and scoring.get("threshold_candidates") == 201
        and type(scoring.get("threshold_candidates")) is int
        and scoring.get("probability_temperature") == 0.05
        and type(scoring.get("probability_temperature")) is float
        and scoring.get("no_outer_tuning") is True
        and scoring.get("no_exact_c002b_reproduction_claim") is True,
        "F005 scoring grid or original-heldout protocol changed",
    )
    arm_ids = tuple(row.get("id") for row in config.get("arms", []))
    _require(arm_ids == ARM_IDS, "F005 scoring requires the four preregistered arms")


def reload_arm_selection_seal(path: Path, contract: dict, outer: int) -> dict:
    """Reload and authenticate the known-only arm choice before unknown scoring."""
    seal, payload = _read_regular_json(path)
    required = {
        "schema_version", "experiment_signature", "outer_fold", "selected_arm",
        "scientific_conclusion", "arm_metrics", "selection_order",
        "known_calibration_indices_sha256", "known_query_rows",
        "observed_known_labels", "absent_known_labels",
        "unknown_calibration_indices_materialized",
        "unknown_reference_cohort_materialized", "unknown_similarity_computed",
        "outer_rows_or_labels_read", "refit_after_selection", "seal_sha256",
    }
    _require(set(seal) == required, "F005 arm-selection seal schema changed")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    tie_order = contract["config"].get("arm_selection", {}).get(
        "arm_tie_order", list(ARM_IDS)
    )
    metrics = seal.get("arm_metrics", {})
    _require(
        seal["schema_version"] == 1
        and seal["experiment_signature"] == contract.get("signature")
        and type(outer) is int and outer in contract["config"]["fold_ids"]
        and seal["outer_fold"] == outer
        and seal["selected_arm"] in ARM_IDS
        and tie_order == list(ARM_IDS)
        and seal["selection_order"] == [
            "macro_f1_observed_known_labels", "top1_accuracy", "fixed_arm_tie_order"
        ]
        and set(metrics) == set(ARM_IDS)
        and all(
            isinstance(metrics[arm], dict)
            and isinstance(metrics[arm].get("macro_f1_observed_known_labels"), (int, float))
            and isinstance(metrics[arm].get("top1_accuracy"), (int, float))
            and math.isfinite(float(metrics[arm]["macro_f1_observed_known_labels"]))
            and math.isfinite(float(metrics[arm]["top1_accuracy"]))
            and 0 <= metrics[arm]["macro_f1_observed_known_labels"] <= 1
            and 0 <= metrics[arm]["top1_accuracy"] <= 1
            for arm in ARM_IDS
        )
        and _is_sha256(seal["known_calibration_indices_sha256"])
        and seal["unknown_calibration_indices_materialized"] is False
        and seal["unknown_reference_cohort_materialized"] is False
        and seal["unknown_similarity_computed"] is False
        and seal["outer_rows_or_labels_read"] is False
        and seal["refit_after_selection"] is False
        and seal["seal_sha256"] == _sha(body),
        "F005 arm-selection seal identity or anti-leak claim is invalid",
    )
    winner = max(
        tie_order,
        key=lambda arm: (
            metrics[arm]["macro_f1_observed_known_labels"],
            metrics[arm]["top1_accuracy"],
            -tie_order.index(arm),
        ),
    )
    expected_conclusion = (
        "consistency_not_supported_on_known_calibration"
        if winner == "control" else "consistency_candidate_selected"
    )
    _require(
        seal["selected_arm"] == winner
        and seal["scientific_conclusion"] == expected_conclusion,
        "F005 sealed arm is not the recomputed known-query winner",
    )
    return {
        "kind": "f005_arm_selection_disk_reload",
        "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "seal_sha256": seal["seal_sha256"],
        "disk_reloaded": True,
        "seal": seal,
    }


def _verify_arm_reload(reload: dict, contract: dict, outer: int) -> dict:
    _require(
        isinstance(reload, dict)
        and reload.get("kind") == "f005_arm_selection_disk_reload"
        and reload.get("disk_reloaded") is True,
        "Full heldout scoring requires a disk-reloaded arm-selection seal",
    )
    current = reload_arm_selection_seal(Path(reload["path"]), contract, outer)
    _require(
        current["file_sha256"] == reload.get("file_sha256")
        and current["seal_sha256"] == reload.get("seal_sha256")
        and current["seal"] == reload.get("seal"),
        "Arm-selection seal changed after the caller reloaded it",
    )
    return current


class _OuterTruthForbidden(dict):
    """A manifest row that raises if pretruth code reads its class label."""

    def __getitem__(self, key):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before all inner policies were sealed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "speaker_id":
            raise RuntimeError("Outer truth was accessed before all inner policies were sealed")
        return super().get(key, default)


def _guard_outer_truth(contract: dict, outer: int) -> dict:
    roles = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    outer_names = {
        row["audio_file"] for row in roles if truth(row["outer_evaluation_included"])
    }
    _require(bool(outer_names), "F005 outer fold has no evaluation rows")
    guarded = dict(contract)
    guarded["manifest"] = [
        _OuterTruthForbidden(row) if row["audio_file"] in outer_names else row
        for row in contract["manifest"]
    ]
    guarded["roles"] = [
        _OuterTruthForbidden(row)
        if int(row["outer_fold"]) == outer and truth(row["outer_evaluation_included"])
        else row
        for row in contract["roles"]
    ]
    return guarded


def _validate_embedding_set(public, frozen, arms: dict, valid, row_count: int,
                            selected_arm: str) -> None:
    public, frozen, mask = np.asarray(public), np.asarray(frozen), np.asarray(valid)
    _require(
        public.shape == (row_count, 512) and frozen.shape == (row_count, 192)
        and public.dtype == np.float32 and frozen.dtype == np.float32
        and mask.shape == (row_count,) and mask.dtype == np.bool_
        and set(arms) == {"control", selected_arm},
        "F005 scoring embedding dimensions, validity, or arm inventory changed",
    )
    for name, values in {"public": public, "frozen": frozen, **arms}.items():
        array = np.asarray(values)
        expected = (row_count, 512 if name == "public" else 192)
        _require(
            array.shape == expected and array.dtype == np.float32
            and np.isfinite(array).all() and not np.any(array[~mask]),
            f"F005 {name} embeddings are not aligned finite float32 rows",
        )
        norms = np.linalg.norm(array[mask], axis=1)
        _require(
            np.allclose(norms, 1.0, atol=1e-5),
            f"F005 {name} valid embeddings must be unit vectors",
        )


def _score_candidates(contract: dict, outer: int, public, advanced, valid) -> dict:
    candidates = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            embeddings = np.asarray(public)
        elif alpha == 1.0:
            embeddings = np.asarray(advanced)
        else:
            embeddings = weighted_encoder_pair(public, advanced, valid, alpha)
        candidates[alpha] = heldout_reference_scores(contract, embeddings, valid, outer)
    anchor = candidates[0.0]
    for score in candidates.values():
        _require(
            np.array_equal(score["calibration_indices"], anchor["calibration_indices"])
            and np.array_equal(score["outer_indices"], anchor["outer_indices"])
            and np.array_equal(score["outer_valid"], anchor["outer_valid"])
            and score["known_labels"] == anchor["known_labels"],
            "F005 heldout candidate rows or known-label columns are misaligned",
        )
    return candidates


def _score_selected_alpha(contract: dict, outer: int, public, advanced, valid,
                          alpha: float) -> dict:
    _require(alpha in ALPHAS, "F005 sealed alpha is outside the preregistered grid")
    if alpha == 0.0:
        embeddings = np.asarray(public)
    elif alpha == 1.0:
        embeddings = np.asarray(advanced)
    else:
        embeddings = weighted_encoder_pair(public, advanced, valid, alpha)
    return heldout_reference_scores(contract, embeddings, valid, outer)


def _score_evidence(score: dict) -> dict:
    required = (
        "calibration_indices", "outer_indices", "inner_known_scores",
        "outer_known_scores", "inner_unknown_similarity",
        "outer_unknown_similarity", "outer_valid",
    )
    evidence = {key: _array_receipt(score[key]) for key in required}
    evidence["known_labels_sha256"] = _sha(score["known_labels"])
    evidence["protocol"] = score.get("provenance", {}).get("protocol")
    evidence["outer_fold"] = score.get("provenance", {}).get("outer_fold")
    evidence["receipt_sha256"] = _sha(evidence)
    return evidence


def _normalized_public_metadata(row: dict, group_id: object) -> dict:
    duration = row.get("duration_seconds")
    try:
        duration_value = float(duration)
    except (TypeError, ValueError) as error:
        raise ValueError("F005 duration metadata must be numeric") from error
    _require(
        math.isfinite(duration_value) and duration_value >= 0
        and isinstance(row.get("audio_file"), str) and bool(row["audio_file"])
        and isinstance(group_id, str) and bool(group_id)
        and "has_nonzero_signal" in row,
        "F005 outer public metadata are incomplete",
    )
    return {
        "audio_file": row["audio_file"],
        "group_id": group_id,
        "duration_seconds": duration_value,
        "has_nonzero_signal": truth(row["has_nonzero_signal"]),
    }


def _outer_public_metadata(contract: dict, outer: int,
                           outer_indices: np.ndarray) -> list[dict]:
    split = {row["audio_file"]: row for row in contract["folds"]}
    names = [row["audio_file"] for row in contract["manifest"]]
    _require(len(split) == len(names) and set(split) == set(names),
             "F005 split/manifest public metadata are misaligned")
    rows = []
    for value in outer_indices:
        row = contract["manifest"][int(value)]
        fold = split[row["audio_file"]]
        _require(int(fold["fold"]) == outer,
                 "F005 outer metadata contains a row from another fold")
        rows.append(_normalized_public_metadata(row, fold.get("group_id")))
    return rows


def _expected_outer_indices(contract: dict, outer: int) -> np.ndarray:
    positions = {
        row["audio_file"]: index for index, row in enumerate(contract["manifest"])
    }
    _require(len(positions) == len(contract["manifest"]),
             "F005 manifest filenames must be unique")
    roles = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    indices = np.asarray([
        positions[row["audio_file"]] for row in roles
        if truth(row["outer_evaluation_included"])
    ], dtype=np.int64)
    _require(len(indices) > 0 and len(indices) == len(set(indices.tolist())),
             "F005 outer role rows are empty or duplicated")
    return indices


def _fit_comparator_policy(contract: dict, candidates: dict) -> tuple[dict, dict]:
    labels, manifest = contract["labels"], contract["manifest"]
    query = candidates[0.0]["calibration_indices"]
    label_index = {label: index for index, label in enumerate(labels)}
    inner_truth = np.asarray(
        [label_index[manifest[int(index)]["speaker_id"]] for index in query],
        dtype=np.int64,
    )
    _require(
        np.any(inner_truth == 0) and np.any(inner_truth > 0),
        "F005 gate fitting requires disjoint known and unknown calibration rows",
    )
    selected, curves = select_inner_alpha(
        {
            alpha: {
                "known": score["inner_known_scores"],
                "unknown": score["inner_unknown_similarity"],
            }
            for alpha, score in candidates.items()
        },
        inner_truth,
        classes=len(labels),
    )
    alpha = selected["advanced_weight"]
    chosen = candidates[alpha]
    policy = {
        "advanced_weight": alpha,
        "calibration": selected["calibration"],
        "alpha_tie_order": selected["alpha_tie_order"],
        "selection_scope": "original_disjoint_group_excluded_calibration_queries_only",
        "calibration_indices_sha256": _array_receipt(query)["sha256"],
        "inner_truth_sha256": _array_receipt(inner_truth)["sha256"],
        "candidate_curves_sha256": _sha(curves),
        "score_evidence": _score_evidence(chosen),
    }
    return policy, chosen


def reload_policy_seal(path: Path, contract: dict, outer: int,
                       arm_reload: dict | None = None) -> dict:
    """Reload the three frozen inner policies and verify their complete identity."""
    seal, payload = _read_regular_json(path)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    expected_indices = _expected_outer_indices(contract, outer)
    expected_metadata = _outer_public_metadata(contract, outer, expected_indices)
    _require(
        seal.get("schema_version") == POLICY_SCHEMA
        and seal.get("experiment_signature") == contract.get("signature")
        and seal.get("outer_fold") == outer
        and seal.get("outer_truth_read") is False
        and seal.get("outer_truth_accepted_by_pretruth_api") is False
        and seal.get("unknown_scoring_started_after_arm_seal_reload") is True
        and seal.get("scoring_protocol") == "original_heldout_queries_expanded_gallery_v1"
        and set(seal.get("policies", {})) == set(COMPARATORS)
        and seal.get("labels_sha256") == _sha(contract["labels"])
        and seal.get("label_count") == len(contract["labels"])
        and seal.get("outer_public_metadata") == expected_metadata
        and seal.get("outer_public_metadata_sha256") == _sha(expected_metadata)
        and seal.get("seal_sha256") == _sha(body),
        "F005 policy seal identity, protocol, or anti-leak claim is invalid",
    )
    if arm_reload is not None:
        arm = _verify_arm_reload(arm_reload, contract, outer)
        _require(
            seal.get("arm_selection_seal_sha256") == arm["seal_sha256"]
            and seal.get("arm_selection_file_sha256") == arm["file_sha256"]
            and seal.get("selected_arm") == arm["seal"]["selected_arm"],
            "F005 policy seal refers to a different arm-selection seal",
        )
    return {
        "kind": "f005_policy_disk_reload",
        "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "seal_sha256": seal["seal_sha256"],
        "disk_reloaded": True,
        "seal": seal,
    }


def prepare_and_seal_inner_policies(
        contract: dict, outer: int, *, public_embeddings,
        frozen_advanced_embeddings, advanced_embeddings_by_arm: dict,
        valid, arm_selection_reload: dict, policy_seal_path: Path) -> dict:
    """Build three full heldout scorers only after known-arm seal reload.

    Outer truth is intentionally absent from the signature.  Outer manifest
    labels are replaced by raising sentinels before the shared scorer is called.
    Unknown calibration and unknown references first become visible here, after
    the selected arm is immutable on disk.
    """
    _validate_scoring_config(contract["config"])
    arm = _verify_arm_reload(arm_selection_reload, contract, outer)
    selected_arm = arm["seal"]["selected_arm"]
    _validate_embedding_set(
        public_embeddings, frozen_advanced_embeddings,
        advanced_embeddings_by_arm, valid, len(contract["manifest"]), selected_arm,
    )
    guarded = _guard_outer_truth(contract, outer)
    advanced = {
        "frozen_same_protocol": frozen_advanced_embeddings,
        "fresh_control": advanced_embeddings_by_arm["control"],
        "selected_arm": advanced_embeddings_by_arm[selected_arm],
    }
    policies, score_bundles = {}, {}
    for comparator in COMPARATORS:
        candidates = _score_candidates(
            guarded, outer, public_embeddings, advanced[comparator], valid,
        )
        policy, chosen = _fit_comparator_policy(guarded, candidates)
        policies[comparator] = {"comparator": comparator, **policy}
        score_bundles[comparator] = chosen
    anchor = score_bundles[COMPARATORS[0]]
    for score in score_bundles.values():
        _require(
            np.array_equal(score["calibration_indices"], anchor["calibration_indices"])
            and np.array_equal(score["outer_indices"], anchor["outer_indices"]),
            "F005 comparator query rows differ",
        )
    outer_indices = np.asarray(anchor["outer_indices"], dtype=np.int64)
    _require(
        np.array_equal(outer_indices, _expected_outer_indices(contract, outer)),
        "F005 scorer outer order differs from immutable roles",
    )
    outer_metadata = _outer_public_metadata(contract, outer, outer_indices)
    labels_sha256 = _sha(contract["labels"])
    body = {
        "schema_version": POLICY_SCHEMA,
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "selected_arm": selected_arm,
        "arm_selection_seal_sha256": arm["seal_sha256"],
        "arm_selection_file_sha256": arm["file_sha256"],
        "scoring_protocol": "original_heldout_queries_expanded_gallery_v1",
        "comparators": list(COMPARATORS),
        "policies": policies,
        "labels_sha256": labels_sha256,
        "label_count": len(contract["labels"]),
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "unknown_scoring_started_after_arm_seal_reload": True,
        "outer_truth_read": False,
        "outer_truth_accepted_by_pretruth_api": False,
        "policy_selection_complete_before_outer_truth": True,
    }
    seal = {**body, "seal_sha256": _sha(body)}
    _write_new_json(policy_seal_path, seal)
    reloaded = reload_policy_seal(
        policy_seal_path, contract, outer, arm_reload=arm,
    )
    return {
        "kind": "f005_pretruth_scoring_bundle",
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "selected_arm": selected_arm,
        "labels_sha256": labels_sha256,
        "outer_public_metadata": outer_metadata,
        "outer_public_metadata_sha256": _sha(outer_metadata),
        "outer_files": [row["audio_file"] for row in outer_metadata],
        "outer_indices_sha256": _array_receipt(outer_indices)["sha256"],
        "score_bundles": score_bundles,
        "policy_reload": reloaded,
        "outer_truth_read": False,
    }


def rebuild_pretruth_from_policy_seal(
        contract: dict, outer: int, *, public_embeddings,
        frozen_advanced_embeddings, advanced_embeddings_by_arm: dict, valid,
        arm_selection_reload: dict, policy_reload: dict) -> dict:
    """Recover score arrays after interruption without refitting or replacing seals.

    Only the alpha already fixed inside each disk-reloaded policy is scored.
    Rebuilt score evidence must match every sealed array hash before the bundle
    can reach the outer-evaluation API.
    """
    _validate_scoring_config(contract["config"])
    arm = _verify_arm_reload(arm_selection_reload, contract, outer)
    policies = reload_policy_seal(
        Path(policy_reload["path"]), contract, outer, arm_reload=arm,
    )
    _require(
        policies["file_sha256"] == policy_reload.get("file_sha256")
        and policies["seal_sha256"] == policy_reload.get("seal_sha256"),
        "F005 recovery received a stale policy-reload token",
    )
    selected_arm = arm["seal"]["selected_arm"]
    _validate_embedding_set(
        public_embeddings, frozen_advanced_embeddings,
        advanced_embeddings_by_arm, valid, len(contract["manifest"]), selected_arm,
    )
    guarded = _guard_outer_truth(contract, outer)
    advanced = {
        "frozen_same_protocol": frozen_advanced_embeddings,
        "fresh_control": advanced_embeddings_by_arm["control"],
        "selected_arm": advanced_embeddings_by_arm[selected_arm],
    }
    score_bundles = {}
    for comparator in COMPARATORS:
        policy = policies["seal"]["policies"][comparator]
        score = _score_selected_alpha(
            guarded, outer, public_embeddings, advanced[comparator], valid,
            policy["advanced_weight"],
        )
        _require(
            _score_evidence(score) == policy["score_evidence"],
            f"F005 recovered {comparator} scores differ from the sealed evidence",
        )
        score_bundles[comparator] = score
    anchor = score_bundles[COMPARATORS[0]]
    outer_indices = np.asarray(anchor["outer_indices"], dtype=np.int64)
    _require(
        all(
            np.array_equal(score["calibration_indices"], anchor["calibration_indices"])
            and np.array_equal(score["outer_indices"], outer_indices)
            for score in score_bundles.values()
        )
        and np.array_equal(outer_indices, _expected_outer_indices(contract, outer)),
        "F005 recovered comparator rows are misaligned",
    )
    metadata = _outer_public_metadata(contract, outer, outer_indices)
    return {
        "kind": "f005_pretruth_scoring_bundle",
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "selected_arm": selected_arm,
        "labels_sha256": _sha(contract["labels"]),
        "outer_public_metadata": metadata,
        "outer_public_metadata_sha256": _sha(metadata),
        "outer_files": [row["audio_file"] for row in metadata],
        "outer_indices_sha256": _array_receipt(outer_indices)["sha256"],
        "score_bundles": score_bundles,
        "policy_reload": policies,
        "outer_truth_read": False,
        "recovered_without_policy_refit": True,
    }


def _verify_policy_reload(reload: dict, pretruth: dict) -> dict:
    _require(
        isinstance(reload, dict) and reload.get("kind") == "f005_policy_disk_reload"
        and reload.get("disk_reloaded") is True,
        "Outer evaluation requires a disk-reloaded policy seal",
    )
    seal, payload = _read_regular_json(Path(reload["path"]))
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    _require(
        hashlib.sha256(payload).hexdigest() == reload.get("file_sha256")
        and seal.get("seal_sha256") == reload.get("seal_sha256") == _sha(body)
        and seal == reload.get("seal")
        and seal.get("experiment_signature") == pretruth.get("experiment_signature")
        and seal.get("outer_fold") == pretruth.get("outer_fold")
        and seal.get("selected_arm") == pretruth.get("selected_arm")
        and seal.get("outer_truth_read") is False,
        "F005 policy seal changed after disk reload",
    )
    return seal


def _duration_slices(reference: list[dict], predictions: list[dict],
                     labels: list[str]) -> dict:
    bins = (
        ("under_3s", lambda seconds: seconds < 3.0),
        ("3_to_5s", lambda seconds: 3.0 <= seconds < 5.0),
        ("5_to_8s", lambda seconds: 5.0 <= seconds < 8.0),
        ("8_to_30s", lambda seconds: 8.0 <= seconds < 30.0),
        ("at_least_30s", lambda seconds: seconds >= 30.0),
    )
    predicted = {row["audio_file"]: row["speaker_id"] for row in predictions}
    result = {}
    for name, predicate in bins:
        rows = [row for row in reference if predicate(float(row["duration_seconds"]))]
        if rows:
            metrics = score_predictions(
                rows,
                [{"audio_file": row["audio_file"], "speaker_id": predicted[row["audio_file"]]}
                 for row in rows],
                labels,
            )
            result[name] = {
                key: metrics[key] for key in ("row_count", "macro_f1", "accuracy", "errors")
            }
    return result


def evaluate_outer_once(pretruth: dict, policy_reload: dict,
                        outer_truth_rows: list[dict], labels: list[str], *,
                        evaluation_path: Path, probability_temperature: float = 0.05) -> dict:
    """Consume outer truth once, after all three policies are sealed and reloaded."""
    # Everything above this line is pretruth evidence.  Authenticate it before
    # iterating, indexing, or otherwise touching outer_truth_rows.
    _require(
        pretruth.get("kind") == "f005_pretruth_scoring_bundle"
        and pretruth.get("outer_truth_read") is False
        and set(pretruth.get("score_bundles", {})) == set(COMPARATORS),
        "F005 pretruth score bundle is malformed",
    )
    seal = _verify_policy_reload(policy_reload, pretruth)
    _require(
        probability_temperature == 0.05 and type(probability_temperature) is float,
        "F005 probability temperature changed",
    )
    for comparator in COMPARATORS:
        _require(
            _score_evidence(pretruth["score_bundles"][comparator])
            == seal["policies"][comparator]["score_evidence"],
            f"F005 {comparator} score evidence changed after policy sealing",
        )
    _require(
        isinstance(labels, list)
        and _sha(labels) == seal.get("labels_sha256")
        and _sha(labels) == pretruth.get("labels_sha256")
        and len(labels) == seal.get("label_count"),
        "F005 label order differs from the sealed policy",
    )
    sealed_metadata = seal.get("outer_public_metadata")
    _require(
        isinstance(sealed_metadata, list)
        and sealed_metadata == pretruth.get("outer_public_metadata")
        and _sha(sealed_metadata) == seal.get("outer_public_metadata_sha256")
        and _sha(sealed_metadata) == pretruth.get("outer_public_metadata_sha256"),
        "F005 sealed outer public metadata changed",
    )
    evaluation_path = Path(evaluation_path)
    if evaluation_path.exists() or evaluation_path.is_symlink():
        raise FileExistsError(
            f"F005 outer fold was already evaluated once: {evaluation_path}"
        )

    _require(isinstance(outer_truth_rows, list), "Outer truth rows must be a list")
    observed_metadata = []
    for row in outer_truth_rows:
        _require(isinstance(row, dict), "Every outer truth row must be an object")
        observed_metadata.append(_normalized_public_metadata(row, row.get("group_id")))
    _require(
        observed_metadata == sealed_metadata,
        "Outer public metadata or row order differs from the sealed fold",
    )
    # This is the first speaker_id access in the function.
    reference = []
    for row in outer_truth_rows:
        speaker = row.get("speaker_id")
        _require(speaker in labels, "Outer truth label is outside the sealed label map")
        reference.append(row)
    results = {}
    for comparator in COMPARATORS:
        score = pretruth["score_bundles"][comparator]
        policy = seal["policies"][comparator]
        probabilities = reference_probabilities(
            score["outer_known_scores"], score["outer_unknown_similarity"],
            policy["calibration"], score["outer_valid"], probability_temperature,
        )
        indices = probabilities.argmax(axis=1)
        predictions = [
            {"audio_file": row["audio_file"], "speaker_id": labels[int(index)]}
            for row, index in zip(reference, indices, strict=True)
        ]
        top_known = np.asarray(score["outer_known_scores"]).argmax(axis=1) + 1
        known_top1_predictions = [
            {
                "audio_file": row["audio_file"],
                "speaker_id": labels[int(index)] if bool(valid) else labels[0],
            }
            for row, index, valid in zip(
                reference, top_known, score["outer_valid"], strict=True,
            )
        ]
        metrics = score_predictions(reference, predictions, labels)
        results[comparator] = {
            "policy": policy,
            "predictions": predictions,
            "known_top1_predictions": known_top1_predictions,
            "metrics": metrics,
            "duration_slices": _duration_slices(reference, predictions, labels),
        }
    body = {
        "schema_version": EVALUATION_SCHEMA,
        "experiment_signature": pretruth["experiment_signature"],
        "outer_fold": pretruth["outer_fold"],
        "selected_arm": pretruth["selected_arm"],
        "policy_seal_sha256": seal["seal_sha256"],
        "policy_file_sha256": policy_reload["file_sha256"],
        "outer_reference": reference,
        "comparators": results,
        "one_shot_outer_evaluation": True,
        "outer_truth_first_access_stage": "after_all_three_policy_seals_reloaded",
    }
    result = {**body, "evaluation_sha256": _sha(body)}
    _write_new_json(evaluation_path, result)
    reloaded, payload = _read_regular_json(evaluation_path)
    _require(
        reloaded == result and hashlib.sha256(payload).hexdigest()
        == hashlib.sha256(Path(evaluation_path).read_bytes()).hexdigest(),
        "F005 outer evaluation did not round-trip exactly",
    )
    return result


def _index_prediction_rows(rows: list[dict], expected: set[str], labels: list[str],
                           name: str) -> dict[str, str]:
    _require(isinstance(rows, list), f"{name} predictions must be a list")
    indexed = {}
    for row in rows:
        _require(
            isinstance(row, dict) and isinstance(row.get("audio_file"), str)
            and row["audio_file"] not in indexed and row.get("speaker_id") in labels,
            f"{name} prediction identity or label is invalid",
        )
        indexed[row["audio_file"]] = row["speaker_id"]
    _require(set(indexed) == expected, f"{name} predictions are not exactly aligned")
    return indexed


def _short_known_top1(reference: list[dict], top1: dict[str, str],
                      cutoff_seconds: float) -> dict:
    selected = [
        row for row in reference
        if row["speaker_id"] != "unknown" and truth(row["has_nonzero_signal"])
        and float(row["duration_seconds"]) < cutoff_seconds
    ]
    _require(bool(selected), "F005 short-known diagnostic population is empty")
    correct = sum(top1[row["audio_file"]] == row["speaker_id"] for row in selected)
    return {
        "definition": f"nonzero known duration < {cutoff_seconds:g}s; known argmax before rejection",
        "rows": len(selected), "correct": correct, "accuracy": correct / len(selected),
    }


def _promotion_config(config: dict) -> tuple[dict, dict]:
    bootstrap, promotion = config.get("bootstrap", {}), config.get("promotion", {})
    _require(
        bootstrap.get("kind") == "paired_whole_content_group_unstratified"
        and bootstrap.get("mixed_label_groups_preserved") is True
        and bootstrap.get("true_class_purity_assumed") is False
        and type(bootstrap.get("seed")) is int
        and type(bootstrap.get("replicates")) is int
        and bootstrap["replicates"] >= 100
        and 0 < bootstrap.get("lower_quantile", 0) < bootstrap.get("upper_quantile", 1) < 1,
        "F005 whole-group bootstrap configuration changed",
    )
    required = {
        "minimum_treatment_delta_vs_fresh_control",
        "minimum_selected_delta_vs_c002b",
        "minimum_control_delta_vs_c002b_when_selected",
        "minimum_accuracy_delta_vs_c002b",
        "minimum_short_known_top1_delta_vs_c002b",
        "minimum_each_fold_delta_vs_control", "minimum_each_fold_delta_vs_c002b",
        "maximum_unknown_to_known_increase_vs_c002b",
        "maximum_known_to_other_known_increase_vs_c002b",
        "minimum_group_bootstrap_lower_bound_vs_control",
        "minimum_group_bootstrap_lower_bound_vs_c002b",
        "require_exact_cpu_cuda_prediction_parity", "all_conditions_required",
    }
    _require(required.issubset(promotion), "F005 promotion configuration is incomplete")
    _require(
        promotion["minimum_treatment_delta_vs_fresh_control"] == 0.003
        and promotion["minimum_selected_delta_vs_c002b"] == 0.003
        and promotion["minimum_control_delta_vs_c002b_when_selected"] == 0.003
        and promotion["minimum_accuracy_delta_vs_c002b"] == 0.0
        and promotion["minimum_short_known_top1_delta_vs_c002b"] == 0.0
        and promotion["minimum_each_fold_delta_vs_control"] == -0.001
        and promotion["minimum_each_fold_delta_vs_c002b"] == -0.001
        and promotion["maximum_unknown_to_known_increase_vs_c002b"] == 2
        and promotion["maximum_known_to_other_known_increase_vs_c002b"] == 0
        and promotion["minimum_group_bootstrap_lower_bound_vs_control"] == -0.0005
        and promotion["minimum_group_bootstrap_lower_bound_vs_c002b"] == -0.0005
        and promotion["require_exact_cpu_cuda_prediction_parity"] is True
        and promotion["all_conditions_required"] is True,
        "F005 strict promotion thresholds changed",
    )
    goal_target = promotion.get(
        "goal_target_oof_macro_f1", promotion.get("target_oof_macro_f1")
    )
    _require(
        goal_target == 0.965
        and promotion.get("incumbent_promotion_is_distinct_from_goal_completion") is True
        and promotion.get("otherwise") == "retain_c002b",
        "F005 development goal or incumbent-retention policy changed",
    )
    return bootstrap, promotion


def aggregate_oof_and_decide(contract: dict, fold_results: list[dict], *,
                             historical_c002b_predictions: list[dict],
                             historical_c002b_known_top1_predictions: list[dict],
                             exact_cpu_cuda_prediction_parity: bool) -> dict:
    """Pool sealed one-shot folds and apply every conditional promotion guard."""
    config, labels, manifest = contract["config"], contract["labels"], contract["manifest"]
    bootstrap_config, rule = _promotion_config(config)
    folds = {int(row["outer_fold"]): row for row in fold_results}
    _require(
        len(folds) == len(fold_results) and set(folds) == set(config["fold_ids"]),
        "F005 OOF aggregation requires one result for every configured outer fold",
    )
    for outer, result in folds.items():
        body = {key: value for key, value in result.items() if key != "evaluation_sha256"}
        _require(
            result.get("schema_version") == EVALUATION_SCHEMA
            and result.get("experiment_signature") == contract["signature"]
            and result.get("one_shot_outer_evaluation") is True
            and result.get("evaluation_sha256") == _sha(body)
            and set(result.get("comparators", {})) == set(COMPARATORS)
            and result.get("selected_arm") in ARM_IDS,
            f"F005 fold {outer} evaluation receipt is invalid",
        )
    expected_names = {row["audio_file"] for row in manifest}
    _require(len(expected_names) == len(manifest), "F005 manifest contains duplicate filenames")
    references_by_name = {row["audio_file"]: row for row in manifest}
    split = {row["audio_file"]: row for row in contract["folds"]}
    _require(set(split) == expected_names, "F005 split rows are not aligned with the manifest")
    evaluation_reference = {}
    for outer, result in folds.items():
        for row in result["outer_reference"]:
            name = row["audio_file"]
            source = references_by_name.get(name, {})
            public_metadata_matches = False
            if name in split:
                expected_metadata = _normalized_public_metadata(
                    source, split[name].get("group_id")
                )
                observed_metadata = _normalized_public_metadata(
                    row, row.get("group_id")
                )
                public_metadata_matches = (
                    type(split[name].get("fold")) in (int, str)
                    and int(split[name]["fold"]) == outer
                    and observed_metadata == expected_metadata
                )
            _require(
                name not in evaluation_reference and name in references_by_name
                and row["speaker_id"] == source["speaker_id"]
                and public_metadata_matches,
                "F005 outer truth or public metadata overlap/differ from the contract",
            )
            evaluation_reference[name] = row
    _require(set(evaluation_reference) == expected_names, "F005 fold union is not complete OOF")
    reference = [evaluation_reference[row["audio_file"]] for row in manifest]

    prediction_maps, top1_maps = {}, {}
    for comparator in COMPARATORS:
        rows = [item for result in folds.values()
                for item in result["comparators"][comparator]["predictions"]]
        top1 = [item for result in folds.values()
                for item in result["comparators"][comparator]["known_top1_predictions"]]
        prediction_maps[comparator] = _index_prediction_rows(
            rows, expected_names, labels, comparator,
        )
        top1_maps[comparator] = _index_prediction_rows(
            top1, expected_names, labels, comparator + " known-top1",
        )
    for outer, result in folds.items():
        if result["selected_arm"] == "control":
            selected_fold = result["comparators"]["selected_arm"]
            control_fold = result["comparators"]["fresh_control"]
            _require(
                selected_fold["predictions"] == control_fold["predictions"]
                and selected_fold["known_top1_predictions"]
                == control_fold["known_top1_predictions"],
                f"F005 fold {outer} selected control differs from its candidate evidence",
            )
    prediction_maps["historical_c002b"] = _index_prediction_rows(
        historical_c002b_predictions, expected_names, labels, "historical C002b",
    )
    top1_maps["historical_c002b"] = _index_prediction_rows(
        historical_c002b_known_top1_predictions, expected_names, labels,
        "historical C002b known-top1",
    )
    predictions = {
        name: [{"audio_file": row["audio_file"], "speaker_id": mapping[row["audio_file"]]}
               for row in reference]
        for name, mapping in prediction_maps.items()
    }
    metrics = {
        name: score_predictions(reference, rows, labels)
        for name, rows in predictions.items()
    }
    fold_metrics = {}
    for outer, result in folds.items():
        fold_reference = result["outer_reference"]
        fold_metrics[str(outer)] = {
            name: score_predictions(
                fold_reference,
                [{"audio_file": row["audio_file"], "speaker_id": mapping[row["audio_file"]]}
                 for row in fold_reference],
                labels,
            )
            for name, mapping in prediction_maps.items()
        }

    label_index = {label: index for index, label in enumerate(labels)}
    truth_indices = np.asarray([label_index[row["speaker_id"]] for row in reference], dtype=np.int64)
    prediction_indices = {
        name: np.asarray([label_index[row["speaker_id"]] for row in rows], dtype=np.int64)
        for name, rows in predictions.items()
    }
    groups = np.asarray([split[row["audio_file"]]["group_id"] for row in reference], dtype=object)
    bootstrap_arguments = {
        "class_count": len(labels), "seed": bootstrap_config["seed"],
        "replicates": bootstrap_config["replicates"],
        "quantiles": (bootstrap_config["lower_quantile"], bootstrap_config["upper_quantile"]),
    }
    bootstraps = {
        "selected_vs_fresh_control": paired_whole_group_cluster_bootstrap(
            truth_indices, prediction_indices["fresh_control"],
            prediction_indices["selected_arm"], groups, **bootstrap_arguments,
        )["receipt"],
        "selected_vs_c002b": paired_whole_group_cluster_bootstrap(
            truth_indices, prediction_indices["historical_c002b"],
            prediction_indices["selected_arm"], groups, **bootstrap_arguments,
        )["receipt"],
    }
    selected_arms = {str(outer): folds[outer]["selected_arm"] for outer in sorted(folds)}
    selection_kind = (
        "control" if set(selected_arms.values()) == {"control"} else "treatment"
    )
    if selection_kind == "control":
        _require(
            prediction_maps["selected_arm"] == prediction_maps["fresh_control"],
            "A selected-control procedure must exactly equal fresh-control predictions",
        )
    selected, control, c002b = (
        metrics["selected_arm"], metrics["fresh_control"], metrics["historical_c002b"]
    )
    delta_control = selected["macro_f1"] - control["macro_f1"]
    delta_c002b = selected["macro_f1"] - c002b["macro_f1"]
    fold_delta_control = [
        fold_metrics[str(outer)]["selected_arm"]["macro_f1"]
        - fold_metrics[str(outer)]["fresh_control"]["macro_f1"]
        for outer in sorted(folds)
    ]
    fold_delta_c002b = [
        fold_metrics[str(outer)]["selected_arm"]["macro_f1"]
        - fold_metrics[str(outer)]["historical_c002b"]["macro_f1"]
        for outer in sorted(folds)
    ]
    cutoff = float(config["views"]["long_seconds"])
    short_selected = _short_known_top1(reference, top1_maps["selected_arm"], cutoff)
    short_c002b = _short_known_top1(reference, top1_maps["historical_c002b"], cutoff)
    conditional_gain = (
        delta_c002b >= rule["minimum_control_delta_vs_c002b_when_selected"]
        if selection_kind == "control" else
        delta_control >= rule["minimum_treatment_delta_vs_fresh_control"]
        and delta_c002b >= rule["minimum_selected_delta_vs_c002b"]
    )
    conditions = {
        "conditional_pooled_macro_f1_gain": conditional_gain,
        "pooled_accuracy_nonnegative_vs_c002b": (
            selected["accuracy"] - c002b["accuracy"]
            >= rule["minimum_accuracy_delta_vs_c002b"]
        ),
        "each_fold_vs_fresh_control": min(fold_delta_control)
        >= rule["minimum_each_fold_delta_vs_control"],
        "each_fold_vs_c002b": min(fold_delta_c002b)
        >= rule["minimum_each_fold_delta_vs_c002b"],
        "known_to_other_known_guard": (
            selected["errors"]["known_to_other_known"]
            - c002b["errors"]["known_to_other_known"]
            <= rule["maximum_known_to_other_known_increase_vs_c002b"]
        ),
        "unknown_to_known_guard": (
            selected["errors"]["unknown_to_known"]
            - c002b["errors"]["unknown_to_known"]
            <= rule["maximum_unknown_to_known_increase_vs_c002b"]
        ),
        "short_known_top1_nonnegative_vs_c002b": (
            short_selected["accuracy"] - short_c002b["accuracy"]
            >= rule["minimum_short_known_top1_delta_vs_c002b"]
        ),
        "whole_group_bootstrap_vs_control": (
            bootstraps["selected_vs_fresh_control"]["confidence_interval"]
            ["lower_delta_macro_f1"]
            >= rule["minimum_group_bootstrap_lower_bound_vs_control"]
        ),
        "whole_group_bootstrap_vs_c002b": (
            bootstraps["selected_vs_c002b"]["confidence_interval"]
            ["lower_delta_macro_f1"]
            >= rule["minimum_group_bootstrap_lower_bound_vs_c002b"]
        ),
        "exact_cpu_cuda_prediction_parity": exact_cpu_cuda_prediction_parity is True,
    }
    promoted = all(conditions.values())
    goal_target = rule.get(
        "goal_target_oof_macro_f1", rule.get("target_oof_macro_f1")
    )
    _require(
        isinstance(goal_target, (int, float)) and not isinstance(goal_target, bool)
        and math.isfinite(float(goal_target)) and 0 <= float(goal_target) <= 1,
        "F005 goal OOF Macro-F1 is missing or invalid",
    )
    metric_goal_conditions = {
        "oof_macro_f1_at_least_target": selected["macro_f1"] >= float(goal_target),
        "whole_group_bootstrap_lower_bound_vs_c002b_positive": (
            bootstraps["selected_vs_c002b"]["confidence_interval"]
            ["lower_delta_macro_f1"] > 0.0
        ),
        "both_outer_folds_positive_vs_c002b": min(fold_delta_c002b) > 0.0,
    }
    development_metric_goal_reached = promoted and all(metric_goal_conditions.values())
    return {
        "schema_version": "f005-oof-promotion-decision-v1",
        "experiment_signature": contract["signature"],
        "selected_arms_by_outer": selected_arms,
        "selection_kind": selection_kind,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "duration_slices": {
            name: _duration_slices(reference, rows, labels)
            for name, rows in predictions.items()
        },
        "short_known_top1": {
            "selected_arm": short_selected, "historical_c002b": short_c002b,
            "delta": short_selected["accuracy"] - short_c002b["accuracy"],
        },
        "deltas": {
            "selected_vs_fresh_control_macro_f1": delta_control,
            "selected_vs_c002b_macro_f1": delta_c002b,
            "selected_vs_c002b_accuracy": selected["accuracy"] - c002b["accuracy"],
            "fold_vs_fresh_control": fold_delta_control,
            "fold_vs_c002b": fold_delta_c002b,
        },
        "bootstraps": bootstraps,
        "goal_target_oof_macro_f1": float(goal_target),
        "metric_goal_conditions": metric_goal_conditions,
        "development_metric_goal_reached": development_metric_goal_reached,
        "goal_reached": False,
        "goal_status": "pending_clean_reproduction_and_offline_package_qa",
        "goal_outstanding": ["clean_reproduction", "offline_package_qa"],
        "conditions": conditions,
        "all_conditions_passed": promoted,
        "promote_new_incumbent": promoted,
        "decision": "promote_selected_f005" if promoted else "retain_c002b",
        "rule": rule,
        "historical_comparator_note": (
            "C002b is the immutable actual-best comparator; frozen_same_protocol is "
            "reported separately and is not claimed to reproduce C002b."
        ),
    }
