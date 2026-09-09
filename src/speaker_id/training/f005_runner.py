"""Pure planning, selection and sealing primitives for F005.

No function in this module imports Torch, opens MLflow, reads audio, or reads an
original outer label.  The CUDA worker consumes these deterministic receipts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.models.campp import file_sha256
from speaker_id.tracking.snapshot import write_json
from speaker_id.training.f005_contract import (
    ADVANCED_DIMENSION, ADVANCED_WEIGHTS_SHA256, ARM_IDS, TREATMENT_IDS, canonical,
)
from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_total_steps


def _index_hash(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _seed(*values) -> int:
    return int.from_bytes(hashlib.sha256(canonical(values)).digest()[:8], "little")


def fit_rows(contract: dict, outer: int) -> list[dict]:
    """Return the exact known encoder-fit population in stable role order."""
    config, labels = contract["config"], contract["labels"]
    if type(outer) is not int or outer not in config["fold_ids"]:
        raise ValueError("F005 outer fold is not configured")
    known = set(labels[1:])
    rows = [
        row for row in contract["roles"]
        if int(row["outer_fold"]) == outer
        and truth(row["encoder_fit_allowed"])
        and row["speaker_id"] in known
    ]
    names = [row["audio_file"] for row in rows]
    if (not rows or len(names) != len(set(names))
            or set(row["speaker_id"] for row in rows) != known):
        raise ValueError("F005 requires every known speaker and only permitted fit rows")
    return rows


def pairing_identity(contract: dict, outer: int) -> dict:
    """Identity shared by all four arms; arm coefficients are deliberately absent."""
    rows = fit_rows(contract, outer)
    config = contract["config"]
    body = {
        "schema_version": 1,
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "scope": "all_known_encoder_fit_allowed_rows_once",
        "fit_rows": [{key: row[key] for key in ("audio_file", "speaker_id", "group_id")} for row in rows],
        "fit_audio_order_sha256": hashlib.sha256(canonical([row["audio_file"] for row in rows])).hexdigest(),
        "views": config["views"],
        "fit": config["fit"],
        "seed": config["seed"],
        "base_model": {"embedding_dim": ADVANCED_DIMENSION, "weights_sha256": ADVANCED_WEIGHTS_SHA256},
        "sampling": "step_counter_seeded_uniform_speaker_then_uniform_permitted_file_v1",
        "head_initialization": "same_seed_after_verified_advanced_weight_load_for_every_arm",
    }
    return {**body, "signature": hashlib.sha256(canonical(body)).hexdigest()}


def training_step_plan(contract: dict, outer: int, step: int) -> list[dict]:
    """Build one arm-independent effective batch and nested-crop seed plan."""
    config, labels = contract["config"], contract["labels"]
    total = adaptation_total_steps(config["fit"])
    if type(step) is not int or not 0 <= step < total:
        raise ValueError("F005 plan step is outside the fixed schedule")
    rows = fit_rows(contract, outer)
    by_label = {label: [] for label in labels[1:]}
    for row in rows:
        by_label[row["speaker_id"]].append(row)
    if not all(by_label.values()):
        raise ValueError("F005 batch planning lost a known identity")
    count = config["fit"]["batch_pairs"]
    rng = np.random.default_rng(_seed(config["seed"], "F005-batch", outer, step))
    selected = rng.choice(labels[1:], count, replace=count > len(labels) - 1)
    targets = {label: index for index, label in enumerate(labels[1:])}
    plan = []
    for slot, value in enumerate(selected):
        label = str(value)
        choices = by_label[label]
        row = choices[int(rng.integers(0, len(choices)))]
        plan.append({
            "slot": slot,
            "audio_file": row["audio_file"],
            "speaker_id": label,
            "target": targets[label],
            "group_id": row["group_id"],
            "crop_seed": int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)),
        })
    return plan


def plan_range_sha256(contract: dict, outer: int, start: int, stop: int) -> str:
    total = adaptation_total_steps(contract["config"]["fit"])
    if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= total):
        raise ValueError("F005 plan range is invalid")
    digest = hashlib.sha256()
    for step in range(start, stop):
        digest.update(canonical({"step": step, "batch": training_step_plan(contract, outer, step)}))
    return digest.hexdigest()


def complete_plan_sha256(contract: dict, outer: int) -> str:
    return plan_range_sha256(contract, outer, 0, adaptation_total_steps(contract["config"]["fit"]))


def shared_head_identity(contract: dict, outer: int, *, plan_sha256: str | None = None) -> dict:
    fit = contract["config"]["fit"]
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    pairing = pairing_identity(contract, outer)
    plan_hash = plan_sha256 or plan_range_sha256(contract, outer, 0, head_steps)
    body = {
        "schema_version": 1, "stage": "shared_head", "experiment_signature": contract["signature"],
        "outer_fold": outer, "pairing_signature": pairing["signature"],
        "head_plan_sha256": plan_hash, "completed_steps": head_steps,
        "objective": "dual_short_long_aam_no_consistency",
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256, "embedding_dim": ADVANCED_DIMENSION,
        "fork_policy": "one_checkpoint_byte_identical_input_to_all_four_tail_arms",
        "checkpoint_scope": "server_only_until_promotion",
    }
    return {**body, "signature": hashlib.sha256(canonical(body)).hexdigest()}


def arm_identity(contract: dict, outer: int, arm_id: str, *,
                 shared_head_checkpoint_sha256: str, tail_plan_sha256: str | None = None) -> dict:
    """Bind an arm checkpoint while retaining the shared pairing signature."""
    arms = {arm["id"]: arm for arm in contract["config"]["arms"]}
    if arm_id not in arms or tuple(arms) != ARM_IDS:
        raise ValueError("Unknown or reordered F005 arm")
    pairing = pairing_identity(contract, outer)
    head_steps = contract["config"]["fit"]["adaptation_schedule"]["head_only_steps"]
    plan_hash = tail_plan_sha256 or plan_range_sha256(
        contract, outer, head_steps, adaptation_total_steps(contract["config"]["fit"])
    )
    for name, value in (("tail plan", plan_hash), ("shared head checkpoint", shared_head_checkpoint_sha256)):
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"F005 {name} requires a lowercase SHA256 identity")
    body = {
        "schema_version": 1,
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "arm": arms[arm_id],
        "pairing_signature": pairing["signature"],
        "tail_plan_sha256": plan_hash,
        "shared_head_checkpoint_sha256": shared_head_checkpoint_sha256,
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256,
        "embedding_dim": ADVANCED_DIMENSION,
        "checkpoint_scope": "server_only_until_promotion",
    }
    return {**body, "signature": hashlib.sha256(canonical(body)).hexdigest()}


def execution_plan(contract: dict, *, include_plan_hashes: bool = False) -> dict:
    """Return two shared-head and eight forked-tail units; no refit exists."""
    units = []
    for outer in contract["config"]["fold_ids"]:
        head_steps = contract["config"]["fit"]["adaptation_schedule"]["head_only_steps"]
        head_plan = plan_range_sha256(contract, outer, 0, head_steps) if include_plan_hashes else None
        tail_plan = (plan_range_sha256(contract, outer, head_steps,
                                      adaptation_total_steps(contract["config"]["fit"]))
                     if include_plan_hashes else None)
        pairing = pairing_identity(contract, outer)
        head = shared_head_identity(contract, outer, plan_sha256=head_plan) if head_plan else None
        units.append({"outer_fold": outer, "stage": "shared_head", "arm_id": None,
                      "pairing_signature": pairing["signature"], "plan_sha256": head_plan,
                      "unit_signature": None if head is None else head["signature"]})
        for arm_id in ARM_IDS:
            units.append({
                "outer_fold": outer, "stage": "tail", "arm_id": arm_id, "scope": "all_fit",
                "pairing_signature": pairing["signature"],
                "plan_sha256": tail_plan, "shared_head_checkpoint_sha256": "pending_head_completion",
                "unit_signature": None,
            })
    return {
        "schema_version": 1,
        "experiment_signature": contract["signature"],
        "units": units,
        "unit_count": len(units), "shared_head_units": len(contract["config"]["fold_ids"]),
        "tail_arm_units": len(contract["config"]["fold_ids"]) * len(ARM_IDS),
        "refit_after_selection": False,
        "outer_evaluation_units": 0,
        "outer_truth_read": False,
    }


def shared_head_checkpoint_metadata(identity: dict, fit: dict, completed_steps: int | None = None) -> dict:
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    completed_steps = head_steps if completed_steps is None else completed_steps
    if (identity.get("stage") != "shared_head" or identity.get("completed_steps") != head_steps
            or identity.get("embedding_dim") != ADVANCED_DIMENSION
            or type(completed_steps) is not int or not 0 <= completed_steps <= head_steps):
        raise ValueError("F005 shared head identity is malformed")
    body = {
        "format_version": 1, "stage": "shared_head",
        "experiment_signature": identity["experiment_signature"],
        "unit_signature": identity["signature"], "pairing_signature": identity["pairing_signature"],
        "head_plan_sha256": identity["head_plan_sha256"], "outer_fold": identity["outer_fold"],
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256, "embedding_dim": ADVANCED_DIMENSION,
        "completed_steps": completed_steps, "total_steps": adaptation_total_steps(fit),
        "schedule_state": adaptation_checkpoint_state(fit, completed_steps),
        "tail_fork_arms": list(ARM_IDS), "byte_identical_fork_source": completed_steps == head_steps,
        "mlflow_upload_allowed": False, "local_transfer_allowed": False,
    }
    return {**body, "metadata_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def checkpoint_metadata(identity: dict, fit: dict, completed_steps: int) -> dict:
    """Metadata stored beside tensor state and checked before any resume load."""
    total = adaptation_total_steps(fit)
    if type(completed_steps) is not int or not 0 <= completed_steps <= total:
        raise ValueError("F005 checkpoint completed step is outside its recipe")
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    if (identity.get("embedding_dim") != ADVANCED_DIMENSION or identity.get("advanced_weights_sha256") != ADVANCED_WEIGHTS_SHA256
            or type(completed_steps) is not int or not head_steps <= completed_steps <= total):
        raise ValueError("F005 checkpoint identity is not advanced192")
    body = {
        "format_version": 1,
        "experiment_signature": identity["experiment_signature"],
        "stage": "tail", "arm_signature": identity["signature"],
        "pairing_signature": identity["pairing_signature"],
        "tail_plan_sha256": identity["tail_plan_sha256"],
        "shared_head_checkpoint_sha256": identity["shared_head_checkpoint_sha256"],
        "outer_fold": identity["outer_fold"],
        "arm_id": identity["arm"]["id"],
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256,
        "embedding_dim": ADVANCED_DIMENSION,
        "completed_steps": completed_steps,
        "total_steps": total,
        "schedule_state": adaptation_checkpoint_state(fit, completed_steps),
        "next_batch_is_counter_derived": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    return {**body, "metadata_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def validate_resume_payload(payload: dict, expected_identity: dict, fit: dict) -> dict:
    """Reject cross-arm/fold/plan/model resumes before loading tensor state."""
    required = {"metadata", "encoder", "head", "optimizer", "torch_rng", "cuda_rng"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("F005 checkpoint payload fields differ from the fixed format")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or type(metadata.get("completed_steps")) is not int:
        raise ValueError("F005 checkpoint metadata is incomplete")
    expected = checkpoint_metadata(expected_identity, fit, metadata["completed_steps"])
    if metadata != expected:
        raise ValueError("F005 resume checkpoint belongs to another arm, fold, plan, model or step")
    return metadata


def validate_shared_head_payload(payload: dict, expected_identity: dict, fit: dict) -> dict:
    required = {"metadata", "encoder", "head", "optimizer", "torch_rng", "cuda_rng"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("F005 shared-head checkpoint fields differ from the fixed format")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or type(metadata.get("completed_steps")) is not int:
        raise ValueError("F005 shared-head checkpoint metadata is incomplete")
    if metadata != shared_head_checkpoint_metadata(expected_identity, fit, metadata["completed_steps"]):
        raise ValueError("F005 tail fork source is not the exact shared-head checkpoint")
    return metadata


def _known_query_metrics(truth_indices: np.ndarray, guess_indices: np.ndarray) -> dict:
    observed = sorted(set(int(value) for value in truth_indices))
    support = Counter(int(value) for value in truth_indices)
    assigned = Counter(int(value) for value in guess_indices)
    correct = Counter(int(t) for t, g in zip(truth_indices, guess_indices) if t == g)
    f1 = [2 * correct[label] / (support[label] + assigned[label])
          if support[label] + assigned[label] else 0.0 for label in observed]
    return {
        "row_count": len(truth_indices),
        "observed_known_labels": len(observed),
        "macro_f1_observed_known_labels": float(np.mean(f1)),
        "top1_accuracy": float(np.mean(truth_indices == guess_indices)),
        "errors": int(np.sum(truth_indices != guess_indices)),
    }


def known_selection_scores(embeddings, valid, contract: dict, outer: int) -> dict:
    """Build only known-query/max-known-reference evidence before arm sealing.

    This mirrors the known side of ``heldout_reference_scores``: all eligible
    outer-training known references are available and each query's complete
    content group is removed before every class maximum.  It deliberately never
    creates unknown query indices, an unknown cohort, or unknown similarities.
    """
    values, mask = np.asarray(embeddings), np.asarray(valid)
    manifest, folds, labels = contract["manifest"], contract["folds"], contract["labels"]
    if (values.shape != (len(manifest), ADVANCED_DIMENSION) or values.dtype != np.float32
            or mask.shape != (len(manifest),) or mask.dtype != np.bool_
            or not np.isfinite(values).all()):
        raise ValueError("F005 known selector requires aligned finite advanced192 arrays")
    norms = np.linalg.norm(values[mask], axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("F005 known selector requires unit valid embeddings")
    if np.any(values[~mask]):
        raise ValueError("F005 invalid embeddings must retain exact zero fallback")
    names = [row["audio_file"] for row in manifest]
    by_name = {name: index for index, name in enumerate(names)}
    split = {row["audio_file"]: row for row in folds}
    roles_list = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    roles = {row["audio_file"]: row for row in roles_list}
    if (len(by_name) != len(names) or len(split) != len(folds) or len(roles) != len(roles_list)
            or set(by_name) != set(split) or set(by_name) != set(roles)):
        raise ValueError("F005 known selector manifest/fold/role alignment changed")
    known = set(labels[1:])
    group_assignments, fit_groups = defaultdict(set), set()
    groups, assigned, eligible = [], [], []
    for index, name in enumerate(names):
        row, role = split[name], roles[name]
        group, fold = row["group_id"], int(row["fold"])
        if not isinstance(group, str) or not group or role["group_id"] != group:
            raise ValueError("F005 known selector requires aligned nonempty content groups")
        fitting, enrolling, querying, evaluating = [truth(role[key]) for key in
            ("encoder_fit_allowed", "enrollment_allowed", "calibration_query", "outer_evaluation_included")]
        is_eligible = truth(row["train_eligible"])
        if (evaluating != (fold == outer) or (evaluating and (fitting or enrolling or querying))
                or (querying and (fitting or enrolling))
                or ((fitting or enrolling or querying) and (not is_eligible or not mask[index]))):
            raise ValueError("F005 known selector detected heldout role leakage or invalid support")
        group_assignments[group].add((fold, fitting, enrolling, querying, evaluating))
        if fitting: fit_groups.add(group)
        groups.append(group); assigned.append(fold); eligible.append(is_eligible)
    if any(len(values) != 1 for values in group_assignments.values()):
        raise ValueError("F005 known selector found a content group crossing heldout roles")
    groups, assigned, eligible = np.asarray(groups, dtype=object), np.asarray(assigned), np.asarray(eligible)
    all_references = np.flatnonzero((assigned != outer) & eligible & mask)
    query_indices = np.asarray([
        by_name[row["audio_file"]] for row in roles_list
        if truth(row["calibration_query"]) and row["speaker_id"] in known
    ], dtype=np.int64)
    if not len(all_references) or not len(query_indices) or not set(query_indices).issubset(set(all_references)):
        raise ValueError("F005 known selector has no permitted references or known queries")
    query_groups, outer_groups = set(groups[query_indices]), set(groups[assigned == outer])
    if query_groups & fit_groups or set(groups[all_references]) & outer_groups:
        raise ValueError("F005 known selector query/fit/outer groups are not disjoint")
    all_reference_labels = np.asarray([manifest[int(i)]["speaker_id"] for i in all_references], dtype=object)
    for group in set(groups[all_references]):
        if len(set(all_reference_labels[groups[all_references] == group])) != 1:
            raise ValueError("F005 known selector found a conflicting-label reference group")
    reference_indices = all_references[np.isin(all_reference_labels, labels[1:])]
    normalized = values.copy()
    # Contract requires unit vectors, so no new transform is applied.
    reference_labels = np.asarray([manifest[int(i)]["speaker_id"] for i in reference_indices], dtype=object)
    scores = np.empty((len(query_indices), len(labels) - 1), dtype=np.float32)
    support = np.empty_like(scores, dtype=np.int64)
    similarities = normalized[query_indices] @ normalized[reference_indices].T
    same_group = groups[query_indices, None] == groups[reference_indices][None, :]
    similarities[same_group] = -np.inf
    for target, label in enumerate(labels[1:]):
        columns = reference_labels == label
        if not np.any(columns):
            raise ValueError("F005 known selector lost a reference class")
        scores[:, target] = similarities[:, columns].max(axis=1)
        support[:, target] = np.sum(~same_group[:, columns], axis=1)
    if not np.isfinite(scores).all() or np.any(support < 1):
        raise ValueError("Whole-group exclusion left an empty known reference class")
    np.clip(scores, -1.0, 1.0, out=scores)
    return {
        "known_calibration_indices": query_indices,
        "known_scores": scores,
        "known_labels": labels[1:],
        "reference_support": support,
        "provenance": {
            "protocol": "f005_preselection_known_only_expanded_heldout_gallery_v1",
            "authoritative_postselection_scorer": "speaker_id.training.heldout_references.heldout_reference_scores",
            "reference_scope": "all_eligible_outer_training_known_references",
            "whole_query_group_excluded": True,
            "unknown_query_indices_materialized": False,
            "unknown_reference_cohort_materialized": False,
            "unknown_similarity_computed": False,
            "outer_labels_read": False,
            "outer_fold": outer,
        },
    }


def select_arm(contract: dict, outer: int, known_scores_by_arm: dict[str, dict]) -> dict:
    """Choose among control and three treatments using known queries only."""
    if set(known_scores_by_arm) != set(ARM_IDS):
        raise ValueError("F005 selection requires all four paired arms")
    config, manifest, labels = contract["config"], contract["manifest"], contract["labels"]
    label_index = {label: index for index, label in enumerate(labels)}
    roles = {row["audio_file"]: row for row in contract["roles"] if int(row["outer_fold"]) == outer}
    anchor = known_scores_by_arm["control"]
    indices = np.asarray(anchor["known_calibration_indices"])
    if indices.dtype.kind not in "iu" or indices.ndim != 1 or not len(indices):
        raise ValueError("F005 known calibration indices are malformed")
    for arm_id, scores in known_scores_by_arm.items():
        if (scores.get("provenance", {}).get("protocol") != "f005_preselection_known_only_expanded_heldout_gallery_v1"
                or scores["provenance"].get("unknown_query_indices_materialized") is not False
                or scores["provenance"].get("unknown_reference_cohort_materialized") is not False
                or scores["provenance"].get("unknown_similarity_computed") is not False
                or scores.get("known_labels") != labels[1:]
                or not np.array_equal(np.asarray(scores["known_calibration_indices"]), indices)
                or np.asarray(scores["known_scores"]).shape != (len(indices), len(labels) - 1)
                or not np.isfinite(scores["known_scores"]).all()):
            raise ValueError(f"F005 arm {arm_id} is not aligned known-only heldout evidence")
    true_known = []
    for index in indices:
        row = manifest[int(index)]
        role = roles.get(row["audio_file"])
        if (role is None or not truth(role["calibration_query"])
                or row["speaker_id"] not in set(labels[1:])):
            raise ValueError("F005 arm selection saw a non-known-calibration row")
        true_known.append(label_index[row["speaker_id"]])
    true_known = np.asarray(true_known, dtype=np.int64)
    expected = contract["config"]["arm_selection"]
    if (len(set(true_known.tolist())) != expected["expected_observed_known_labels_by_outer_fold"][str(outer)]
            or (len(labels) - 1 - len(set(true_known.tolist())))
            != expected["expected_absent_known_labels_by_outer_fold"][str(outer)]):
        raise ValueError("F005 known-query coverage differs from preregistration")
    metrics = {}
    for arm_id, scores in known_scores_by_arm.items():
        matrix = np.asarray(scores["known_scores"])
        guess = matrix.argmax(axis=1).astype(np.int64) + 1
        metrics[arm_id] = _known_query_metrics(true_known, guess)
    tie = expected["arm_tie_order"]
    selected = max(tie, key=lambda arm: (
        metrics[arm]["macro_f1_observed_known_labels"], metrics[arm]["top1_accuracy"], -tie.index(arm)
    ))
    body = {
        "schema_version": 1,
        "experiment_signature": contract["signature"],
        "outer_fold": outer,
        "selected_arm": selected,
        "scientific_conclusion": ("consistency_not_supported_on_known_calibration"
                                  if selected == "control" else "consistency_candidate_selected"),
        "arm_metrics": {arm: metrics[arm] for arm in ARM_IDS},
        "selection_order": ["macro_f1_observed_known_labels", "top1_accuracy", "fixed_arm_tie_order"],
        "known_calibration_indices_sha256": _index_hash(indices),
        "known_query_rows": len(indices),
        "observed_known_labels": len(set(true_known.tolist())),
        "absent_known_labels": sorted(set(range(1, len(labels))) - set(true_known.tolist())),
        "unknown_calibration_indices_materialized": False,
        "unknown_reference_cohort_materialized": False,
        "unknown_similarity_computed": False,
        "outer_rows_or_labels_read": False,
        "refit_after_selection": False,
    }
    return {**body, "seal_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def write_and_reload_seal(path: Path, seal: dict) -> dict:
    """Atomically persist a pre-truth seal and verify exact bytes/identity."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("F005 refuses to replace an existing selection seal")
    expected = seal.get("seal_sha256")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if not isinstance(expected, str) or hashlib.sha256(canonical(body)).hexdigest() != expected:
        raise ValueError("F005 selection seal identity is invalid")
    expected_bytes = (json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n").encode("utf-8")
    write_json(path, seal)
    actual_bytes = path.read_bytes()
    reloaded = json.loads(actual_bytes.decode("utf-8"))
    if reloaded != seal or actual_bytes != expected_bytes:
        raise RuntimeError("F005 selection seal did not round-trip")
    return {"path": str(path), "file_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "expected_bytes_sha256": hashlib.sha256(expected_bytes).hexdigest(), "seal_sha256": expected}


def probe_receipt(contract: dict, *, plan_steps: tuple[int, ...] = (0, 599, 600, 699, 1099)) -> dict:
    """Bounded, no-audio/no-Torch proof of pairing and schedule boundaries."""
    total = adaptation_total_steps(contract["config"]["fit"])
    if any(type(step) is not int or not 0 <= step < total for step in plan_steps):
        raise ValueError("F005 probe step is outside the schedule")
    folds = []
    for outer in contract["config"]["fold_ids"]:
        pairing = pairing_identity(contract, outer)
        plans = {str(step): training_step_plan(contract, outer, step) for step in plan_steps}
        identities = [arm_identity(contract, outer, arm, shared_head_checkpoint_sha256="0" * 64,
                                   tail_plan_sha256="0" * 64) for arm in ARM_IDS]
        if len({item["pairing_signature"] for item in identities}) != 1:
            raise RuntimeError("F005 arms do not share their data/crop plan identity")
        folds.append({
            "outer_fold": outer,
            "pairing_signature": pairing["signature"],
            "fit_rows": len(fit_rows(contract, outer)),
            "probe_steps": {step: hashlib.sha256(canonical(plan)).hexdigest() for step, plan in plans.items()},
            "first_step_plan": plans[str(plan_steps[0])],
            "arm_signatures_are_distinct": len({item["signature"] for item in identities}) == len(ARM_IDS),
            "all_arms_share_batch_and_crop_plan": True,
        })
    return {
        "status": "validated_no_cuda_no_audio_no_training",
        "experiment_signature": contract["signature"],
        "trainable_endpoint": {"name": "advanced_campp", "embedding_dim": 192,
                               "weights_sha256": ADVANCED_WEIGHTS_SHA256},
        "frozen_endpoint": {"name": "public_campp", "embedding_dim": 512},
        "shared_head_steps_per_outer": contract["config"]["fit"]["adaptation_schedule"]["head_only_steps"],
        "tail_steps_per_arm": total - contract["config"]["fit"]["adaptation_schedule"]["head_only_steps"],
        "work_units": len(contract["config"]["fold_ids"]) * (1 + len(ARM_IDS)),
        "shared_head_checkpoint_forked_byte_identically": True,
        "folds": folds,
        "mlflow_model_or_optimizer_artifacts_allowed": False,
        "outer_truth_read": False,
    }
