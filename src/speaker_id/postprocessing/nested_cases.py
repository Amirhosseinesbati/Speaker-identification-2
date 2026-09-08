"""S012 meta cases with validation groups absent from the complete fit path.

Original outer labels are never read. Each meta split fits its baseline and
learner features using only its training gallery, then scores meta validation.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from speaker_id.postprocessing.scoring import prepare_fold, select_baseline


META_FOLDS = 3
META_SALT = "S012-nested-gallery-v1"
FEATURE_COUNT = 28


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _hash(*items):
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def make_meta_plan(valid, manifest, folds, labels, outer):
    """Create deterministic supervised group assignments inside original R only."""
    valid = np.asarray(valid)
    _require(valid.shape == (len(manifest),) and valid.dtype == np.bool_, "Validity must align with manifest")
    _require(isinstance(labels, list) and len(labels) >= 3 and labels[0] == "unknown"
             and len(set(labels)) == len(labels) and labels[1:] == sorted(labels[1:]), "Invalid known-label columns")
    names = [row["audio_file"] for row in manifest]
    indexed = {row["audio_file"]: row for row in folds}
    _require(len(set(names)) == len(names) and len(indexed) == len(folds)
             and set(indexed) == set(names), "Manifest/fold coverage must be unique and identical")
    assigned, groups, eligible = [], [], []
    group_folds = defaultdict(set)
    for name in names:
        row = indexed[name]
        fold, group = int(row["fold"]), row["group_id"]
        _require(isinstance(group, str) and group, "Every row requires a nonempty content group")
        assigned.append(fold)
        groups.append(group)
        eligible.append(_truth(row["train_eligible"]))
        group_folds[group].add(fold)
    _require(all(len(value) == 1 for value in group_folds.values()), "An original group crosses outer folds")
    assigned, groups = np.asarray(assigned), np.asarray(groups)
    _require(outer in assigned and np.any(assigned != outer), "Original outer fold must have training and evaluation rows")
    references = np.flatnonzero((assigned != outer) & np.asarray(eligible) & valid)
    _require(len(references) > 0, "No eligible outer-training references")
    # No label lookup happens before restriction to R.
    labels_by_index = {int(i): manifest[int(i)]["speaker_id"] for i in references}
    _require(set(labels_by_index.values()) == set(labels), "Every label must have eligible training support")
    grouped = defaultdict(set)
    label_groups = defaultdict(set)
    for index, label in labels_by_index.items():
        grouped[str(groups[index])].add(label)
        label_groups[label].add(str(groups[index]))
        if "speaker_id" in indexed[names[index]]:
            _require(indexed[names[index]]["speaker_id"] == label, "Training fold and manifest labels disagree")
    _require(all(len(value) == 1 for value in grouped.values()), "Eligible training content group has conflicting labels")
    _require(len(label_groups["unknown"]) >= 4, "Three meta folds need at least four independent unknown groups")
    assignment_by_group = {}
    reference_only_groups = []
    for label in labels:
        members = sorted(label_groups[label], key=lambda group: (_hash(META_SALT, "group", label, group), group))
        if label != "unknown" and len(members) == 1:
            reference_only_groups.extend(members)
            continue
        offset = int(_hash(META_SALT, "offset", label)[:8], 16) % META_FOLDS
        for position, group in enumerate(members):
            assignment_by_group[group] = (position + offset) % META_FOLDS
    query = np.asarray([i for i in references if str(groups[i]) in assignment_by_group], dtype=np.int64)
    assignments = np.asarray([assignment_by_group[str(groups[i])] for i in query], dtype=np.int64)
    _require(len(query) and set(assignments) == set(range(META_FOLDS)), "All three meta validation folds must contain queries")
    cases = []
    for meta_fold in range(META_FOLDS):
        validation = query[assignments == meta_fold]
        validation_groups = set(groups[validation])
        training = references[~np.isin(groups[references], list(validation_groups))]
        train_labels = {labels_by_index[int(i)] for i in training}
        unknown_groups = {str(groups[i]) for i in training if labels_by_index[int(i)] == "unknown"}
        _require(train_labels == set(labels) and len(unknown_groups) >= 2,
                 "Meta validation removed a known identity or required unknown cohort support")
        _require(not (set(groups[training]) & validation_groups), "Meta validation group leaked into training gallery")
        cases.append({"meta_fold": meta_fold, "reference_global_indices": training,
                      "validation_global_indices": validation})
    return {"cases": cases, "query_global_indices": query, "assignments": assignments,
            "reference_global_indices": references, "groups": groups,
            "provenance": {"protocol": "S012_threefold_whole_group_gallery_isolated_meta_validation_v1",
                "outer_fold": int(outer), "meta_folds": META_FOLDS, "assignment_salt": META_SALT,
                "assignment_rule": "Per-label salted-hash sorted content groups; cyclic threefold assignment with salted label offset",
                "known_singleton_groups_reference_only": sorted(reference_only_groups),
                "original_outer_labels_read": False, "all_fit_reference_groups_exclude_meta_validation": True,
                "outer_training_reference_count": len(references), "meta_query_count": len(query),
                "reference_indices_sha256": _index_hash(references), "query_indices_sha256": _index_hash(query),
                "assignments_sha256": _index_hash(assignments)}}


def _validate_feature_case(value, expected_indices, global_order, scope, classes):
    required = {"features", "feature_names", "guess", "margin", "valid", "indices", "known_scores", "unknown_similarity"}
    _require(isinstance(value, dict) and required.issubset(value) and "truth" not in value,
             "Feature builder must expose the fixed fields and must not receive or emit query truth")
    features = np.asarray(value["features"])
    indices = np.asarray(value["indices"])
    count = len(expected_indices)
    _require(features.shape == (count, FEATURE_COUNT) and features.dtype == np.float64
             and np.isfinite(features).all(), "Expected finite float64 28-column features")
    names = value["feature_names"]
    _require(isinstance(names, list) and len(names) == FEATURE_COUNT and len(set(names)) == FEATURE_COUNT
             and all(isinstance(name, str) and name for name in names), "Feature order/names must be explicit and unique")
    _require(indices.dtype.kind in "iu" and np.array_equal(indices, expected_indices), "Feature row order differs from exact scoring queries")
    valid, guess = np.asarray(value["valid"]), np.asarray(value["guess"])
    _require(valid.shape == (count,) and valid.dtype == np.bool_ and valid.all(),
             "Meta queries must be the original eligible valid rows")
    _require(guess.shape == (count,) and guess.dtype.kind in "iu" and np.all((guess >= 1) & (guess <= classes)),
             "Known guesses must be fixed one-based known label indices")
    _require(np.asarray(value["margin"]).shape == (count,) and np.isfinite(value["margin"]).all(), "Nonfinite or misaligned baseline gate margins")
    known, unknown = np.asarray(value["known_scores"]), np.asarray(value["unknown_similarity"])
    _require(known.shape == (count, classes) and known.dtype == np.float32 and np.isfinite(known).all()
             and unknown.shape == (count,) and unknown.dtype == np.float32 and np.isfinite(unknown).all(),
             "Known/background feature score arrays differ from the declared contract")
    _require(np.array_equal(guess, known.argmax(axis=1) + 1), "Features changed the raw known winner")
    result = dict(value)
    result["local_indices"] = indices.astype(np.int64, copy=True)
    result["global_indices"] = global_order[indices].copy()
    result["scope"] = scope
    return result


def _prepare_case(public, advanced, valid, manifest, labels, groups, plan_case, feature_builder):
    """A fixed-plan case: changing validation values cannot affect its fit path."""
    train = np.asarray(plan_case["reference_global_indices"], dtype=np.int64)
    validation = np.asarray(plan_case["validation_global_indices"], dtype=np.int64)
    _require(not (set(groups[train]) & set(groups[validation])), "Validation group leaked into complete training-reference path")
    global_order = np.r_[train, validation]
    _require(len(np.unique(global_order)) == len(global_order) and valid[global_order].all(), "Meta subset must contain distinct eligible rows")
    local_manifest = [manifest[int(i)] for i in global_order]
    local_folds = [{"audio_file": manifest[int(i)]["audio_file"], "group_id": str(groups[i]),
                    "fold": 1 if position < len(train) else 0,
                    "train_eligible": True, "evaluation_included": True}
                   for position, i in enumerate(global_order)]
    prepared = prepare_fold(public[global_order], advanced[global_order], valid[global_order],
                            local_manifest, local_folds, labels, outer=0)
    anchor = prepared["scores_by_alpha"][0.0]
    actual_refs = global_order[np.asarray(anchor["provenance"]["reference_indices"], dtype=np.int64)]
    _require(np.array_equal(actual_refs, train), "Meta scorer used a different reference pool")
    _require(np.any(prepared["inner_truth"] == 0) and np.any(prepared["inner_truth"] > 0),
             "Reduced meta pool lacks honest known/unknown calibration queries")
    baseline = select_baseline(prepared)
    # The builder API does not accept labels. Validation truth is attached only
    # after baseline fitting and both sets of predicted features are complete.
    feature_prepared = {key: value for key, value in prepared.items() if key != "inner_truth"}
    fit = _validate_feature_case(feature_builder(feature_prepared, baseline, "inner"),
        anchor["calibration_indices"], global_order, "inner", len(labels) - 1)
    heldout = _validate_feature_case(feature_builder(feature_prepared, baseline, "outer"),
        anchor["outer_indices"], global_order, "outer", len(labels) - 1)
    _require(fit["feature_names"] == heldout["feature_names"], "Fit/validation feature columns differ")
    fit["truth"] = prepared["inner_truth"].copy()
    fit["groups"] = groups[fit["global_indices"]].copy()
    label_index = {label: i for i, label in enumerate(labels)}
    heldout["truth"] = np.asarray([label_index[manifest[int(i)]["speaker_id"]]
                                  for i in heldout["global_indices"]], dtype=np.int64)
    heldout["groups"] = groups[heldout["global_indices"]].copy()
    _require(np.array_equal(heldout["global_indices"], validation), "Meta validation coverage or ordering changed")
    fit_indices = fit["global_indices"]
    skipped = np.setdiff1d(train, fit_indices)
    fit_groups, validation_groups = set(groups[fit_indices]), set(groups[validation])
    _require(not (fit_groups & validation_groups), "Validation groups entered learner-fit queries")
    support = anchor["reference_counts"]
    provenance = {"meta_fold": int(plan_case["meta_fold"]),
        "baseline_fit_scope": "Only meta-training leave-whole-query-group-out rows",
        "learner_fit_feature_scope": "Only meta-training gallery/cohort; complete query group removed",
        "validation_feature_scope": "Restored meta-training gallery; every meta-validation group absent",
        "original_outer_labels_read": False, "meta_validation_labels_used_for_baseline_or_fit_features": False,
        "reference_files": len(train), "fit_queries": len(fit_indices), "validation_queries": len(validation),
        "known_fit_queries": int((fit["truth"] > 0).sum()), "unknown_fit_queries": int((fit["truth"] == 0).sum()),
        "known_fit_classes_represented": int(len(np.unique(fit["truth"][fit["truth"] > 0]))),
        "known_reference_classes": len(labels) - 1,
        "known_singleton_classes_in_meta_training": list(support["singleton_known_labels"]),
        "reference_only_global_indices": skipped.tolist(),
        "reference_indices_sha256": _index_hash(train), "fit_indices_sha256": _index_hash(fit_indices),
        "validation_indices_sha256": _index_hash(validation),
        "reference_groups": sorted(set(str(groups[i]) for i in train)),
        "validation_groups": sorted(str(group) for group in validation_groups),
        "fit_reference_groups_disjoint_from_validation": True,
        "support_shift_limit": "Whole-meta-group removal lowers fit support further; this prevents contamination but does not remove support-domain mismatch"}
    return {"fit": fit, "validation": heldout, "baseline": baseline,
            "meta_fold": int(plan_case["meta_fold"]), "fit_global_indices": fit_indices.copy(),
            "validation_global_indices": validation.copy(), "reference_global_indices": train.copy(),
            "local_to_global_indices": global_order, "provenance": provenance}


def make_nested_cases(public, advanced, valid, manifest, folds, labels, outer, feature_builder,
                      on_progress=None, prepared=None):
    """Construct three honest meta-validation cases inside one original fold.

    ``feature_builder(prepared, baseline, scope)`` receives scope ``inner`` or
    ``outer`` and returns finite float64 28-column features, names, raw known
    guesses, baseline margins, validity, local indices and known/unknown scores.
    It must not read query truth; this layer attaches truth only after feature
    generation. ``prepared`` optionally reuses the already verified full-fold
    baseline preparation; it is never a feature source for the reduced cases.
    """
    public, advanced, valid = np.asarray(public), np.asarray(advanced), np.asarray(valid)
    _require(public.shape == (len(manifest), 512) and advanced.shape == (len(manifest), 192)
             and public.dtype == np.float32 and advanced.dtype == np.float32
             and np.isfinite(public).all() and np.isfinite(advanced).all(), "Invalid fixed public source arrays")
    plan = make_meta_plan(valid, manifest, folds, labels, outer)
    if prepared is None:
        prepared = prepare_fold(public, advanced, valid, manifest, folds, labels, outer)
    anchor = prepared["scores_by_alpha"][0.0]
    _require(np.array_equal(anchor["calibration_indices"], plan["query_global_indices"])
             and np.array_equal(np.asarray(anchor["provenance"]["reference_indices"]), plan["reference_global_indices"])
             and anchor["known_labels"] == labels[1:], "Full-fold baseline and nested protocol disagree about original query/reference identity")
    cases = []
    for plan_case in plan["cases"]:
        if on_progress:
            on_progress({"stage": "nested_case", "meta_fold": plan_case["meta_fold"], "status": "started"})
        case = _prepare_case(public, advanced, valid, manifest, labels, plan["groups"], plan_case, feature_builder)
        cases.append(case)
        if on_progress:
            on_progress({"stage": "nested_case", "meta_fold": plan_case["meta_fold"], "status": "complete",
                         "fit_queries": len(case["fit_global_indices"]), "validation_queries": len(case["validation_global_indices"])})
    observed = np.concatenate([case["validation_global_indices"] for case in cases])
    _require(len(np.unique(observed)) == len(observed)
             and np.array_equal(np.sort(observed), plan["query_global_indices"]), "Meta validation must cover each original eligible calibration query exactly once")
    _require(all(case["fit"]["feature_names"] == cases[0]["fit"]["feature_names"] for case in cases), "Feature schema changed across meta folds")
    return {"cases": cases, "query_global_indices": plan["query_global_indices"],
            "assignments": plan["assignments"], "provenance": plan["provenance"]}
