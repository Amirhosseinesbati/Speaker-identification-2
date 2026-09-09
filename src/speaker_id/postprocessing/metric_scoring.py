"""Honest nested calibration of six frozen-encoder geometries.

Transform fitting sees only known content groups in each permitted reference
pool. No globally fitted transform is used to generate inner calibration scores.
This module defines methods only; it does not authorize an experiment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from speaker_id.postprocessing.frozen_metric import METRIC_SPECS, fit_transform, transform
from speaker_id.postprocessing.nested_cases import make_meta_plan, META_FOLDS, META_SALT
from speaker_id.training.candidate_fusion import weighted_encoder_pair
from speaker_id.training.reference_scoring import calibrate_gate, gate_scores
from speaker_id.training.scoring import macro_f1_indices

ADVANCED_WEIGHT = .5
UNKNOWN_WEIGHTS = [0., .25, .5, .75, 1.]
MARGIN_WEIGHTS = [0., .5]
THRESHOLD_QUANTILES = 201
MINIMUM_META_GAIN = .001
MAXIMUM_META_FOLD_LOSS = .002


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_sha(value):
    value = np.asarray(value)
    _require(value.dtype.kind != "O", "Object arrays cannot identify metric source evidence")
    return hashlib.sha256(str(value.dtype).encode() + json.dumps(value.shape).encode() + value.tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ranking(scores, truth):
    known = truth > 0
    _require(known.any(), "Ranking diagnostics require known queries")
    order = np.argsort(-scores[known], axis=1, kind="stable") + 1
    ranks = np.argmax(order == truth[known, None], axis=1) + 1
    return {"known_queries": int(known.sum()), "known_top1_correct": int((ranks == 1).sum()),
        "known_top1_accuracy": float(np.mean(ranks == 1)),
        "known_top2_accuracy": float(np.mean(ranks <= min(2, scores.shape[1]))),
        "known_top5_accuracy": float(np.mean(ranks <= min(5, scores.shape[1]))),
        "known_mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "identity_selection": "maximum_cosine_to_a_permitted_reference_then_lowest_class_index_on_tie"}


def _score_references(queries, references, targets, classes):
    _require(queries.ndim == references.ndim == 2 and queries.shape[1] == references.shape[1]
        and queries.dtype == references.dtype == np.float32
        and np.isfinite(queries).all() and np.isfinite(references).all(), "Cosine matching requires finite float32 rows")
    targets = np.asarray(targets)
    _require(targets.shape == (len(references),) and targets.dtype.kind in "iu"
        and set(targets.tolist()) == set(range(classes + 1)), "Every known identity and unknown background need references")
    similarities = np.clip(queries @ references.T, -1., 1.)
    known = np.column_stack([similarities[:, targets == label].max(axis=1) for label in range(1, classes + 1)])
    unknown = similarities[:, targets == 0].max(axis=1)
    _require(known.dtype == np.float32 and unknown.dtype == np.float32
        and np.isfinite(known).all() and np.isfinite(unknown).all(), "Invalid transformed reference scores")
    return known, unknown


def _fit_score_case(values, targets, groups, references, queries, spec, classes):
    """Fit, apply and score one complete heldout-group case; no query truth input."""
    references, queries = np.asarray(references), np.asarray(queries)
    _require(values.ndim == 2 and values.dtype == np.float32 and np.isfinite(values).all(), "Invalid frozen source vectors")
    _require(references.ndim == queries.ndim == 1 and references.dtype.kind in "iu" and queries.dtype.kind in "iu"
        and len(references) > 0 and len(queries) > 0 and len(np.unique(references)) == len(references)
        and len(np.unique(queries)) == len(queries) and references.min() >= 0 and queries.min() >= 0
        and references.max() < len(values) and queries.max() < len(values), "Invalid permitted reference/query row identities")
    _require(not (set(groups[references]) & set(groups[queries])), "A heldout content group entered the complete metric fitting pool")
    reference_targets = targets[references]
    _require(set(reference_targets.tolist()) == set(range(classes + 1)), "A required class or background reference is missing")
    _require(np.any(values[references] != 0, axis=1).all(), "Eligible reference cannot have a zero source vector")
    payload = fit_transform(values[references], reference_targets, groups[references], spec)
    metadata = payload["metadata"]
    fit_groups = {row["group"] for row in metadata["known_groups"]}
    expected_groups = set(groups[references[reference_targets > 0]])
    _require(fit_groups == expected_groups and not (fit_groups & set(groups[queries]))
        and metadata["known_labels"] == list(range(1, classes + 1))
        and metadata["unknown_rows_ignored"] == int((reference_targets == 0).sum()), "Metric fitter used an unsupported population")
    changed_references = transform(payload, values[references])
    changed_queries = transform(payload, values[queries])
    known, unknown = _score_references(changed_queries, changed_references, reference_targets, classes)
    support = np.bincount(reference_targets, minlength=classes + 1)
    return {"payload": payload, "reference_global_indices": references.copy(), "query_global_indices": queries.copy(),
        "known_scores": known, "unknown_similarity": unknown,
        "transformed_references": changed_references, "transformed_queries": changed_queries,
        "reference_targets": reference_targets.copy(), "reference_group_ids": groups[references].copy(),
        "query_group_ids": groups[queries].copy(), "provenance": {
            "fit_scope": "known_groups_in_permitted_reference_pool_only",
            "query_groups_absent_from_mean_covariance_and_gallery": True,
            "unknown_rows_fit_as_one_speaker": False, "known_reference_files": int(support[1:].sum()),
            "unknown_reference_files": int(support[0]), "known_reference_count_min": int(support[1:].min()),
            "known_reference_count_mean": float(support[1:].mean()), "known_reference_count_max": int(support[1:].max()),
            "reference_counts": support.tolist(), "reference_indices_sha256": _array_sha(references),
            "query_indices_sha256": _array_sha(queries), "metric_mean_sha256": metadata["mean_sha256"],
            "metric_matrix_sha256": metadata["matrix_sha256"], "matching_dtype": "float32", "metric_fit_dtype": "float64"}}


def _calibrate_meta(known, unknown, truth, assignments, classes):
    _require(known.shape == (len(truth), classes - 1) and unknown.shape == truth.shape
        and assignments.shape == truth.shape and set(assignments.tolist()) == set(range(META_FOLDS))
        and np.isfinite(known).all() and np.isfinite(unknown).all(), "Metric meta scores or coverage are invalid")
    selected, original = calibrate_gate(known, truth, unknown, UNKNOWN_WEIGHTS, MARGIN_WEIGHTS,
        candidates=THRESHOLD_QUANTILES, classes=classes)
    guess = known.argmax(axis=1) + 1
    curves = []
    gates = {}
    for row in original:
        key = row["unknown_weight"], row["margin_weight"]
        if key not in gates:
            gates[key] = gate_scores(known, unknown, *key)
        prediction = np.where(gates[key] > row["threshold"], guess, 0)
        curves.append({"unknown_weight": row["unknown_weight"], "margin_weight": row["margin_weight"],
            "threshold": row["threshold"], "meta_macro_f1_447": row["inner_macro_f1_447"],
            "meta_fold_macro_f1_447": [macro_f1_indices(truth[assignments == f], prediction[assignments == f], classes)
                for f in range(META_FOLDS)]})
    best = next(row for row in curves if all(row[k] == selected[k] for k in ("unknown_weight", "margin_weight", "threshold")))
    calibration = {"kind": "nested_metric_gate", **best,
        "scope": "pooled_nested_heldout_geometry_scores_with_optimized_gate"}
    margin = gates[(best["unknown_weight"], best["margin_weight"])] - best["threshold"]
    return calibration, curves, np.where(margin > 0, guess, 0).astype(np.int64), margin


def select_metric_policy(candidates):
    """Compare fixed geometries to their identity control, never historical fit F1."""
    _require(len(candidates) == len(METRIC_SPECS)
        and [r["id"] for r in candidates] == [s["id"] for s in METRIC_SPECS], "All six ordered metric recipes are required")
    identity = candidates[0]
    for order, row in enumerate(candidates):
        row["candidate_order"] = order
        row["meta_gain_vs_identity"] = row["calibration"]["meta_macro_f1_447"] - identity["calibration"]["meta_macro_f1_447"]
        row["meta_fold_delta_vs_identity"] = [a - b for a, b in zip(row["calibration"]["meta_fold_macro_f1_447"],
            identity["calibration"]["meta_fold_macro_f1_447"], strict=True)]
        row["promotion_eligible"] = order > 0 and row["meta_gain_vs_identity"] >= MINIMUM_META_GAIN \
            and min(row["meta_fold_delta_vs_identity"]) >= -MAXIMUM_META_FOLD_LOSS
    eligible = [row for row in candidates if row["promotion_eligible"]]
    best = max(eligible, key=lambda row: (row["calibration"]["meta_macro_f1_447"], -row["candidate_order"])) if eligible else None
    return {"identity_control": "identity", "overall_metric_id": None if best is None else best["id"],
        "baseline_retained": best is None, "fallback": "exact_historical_S008c_baseline",
        "comparison_scope": "same_nested_geometry_fold_assignments_and_separately_inner_selected_gate",
        "minimum_pooled_gain": MINIMUM_META_GAIN, "maximum_meta_fold_loss": MAXIMUM_META_FOLD_LOSS,
        "original_outer_labels_used": False,
        "identity_control_is_historical_baseline": False}


def _selection_digest(search):
    return _json_sha({"candidates": search["candidate_summary"], "selection": search["selection"],
        "provenance": search["provenance"], "source_fingerprints": search["source_fingerprints"],
        "labels": search["labels"], "historical_baseline_binding": search["historical_baseline_binding"]})


def _prepare_from_values(prepared, baseline, values, valid, manifest, folds, labels, outer, on_progress=None):
    """Shared dimension-independent core for synthetic audits and the 704d wrapper."""
    values = np.asarray(values)
    _require(values.ndim == 2 and values.dtype == np.float32 and len(values) == len(manifest)
        and np.isfinite(values).all(), "Invalid fused source matrix")
    plan = make_meta_plan(valid, manifest, folds, labels, outer)
    anchor = prepared["scores_by_alpha"][0.]
    references, query, assignments = plan["reference_global_indices"], plan["query_global_indices"], plan["assignments"]
    _require(np.array_equal(anchor["calibration_indices"], query)
        and np.array_equal(np.asarray(anchor["provenance"]["reference_indices"]), references)
        and anchor["known_labels"] == labels[1:] and prepared["classes"] == len(labels) - 1,
        "Prepared historical fold and metric reference/query identities disagree")
    outer_indices = np.asarray(anchor["outer_indices"], dtype=np.int64)
    _require(not (set(plan["groups"][references]) & set(plan["groups"][outer_indices])), "Original outer group entered metric reference pool")
    targets = np.full(len(values), -1, dtype=np.int64)
    positions = {label: i for i, label in enumerate(labels)}
    # Original outer labels are never read: targets outside R remain sentinel -1.
    targets[references] = [positions[manifest[int(i)]["speaker_id"]] for i in references]
    truth = targets[query].copy()
    _require(np.array_equal(prepared["inner_truth"], truth), "Historical and nested metric training targets differ")
    cases, pooled, candidates = {}, {}, []
    for spec in METRIC_SPECS:
        identifier = spec["id"]
        by_index = {int(index): position for position, index in enumerate(query)}
        known = np.full((len(query), len(labels) - 1), np.nan, dtype=np.float32)
        unknown = np.full(len(query), np.nan, dtype=np.float32)
        cases[identifier] = []
        for entry in plan["cases"]:
            if on_progress:
                on_progress({"stage": "nested_metric", "metric_id": identifier, "meta_fold": entry["meta_fold"], "status": "started"})
            case = _fit_score_case(values, targets, plan["groups"], entry["reference_global_indices"],
                entry["validation_global_indices"], spec, len(labels) - 1)
            case["meta_fold"] = entry["meta_fold"]
            rows = np.asarray([by_index[int(i)] for i in case["query_global_indices"]], dtype=np.int64)
            known[rows], unknown[rows] = case["known_scores"], case["unknown_similarity"]
            case["truth"] = truth[rows].copy()
            cases[identifier].append(case)
            if on_progress:
                on_progress({"stage": "nested_metric", "metric_id": identifier, "meta_fold": entry["meta_fold"], "status": "complete",
                    "reference_rows": len(case["reference_global_indices"]), "query_rows": len(rows)})
        calibration, curves, predictions, margin = _calibrate_meta(known, unknown, truth, assignments, len(labels))
        ranking = _ranking(known, truth)
        ranking_by_fold = [_ranking(known[assignments == f], truth[assignments == f]) for f in range(META_FOLDS)]
        candidates.append({"id": identifier, "spec": deepcopy(spec), "advanced_weight": ADVANCED_WEIGHT,
            "calibration": calibration, "ranking": ranking, "meta_fold_ranking": ranking_by_fold})
        pooled[identifier] = {"known_scores": known, "unknown_similarity": unknown,
            "predictions": predictions, "margin": margin, "calibration": calibration, "curves": curves}
    selection = select_metric_policy(candidates)
    source = {"fused_vectors": values.copy(), "targets": targets, "groups": plan["groups"].copy(),
        "references": references.copy(), "query_indices": query.copy(), "outer_indices": outer_indices.copy(),
        "valid": np.asarray(valid).copy(), "inner_truth": truth.copy(), "assignments": assignments.copy()}
    fingerprints = {name: _array_sha(array) for name, array in source.items()}
    for array in source.values():
        array.setflags(write=False)
    result = {"cases": cases, "plan": plan, "pooled_meta": pooled, "candidate_summary": candidates,
        "selection": selection, "source": source, "source_fingerprints": fingerprints,
        "labels": list(labels), "historical_baseline_binding": {
            "policy": deepcopy(baseline["policy"]), "calibration": deepcopy(baseline["calibration"]),
            "probabilities_sha256": _array_sha(baseline["probabilities"])},
        "provenance": {"advanced_weight": ADVANCED_WEIGHT, "metric_specs": deepcopy(METRIC_SPECS),
            "nested_assignment_salt": META_SALT, "meta_folds": META_FOLDS, "outer_fold": int(outer),
            "evaluation_classes": len(labels), "identity_control": "float64_L2_then_float32_cosine_not_bitexact_S008c",
            "known_only_metric_fitting": True, "mean_covariance_gallery_exclude_whole_meta_validation_groups": True,
            "original_outer_labels_read": False, "outer_scores_computed": False,
            "gate_grid": {"unknown_weights": UNKNOWN_WEIGHTS, "margin_weights": MARGIN_WEIGHTS, "threshold_quantiles": THRESHOLD_QUANTILES},
            "selection_bias_limit": "Pooled meta curves are optimized selection scores, not an independent final test",
            "support_limit": "Nested gallery removal reduces speaker support; singleton groups remain reference-only"}}
    result["selection_sha256"] = _selection_digest(result)
    return result


def prepare_metric_search(prepared, baseline, public, advanced, valid, manifest, folds, labels, outer, on_progress=None):
    """Fit only nested geometries, then freeze all six gate policies and selection."""
    public, advanced = np.asarray(public), np.asarray(advanced)
    _require(public.dtype == advanced.dtype == np.float32, "Public encoder caches must remain float32")
    values = weighted_encoder_pair(public, advanced, valid, ADVANCED_WEIGHT)
    _require(values.shape == (len(manifest), 704), "The fixed 512+192 equal-weight fusion is required")
    return _prepare_from_values(prepared, baseline, values, valid, manifest, folds, labels, outer, on_progress)


def evaluate_metric_outer(search, baseline, on_progress=None):
    """After frozen nested selection, refit six geometries on R and score outer.

    No outer truth is accepted. All six settings are exploratory outputs; the
    overall result follows the already sealed selection or exact S008c fallback.
    """
    from speaker_id.postprocessing.decision_scoring import decision_probabilities
    _require(search["selection_sha256"] == _selection_digest(search), "Metric selection changed after inner-only freezing")
    source = search["source"]
    _require(all(_array_sha(value) == search["source_fingerprints"][name] for name, value in source.items()),
        "Metric source arrays changed after selection")
    historical = search["historical_baseline_binding"]
    _require(baseline["policy"] == historical["policy"] and baseline["calibration"] == historical["calibration"]
        and _array_sha(baseline["probabilities"]) == historical["probabilities_sha256"], "Historical fallback changed after selection")
    _require(search["provenance"]["metric_specs"] == METRIC_SPECS
        and search["provenance"]["advanced_weight"] == ADVANCED_WEIGHT, "Geometry recipe or fixed fusion changed")
    results, cases = {}, {}
    classes = len(search["labels"]) - 1
    for spec, candidate in zip(METRIC_SPECS, search["candidate_summary"], strict=True):
        identifier = spec["id"]
        if on_progress:
            on_progress({"stage": "full_reference_metric", "metric_id": identifier, "status": "started"})
        case = _fit_score_case(source["fused_vectors"], source["targets"], source["groups"], source["references"],
            source["outer_indices"], spec, classes)
        calibration = candidate["calibration"]
        known, unknown = case["known_scores"], case["unknown_similarity"]
        margin = gate_scores(known, unknown, calibration["unknown_weight"], calibration["margin_weight"]) - calibration["threshold"]
        valid = source["valid"][source["outer_indices"]]
        probabilities = decision_probabilities(known, margin, valid)
        expected = np.where(valid & (margin > 0), known.argmax(axis=1) + 1, 0)
        _require(np.array_equal(probabilities.argmax(axis=1), expected), "Metric probability conversion changed gate decisions")
        scores = {"outer_known_scores": known, "outer_unknown_similarity": unknown, "outer_valid": valid.copy(),
            "outer_indices": source["outer_indices"].copy(), "outer_margin": margin, "provenance": case["provenance"]}
        policy = {"id": identifier, "family": "learned_geometry", "kind": "nested_metric_gate", "spec": deepcopy(spec),
            "advanced_weight": ADVANCED_WEIGHT, "calibration": deepcopy(calibration),
            "mean_sha256": case["payload"]["metadata"]["mean_sha256"], "matrix_sha256": case["payload"]["metadata"]["matrix_sha256"],
            "selection_sha256": search["selection_sha256"]}
        results[identifier] = {"policy": policy, "calibration": deepcopy(calibration),
            "selection_meta_macro_f1_447": calibration["meta_macro_f1_447"],
            "meta_selection": deepcopy(candidate), "probabilities": probabilities, "scores": scores,
            "payload": case["payload"], "baseline_retained": False}
        cases[identifier] = case
        if on_progress:
            on_progress({"stage": "full_reference_metric", "metric_id": identifier, "status": "complete"})
    selected = search["selection"]["overall_metric_id"]
    overall = results[selected] if selected is not None else {**baseline, "baseline_retained": True,
        "payload": None, "metric_selection": deepcopy(search["selection"]),
        "selection_meta_macro_f1_447": search["candidate_summary"][0]["calibration"]["meta_macro_f1_447"],
        "selection_meta_score_role": "identity_metric_control_used_for_fallback_decision_not_historical_baseline_meta_F1"}
    return {"results": results, "overall": overall, "full_reference_cases": cases,
        "selection": deepcopy(search["selection"]), "selection_sha256": search["selection_sha256"],
        "original_outer_labels_read": False, "outer_scores_computed_after_selection_freeze": True,
        "historical_fallback_probability_bytes_preserved": selected is None and np.array_equal(overall["probabilities"], baseline["probabilities"])}
