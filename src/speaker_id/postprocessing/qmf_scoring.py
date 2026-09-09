"""Deterministic, nested-safe quality-measure fusion for open-set gating.

The model in this module is deliberately small: a standardized linear
logistic regressor predicts ``is_known``.  Its output may only accept or reject
the already selected best known identity; it can never reorder known speakers.
All functions operate on numerical arrays and receive no speaker or outer-fold
metadata beyond the explicit binary fit target and content-group identifiers.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from speaker_id.postprocessing.decision_scoring import FEATURE_NAMES as DECISION_FEATURE_ORDER
from speaker_id.training.scoring import macro_f1_indices


MODEL_SCHEMA_VERSION = 1
SCORE_ONLY = "scores_only"
SCORE_QUALITY = "scores_quality"
FEATURE_SETS = (SCORE_ONLY, SCORE_QUALITY)
SCORE_ONLY_FEATURE_ORDER = (
    "fused_known_top",
    "fused_known_gap",
    "fused_unknown_top",
    "fused_unknown_top3_mean",
    "fused_unknown_top50_mean",
    "fused_unknown_top50_std",
    "public_known_minus_unknown",
    "advanced_known_minus_unknown",
    "encoder_winner_agreement",
)
SCORE_QUALITY_FEATURE_ORDER = SCORE_ONLY_FEATURE_ORDER + (
    "log1p_duration_capped180",
    "rms_dbfs_clipped",
)
FEATURE_ORDER_BY_SET = {
    SCORE_ONLY: SCORE_ONLY_FEATURE_ORDER,
    SCORE_QUALITY: SCORE_QUALITY_FEATURE_ORDER,
}
DEFAULT_L2_PENALTY = 0.1
DEFAULT_SCALE_FLOOR = 1e-6
DEFAULT_THRESHOLD_QUANTILES = 201
DEFAULT_TEMPERATURE = 0.05
POSITIVE_MARGIN_FLOOR = 1e-12
META_FOLDS = 3
META_MINIMUM_POOLED_GAIN = 0.001
META_MAXIMUM_FOLD_LOSS = 0.002


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _features(value: Any, *, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and len(result) > 0, "QMF features must be a nonempty matrix")
    if columns is not None:
        _require(result.shape[1] == columns, "QMF feature dimension changed")
    _require(result.shape[1] > 0 and np.isfinite(result).all(), "QMF features must be finite")
    return result


def _feature_order(value: Sequence[str], columns: int) -> list[str]:
    _require(isinstance(value, (list, tuple)) and len(value) == columns,
             "QMF feature order must name every feature")
    result = list(value)
    _require(all(type(name) is str and name for name in result) and len(set(result)) == len(result),
             "QMF feature names must be nonempty and unique")
    return result


def _canonical_feature_set(order: Sequence[str]) -> str:
    matches = [name for name, expected in FEATURE_ORDER_BY_SET.items() if list(order) == list(expected)]
    _require(len(matches) == 1, "QMF feature order is not a preregistered feature set")
    return matches[0]


def qmf_feature_view(features: Any, source_feature_order: Sequence[str], feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Select the exact preregistered QMF columns from S012 decision features."""
    _require(feature_set in FEATURE_SETS, "Unknown QMF feature set")
    source_order = _feature_order(source_feature_order, len(DECISION_FEATURE_ORDER))
    _require(source_order == list(DECISION_FEATURE_ORDER), "S012 decision feature schema changed")
    source = _features(features, columns=len(source_order))
    output_order = list(FEATURE_ORDER_BY_SET[feature_set])
    indices = [source_order.index(name) for name in output_order]
    selected = source[:, indices].copy()
    _require(selected.shape == (len(source), len(output_order)) and np.isfinite(selected).all(),
             "QMF selected feature view is malformed")
    return selected, output_order


def is_known_targets(truth: Any, *, classes: int = 447) -> np.ndarray:
    """Map fixed 447-class truth directly to the binary open-set target."""
    values = np.asarray(truth)
    _require(values.ndim == 1 and len(values) > 0 and values.dtype.kind in "iu",
             "QMF truth must be a nonempty integer vector")
    _require(type(classes) is int and classes >= 3
             and np.all((values >= 0) & (values < classes)), "QMF truth is outside the label map")
    return (values != 0).astype(np.float64)


def group_equal_binary_balanced_weights(is_known: Any, groups: Any) -> np.ndarray:
    """Give groups equal mass within each binary class, then classes equal mass.

    The returned weights have mean one.  Repeating rows inside a content group
    therefore cannot increase that group's total influence.
    """
    target = np.asarray(is_known)
    group_values = np.asarray(groups)
    _require(target.ndim == 1 and len(target) > 0 and group_values.shape == target.shape,
             "QMF targets and groups must be aligned vectors")
    if target.dtype == np.bool_:
        binary = target.astype(np.int64)
    else:
        _require(target.dtype.kind in "iuf" and np.isfinite(target).all()
                 and np.all((target == 0) | (target == 1)), "QMF target must be exactly binary")
        binary = target.astype(np.int64)
    _require(set(binary.tolist()) == {0, 1}, "QMF fitting requires known and unknown examples")

    # Convert to strings only after rejecting missing/empty values.  Object
    # identity must not influence grouping or deterministic ordering.
    flat_groups = group_values.tolist()
    _require(all(type(group) is str and group for group in flat_groups),
             "Every QMF fit row requires a nonempty string content group")
    group_to_rows: dict[str, list[int]] = {}
    for index, group in enumerate(flat_groups):
        group_to_rows.setdefault(group, []).append(index)
    for rows in group_to_rows.values():
        _require(len(set(binary[rows].tolist())) == 1,
                 "A QMF content group cannot mix known and unknown targets")

    weights = np.empty(len(binary), dtype=np.float64)
    for rows in group_to_rows.values():
        weights[rows] = 1.0 / len(rows)
    for label in (0, 1):
        selected = binary == label
        mass = float(weights[selected].sum())
        _require(np.isfinite(mass) and mass > 0, "QMF class has no group weight")
        weights[selected] *= (len(binary) / 2.0) / mass
    _require(np.isfinite(weights).all() and np.all(weights > 0)
             and np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12),
             "QMF weights are not finite and normalized")
    return weights


def _validate_model(model: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[str]]:
    required = {
        "schema_version", "mean", "scale", "coef", "intercept", "feature_order",
        "l2_penalty", "scale_floor", "target", "weighting", "feature_set",
    }
    _require(isinstance(model, Mapping) and set(model) == required, "QMF model schema changed")
    _require(type(model["schema_version"]) is int and model["schema_version"] == MODEL_SCHEMA_VERSION
             and model["target"] == "is_known"
             and model["weighting"] == "group_equal_then_binary_balanced",
             "QMF model identity changed")
    order = _feature_order(model["feature_order"], len(model["feature_order"]))
    _require(model["feature_set"] == _canonical_feature_set(order), "QMF feature-set binding changed")
    columns = len(order)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    _require(mean.shape == scale.shape == coef.shape == (columns,)
             and np.isfinite(mean).all() and np.isfinite(scale).all()
             and np.isfinite(coef).all() and np.all(scale > 0), "Malformed QMF coefficients")
    intercept = model["intercept"]
    penalty, floor = model["l2_penalty"], model["scale_floor"]
    _require(type(intercept) in (int, float) and np.isfinite(intercept)
             and type(penalty) in (int, float) and np.isfinite(penalty) and penalty > 0
             and type(floor) in (int, float) and np.isfinite(floor) and floor > 0
             and np.all(scale >= floor), "Malformed QMF scalar parameters")
    # This also rejects NumPy scalars and any accidental non-JSON metadata.
    try:
        json.dumps(dict(model), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("QMF model is not JSON-safe") from error
    return mean, scale, coef, float(intercept), order


def fit_qmf_logistic(
    features: Any,
    is_known: Any,
    groups: Any,
    feature_order: Sequence[str],
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
) -> dict[str, Any]:
    """Fit and export a deterministic standardized linear logistic model."""
    x = _features(features)
    order = _feature_order(feature_order, x.shape[1])
    feature_set = _canonical_feature_set(order)
    _require(type(l2_penalty) in (int, float) and not isinstance(l2_penalty, bool)
             and np.isfinite(l2_penalty) and l2_penalty > 0,
             "QMF L2 penalty must be a positive finite scalar")
    _require(type(scale_floor) in (int, float) and not isinstance(scale_floor, bool)
             and np.isfinite(scale_floor) and scale_floor > 0,
             "QMF scale floor must be a positive finite scalar")
    target = np.asarray(is_known)
    _require(target.shape == (len(x),), "QMF target length differs from features")
    weights = group_equal_binary_balanced_weights(target, groups)
    target = target.astype(np.float64)

    total = float(weights.sum())
    mean = np.sum(x * weights[:, None], axis=0) / total
    variance = np.sum(((x - mean) ** 2) * weights[:, None], axis=0) / total
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), float(scale_floor))
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x), dtype=np.float64), standardized))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        loss = (np.sum(weights * (np.logaddexp(0.0, logits) - target * logits)) / total
                + 0.5 * float(l2_penalty) * float(parameters[1:] @ parameters[1:]))
        gradient = design.T @ (weights * (expit(logits) - target)) / total
        gradient[1:] += float(l2_penalty) * parameters[1:]
        return float(loss), gradient

    fitted = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    _require(bool(fitted.success) and fitted.x.shape == (design.shape[1],)
             and np.isfinite(fitted.x).all(), "QMF logistic optimization did not converge")
    model: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "coef": [float(value) for value in fitted.x[1:]],
        "intercept": float(fitted.x[0]),
        "feature_order": order,
        "feature_set": feature_set,
        "l2_penalty": float(l2_penalty),
        "scale_floor": float(scale_floor),
        "target": "is_known",
        "weighting": "group_equal_then_binary_balanced",
    }
    _validate_model(model)
    return model


def predict_qmf_logit(features: Any, model: Mapping[str, Any], *,
                       feature_order: Sequence[str]) -> np.ndarray:
    """Apply an exported model, rejecting feature reordering or drift."""
    mean, scale, coef, intercept, order = _validate_model(model)
    _require(_feature_order(feature_order, len(order)) == order, "QMF feature order changed")
    x = _features(features, columns=len(order))
    logits = intercept + ((x - mean) / scale) @ coef
    _require(logits.shape == (len(x),) and np.isfinite(logits).all(), "QMF produced nonfinite logits")
    return logits


def qmf_decisions(known_scores: Any, known_logits: Any, threshold: float, valid: Any) -> np.ndarray:
    """Accept/reject the unchanged best known identity; a boundary tie rejects."""
    scores = np.asarray(known_scores)
    logits = np.asarray(known_logits, dtype=np.float64)
    mask = np.asarray(valid)
    _require(scores.ndim == 2 and len(scores) > 0 and scores.shape[1] >= 2
             and np.isfinite(scores).all(), "QMF requires finite known-class scores")
    _require(logits.shape == (len(scores),) and np.isfinite(logits).all(), "QMF logits are misaligned or nonfinite")
    _require(mask.shape == (len(scores),) and mask.dtype == np.bool_, "QMF validity mask is invalid")
    _require(type(threshold) in (int, float) and not isinstance(threshold, bool)
             and np.isfinite(threshold), "QMF threshold must be finite")
    winner = scores.argmax(axis=1).astype(np.int64) + 1
    return np.where(mask & (logits > float(threshold)), winner, 0)


def qmf_probabilities(
    known_scores: Any,
    known_logits: Any,
    threshold: float,
    valid: Any,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> np.ndarray:
    """Create normalized scores whose argmax is exactly :func:`qmf_decisions`."""
    scores = np.asarray(known_scores, dtype=np.float64)
    logits = np.asarray(known_logits, dtype=np.float64)
    mask = np.asarray(valid)
    expected = qmf_decisions(scores, logits, threshold, mask)
    _require(type(temperature) in (int, float) and not isinstance(temperature, bool)
             and np.isfinite(temperature) and temperature > 0, "QMF temperature must be positive")
    relative_known = scores - scores.max(axis=1, keepdims=True)
    margin = logits - float(threshold)
    display_margin = np.where(margin > 0, np.maximum(margin, POSITIVE_MARGIN_FLOOR), margin)
    all_logits = np.column_stack((-display_margin, relative_known)) / float(temperature)
    all_logits -= all_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(all_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~mask] = 0.0
    probabilities[~mask, 0] = 1.0
    _require(np.isfinite(probabilities).all()
             and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
             and np.array_equal(probabilities.argmax(axis=1), expected),
             "QMF normalized scores changed the accept/reject decision")
    return probabilities


def select_qmf_threshold(
    known_logits: Any,
    truth: Any,
    known_guess: Any,
    valid: Any,
    baseline_predictions: Any,
    *,
    quantiles: int = DEFAULT_THRESHOLD_QUANTILES,
    classes: int = 447,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a threshold within one declared fitting scope.

    For nested validation, callers must pass only one case's fit logits,
    labels, guesses, validity and baseline decisions.  The returned threshold
    may then be applied to that case's untouched validation rows.  Calling
    this primitive on pooled meta-OOF logits and scoring the same rows is an
    invalid use and is intentionally not performed by policy selection.
    """
    logits = np.asarray(known_logits, dtype=np.float64)
    target = np.asarray(truth)
    guess = np.asarray(known_guess)
    mask = np.asarray(valid)
    baseline = np.asarray(baseline_predictions)
    count = len(logits)
    _require(logits.shape == target.shape == guess.shape == mask.shape == baseline.shape == (count,)
             and count > 0 and np.isfinite(logits).all(), "QMF threshold inputs are misaligned")
    _require(target.dtype.kind in "iu" and guess.dtype.kind in "iu" and baseline.dtype.kind in "iu"
             and mask.dtype == np.bool_, "QMF threshold labels/mask have invalid dtypes")
    _require(type(classes) is int and classes >= 3 and np.all((target >= 0) & (target < classes))
             and np.all((guess >= 1) & (guess < classes))
             and np.all((baseline >= 0) & (baseline < classes)), "QMF threshold labels are outside the map")
    _require(type(quantiles) is int and quantiles == DEFAULT_THRESHOLD_QUANTILES,
             "QMF uses exactly 201 threshold quantiles")
    _require(np.all(baseline[~mask] == 0), "Invalid baseline rows must be unknown")
    _require(mask.any(), "QMF threshold selection needs a valid population")
    population = logits[mask]
    thresholds = np.unique(np.r_[float(population.min()) - 1e-6,
                                 np.quantile(population, np.linspace(0.0, 1.0, quantiles)),
                                 float(population.max()) + 1e-6])
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        prediction = np.where(mask & (logits > threshold), guess, 0)
        curve.append({
            "threshold": float(threshold),
            "meta_macro_f1_447": macro_f1_indices(target, prediction, classes),
            "changed_from_baseline": int(np.sum(prediction != baseline)),
        })
    selected = max(curve, key=lambda row: (
        row["meta_macro_f1_447"], -row["changed_from_baseline"], row["threshold"]
    ))
    return dict(selected), curve


def _nested_fold_fits(value: Any, assignments: np.ndarray) -> list[dict[str, Any]]:
    """Validate provenance for three independently fit case thresholds.

    This metadata is intentionally strict.  In particular, a threshold made
    from pooled meta-OOF logits is not representable by this schema.  The
    orchestration layer must fit both the logistic model and its threshold on
    each case's ``fit`` rows, then apply both to that case's validation rows.
    """
    required = {
        "heldout_meta_fold", "fit_meta_folds", "fit_rows", "validation_rows",
        "threshold", "threshold_fit_scope",
    }
    _require(isinstance(value, list) and len(value) == META_FOLDS,
             "QMF candidate needs exactly three case-fit threshold records")
    result: list[dict[str, Any]] = []
    for expected_fold, record in enumerate(value):
        _require(isinstance(record, Mapping) and set(record) == required,
                 "Malformed QMF case-fit threshold provenance")
        heldout = record["heldout_meta_fold"]
        fit_folds = record["fit_meta_folds"]
        fit_rows = record["fit_rows"]
        validation_rows = record["validation_rows"]
        threshold = record["threshold"]
        _require(type(heldout) is int and heldout == expected_fold,
                 "QMF case-fit threshold records must be ordered meta folds 0, 1, 2")
        _require(isinstance(fit_folds, list)
                 and all(type(fold) is int for fold in fit_folds)
                 and fit_folds == [fold for fold in range(META_FOLDS) if fold != heldout],
                 "QMF case-fit folds must exclude exactly the held-out meta fold")
        _require(type(fit_rows) is int and fit_rows > 0
                 and type(validation_rows) is int
                 and validation_rows == int(np.sum(assignments == heldout)),
                 "QMF case-fit row counts are invalid")
        _require(type(threshold) in (int, float) and not isinstance(threshold, bool)
                 and np.isfinite(threshold), "QMF case-fit threshold must be finite")
        _require(record["threshold_fit_scope"] == "case_fit_only",
                 "QMF threshold must be selected only from the same case fit rows")
        normalized = {
            "heldout_meta_fold": heldout,
            "fit_meta_folds": list(fit_folds),
            "fit_rows": fit_rows,
            "validation_rows": validation_rows,
            "threshold": float(threshold),
            "threshold_fit_scope": "case_fit_only",
        }
        try:
            json.dumps(normalized, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("QMF case-fit provenance is not JSON-safe") from error
        result.append(normalized)
    return result


def select_qmf_policy(
    truth: Any,
    known_guess: Any,
    valid: Any,
    baseline_predictions: Any,
    assignments: Any,
    candidates: Iterable[Mapping[str, Any]],
    *,
    classes: int = 447,
) -> dict[str, Any]:
    """Select a policy from fully nested, held-out case predictions.

    This function deliberately accepts predictions instead of logits.  Each
    candidate must already have fit its logistic model *and* threshold on each
    case's fit rows and applied both to that case's untouched validation rows.
    Consequently, neither validation truth nor validation-derived logits can
    enter model fitting, scaling, threshold fitting, or policy calibration.
    """
    target = np.asarray(truth)
    guess = np.asarray(known_guess)
    mask = np.asarray(valid)
    baseline = np.asarray(baseline_predictions)
    assigned = np.asarray(assignments)
    _require(target.shape == guess.shape == mask.shape == baseline.shape == assigned.shape and target.ndim == 1,
             "QMF policy inputs are misaligned")
    _require(target.dtype.kind in "iu" and guess.dtype.kind in "iu" and baseline.dtype.kind in "iu"
             and mask.dtype == np.bool_ and assigned.dtype.kind in "iu",
             "QMF policy labels, assignments or mask have invalid dtypes")
    _require(type(classes) is int and classes >= 3
             and np.all((target >= 0) & (target < classes))
             and np.all((guess >= 1) & (guess < classes))
             and np.all((baseline >= 0) & (baseline < classes))
             and np.all(baseline[~mask] == 0)
             and set(assigned.tolist()) == set(range(META_FOLDS)),
             "QMF policy labels or three-fold assignments are invalid")
    baseline_f1 = macro_f1_indices(target, baseline, classes)
    baseline_fold_f1 = [
        macro_f1_indices(target[assigned == fold], baseline[assigned == fold], classes)
        for fold in range(META_FOLDS)
    ]
    rows: list[dict[str, Any]] = [{
        "id": "baseline",
        "kind": "baseline",
        "feature_set": "baseline",
        "meta_macro_f1_447": baseline_f1,
        "meta_fold_macro_f1_447": baseline_fold_f1,
        "meta_gain": 0.0,
        "meta_fold_delta": [0.0] * META_FOLDS,
        "eligible_for_selection": True,
        "changed_from_baseline": 0,
        "predictions": baseline.copy(),
        "fold_fits": [],
    }]
    observed_ids: set[str] = set()
    for position, candidate in enumerate(candidates):
        _require(isinstance(candidate, Mapping) and set(candidate) == {
            "id", "feature_set", "crossfit_predictions", "fold_fits"
        },
                 "Malformed QMF candidate")
        candidate_id, feature_set = candidate["id"], candidate["feature_set"]
        _require(type(candidate_id) is str and candidate_id and candidate_id != "baseline"
                 and candidate_id not in observed_ids, "QMF candidate IDs must be unique")
        _require(feature_set in FEATURE_SETS, "Unknown QMF feature set")
        observed_ids.add(candidate_id)
        prediction = np.asarray(candidate["crossfit_predictions"])
        _require(prediction.shape == target.shape and prediction.dtype.kind in "iu"
                 and np.all((prediction >= 0) & (prediction < classes)),
                 "QMF cross-fit predictions are malformed")
        _require(np.all(prediction[~mask] == 0), "Invalid QMF rows must be unknown")
        accepted = prediction != 0
        _require(np.array_equal(prediction[accepted], guess[accepted]),
                 "QMF may only accept or reject the fixed known winner")
        fold_fits = _nested_fold_fits(candidate["fold_fits"], assigned)
        pooled_f1 = macro_f1_indices(target, prediction, classes)
        fold_f1 = [
            macro_f1_indices(target[assigned == fold], prediction[assigned == fold], classes)
            for fold in range(META_FOLDS)
        ]
        gain = pooled_f1 - baseline_f1
        fold_delta = [value - base for value, base in zip(fold_f1, baseline_fold_f1, strict=True)]
        rows.append({"id": candidate_id, "kind": "qmf", "feature_set": feature_set,
                     "candidate_order": position,
                     "meta_macro_f1_447": pooled_f1,
                     "meta_fold_macro_f1_447": fold_f1,
                     "meta_gain": gain,
                     "meta_fold_delta": fold_delta,
                     "eligible_for_selection": (gain >= META_MINIMUM_POOLED_GAIN
                                                and min(fold_delta) >= -META_MAXIMUM_FOLD_LOSS),
                     "changed_from_baseline": int(np.sum(prediction != baseline)),
                     "predictions": prediction.copy(),
                     "fold_fits": fold_fits,
                     "threshold_protocol": "case_fit_logits_and_truth_apply_untouched_validation"})

    priority = {"baseline": 2, SCORE_ONLY: 1, SCORE_QUALITY: 0}
    eligible = [row for row in rows[1:] if row["eligible_for_selection"]]
    selected = max(eligible, key=lambda row: (
        row["meta_macro_f1_447"], priority[row["feature_set"]],
        -row["changed_from_baseline"], -row["candidate_order"],
    )) if eligible else rows[0]
    return {"selected": selected, "candidates": rows,
            "tie_order": ["baseline", SCORE_ONLY, SCORE_QUALITY],
            "minimum_pooled_meta_gain": META_MINIMUM_POOLED_GAIN,
            "maximum_meta_fold_loss": META_MAXIMUM_FOLD_LOSS,
            "baseline_fallback": not bool(eligible),
            "selection_predictions_fully_nested": True,
            "case_threshold_fit_scope": "case_fit_only",
            "final_threshold_required_after_policy_selection": True,
            "outer_threshold_fit_scope": "full_inner_fit_logits_and_truth_after_recipe_selection",
            "outer_labels_used": False}
