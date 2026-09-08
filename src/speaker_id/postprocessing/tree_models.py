"""Six preregistered S012 CPU tree heads and verified portable exports."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from speaker_id.inference.decision_trees import feature_matrix, predict_export

SEED = 20260908
SKLEARN_VERSION = "1.8.0"

MODEL_SPECS = [
    {"id": "dt_depth2", "family": "decision_tree",
     "params": {"max_depth": 2, "min_samples_leaf": 40}},
    {"id": "dt_depth3", "family": "decision_tree",
     "params": {"max_depth": 3, "min_samples_leaf": 80}},
    {"id": "rf128_depth5", "family": "random_forest",
     "params": {"n_estimators": 128, "max_depth": 5, "min_samples_leaf": 30,
                "max_features": 0.8, "n_jobs": 4}},
    {"id": "extra256_depth5", "family": "extra_trees",
     "params": {"n_estimators": 256, "max_depth": 5, "min_samples_leaf": 30,
                "max_features": 1.0, "n_jobs": 4}},
    {"id": "gb64_stump", "family": "gradient_boosting",
     "params": {"n_estimators": 64, "max_depth": 1, "min_samples_leaf": 40,
                "learning_rate": 0.05}},
    {"id": "gb128_depth2", "family": "gradient_boosting",
     "params": {"n_estimators": 128, "max_depth": 2, "min_samples_leaf": 60,
                "learning_rate": 0.03}},
]


def _binary_target(target, rows):
    result = np.asarray(target)
    if (result.ndim != 1 or len(result) != rows or result.dtype.kind not in "biuf"
            or not np.isfinite(result).all() or not np.isin(result, [0, 1]).all()):
        raise ValueError("Correctness target must be an aligned finite binary vector")
    result = result.astype(np.int64)
    counts = np.bincount(result, minlength=2)
    if not counts.all():
        raise ValueError("Tree fitting requires both binary correctness classes")
    weights = rows / (2.0 * counts)
    return result, counts, weights[result], weights


def _estimator(spec):
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if sklearn.__version__ != SKLEARN_VERSION:
        raise RuntimeError(f"S012 tree fitting requires scikit-learn {SKLEARN_VERSION}")
    constructors = {"decision_tree": DecisionTreeClassifier,
                    "random_forest": RandomForestClassifier,
                    "extra_trees": ExtraTreesClassifier,
                    "gradient_boosting": GradientBoostingClassifier}
    params = deepcopy(spec["params"])
    params["random_state"] = SEED
    if spec["family"] == "gradient_boosting":
        params.update(loss="log_loss", subsample=1.0, n_iter_no_change=None)
    else:
        params["class_weight"] = None
    return constructors[spec["family"]](**params)


def _export_tree(estimator, *, classifier):
    tree = estimator.tree_
    values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
    if classifier:
        if not np.array_equal(estimator.classes_, [0, 1]) or values.shape[1] != 2:
            raise ValueError("Tree class order differs from binary correctness")
        total = values.sum(axis=1)
        if not np.isfinite(total).all() or not np.allclose(total, 1.0, rtol=0, atol=1e-12):
            raise ValueError("Invalid native weighted classification probabilities")
        # In sklearn 1.8 these are already weighted proportions, and
        # predict_proba returns them without renormalization. Preserve them:
        # dividing again could perturb a binary decision at exactly 0.5.
        value = values[:, 1]
    else:
        if values.shape[1] != 1:
            raise ValueError("Boosting requires scalar regression trees")
        value = values[:, 0]
    return {"children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": value.tolist()}


def _check_translation(payload, estimator, features):
    native = estimator.predict_proba(features)[:, 1]
    portable = predict_export(payload, features)
    maximum = float(np.max(np.abs(portable - native), initial=0))
    disagreements = int(np.count_nonzero((portable > 0.5) != (native > 0.5)))
    if maximum > 1e-12 or disagreements:
        raise ValueError("Portable tree export does not reproduce native probabilities/decisions")
    return {"performed": True, "rows": len(features),
            "max_abs_error": maximum, "probability_tie_boundary_disagreements": disagreements,
            "tolerance": 1e-12}


def fit_export(spec, X, target, check_features=None):
    """Fit binary-balanced correctness labels; export and verify NumPy parity.

    check_features carries no labels and is used only for arithmetic translation
    QA after fitting. It never enters fitting, weighting or model selection.
    """
    if (not isinstance(spec, dict) or not any(spec == expected for expected in MODEL_SPECS)):
        raise ValueError("Tree model must match a preregistered S012 specification")
    spec = deepcopy(spec)
    values = feature_matrix(X)
    labels, counts, sample_weights, class_weights = _binary_target(target, len(values))
    check = None if check_features is None else feature_matrix(check_features, values.shape[1])
    estimator = _estimator(spec)
    estimator.fit(values, labels, sample_weight=sample_weights)
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("Fitted class order differs from binary correctness")
    family = spec["family"]
    if family == "gradient_boosting":
        trees = [_export_tree(tree, classifier=False) for tree in estimator.estimators_[:, 0]]
        # Version-bound extraction of the exact fitted initializer, not a
        # reconstruction from unweighted labels or an assumed zero intercept.
        initial = np.asarray(estimator._raw_predict_init(values), dtype=np.float64).reshape(-1)
        if not np.isfinite(initial).all() or not np.all(initial == initial[0]):
            raise ValueError("Only a finite constant boosting initializer is portable")
        aggregation, base, rate = "additive_logit", float(initial[0]), float(estimator.learning_rate)
    elif family == "decision_tree":
        trees = [_export_tree(estimator, classifier=True)]
        aggregation, base, rate = "single", 0.0, 1.0
    else:
        trees = [_export_tree(tree, classifier=True) for tree in estimator.estimators_]
        aggregation, base, rate = "mean", 0.0, 1.0
    payload = {
        "schema_version": 1, "id": spec["id"], "family": family,
        "n_features": int(values.shape[1]), "classes": [0, 1],
        "input_dtype": "float32", "aggregation": aggregation,
        "base_margin": base, "learning_rate": rate, "trees": trees,
        "feature_importances": np.asarray(estimator.feature_importances_, dtype=np.float64).tolist(),
        "metadata": {
            "sklearn_version": SKLEARN_VERSION, "random_state": SEED,
            "training_rows": len(values), "class_counts": counts.tolist(),
            "target": "top_known_identity_correct", "loss_weighting": "binary_balanced",
            "class_sample_weights": class_weights.tolist(),
            "weighted_class_mass": [float(sample_weights[labels == c].sum()) for c in (0, 1)],
            "model_spec": spec, "fit_params": estimator.get_params(deep=False),
            "tree_count": len(trees), "feature_dtype": "float32",
            "leaf_probability_convention": "native_sklearn_1_8_weighted_proportions",
            "check_features_use": "translation_only_no_labels_no_fitting",
        },
    }
    fit_parity = _check_translation(payload, estimator, values)
    if check is not None and len(check):
        check_parity = _check_translation(payload, estimator, check)
    else:
        check_parity = {"performed": check is not None, "rows": 0,
                        "max_abs_error": 0.0, "probability_tie_boundary_disagreements": 0,
                        "tolerance": 1e-12}
    payload["metadata"]["export_parity"] = {"fit": fit_parity, "check_features": check_parity}
    return payload
