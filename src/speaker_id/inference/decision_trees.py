"""Portable binary decision-tree inference with no scikit-learn dependency.

The payload contains numeric node arrays, never a pickle or executable object.
All feature comparisons reproduce scikit-learn's float32 input conversion.
"""
from __future__ import annotations

import json
import math

import numpy as np


_PAYLOAD_KEYS = {
    "schema_version", "id", "family", "n_features", "classes", "input_dtype",
    "aggregation", "base_margin", "learning_rate", "trees",
    "feature_importances", "metadata",
}
_TREE_KEYS = {"children_left", "children_right", "feature", "threshold", "value"}
_AGGREGATIONS = {
    "decision_tree": "single",
    "random_forest": "mean",
    "extra_trees": "mean",
    "gradient_boosting": "additive_logit",
}


def feature_matrix(values, n_features=None):
    """Validate numeric, finite input before and after float32 conversion."""
    try:
        original = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be a numeric matrix") from exc
    if (original.ndim != 2 or original.shape[1] < 1
            or original.dtype.kind not in "biuf"):
        raise ValueError("Features must be a nonempty-width real numeric matrix")
    if n_features is not None and original.shape[1] != n_features:
        raise ValueError("Feature width differs from the exported model")
    if not np.isfinite(original).all():
        raise ValueError("Features must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(original, dtype=np.float32, order="C")
    if not np.isfinite(result).all():
        raise ValueError("Features overflow float32")
    return result


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite JSON number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite JSON number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite JSON number")
    return converted


def _integer_vector(values, name, length=None):
    if (not isinstance(values, list)
            or any(type(value) is not int for value in values)
            or (length is not None and len(values) != length)):
        raise ValueError(f"{name} must be an aligned JSON integer list")
    try:
        return np.asarray(values, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid integer") from exc


def _numeric_vector(values, name, length):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} must be an aligned JSON number list")
    return np.asarray([_number(value, name) for value in values], dtype=np.float64)


def _validate_tree(tree, n_features, probabilities):
    if not isinstance(tree, dict) or set(tree) != _TREE_KEYS:
        raise ValueError("A tree must contain exactly the declared node arrays")
    left = _integer_vector(tree["children_left"], "children_left")
    count = len(left)
    if count < 1:
        raise ValueError("A tree must have at least one node")
    right = _integer_vector(tree["children_right"], "children_right", count)
    feature = _integer_vector(tree["feature"], "feature", count)
    threshold = _numeric_vector(tree["threshold"], "threshold", count)
    value = _numeric_vector(tree["value"], "value", count)
    if probabilities and ((value < 0).any() or (value > 1).any()):
        raise ValueError("Classification node probabilities must be in [0, 1]")
    leaves = left == -1
    if not np.array_equal(leaves, right == -1):
        raise ValueError("Both children must mark a leaf together")
    if (feature[leaves] != -2).any() or (threshold[leaves] != -2).any():
        raise ValueError("Leaf feature and threshold sentinels must be -2")
    branch = np.flatnonzero(~leaves)
    if ((left[branch] < 0).any() or (right[branch] < 0).any()
            or (left[branch] >= count).any() or (right[branch] >= count).any()
            or (left[branch] == right[branch]).any()
            or (feature[branch] < 0).any()
            or (feature[branch] >= n_features).any()):
        raise ValueError("Invalid branch child or feature index")
    # A rooted tree has one parent per nonroot node. Also visit every node:
    # parent counts alone do not exclude a disconnected cyclic component.
    parents = np.bincount(np.r_[left[branch], right[branch]], minlength=count)
    if parents[0] != 0 or (parents[1:] != 1).any():
        raise ValueError("Tree nodes must have unique parents and root zero")
    seen = np.zeros(count, dtype=bool)
    stack = [0]
    while stack:
        node = stack.pop()
        if seen[node]:
            raise ValueError("Tree contains a cycle or repeated child")
        seen[node] = True
        if left[node] != -1:
            stack.extend((int(left[node]), int(right[node])))
    if not seen.all():
        raise ValueError("Tree has unreachable nodes or a disconnected cycle")
    return left, right, feature, threshold, value


def validate_export(payload):
    """Validate schema and complete tree topology; return prepared node arrays."""
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("Invalid exported decision-model schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported decision-model schema version")
    if not isinstance(payload["id"], str) or not payload["id"]:
        raise ValueError("Model id must be a nonempty string")
    family = payload["family"]
    if not isinstance(family, str) or family not in _AGGREGATIONS:
        raise ValueError("Unknown decision-model family")
    n_features = payload["n_features"]
    if type(n_features) is not int or n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if payload["input_dtype"] != "float32":
        raise ValueError("Decision models require float32 feature comparisons")
    classes = payload["classes"]
    if (not isinstance(classes, list) or len(classes) != 2
            or any(type(value) is not int for value in classes) or classes != [0, 1]):
        raise ValueError("Classes must be ordered binary correctness labels [0, 1]")
    aggregation = payload["aggregation"]
    if aggregation != _AGGREGATIONS[family]:
        raise ValueError("Model family and aggregation disagree")
    base = _number(payload["base_margin"], "base_margin")
    rate = _number(payload["learning_rate"], "learning_rate")
    if rate <= 0:
        raise ValueError("learning_rate must be positive")
    if aggregation != "additive_logit" and (base != 0 or rate != 1):
        raise ValueError("Classification trees cannot have a boosting offset/rate")
    trees = payload["trees"]
    if not isinstance(trees, list) or not trees:
        raise ValueError("An exported ensemble must contain at least one tree")
    if aggregation == "single" and len(trees) != 1:
        raise ValueError("A decision tree must contain exactly one tree")
    importance = _numeric_vector(payload["feature_importances"],
                                 "feature_importances", n_features)
    total = importance.sum()
    if (importance < 0).any() or (total != 0 and not np.isclose(total, 1, rtol=0, atol=1e-8)):
        raise ValueError("Feature importances must be nonnegative and sum to zero or one")
    if not isinstance(payload["metadata"], dict):
        raise ValueError("Model metadata must be a JSON object")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Payload must contain finite JSON-serializable values") from exc
    arrays = [_validate_tree(tree, n_features, aggregation != "additive_logit")
              for tree in trees]
    return arrays


def _tree_values(arrays, values):
    left, right, features, thresholds, outputs = arrays
    nodes = np.zeros(len(values), dtype=np.int64)
    active = np.flatnonzero(left[nodes] != -1)
    while len(active):
        branch = nodes[active]
        go_left = values[active, features[branch]] <= thresholds[branch]
        nodes[active] = np.where(go_left, left[branch], right[branch])
        active = active[left[nodes[active]] != -1]
    return outputs[nodes]


def predict_export(payload, X):
    """Return float64 P(top known identity is correct) from JSON node arrays."""
    arrays = validate_export(payload)
    values = feature_matrix(X, payload["n_features"])
    aggregation = payload["aggregation"]
    try:
        with np.errstate(over="raise", invalid="raise"):
            if aggregation == "single":
                probabilities = _tree_values(arrays[0], values)
            elif aggregation == "mean":
                probabilities = np.zeros(len(values), dtype=np.float64)
                for tree in arrays:
                    probabilities += _tree_values(tree, values)
                probabilities /= len(arrays)
            else:
                raw = np.full(len(values), payload["base_margin"], dtype=np.float64)
                for tree in arrays:
                    raw += payload["learning_rate"] * _tree_values(tree, values)
                probabilities = np.empty(len(values), dtype=np.float64)
                positive = raw >= 0
                probabilities[positive] = 1 / (1 + np.exp(-raw[positive]))
                exp_negative = np.exp(raw[~positive])
                probabilities[~positive] = exp_negative / (1 + exp_negative)
    except FloatingPointError as exc:
        raise ValueError("Nonfinite arithmetic in decision model") from exc
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite decision-model probability")
    return np.asarray(probabilities, dtype=np.float64)
