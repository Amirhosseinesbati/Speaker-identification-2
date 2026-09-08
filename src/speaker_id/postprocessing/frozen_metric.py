"""Known-speaker, group-balanced partial WCCN for allowed training rows only.

This module cannot infer an evaluation split. The caller must exclude every
heldout group from the complete fit population, not only from the gallery.
No sklearn, encoder, GPU or scoring backend is imported.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Integral

import numpy as np


METRIC_SPECS = [
    {"id": "identity", "kind": "identity"},
    {"id": "centering_only", "kind": "centering"},
    *[{"id": f"wccn_l{int(shrinkage * 100):02d}_p{int(power * 100):02d}",
       "kind": "partial_wccn", "shrinkage": shrinkage, "power": power}
      for shrinkage in (0.5, 0.9) for power in (0.25, 0.5)],
]
EIGENVALUE_FLOOR_RATIO = 1e-6
GROUP_MEAN_NORM_FLOOR = 1e-12
TRANSFORM_NORM_FLOOR = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _values(values):
    values = np.asarray(values)
    _require(values.ndim == 2 and values.shape[1] > 0 and values.dtype == np.float32
             and np.isfinite(values).all(), "Expected a finite float32 matrix with positive feature width")
    return values


def _labels_and_groups(labels, groups, rows):
    label_values = np.asarray(labels, dtype=object)
    group_values = np.asarray(groups, dtype=object)
    _require(label_values.shape == (rows,) and group_values.shape == (rows,),
             "Labels and content groups must align with allowed training rows")
    _require(all(isinstance(g, str) and g for g in group_values),
             "Content group identifiers must be nonempty strings")
    if all(isinstance(label, str) and label for label in label_values):
        labels = [str(label) for label in label_values]
        unknown = "unknown"
    elif all(isinstance(label, Integral) and not isinstance(label, (bool, np.bool_))
             and int(label) >= 0 for label in label_values):
        labels = [int(label) for label in label_values]
        unknown = 0
    else:
        raise ValueError("Labels must be uniform nonempty strings or nonnegative integer indices")
    groups = [str(group) for group in group_values]
    group_label = {}
    for label, group in zip(labels, groups):
        if group in group_label:
            _require(group_label[group] == label, "A content group has conflicting speaker labels")
        group_label[group] = label
    return labels, groups, unknown


def _normalize_nonzero(values):
    """Normalize in float64; exact all-zero input rows remain exactly zero."""
    values = np.asarray(values, dtype=np.float64)
    active = np.any(values != 0, axis=1)
    output = np.zeros_like(values)
    norms = np.linalg.norm(values[active], axis=1)
    _require(np.isfinite(norms).all() and np.all(norms > 0), "Nonzero row norm is undefined")
    output[active] = values[active] / norms[:, None]
    return output, active


def _sha_array(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _canonical_groups(values, labels, groups, unknown):
    active = np.any(values != 0, axis=1)
    grouped = {}
    for i, (label, group) in enumerate(zip(labels, groups)):
        if label != unknown and active[i]:
            grouped.setdefault((label, group), []).append(i)
    by_class, records = {}, []
    for (label, group), positions in sorted(grouped.items()):
        # Exact repeated files cannot alter their content group's mean. Sorting
        # unique rows also fixes summation order under input/file permutation.
        unique = np.unique(values[positions], axis=0)
        unique = np.where(unique == 0, np.float32(0), unique).astype(np.float32)
        normalized, _ = _normalize_nonzero(unique)
        mean = normalized.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        _require(np.isfinite(norm) and norm > GROUP_MEAN_NORM_FLOOR,
                 "A known content-group mean is degenerate or cancels")
        vector = mean / norm
        by_class.setdefault(label, []).append(vector)
        records.append({"label": label, "group": group, "input_rows": len(positions),
                        "unique_nonzero_vectors": len(unique),
                        "unique_vectors_sha256": hashlib.sha256(
                            np.asarray(unique, dtype="<f4").tobytes()).hexdigest(),
                        "group_vector_sha256": _sha_array(vector)})
    return by_class, records, active


def fit_transform(values, labels, groups, spec):
    """Fit a numeric mean/matrix payload from caller-authorized training rows.

    Inputs are float32 frozen vectors of any dimension, with uniformly typed
    string speaker labels ('unknown' excluded) or integer labels (0 excluded).
    Exact zero rows do not enter fitting. Each nonzero row is length normalized
    before computing a deduplicated normalized group mean. The returned dict has
    JSON-only metadata plus float64 mean and matrix arrays; no transformed query
    scores or predictions are generated.
    """
    _require(isinstance(spec, dict) and any(spec == entry for entry in METRIC_SPECS),
             "Metric spec must match one of the six fixed recipes")
    spec = deepcopy(spec)
    values = _values(values)
    labels, groups, unknown = _labels_and_groups(labels, groups, len(values))
    dimension = values.shape[1]
    by_class, records, active = _canonical_groups(values, labels, groups, unknown)
    known_labels = sorted(by_class)
    contributing = [label for label in known_labels if len(by_class[label]) >= 2]
    if spec["kind"] != "identity":
        _require(bool(known_labels), "A learned centering/metric needs nonzero known-speaker groups")
    mean = np.zeros(dimension, dtype=np.float64)
    matrix = np.eye(dimension, dtype=np.float64)
    diagnostics = None
    if spec["kind"] != "identity":
        means = [np.mean(np.asarray(by_class[label]), axis=0) for label in known_labels]
        mean = np.mean(np.asarray(means), axis=0)
    if spec["kind"] == "partial_wccn":
        _require(bool(contributing), "Within-speaker covariance needs a class with two independent groups")
        within = np.zeros((dimension, dimension), dtype=np.float64)
        for label in contributing:
            group_vectors = np.asarray(by_class[label], dtype=np.float64)
            residuals = group_vectors - group_vectors.mean(axis=0)
            within += (residuals.T @ residuals) / (len(group_vectors) - 1)
        within /= len(contributing)
        within = (within + within.T) / 2
        nu = float(np.trace(within) / dimension)
        _require(np.isfinite(nu) and nu > 0, "Within-speaker variance must be positive and finite")
        shrinkage = spec["shrinkage"]
        regularized = (1 - shrinkage) * within + shrinkage * nu * np.eye(dimension)
        regularized = (regularized + regularized.T) / 2
        eigenvalues, directions = np.linalg.eigh(regularized)
        floor = EIGENVALUE_FLOOR_RATIO * nu
        _require(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0
                 and np.isfinite(directions).all() and floor > 0,
                 "Shrunk covariance is not finite positive definite")
        used = np.maximum(eigenvalues, floor)
        scales = used ** (-spec["power"])
        matrix = (directions * scales) @ directions.T
        matrix = (matrix + matrix.T) / 2
        _require(np.isfinite(matrix).all(), "Metric matrix is nonfinite")
        # Numeric rank is a diagnostic, never a selection or projection knob.
        within_eigenvalues = np.linalg.eigvalsh(within)
        largest = float(np.max(np.abs(within_eigenvalues), initial=0))
        rank_tolerance = max(1, dimension) * np.finfo(np.float64).eps * largest
        diagnostics = {
            "within_covariance_sha256": _sha_array(within),
            "within_trace": float(np.trace(within)), "average_within_variance": nu,
            "residual_degrees_of_freedom": sum(len(by_class[c]) - 1 for c in contributing),
            "within_numeric_rank": int(np.sum(within_eigenvalues > rank_tolerance)),
            "within_rank_tolerance": rank_tolerance,
            "within_min_eigenvalue": float(within_eigenvalues.min()),
            "within_max_eigenvalue": float(within_eigenvalues.max()),
            "regularized_min_eigenvalue": float(eigenvalues.min()),
            "regularized_max_eigenvalue": float(eigenvalues.max()),
            "regularized_condition_number": float(eigenvalues.max() / eigenvalues.min()),
            "used_condition_number": float(used.max() / used.min()),
            "eigenvalue_floor": floor, "floor_applied_count": int(np.sum(eigenvalues < floor)),
        }
    metadata = {
        "schema_version": 1, "spec": spec, "n_features": dimension,
        "fit_scope": "caller_supplied_allowed_training_only_no_split_inferred",
        "input_normalization": "float64_L2_per_nonzero_row_before_centering",
        "group_weighting": "one_normalized_mean_of_unique_float32_rows_per_known_content_group",
        "class_mean_weighting": "equal_weight_per_known_speaker",
        "within_covariance_weighting": "equal_weight_per_known_speaker_with_at_least_two_groups",
        "unknown_label": unknown, "unknown_rows_ignored": sum(label == unknown for label in labels),
        "known_zero_rows_ignored": sum(label != unknown and not active[i] for i, label in enumerate(labels)),
        "input_rows": len(values), "known_labels": known_labels,
        "known_groups": records, "known_group_count": len(records),
        "known_groups_per_class": [{"label": c, "groups": len(by_class[c])} for c in known_labels],
        "within_contributing_labels": contributing,
        "within_contributing_classes": len(contributing),
        "diagnostics": diagnostics, "eigenvalue_floor_ratio": EIGENVALUE_FLOOR_RATIO,
        "group_mean_norm_floor": GROUP_MEAN_NORM_FLOOR,
        "transformed_norm_floor": TRANSFORM_NORM_FLOOR,
        "zero_policy": "exact_input_zero_preserved_nonzero_degenerate_transform_rejected",
        "matrix_convention": "column_operator_z=A@(L2(x)-mean); rows_use_centered@A.T",
        "mean_sha256": _sha_array(mean), "matrix_sha256": _sha_array(matrix),
    }
    # Counts preserve raw provenance; this separate digest ignores duplicated
    # input rows and their ordering, binding only actual fitted group vectors.
    canonical = [{"label": r["label"], "group": r["group"],
                  "group_vector_sha256": r["group_vector_sha256"]} for r in records]
    metadata["canonical_fit_groups_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    json.dumps(metadata, allow_nan=False)
    return {"metadata": metadata, "mean": mean, "matrix": matrix}


def _validate_payload(payload):
    _require(isinstance(payload, dict) and set(payload) == {"metadata", "mean", "matrix"},
             "Invalid metric payload schema")
    metadata = payload["metadata"]
    _require(isinstance(metadata, dict) and type(metadata.get("schema_version")) is int
             and metadata["schema_version"] == 1, "Unsupported metric payload version")
    dimension = metadata.get("n_features")
    _require(type(dimension) is int and dimension > 0, "Invalid metric feature dimension")
    spec = metadata.get("spec")
    _require(isinstance(spec, dict) and any(spec == entry for entry in METRIC_SPECS),
             "Unregistered metric payload spec")
    mean, matrix = np.asarray(payload["mean"]), np.asarray(payload["matrix"])
    _require(mean.shape == (dimension,) and mean.dtype == np.float64
             and matrix.shape == (dimension, dimension) and matrix.dtype == np.float64
             and np.isfinite(mean).all() and np.isfinite(matrix).all(), "Invalid metric numeric arrays")
    _require(np.allclose(matrix, matrix.T, rtol=0, atol=1e-12),
             "Metric matrix must be symmetric")
    _require(metadata.get("mean_sha256") == _sha_array(mean)
             and metadata.get("matrix_sha256") == _sha_array(matrix),
             "Metric arrays differ from their fitted receipt")
    _require(metadata.get("input_normalization") == "float64_L2_per_nonzero_row_before_centering"
             and metadata.get("transformed_norm_floor") == TRANSFORM_NORM_FLOOR,
             "Metric normalization convention differs")
    if spec["kind"] != "partial_wccn":
        _require(np.array_equal(matrix, np.eye(dimension)), "A control metric must have an identity matrix")
    if spec["kind"] == "identity":
        _require(not np.any(mean), "Identity control cannot have a fitted mean")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Metric metadata must contain finite JSON values") from exc
    return metadata, mean, matrix


def transform(payload, values):
    """Apply the fitted metric and return normalized float32 rows, preserving zeros."""
    metadata, mean, matrix = _validate_payload(payload)
    values = _values(values)
    _require(values.shape[1] == metadata["n_features"], "Query and metric feature widths differ")
    normalized, active = _normalize_nonzero(values)
    result = np.zeros(values.shape, dtype=np.float32)
    if not np.any(active):
        return result
    changed = (normalized[active] - mean) @ matrix.T
    norms = np.linalg.norm(changed, axis=1)
    _require(np.isfinite(changed).all() and np.isfinite(norms).all()
             and np.all(norms > TRANSFORM_NORM_FLOOR),
             "A nonzero input becomes a degenerate or nonfinite transformed vector")
    result[active] = (changed / norms[:, None]).astype(np.float32)
    _require(np.isfinite(result).all(), "Transformed float32 output is nonfinite")
    return result
