"""Deterministic paired cluster bootstrap over whole content groups.

The resampling unit is one complete content group.  A draw repeats every row in
that group, including groups whose rows have different true labels.  This is a
plain (unstratified) one-stage cluster bootstrap: each replicate draws exactly
the observed number of groups, with replacement, from all observed groups.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from typing import Sequence

import numpy as np


SCHEMA_VERSION = "paired-whole-content-group-cluster-bootstrap-v1"
RNG_ALGORITHM = "numpy.random.PCG64"
QUANTILE_METHOD = "linear"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float64_array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_index_vector(
        value: np.ndarray, name: str, row_count: int | None,
        class_count: int) -> np.ndarray:
    _require(isinstance(value, np.ndarray), f"{name} must be a numpy array")
    _require(value.ndim == 1, f"{name} must be one-dimensional")
    _require(value.dtype.kind in "iu", f"{name} must use an integer dtype")
    _require(len(value) > 0, f"{name} must be nonempty")
    if row_count is not None:
        _require(len(value) == row_count, "Bootstrap vectors must be aligned")
    _require(
        bool(np.all(value >= 0)) and bool(np.all(value < class_count)),
        f"{name} indices must be in [0, class_count)",
    )
    return np.asarray(value, dtype=np.int64)


def _validate_group_ids(value: np.ndarray, row_count: int) -> list[str]:
    _require(isinstance(value, np.ndarray), "group_ids must be a numpy array")
    _require(value.ndim == 1, "group_ids must be one-dimensional")
    _require(len(value) == row_count, "Bootstrap vectors must be aligned")
    groups: list[str] = []
    for item in value:
        _require(
            isinstance(item, (str, np.str_)),
            "Every group ID must be a nonempty string",
        )
        group = str(item)
        _require(
            bool(group) and group == group.strip(),
            "Every group ID must be a nonempty, trimmed string",
        )
        groups.append(group)
    return groups


def _validate_configuration(
        class_count: int, seed: int, replicates: int,
        quantiles: Sequence[Real]) -> tuple[float, float]:
    _require(
        type(class_count) is int and 1 <= class_count <= np.iinfo(np.int32).max,
        "class_count must be a positive integer",
    )
    _require(
        type(seed) is int and 0 <= seed <= np.iinfo(np.uint64).max,
        "seed must be an integer in the uint64 range",
    )
    _require(
        type(replicates) is int and replicates >= 100,
        "replicates must be an integer of at least 100",
    )
    _require(
        isinstance(quantiles, Sequence)
        and not isinstance(quantiles, (str, bytes))
        and len(quantiles) == 2,
        "quantiles must contain exactly two numeric values",
    )
    lower, upper = quantiles
    _require(
        isinstance(lower, Real) and not isinstance(lower, (bool, np.bool_))
        and isinstance(upper, Real) and not isinstance(upper, (bool, np.bool_)),
        "quantiles must contain exactly two numeric values",
    )
    lower, upper = float(lower), float(upper)
    _require(
        math.isfinite(lower) and math.isfinite(upper)
        and 0.0 < lower < upper < 1.0,
        "quantiles must be finite and satisfy 0 < lower < upper < 1",
    )
    return lower, upper


def _macro_f1_from_weights(
        truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray,
        class_count: int) -> float:
    support = np.bincount(truth, weights=weights, minlength=class_count)
    predicted = np.bincount(
        prediction, weights=weights, minlength=class_count)
    correct_mask = truth == prediction
    true_positive = np.bincount(
        truth[correct_mask], weights=weights[correct_mask],
        minlength=class_count,
    )
    denominator = support + predicted
    per_class = np.divide(
        2.0 * true_positive, denominator,
        out=np.zeros(class_count, dtype=np.float64),
        where=denominator > 0,
    )
    return float(per_class.mean())


def _canonical_group_layout(
        ordered_group_ids: list[str], group_to_rows: dict[str, list[int]],
        truth: np.ndarray, before: np.ndarray, after: np.ndarray) -> list[dict]:
    layout = []
    for group_id in ordered_group_ids:
        triples = sorted(
            (int(truth[index]), int(before[index]), int(after[index]))
            for index in group_to_rows[group_id]
        )
        layout.append({"group_id": group_id, "rows": triples})
    return layout


def paired_whole_group_cluster_bootstrap(
        truth: np.ndarray, before_prediction: np.ndarray,
        after_prediction: np.ndarray, group_ids: np.ndarray, *,
        class_count: int, seed: int, replicates: int,
        quantiles: Sequence[Real] = (0.025, 0.975)) -> dict:
    """Return paired Macro-F1 deltas and an auditable deterministic receipt.

    Groups are sorted by their exact string ID to define a stable RNG draw
    order.  For every replicate, ``group_count`` group indices are sampled with
    replacement from all ``group_count`` groups.  The resulting multiplicity is
    applied to every row of each selected group before both predictions are
    scored.  No label-based strata are constructed.
    """
    lower_quantile, upper_quantile = _validate_configuration(
        class_count, seed, replicates, quantiles)
    actual = _validate_index_vector(truth, "truth", None, class_count)
    row_count = len(actual)
    before = _validate_index_vector(
        before_prediction, "before_prediction", row_count, class_count)
    after = _validate_index_vector(
        after_prediction, "after_prediction", row_count, class_count)
    groups = _validate_group_ids(group_ids, row_count)

    group_to_rows: dict[str, list[int]] = {}
    for row_index, group_id in enumerate(groups):
        group_to_rows.setdefault(group_id, []).append(row_index)
    ordered_group_ids = sorted(group_to_rows)
    group_count = len(ordered_group_ids)
    _require(group_count > 0, "At least one content group is required")

    group_index_by_id = {
        group_id: index for index, group_id in enumerate(ordered_group_ids)}
    row_group_indices = np.fromiter(
        (group_index_by_id[group_id] for group_id in groups),
        dtype=np.int64, count=row_count,
    )
    group_sizes = np.bincount(
        row_group_indices, minlength=group_count).astype(np.int64, copy=False)

    mixed_group_ids: list[str] = []
    mixed_group_rows = 0
    maximum_truth_labels_in_group = 1
    for group_id in ordered_group_ids:
        indices = np.asarray(group_to_rows[group_id], dtype=np.int64)
        label_count = int(np.unique(actual[indices]).size)
        maximum_truth_labels_in_group = max(
            maximum_truth_labels_in_group, label_count)
        if label_count > 1:
            mixed_group_ids.append(group_id)
            mixed_group_rows += len(indices)

    unit_weights = np.ones(row_count, dtype=np.float64)
    before_point = _macro_f1_from_weights(
        actual, before, unit_weights, class_count)
    after_point = _macro_f1_from_weights(
        actual, after, unit_weights, class_count)

    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(replicates, dtype=np.float64)
    sampled_row_counts = np.empty(replicates, dtype=np.int64)
    sampled_unique_group_counts = np.empty(replicates, dtype=np.int64)
    draws_hasher = hashlib.sha256()
    for replicate in range(replicates):
        draws = rng.integers(
            0, group_count, size=group_count, dtype=np.int64)
        draws_hasher.update(
            np.ascontiguousarray(draws, dtype="<i8").tobytes(order="C"))
        group_multiplicity = np.bincount(draws, minlength=group_count)
        row_weights = group_multiplicity[row_group_indices].astype(
            np.float64, copy=False)
        sampled_row_counts[replicate] = int(row_weights.sum())
        sampled_unique_group_counts[replicate] = int(
            np.count_nonzero(group_multiplicity))
        deltas[replicate] = (
            _macro_f1_from_weights(actual, after, row_weights, class_count)
            - _macro_f1_from_weights(actual, before, row_weights, class_count)
        )

    lower = float(np.quantile(
        deltas, lower_quantile, method=QUANTILE_METHOD))
    upper = float(np.quantile(
        deltas, upper_quantile, method=QUANTILE_METHOD))
    group_layout = _canonical_group_layout(
        ordered_group_ids, group_to_rows, actual, before, after)
    data_sha256 = _canonical_json_sha256({
        "class_count": class_count,
        "groups": group_layout,
    })
    configuration = {
        "class_count": class_count,
        "seed": seed,
        "replicates": replicates,
        "quantiles": [lower_quantile, upper_quantile],
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "method": "paired_whole_content_group_cluster_bootstrap",
        "paired": True,
        "label_stratified": False,
        "sampling": {
            "unit": "whole_content_group",
            "population": "all_observed_content_groups",
            "replacement": True,
            "groups_drawn_per_replicate": group_count,
            "group_order": "lexicographic_exact_string_id",
            "rng": RNG_ALGORITHM,
        },
        "configuration": configuration,
        "counts": {
            "rows": row_count,
            "groups": group_count,
            "true_classes_observed": int(np.unique(actual).size),
            "minimum_rows_per_group": int(group_sizes.min()),
            "maximum_rows_per_group": int(group_sizes.max()),
            "mixed_label_groups": len(mixed_group_ids),
            "rows_in_mixed_label_groups": int(mixed_group_rows),
            "maximum_true_labels_in_group": maximum_truth_labels_in_group,
            "minimum_sampled_rows": int(sampled_row_counts.min()),
            "maximum_sampled_rows": int(sampled_row_counts.max()),
            "minimum_unique_groups_sampled": int(
                sampled_unique_group_counts.min()),
            "maximum_unique_groups_sampled": int(
                sampled_unique_group_counts.max()),
        },
        "mixed_groups": {
            "preserved_as_indivisible_clusters": True,
            "ids_sha256": _canonical_json_sha256(mixed_group_ids),
        },
        "point_estimate": {
            "before_macro_f1": before_point,
            "after_macro_f1": after_point,
            "delta_macro_f1": after_point - before_point,
        },
        "bootstrap_distribution": {
            "mean_delta_macro_f1": float(deltas.mean()),
            "median_delta_macro_f1": float(np.median(deltas)),
            "minimum_delta_macro_f1": float(deltas.min()),
            "maximum_delta_macro_f1": float(deltas.max()),
        },
        "confidence_interval": {
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "coverage": upper_quantile - lower_quantile,
            "lower_delta_macro_f1": lower,
            "upper_delta_macro_f1": upper,
            "quantile_method": QUANTILE_METHOD,
        },
        "hashes": {
            "semantic_input_sha256": data_sha256,
            "configuration_sha256": _canonical_json_sha256(configuration),
            "sampled_group_draws_sha256": draws_hasher.hexdigest(),
            "deltas_float64_le_sha256": _float64_array_sha256(deltas),
        },
        "hash_contract": (
            "receipt_sha256 is SHA-256 of canonical UTF-8 JSON for this "
            "receipt with receipt_sha256 omitted; keys sorted, compact "
            "separators, ensure_ascii=false, allow_nan=false"
        ),
    }
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    deltas.flags.writeable = False
    return {"deltas": deltas, "receipt": receipt}
