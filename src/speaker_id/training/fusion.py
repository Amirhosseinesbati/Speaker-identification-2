"""Frozen dual-view reference fusion; policy selection accepts INNER scores only."""
from __future__ import annotations

import numpy as np

from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.reference_scoring import calibrate_gate


CANDIDATES = (
    {"id": "full", "kind": "source_control", "short_weight": 0.0},
    {"id": "short", "kind": "source_control", "short_weight": 1.0},
    *({"id": f"pair_{int(w * 100):03}", "kind": "paired_reference", "short_weight": w}
      for w in (.25, .5, .75)),
    *({"id": f"score_{int(w * 100):03}", "kind": "independent_scores", "short_weight": w}
      for w in (.25, .5, .75)),
)
UNKNOWN_WEIGHTS = [0.0, .25, .5, .75, 1.0]
MARGIN_WEIGHTS = [0.0, .5]


def weighted_embedding_pair(short, full, valid, short_weight: float) -> np.ndarray:
    """Cosine of this concatenation equals weighted cosine of the SAME pair."""
    short, full, valid = np.asarray(short), np.asarray(full), np.asarray(valid)
    if (short.ndim != 2 or short.shape != full.shape or valid.shape != (len(short),)
            or valid.dtype != np.bool_ or not np.isfinite(short).all() or not np.isfinite(full).all()
            or not np.isfinite(short_weight) or not 0 <= short_weight <= 1):
        raise ValueError("Dual-view embeddings, validity and weight must align")
    views = []
    for matrix in (short, full):
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(valid & (norms <= 1e-8)) or np.any(matrix[~valid]):
            raise ValueError("Invalid zero/nonzero embedding identity across views")
        normalized = np.zeros_like(matrix, dtype=np.float32)
        normalized[valid] = matrix[valid] / norms[valid, None]
        views.append(normalized)
    return np.concatenate((np.sqrt(short_weight) * views[0],
                           np.sqrt(1 - short_weight) * views[1]), axis=1).astype(np.float32)


def require_aligned_scores(first: dict, second: dict) -> None:
    for key in ("calibration_indices", "outer_indices", "outer_valid"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Dual-view score alignment differs: {key}")
    if first["known_labels"] != second["known_labels"]:
        raise ValueError("Dual-view known class column order differs")
    for key in ("reference_indices", "reference_groups"):
        if first["provenance"][key] != second["provenance"][key]:
            raise ValueError("Dual-view permitted reference pools differ")


def independent_score_pair(short: dict, full: dict, short_weight: float) -> dict:
    require_aligned_scores(short, full)
    if not np.isfinite(short_weight) or not 0 <= short_weight <= 1:
        raise ValueError("Fusion weight must lie in [0, 1]")
    result = dict(full)
    for key in ("inner_known_scores", "inner_unknown_similarity", "outer_known_scores", "outer_unknown_similarity"):
        if short[key].shape != full[key].shape:
            raise ValueError("Dual-view score dimensions differ")
        result[key] = short_weight * short[key] + (1 - short_weight) * full[key]
    result["provenance"] = {**full["provenance"], "fusion": "independent_max_reference_cosine_scores",
                            "short_weight": short_weight}
    return result


def dual_view_scores(full, short, valid, manifest, folds, outer: int, *, classes: int = 446) -> dict:
    """Outer labels are never accessed by this or the shared crossfit module."""
    result = {"full": crossfit_scores(full, valid, manifest, folds, outer, "max_reference", classes),
              "short": crossfit_scores(short, valid, manifest, folds, outer, "max_reference", classes)}
    require_aligned_scores(result["short"], result["full"])
    for candidate in CANDIDATES[2:]:
        weight = candidate["short_weight"]
        if candidate["kind"] == "paired_reference":
            pairs = weighted_embedding_pair(short, full, valid, weight)
            scores = crossfit_scores(pairs, valid, manifest, folds, outer, "max_reference", classes)
            require_aligned_scores(result["full"], scores)
            scores["provenance"] = {**scores["provenance"], "fusion": "sqrt_weighted_unit_embedding_concatenation",
                                     "short_weight": weight}
            result[candidate["id"]] = scores
        else:
            result[candidate["id"]] = independent_score_pair(result["short"], result["full"], weight)
    return result


def select_inner_policy(inner_candidates: dict, inner_truth: np.ndarray, *, classes: int = 447) -> tuple[dict, dict]:
    """Jointly choose view/method/weight/gate; no outer arrays or labels accepted.

    Exact F1 ties prefer the fixed candidate order: full, short, then paired and
    independent mixtures from .25 to .75 short weight. Each candidate retains
    the existing gate's lower-coefficient/conservative-threshold tie rule.
    """
    if set(inner_candidates) != {item["id"] for item in CANDIDATES}:
        raise ValueError("Exactly the eight preregistered inner candidates are required")
    calibrations = {}
    for candidate in CANDIDATES:
        entry = inner_candidates[candidate["id"]]
        if set(entry) != {"known", "unknown"}:
            raise ValueError("Inner selection accepts known and unknown INNER scores only")
        selected, curve = calibrate_gate(entry["known"], inner_truth, entry["unknown"],
                                         UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201, classes)
        calibrations[candidate["id"]] = {"candidate": dict(candidate), "selected": selected, "curve": curve}
    best = max(CANDIDATES, key=lambda item: calibrations[item["id"]]["selected"]["inner_macro_f1_447"])
    return {**best, "calibration": dict(calibrations[best["id"]]["selected"]),
            "selection_data": "outer_training_content_group_excluded_inner_queries_only",
            "candidate_tie_order": [item["id"] for item in CANDIDATES]}, calibrations
