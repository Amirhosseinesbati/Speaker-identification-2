"""Small, explicit open-set scorers calibrated exclusively on inner queries."""
from __future__ import annotations

import numpy as np

from speaker_id.training.scoring import build_prototypes, macro_f1_indices


def known_scores(queries: np.ndarray, references: np.ndarray, targets: np.ndarray,
                 method: str, classes: int = 446) -> np.ndarray:
    if method == "prototype":
        return queries @ build_prototypes(references, targets, classes).T
    if method != "max_reference":
        raise ValueError("Unknown reference aggregation")
    similarities = queries @ references.T
    scores = []
    for label in range(1, classes + 1):
        selected = similarities[:, targets == label]
        if selected.shape[1] == 0:
            raise ValueError("Every known class requires a permitted reference")
        scores.append(selected.max(axis=1))
    return np.column_stack(scores)


def gate_scores(scores: np.ndarray, unknown_similarity: np.ndarray,
                unknown_weight: float, margin_weight: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    unknown_similarity = np.asarray(unknown_similarity, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[1] < 2 or len(scores) != len(unknown_similarity)
            or not np.isfinite(scores).all() or not np.isfinite(unknown_similarity).all()
            or unknown_weight < 0 or margin_weight < 0):
        raise ValueError("Invalid reference/gate scores")
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    top = top_two[:, 1]
    margin = top - top_two[:, 0]
    return top - unknown_weight * unknown_similarity + margin_weight * margin


def calibrate_gate(scores: np.ndarray, truth: np.ndarray, unknown_similarity: np.ndarray,
                   unknown_weights: list[float], margin_weights: list[float],
                   candidates: int = 201, classes: int = 447) -> tuple[dict, list[dict]]:
    """The API accepts calibration data only; outer labels are never an input."""
    if (len(scores) != len(truth) or not np.any(truth == 0) or not np.any(truth > 0)
            or candidates < 2 or not unknown_weights or not margin_weights):
        raise ValueError("Calibration needs both known and unknown queries and a fixed grid")
    guess = scores.argmax(axis=1) + 1
    curve = []
    for unknown_weight in unknown_weights:
        for margin_weight in margin_weights:
            gate = gate_scores(scores, unknown_similarity, unknown_weight, margin_weight)
            thresholds = np.unique(np.concatenate(([float(gate.min()) - 1e-6],
                np.quantile(gate, np.linspace(0, 1, candidates)), [float(gate.max()) + 1e-6])))
            for threshold in thresholds:
                predicted = np.where(gate > threshold, guess, 0)
                curve.append({"unknown_weight": float(unknown_weight), "margin_weight": float(margin_weight),
                              "threshold": float(threshold),
                              "inner_macro_f1_447": macro_f1_indices(truth, predicted, classes)})
    # Tied inner scores prefer fewer added terms, then conservative rejection.
    best = max(curve, key=lambda row: (row["inner_macro_f1_447"], -row["unknown_weight"],
                                      -row["margin_weight"], row["threshold"]))
    return dict(best), curve


def reference_probabilities(scores: np.ndarray, unknown_similarity: np.ndarray,
                            calibration: dict, valid: np.ndarray, temperature: float = .05) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite")
    gate = gate_scores(scores, unknown_similarity, calibration["unknown_weight"], calibration["margin_weight"])
    # Relative known logits retain the original known ranking. The unknown logit
    # is zero exactly at the gate threshold, so index-zero argmax resolves ties.
    relative_known = scores.astype(np.float64) - scores.max(axis=1, keepdims=True).astype(np.float64)
    logits = np.column_stack((calibration["threshold"] - gate, relative_known)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~valid] = 0
    probabilities[~valid, 0] = 1
    return probabilities
