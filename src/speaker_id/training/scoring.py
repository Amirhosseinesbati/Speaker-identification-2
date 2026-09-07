"""Small transparent baseline; all fitted values use inner roles only."""
from __future__ import annotations

import numpy as np


def macro_f1_indices(truth: np.ndarray, prediction: np.ndarray, classes: int = 447) -> float:
    actual = np.bincount(truth, minlength=classes)
    predicted = np.bincount(prediction, minlength=classes)
    correct = np.bincount(truth[truth == prediction], minlength=classes)
    return float(np.divide(2 * correct, actual + predicted,
                           out=np.zeros(classes, dtype=float), where=(actual + predicted) != 0).mean())


def build_prototypes(embeddings: np.ndarray, label_indices: np.ndarray, known_classes: int = 446) -> np.ndarray:
    prototypes = []
    for label in range(1, known_classes + 1):
        selected = embeddings[label_indices == label]
        if not len(selected) or np.any(np.linalg.norm(selected, axis=1) < 1e-8):
            raise ValueError(f"Known class {label} has missing/invalid enrollment")
        vector = selected.mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            raise ValueError("Undefined mean prototype")
        prototypes.append(vector / norm)
    return np.asarray(prototypes, dtype=np.float32)


def fit_threshold(scores: np.ndarray, truth: np.ndarray, candidates: int = 201) -> tuple[float, list[dict]]:
    if scores.ndim != 2 or scores.shape[1] != 446 or len(scores) != len(truth) or not len(truth):
        raise ValueError("Calibration requires nonempty inner query scores over 446 known classes")
    if not np.isfinite(scores).all() or not np.any(truth == 0) or not np.any(truth > 0):
        raise ValueError("Calibration needs finite known and unknown inner queries")
    top = scores.max(axis=1)
    guess = scores.argmax(axis=1) + 1
    # Include both endpoints; equal logits choose index zero, matching the fixed
    # 447-way probability argmax (unknown is the first label).
    thresholds = np.unique(np.concatenate(([float(top.min()) - 1e-6],
                                           np.quantile(top, np.linspace(0, 1, candidates)),
                                           [float(top.max()) + 1e-6])))
    curve = [{"threshold": float(threshold), "inner_macro_f1_447": macro_f1_indices(truth, np.where(top > threshold, guess, 0))}
             for threshold in thresholds]
    # Prefer the more conservative rejection threshold when inner scores tie.
    best = max(curve, key=lambda item: (item["inner_macro_f1_447"], item["threshold"]))
    return best["threshold"], curve


def score_probabilities(scores: np.ndarray, threshold: float, temperature: float = 0.05,
                        valid: np.ndarray | None = None) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 446 or not np.isfinite(scores).all() or temperature <= 0 or not np.isfinite(threshold):
        raise ValueError("Invalid 447-class scoring inputs")
    logits = np.column_stack((np.full(len(scores), threshold), scores)).astype(np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if valid is not None:
        probabilities[~valid] = 0.0
        probabilities[~valid, 0] = 1.0
    # These are a normalized scoring distribution, not a claim of calibrated
    # posterior probabilities; threshold is tuned to the actual argmax decision.
    return probabilities
