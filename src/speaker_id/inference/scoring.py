"""NumPy-only deployed counterpart of the calibrated reference scorer."""
from __future__ import annotations

import math
import numpy as np


def validate_gallery(gallery: dict, *, classes: int = 446, embedding_dim: int = 512) -> dict:
    if type(classes) is not int or classes < 2 or type(embedding_dim) is not int or embedding_dim < 1:
        raise ValueError("Invalid gallery dimensions")
    required = {"known_embeddings", "known_targets", "unknown_embeddings"}
    if set(gallery) != required:
        raise ValueError("Gallery must contain exactly the three declared reference arrays")
    known, targets, unknown = (np.asarray(gallery[key]) for key in
                               ("known_embeddings", "known_targets", "unknown_embeddings"))
    for name, values in (("known", known), ("unknown", unknown)):
        if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != embedding_dim
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError(f"Invalid {name} reference embeddings")
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-4):
            raise ValueError(f"{name} reference embeddings must already have unit norm")
    if targets.dtype != np.int64 or targets.shape != (len(known),):
        raise ValueError("Known targets must be an int64 vector aligned with references")
    if not np.array_equal(np.unique(targets), np.arange(1, classes + 1)):
        raise ValueError("Every known class must have a reference, without out-of-range targets")
    # Preserve the producer's once-normalized float32 bytes; do not renormalize.
    return {"known_embeddings": known, "known_targets": targets, "unknown_embeddings": unknown}


def validate_calibration(calibration: dict, *, require_inference: bool = False) -> dict:
    required = {"unknown_weight", "margin_weight", "threshold"}
    if not isinstance(calibration, dict) or not required.issubset(calibration):
        raise ValueError("Missing calibrated gate coefficients")
    for name in required | {"temperature"}:
        value = calibration.get(name, 0.05)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid calibration value: {name}")
    if min(calibration["unknown_weight"], calibration["margin_weight"]) < 0:
        raise ValueError("Reference evidence weights must be nonnegative")
    if calibration.get("temperature", 0.05) <= 0:
        raise ValueError("Probability temperature must be positive")
    if require_inference:
        if calibration.get("temperature") != 0.05:
            raise ValueError("The portable release requires probability temperature 0.05")
        policy = calibration.get("inference")
        if (not isinstance(policy, dict) or set(policy) != {"seconds", "maximum_windows"}
                or type(policy["seconds"]) not in (float, int) or policy["seconds"] != 180.0
                or type(policy["maximum_windows"]) is not int or policy["maximum_windows"] != 1):
            raise ValueError("The portable release requires exactly one 180-second inference window")
    return calibration


def reference_scores(embeddings: np.ndarray, gallery: dict, *, classes: int = 446) -> tuple[np.ndarray, np.ndarray]:
    dimension = np.asarray(gallery["known_embeddings"]).shape[-1]
    gallery = validate_gallery(gallery, classes=classes, embedding_dim=dimension)
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise ValueError("Query embeddings must be a finite matrix matching the gallery")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-8)
    similarities = normalized @ gallery["known_embeddings"].T
    known = np.full((classes, len(values)), -np.inf, dtype=np.float32)
    np.maximum.at(known, gallery["known_targets"] - 1, similarities.T)
    unknown = (normalized @ gallery["unknown_embeddings"].T).max(axis=1)
    return np.clip(known.T, -1.0, 1.0), np.clip(unknown, -1.0, 1.0)


def score_embeddings(embeddings: np.ndarray, valid: np.ndarray, gallery: dict,
                     calibration: dict, *, classes: int = 446) -> np.ndarray:
    """Return normalized scores with unknown at column zero, including exact ties.

    Invalid/zero/nonfinite query embeddings become deterministic unknown rows.
    Corrupt model assets and invalid calibration remain fatal validation errors.
    """
    validate_calibration(calibration)
    values = np.asarray(embeddings, dtype=np.float32)
    valid = np.asarray(valid)
    if values.ndim != 2 or valid.shape != (len(values),) or valid.dtype != np.bool_:
        raise ValueError("Query validity must be a boolean vector aligned with embeddings")
    usable = valid & np.isfinite(values).all(axis=1)
    safe = np.where(usable[:, None], values, 0).astype(np.float32)
    usable &= np.linalg.norm(safe, axis=1) > 1e-8
    known, unknown = reference_scores(safe, gallery, classes=classes)
    scores = known.astype(np.float64)
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    top, margin = top_two[:, 1], top_two[:, 1] - top_two[:, 0]
    gate = top - calibration["unknown_weight"] * unknown.astype(np.float64) + calibration["margin_weight"] * margin
    logits = np.column_stack((calibration["threshold"] - gate, scores - top[:, None])) / calibration.get("temperature", 0.05)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~usable] = 0
    probabilities[~usable, 0] = 1
    return probabilities
