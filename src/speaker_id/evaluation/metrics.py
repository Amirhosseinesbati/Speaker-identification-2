"""Fixed 447-class, file-level competition metrics without learned parameters."""
from __future__ import annotations

from collections import Counter
import math
from typing import Sequence


def validate_labels(labels: Sequence[str]) -> list[str]:
    labels = list(labels)
    if len(labels) != 447 or len(set(labels)) != 447 or labels[0] != "unknown":
        raise ValueError("Expected the fixed 447-class label map with unknown at index zero")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Labels must be nonempty strings")
    return labels


def _index_rows(rows: Sequence[dict], labels: set[str], name: str) -> dict[str, str]:
    indexed = {}
    for row in rows:
        filename, label = row.get("audio_file"), row.get("speaker_id")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{name}: audio_file must be a nonempty string")
        if filename in indexed:
            raise ValueError(f"{name}: duplicate audio_file {filename}")
        if label not in labels:
            raise ValueError(f"{name}: label outside the fixed label map")
        indexed[filename] = label
    return indexed


def score_predictions(reference: Sequence[dict], predictions: Sequence[dict],
                      labels: Sequence[str]) -> dict:
    """Align by filename, reject incomplete submissions, then pool all file counts.

    All 447 classes enter the macro average, including classes with no support in
    a diagnostic subset. Zero denominators produce zero, never dropped classes.
    Call once on pooled out-of-fold predictions for the primary development score.
    """
    labels = validate_labels(labels)
    actual = _index_rows(reference, set(labels), "reference")
    predicted = _index_rows(predictions, set(labels), "predictions")
    if not actual:
        raise ValueError("Cannot score an empty reference")
    if actual.keys() != predicted.keys():
        raise ValueError("Predictions must cover exactly the original filenames")
    support, predicted_count, correct = Counter(actual.values()), Counter(predicted.values()), Counter()
    errors = Counter()
    for name, truth in actual.items():
        guess = predicted[name]
        if truth == guess:
            correct[truth] += 1
        elif truth == "unknown":
            errors["unknown_to_known"] += 1
        elif guess == "unknown":
            errors["known_to_unknown"] += 1
        else:
            errors["known_to_other_known"] += 1
    per_class = []
    for label in labels:
        tp, total, assigned = correct[label], support[label], predicted_count[label]
        per_class.append({"speaker_id": label, "support": total, "predicted": assigned,
                          "true_positive": tp, "false_positive": assigned - tp,
                          "false_negative": total - tp,
                          "precision": tp / assigned if assigned else 0.0,
                          "recall": tp / total if total else 0.0,
                          "f1": 2 * tp / (total + assigned) if total + assigned else 0.0})
    return {"row_count": len(actual), "class_count": len(labels),
            "macro_f1": sum(row["f1"] for row in per_class) / len(labels),
            "accuracy": sum(correct.values()) / len(actual),
            "errors": {key: errors[key] for key in
                       ("known_to_unknown", "unknown_to_known", "known_to_other_known")},
            "per_class": per_class}


def predictions_from_probabilities(audio_files: Sequence[str], probabilities: Sequence[Sequence[float]],
                                   labels: Sequence[str], tolerance: float = 1e-6) -> list[dict]:
    """Validate distributions and reproduce the final first-index argmax exactly."""
    labels = validate_labels(labels)
    if len(audio_files) != len(probabilities) or len(set(audio_files)) != len(audio_files):
        raise ValueError("Probability rows need one unique filename each")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Probability tolerance must be finite and nonnegative")
    output = []
    for name, values in zip(audio_files, probabilities):
        if not isinstance(name, str) or not name:
            raise ValueError("audio_file must be a nonempty string")
        values = [float(value) for value in values]
        if len(values) != len(labels) or any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError("Each probability row must contain 447 finite values in [0, 1]")
        if abs(math.fsum(values) - 1.0) > tolerance:
            raise ValueError("Each probability row must sum to one")
        index = max(range(len(values)), key=values.__getitem__)
        output.append({"audio_file": name, "speaker_id": labels[index]})
    return output
