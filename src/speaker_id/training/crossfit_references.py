"""Reference-only crossfit for fixed public embeddings, with untouched outer folds.

This module never fits an encoder, transformation, coefficient, or threshold.
Embeddings from an encoder fitted on a calibration query are not valid inputs to
this protocol. Every same-content copy of a query leaves its reference pool.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def crossfit_scores(embeddings, valid, manifest: list[dict], folds: list[dict],
                    outer: int | None, method: str = "prototype", classes: int = 446) -> dict:
    """Score inner group-held-out queries and outer rows with all training refs.

    Input rows in ``embeddings`` and ``valid`` follow ``manifest`` order. ``folds``
    may have a different row order and supplies audio_file, group_id, fold and
    train_eligible. Known label order is sorted from eligible outer-training
    labels only. Outer labels are never inspected. Invalid outer rows remain in
    the output with zero scores and ``outer_valid=False``; the caller must use its
    unknown fallback for these rows.

    ``prototype`` uses a file-weighted mean of normalized reference embeddings,
    then normalizes that mean. ``max_reference`` uses maximum reference cosine
    per known label. Unknown similarity is maximum reference cosine in both
    methods. A calibration query always removes its full content group.

    Returns NumPy score/index arrays plus counts and provenance. Calibration
    known queries require >=2 eligible reference groups for their identity.
    Unknown calibration requires >=2 eligible unknown groups, so no empty-cohort
    score needs to be fabricated. ``outer=None`` explicitly enrolls all eligible
    training data and returns no outer rows; it is final calibration, not an OOF
    evaluation. Existing fold/group assignments remain unmodified.
    """
    if method not in {"prototype", "max_reference"}:
        raise ValueError("method must be prototype or max_reference")
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 1:
        raise ValueError("classes must be a positive integer")
    values = np.asarray(embeddings, dtype=np.float32)
    mask = np.asarray(valid)
    if (values.ndim != 2 or len(values) != len(manifest) or values.shape[1] < 1
            or mask.shape != (len(manifest),) or mask.dtype != np.bool_):
        raise ValueError("Embeddings/boolean validity must align with manifest rows")
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains nonfinite values")
    names = [row["audio_file"] for row in manifest]
    indexed_folds = {row["audio_file"]: row for row in folds}
    if (len(set(names)) != len(names) or len(indexed_folds) != len(folds)
            or set(indexed_folds) != set(names)):
        raise ValueError("Manifest and folds must cover the same unique filenames")
    groups = []
    assigned = []
    group_folds = defaultdict(set)
    eligible = []
    for name in names:
        row = indexed_folds[name]
        group = row["group_id"]
        if not isinstance(group, str) or not group:
            raise ValueError("A nonempty content group is required for every row")
        fold = int(row["fold"])
        groups.append(group)
        assigned.append(fold)
        eligible.append(_truth(row["train_eligible"]))
        group_folds[group].add(fold)
        if "evaluation_included" in row and not _truth(row["evaluation_included"]):
            raise ValueError("Every source row must remain in outer evaluation")
    if any(len(assignments) != 1 for assignments in group_folds.values()):
        raise ValueError("A content group crosses outer fold boundaries")
    assigned = np.asarray(assigned)
    groups = np.asarray(groups, dtype=object)
    if outer is not None and (outer not in assigned or not np.any(assigned != outer)):
        raise ValueError("Requested outer fold must have both training and evaluation rows")
    training = np.ones(len(assigned), dtype=bool) if outer is None else assigned != outer
    references = np.flatnonzero(training & np.asarray(eligible) & mask)
    outer_indices = np.asarray([], dtype=np.int64) if outer is None else np.flatnonzero(assigned == outer)
    if not len(references):
        raise ValueError("No eligible outer-training references")

    norms = np.linalg.norm(values, axis=1)
    if np.any(mask & (norms <= 1e-8)):
        raise ValueError("A valid row has a zero or undefined embedding")
    normalized = np.zeros_like(values)
    normalized[mask] = values[mask] / norms[mask, None]
    # Deliberately read labels only after restricting to training references.
    labels_by_index = {}
    grouped_labels = defaultdict(set)
    for index in references:
        label = manifest[index]["speaker_id"]
        if not isinstance(label, str) or not label:
            raise ValueError("Training references require nonempty labels")
        if "speaker_id" in indexed_folds[names[index]] and indexed_folds[names[index]]["speaker_id"] != label:
            raise ValueError("Training manifest and fold labels disagree")
        labels_by_index[int(index)] = label
        grouped_labels[groups[index]].add(label)
    if any(len(labels) != 1 for labels in grouped_labels.values()):
        raise ValueError("Eligible content group has conflicting training labels")
    known_labels = sorted(set(labels_by_index.values()) - {"unknown"})
    if len(known_labels) != classes:
        raise ValueError(f"Expected {classes} known identities in eligible outer-training references; got {len(known_labels)}")
    label_positions = {label: index for index, label in enumerate(known_labels)}
    known_references = np.asarray([i for i in references if labels_by_index[int(i)] != "unknown"], dtype=np.int64)
    unknown_references = np.asarray([i for i in references if labels_by_index[int(i)] == "unknown"], dtype=np.int64)
    known_groups = {label: {groups[i] for i in known_references if labels_by_index[int(i)] == label}
                    for label in known_labels}
    unknown_groups = set(groups[unknown_references])
    if len(unknown_groups) < 2:
        raise ValueError("Unknown leave-group-out calibration requires at least two eligible unknown groups")
    calibration_indices = np.asarray([
        i for i in references if labels_by_index[int(i)] == "unknown"
        or len(known_groups[labels_by_index[int(i)]]) >= 2
    ], dtype=np.int64)
    known_reference_classes = np.asarray([label_positions[labels_by_index[int(i)]] for i in known_references])
    counts = np.bincount(known_reference_classes, minlength=classes).astype(np.int64)
    inner_counts = np.broadcast_to(counts, (len(calibration_indices), classes)).copy()
    inner = normalized[calibration_indices]
    outer_vectors = normalized[outer_indices]
    inner_known = np.empty((len(inner), classes), dtype=np.float32)
    outer_known = np.empty((len(outer_vectors), classes), dtype=np.float32)

    if method == "prototype":
        sums = np.zeros((classes, values.shape[1]), dtype=np.float32)
        np.add.at(sums, known_reference_classes, normalized[known_references])
        lengths = np.linalg.norm(sums, axis=1)
        if np.any(lengths <= 1e-8):
            raise ValueError("Known reference mean is undefined due to cancellation")
        prototypes = sums / lengths[:, None]
        inner_known[:] = inner @ prototypes.T
        outer_known[:] = outer_vectors @ prototypes.T
        for query_position, query_index in enumerate(calibration_indices):
            label = labels_by_index[int(query_index)]
            if label == "unknown":
                continue
            target = label_positions[label]
            removed = known_references[groups[known_references] == groups[query_index]]
            remaining = sums[target] - normalized[removed].sum(axis=0)
            norm = np.linalg.norm(remaining)
            if norm <= 1e-8:
                raise ValueError("Known leave-group-out reference mean is undefined")
            inner_known[query_position, target] = inner[query_position] @ (remaining / norm)
            inner_counts[query_position, target] -= len(removed)
    else:
        inner_similarities = inner @ normalized[known_references].T
        same_group = groups[calibration_indices, None] == groups[known_references][None, :]
        inner_similarities[same_group] = -np.inf
        outer_similarities = outer_vectors @ normalized[known_references].T
        for target in range(classes):
            columns = known_reference_classes == target
            inner_known[:, target] = inner_similarities[:, columns].max(axis=1)
            outer_known[:, target] = outer_similarities[:, columns].max(axis=1)
            inner_counts[:, target] -= same_group[:, columns].sum(axis=1)

    inner_unknown_scores = inner @ normalized[unknown_references].T
    same_unknown_group = groups[calibration_indices, None] == groups[unknown_references][None, :]
    inner_unknown_scores[same_unknown_group] = -np.inf
    inner_unknown = inner_unknown_scores.max(axis=1)
    outer_unknown = (outer_vectors @ normalized[unknown_references].T).max(axis=1)
    if (not np.isfinite(inner_known).all() or not np.isfinite(outer_known).all()
            or not np.isfinite(inner_unknown).all() or not np.isfinite(outer_unknown).all()
            or np.any(inner_counts < 1)):
        raise ValueError("Crossfit produced an empty reference pool or invalid similarity")
    for scores in (inner_known, outer_known, inner_unknown, outer_unknown):
        np.clip(scores, -1.0, 1.0, out=scores)
    return {
        "calibration_indices": calibration_indices,
        "inner_known_scores": inner_known,
        "inner_unknown_similarity": inner_unknown,
        "outer_indices": outer_indices,
        "outer_known_scores": outer_known,
        "outer_unknown_similarity": outer_unknown,
        "outer_valid": mask[outer_indices].copy(),
        "known_labels": known_labels,
        "reference_counts": {
            "known_files_per_class": counts,
            "known_groups_per_class": np.asarray([len(known_groups[label]) for label in known_labels]),
            "inner_known_files_per_class": inner_counts,
            "unknown_files": len(unknown_references),
            "unknown_groups": len(unknown_groups),
            "inner_unknown_files": len(unknown_references) - same_unknown_group.sum(axis=1),
            "singleton_known_labels": [label for label in known_labels if len(known_groups[label]) == 1],
        },
        "provenance": {
            "protocol": "frozen_public_embedding_leave_content_group_out_v1",
            "outer_fold": None if outer is None else int(outer), "method": method,
            "scope": "all_training_final_calibration" if outer is None else "outer_fold_development",
            "reference_indices": [int(index) for index in references],
            "reference_groups": sorted(set(groups[references])),
            "outer_labels_accessed": False, "threshold_or_coefficient_fitted": False,
            "encoder_requirement": "Public frozen embeddings; no encoder or learned transform may have been fitted on these queries",
            "known_prototype_weighting": "file-weighted normalized-embedding mean" if method == "prototype" else "maximum reference cosine",
            "unknown_reference_method": "maximum cosine, excluding every member of the query content group",
            "invalid_outer_policy": "Preserved with zero scores and outer_valid=False; caller must apply unknown fallback",
            "support_caveat": "Inner known queries have their group removed; full outer-training support is restored for outer evaluation",
        },
    }


def all_training_crossfit_scores(embeddings, valid, manifest: list[dict], folds: list[dict],
                                method: str = "max_reference", classes: int = 446) -> dict:
    """Final scorer calibration using all training references, without a test fold.

    The public encoder must remain frozen. Each calibration query still excludes
    every member of its content group. No synthetic sample or fold is introduced.
    """
    return crossfit_scores(embeddings, valid, manifest, folds, outer=None,
                           method=method, classes=classes)
