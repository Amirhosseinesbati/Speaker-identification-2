"""Pure NumPy primitives for the S033 reference-resampling veto.

The baseline identity prediction is deliberately an input, rather than a score
derived here.  Resampling can therefore only test whether that frozen winner
persists when the permitted reference gallery changes.  It never supplies an
alternative identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral

import numpy as np


_UNIT_ATOL = 1e-5


@dataclass(frozen=True)
class ResampledGalleries:
    """Balanced class-wise reference draws, addressed in reference-row space."""

    source_indices: np.ndarray
    source_labels: np.ndarray
    class_labels: np.ndarray
    indices: np.ndarray
    seed: int
    per_class: int
    replace: bool


def _frozen_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _require_count(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _integer_vector(name: str, values, *, nonempty: bool = True) -> np.ndarray:
    result = np.asarray(values)
    if (result.ndim != 1 or (nonempty and not len(result))
            or not np.issubdtype(result.dtype, np.integer)):
        raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
    return result


def _unknown_label(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("unknown_label must be an integer")
    return int(value)


def _group_vector(name: str, values, expected: int) -> np.ndarray:
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or len(result) != expected:
        raise ValueError(f"{name} must align one-to-one with embeddings")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty content-group strings")
    return result


def _valid_mask(name: str, values, expected: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != expected or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a one-dimensional boolean mask")
    return result


def _unit_embeddings(name: str, values, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if (result.ndim != 2 or result.shape[0] != len(valid) or result.shape[1] < 1
            or not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite two-dimensional floating array")
    result = result.astype(np.float64, copy=False)
    if np.any(valid):
        norms = np.linalg.norm(result[valid], axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=_UNIT_ATOL):
            raise ValueError(f"valid {name} rows must be unit normalized")
    if np.any(result[~valid] != 0.0):
        raise ValueError(f"invalid {name} rows must be exact zeros")
    return result


def make_balanced_resampled_galleries(
    reference_indices,
    reference_labels,
    *,
    n_resamples: int,
    per_class: int,
    seed: int,
    replace: bool = False,
    unknown_label: int = 0,
) -> ResampledGalleries:
    """Draw equally many permitted reference rows per known class.

    ``reference_indices`` addresses a later reference embedding matrix.  Each
    draw is independent and contains exactly ``per_class`` rows for every
    class.  Sampling is without replacement within a draw by default, which
    makes a gallery membership perturbation rather than a duplicate weighting.
    """
    n_resamples = _require_count("n_resamples", n_resamples)
    per_class = _require_count("per_class", per_class)
    seed = _require_count("seed", seed, minimum=0)
    unknown_label = _unknown_label(unknown_label)
    if not isinstance(replace, (bool, np.bool_)):
        raise ValueError("replace must be boolean")

    indices = _integer_vector("reference_indices", reference_indices)
    labels = _integer_vector("reference_labels", reference_labels)
    if len(indices) != len(labels) or np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("reference indices and labels must align, be nonnegative, and be unique")
    if np.any(labels == unknown_label):
        raise ValueError("resampled galleries contain known classes only")
    class_labels = np.unique(labels)
    if len(class_labels) < 2:
        raise ValueError("reference resampling requires at least two known classes")

    generator = np.random.default_rng(seed)
    galleries = np.empty((n_resamples, len(class_labels), per_class), dtype=indices.dtype)
    for class_position, label in enumerate(class_labels):
        candidates = indices[labels == label]
        if not replace and len(candidates) < per_class:
            raise ValueError("per_class exceeds available references for a known class")
        # ``Generator.choice`` applies ``replace`` to its whole requested
        # shape.  Each gallery must instead be an independent class draw.
        for draw in range(n_resamples):
            galleries[draw, class_position, :] = generator.choice(
                candidates, size=per_class, replace=bool(replace)
            )
    return ResampledGalleries(
        source_indices=_frozen_copy(indices),
        source_labels=_frozen_copy(labels),
        class_labels=_frozen_copy(class_labels),
        indices=_frozen_copy(galleries),
        seed=seed,
        per_class=per_class,
        replace=bool(replace),
    )


def _validate_galleries(galleries: ResampledGalleries, n_references: int, unknown_label: int) -> None:
    if not isinstance(galleries, ResampledGalleries):
        raise ValueError("galleries must come from make_balanced_resampled_galleries")
    source_indices = _integer_vector("gallery source_indices", galleries.source_indices)
    source_labels = _integer_vector("gallery source_labels", galleries.source_labels)
    class_labels = _integer_vector("gallery class_labels", galleries.class_labels)
    sampled = np.asarray(galleries.indices)
    seed = _require_count("gallery seed", galleries.seed, minimum=0)
    per_class = _require_count("gallery per_class", galleries.per_class)
    if (len(source_indices) != len(source_labels) or len(np.unique(source_indices)) != len(source_indices)
            or np.any(source_indices < 0) or np.any(source_indices >= n_references)
            or np.any(source_labels == unknown_label) or np.any(class_labels == unknown_label)
            or len(class_labels) < 2 or not np.array_equal(class_labels, np.unique(source_labels))
            or sampled.ndim != 3 or sampled.shape[0] < 1 or sampled.shape[1] != len(class_labels)
            or sampled.shape[2] != per_class or not np.issubdtype(sampled.dtype, np.integer)
            or not isinstance(galleries.replace, (bool, np.bool_))):
        raise ValueError("malformed balanced resampled galleries")
    for class_position, label in enumerate(class_labels):
        permitted = source_indices[source_labels == label]
        if not len(permitted) or not np.isin(sampled[:, class_position, :], permitted).all():
            raise ValueError("gallery rows do not match their registered class")


def _query_draw_rows(
    galleries: ResampledGalleries,
    reference_groups: np.ndarray,
    query_group: str,
    draw_position: int,
    class_position: int,
) -> np.ndarray:
    """Repair a fixed draw using only references permitted for this query.

    A group-excluded calibration query can legitimately share a label with
    some reference rows.  Dropping such rows from a globally sampled gallery
    would sometimes leave a class empty purely by chance.  We deterministically
    fill those slots from the remaining *permitted* rows; if none exist, the
    caller fails closed instead of silently admitting the query group.
    """
    label = int(np.asarray(galleries.class_labels)[class_position])
    source_indices = np.asarray(galleries.source_indices)
    source_labels = np.asarray(galleries.source_labels)
    allowed_pool = source_indices[
        (source_labels == label) & (reference_groups[source_indices] != query_group)
    ]
    per_class = int(galleries.per_class)
    if not len(allowed_pool) or (not galleries.replace and len(allowed_pool) < per_class):
        raise ValueError("group exclusion leaves no complete class gallery")

    base_rows = np.asarray(galleries.indices)[draw_position, class_position]
    retained = base_rows[reference_groups[base_rows] != query_group]
    if not galleries.replace:
        retained = np.unique(retained)
    retained = retained[:per_class]
    needed = per_class - len(retained)
    if not needed:
        return retained
    pool = allowed_pool if galleries.replace else allowed_pool[~np.isin(allowed_pool, retained)]
    if not galleries.replace and len(pool) < needed:
        raise ValueError("group exclusion leaves no complete class gallery")
    digest = hashlib.sha256(
        f"{galleries.seed}|{query_group}|{draw_position}|{class_position}".encode("utf-8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    fill = generator.choice(pool, size=needed, replace=bool(galleries.replace))
    return np.concatenate((retained, np.asarray(fill, dtype=retained.dtype)))


def frozen_winner_stability(
    query_embeddings,
    query_valid,
    query_groups,
    reference_embeddings,
    reference_valid,
    reference_groups,
    galleries: ResampledGalleries,
    baseline_predictions,
    *,
    unknown_label: int = 0,
) -> np.ndarray:
    """Return the fraction of gallery draws retaining each frozen known winner.

    A query's entire content group is removed before each class maximum.  A
    fixed draw which happened to include an excluded row is repaired only from
    that class's remaining allowed references.  If an entire class has no
    permitted reference, the call fails rather than allowing a hidden
    self-reference or a partial class ranking.
    Invalid queries and baseline-unknown rows receive stability zero and are
    never scored.
    """
    unknown_label = _unknown_label(unknown_label)
    raw_queries = np.asarray(query_embeddings)
    raw_references = np.asarray(reference_embeddings)
    if raw_queries.ndim != 2 or raw_references.ndim != 2:
        raise ValueError("query and reference embeddings must be two-dimensional")
    query_valid = _valid_mask("query_valid", query_valid, raw_queries.shape[0])
    reference_valid = _valid_mask("reference_valid", reference_valid, raw_references.shape[0])
    queries = _unit_embeddings("query_embeddings", raw_queries, query_valid)
    references = _unit_embeddings("reference_embeddings", raw_references, reference_valid)
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference embedding dimensions differ")
    query_groups = _group_vector("query_groups", query_groups, len(queries))
    reference_groups = _group_vector("reference_groups", reference_groups, len(references))
    baseline = _integer_vector("baseline_predictions", baseline_predictions, nonempty=False)
    if len(baseline) != len(queries):
        raise ValueError("baseline predictions must align with query rows")
    _validate_galleries(galleries, len(references), unknown_label)
    if not np.all(reference_valid[np.asarray(galleries.source_indices)]):
        raise ValueError("a gallery source row is invalid")

    class_labels = np.asarray(galleries.class_labels)
    known = query_valid & (baseline != unknown_label)
    if np.any(known) and not np.isin(baseline[known], class_labels).all():
        raise ValueError("a valid frozen baseline winner is absent from the gallery classes")

    stability = np.zeros(len(queries), dtype=np.float64)
    sampled = np.asarray(galleries.indices)
    for query_position in np.flatnonzero(known):
        retained = 0
        for draw_position, draw in enumerate(sampled):
            maxima = np.empty(len(class_labels), dtype=np.float64)
            for class_position, _ in enumerate(draw):
                allowed = _query_draw_rows(
                    galleries, reference_groups, query_groups[query_position], draw_position, class_position,
                )
                maxima[class_position] = np.max(queries[query_position] @ references[allowed].T)
            retained += int(class_labels[int(np.argmax(maxima))] == baseline[query_position])
        stability[query_position] = retained / len(sampled)
    return stability


def group_disjoint_frozen_winner_stability(
    query_embeddings,
    query_valid,
    query_groups,
    reference_embeddings,
    reference_valid,
    reference_groups,
    galleries: ResampledGalleries,
    baseline_predictions,
    *,
    device: str = "cpu",
    query_batch_size: int = 128,
    unknown_label: int = 0,
) -> np.ndarray:
    """Vectorized stability for a gallery disjoint from every query group.

    This is the same frozen-winner statistic as :func:`frozen_winner_stability`,
    specialized to a role assignment whose reference groups and query groups are
    disjoint.  That precondition removes the need for per-query draw repair and
    makes a large, inference-only screen practical.  It does not change a
    prediction or manufacture a replacement identity.

    ``device`` is deliberately limited to ``"cpu"`` and ``"cuda"``.  CUDA is
    used only for batched cosine products over already cached embeddings; no
    model, audio, or training state is loaded by this function.
    """
    unknown_label = _unknown_label(unknown_label)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    query_batch_size = _require_count("query_batch_size", query_batch_size)

    raw_queries = np.asarray(query_embeddings)
    raw_references = np.asarray(reference_embeddings)
    if raw_queries.ndim != 2 or raw_references.ndim != 2:
        raise ValueError("query and reference embeddings must be two-dimensional")
    query_valid = _valid_mask("query_valid", query_valid, raw_queries.shape[0])
    reference_valid = _valid_mask("reference_valid", reference_valid, raw_references.shape[0])
    queries = _unit_embeddings("query_embeddings", raw_queries, query_valid)
    references = _unit_embeddings("reference_embeddings", raw_references, reference_valid)
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference embedding dimensions differ")
    query_groups = _group_vector("query_groups", query_groups, len(queries))
    reference_groups = _group_vector("reference_groups", reference_groups, len(references))
    if set(query_groups.tolist()) & set(reference_groups.tolist()):
        raise ValueError("fast stability path requires group-disjoint queries and references")
    baseline = _integer_vector("baseline_predictions", baseline_predictions, nonempty=False)
    if len(baseline) != len(queries):
        raise ValueError("baseline predictions must align with query rows")
    _validate_galleries(galleries, len(references), unknown_label)
    source_indices = np.asarray(galleries.source_indices)
    if not np.all(reference_valid[source_indices]):
        raise ValueError("a gallery source row is invalid")
    class_labels = np.asarray(galleries.class_labels)
    known = query_valid & (baseline != unknown_label)
    if np.any(known) and not np.isin(baseline[known], class_labels).all():
        raise ValueError("a valid frozen baseline winner is absent from the gallery classes")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Vectorized S033 stability requires the project's Torch dependency") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("S033 CUDA stability was requested but CUDA is unavailable")

    # Validation above intentionally uses float64 so malformed unit vectors are
    # caught before device transfer.  C002's frozen embedding cache and its
    # reference scorer are float32, so preserving float32 here matches the
    # source numerical representation while keeping memory bounded.
    sampled = np.asarray(galleries.indices)
    reference_tensor = torch.as_tensor(references.astype(np.float32, copy=False), device=device)
    gallery_tensor = reference_tensor[
        torch.as_tensor(sampled.reshape(-1), dtype=torch.long, device=device)
    ].reshape((*sampled.shape, references.shape[1]))
    class_tensor = torch.as_tensor(class_labels, dtype=torch.long, device=device)
    stability = np.zeros(len(queries), dtype=np.float64)
    positions = np.flatnonzero(known)
    with torch.no_grad():
        for start in range(0, len(positions), query_batch_size):
            batch_positions = positions[start:start + query_batch_size]
            batch = torch.as_tensor(
                queries[batch_positions].astype(np.float32, copy=False), device=device,
            )
            # [batch, resamples, classes, references-per-class]
            similarity = torch.einsum("bd,rcpd->brcp", batch, gallery_tensor)
            winners = class_tensor[similarity.amax(dim=-1).argmax(dim=-1)]
            expected = torch.as_tensor(baseline[batch_positions], dtype=torch.long, device=device)
            stability[batch_positions] = (
                (winners == expected[:, None]).to(dtype=torch.float64).mean(dim=1).cpu().numpy()
            )
    return stability


def apply_stability_veto(
    baseline_predictions,
    stability,
    valid,
    *,
    minimum_stability: float,
    unknown_label: int = 0,
) -> np.ndarray:
    """Reject unstable known predictions while preserving every retained label.

    This operation has no alternative known label input: it can only retain a
    baseline prediction or map it to ``unknown_label``.
    """
    unknown_label = _unknown_label(unknown_label)
    baseline = _integer_vector("baseline_predictions", baseline_predictions, nonempty=False)
    valid = _valid_mask("valid", valid, len(baseline))
    stability = np.asarray(stability)
    if (stability.ndim != 1 or len(stability) != len(baseline)
            or not np.issubdtype(stability.dtype, np.floating) or not np.isfinite(stability).all()
            or np.any(stability < 0.0) or np.any(stability > 1.0)
            or not isinstance(minimum_stability, (float, int, np.floating, np.integer))
            or isinstance(minimum_stability, (bool, np.bool_))
            or not np.isfinite(minimum_stability) or not 0.0 <= float(minimum_stability) <= 1.0):
        raise ValueError("invalid stability-veto inputs")
    result = baseline.copy()
    result[~valid] = unknown_label
    result[valid & (baseline != unknown_label) & (stability < float(minimum_stability))] = unknown_label
    changed = result != baseline
    if np.any(changed & ~((baseline != unknown_label) & (result == unknown_label))):
        raise RuntimeError("stability veto attempted to change a label other than to unknown")
    return result
