"""Label-free, NumPy-only batch decoding primitives for S014.

Powered-ratio alignment is the preregistered S014 method.  Iterative soft
alignment, the reciprocal graph and one-pass query expansion are retained as
portable experimental primitives; no function accepts labels or group data.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class DistributionAlignmentConfig:
    """Configuration for KL projection to a softened official class prior."""

    schema_version: int = 1
    official_unknown_prior: float = 0.5
    official_known_prior: float = 1.0 / 892.0
    strength: float = 0.5
    temperature: float = 1.0
    epsilon: float = 1e-12
    max_iterations: int = 200
    tolerance: float = 1e-8


@dataclass(frozen=True)
class PoweredRatioAlignmentConfig:
    """One-step S014 alignment to a fixed design prior.

    The target masses are an engineering assumption from the local development
    protocol. They are not guaranteed hidden-batch frequencies from the
    competition organizer.
    """

    schema_version: int = 1
    design_unknown_prior: float = 0.5
    design_known_prior: float = 1.0 / 892.0
    unknown_strength: float = 0.0
    known_strength: float = 0.5
    epsilon: float = 1e-12


@dataclass(frozen=True)
class QueryExpansionConfig:
    """Configuration for the dual-view reciprocal-query primitive."""

    schema_version: int = 1
    k: int = 3
    fused_cosine_floor: float = 0.75
    beta: float = 0.25
    block_size: int = 512


@dataclass(frozen=True)
class DistributionAlignmentResult:
    probabilities: np.ndarray
    observed_prior: np.ndarray
    official_prior: np.ndarray
    target_prior: np.ndarray
    achieved_prior: np.ndarray
    duals: np.ndarray
    residual_trace: np.ndarray
    residual: float
    iterations: int
    converged: bool
    valid_rows: int


@dataclass(frozen=True)
class PoweredRatioAlignmentResult:
    probabilities: np.ndarray
    observed_prior: np.ndarray
    design_prior: np.ndarray
    observed_known_conditional: np.ndarray
    design_known_conditional: np.ndarray
    prior_ratios: np.ndarray
    factors: np.ndarray
    adjusted_prior: np.ndarray
    unknown_factor: float
    unknown_probability_may_change_via_row_normalization: bool
    valid_rows: int


@dataclass(frozen=True)
class ReciprocalGraph:
    """Undirected graph whose edges pass reciprocal top-k in both views."""

    size: int
    edges: np.ndarray
    fused_cosine: np.ndarray
    view_a_cutoff: np.ndarray
    view_b_cutoff: np.ndarray
    degrees: np.ndarray


@dataclass(frozen=True)
class QueryExpansionResult:
    view_a: np.ndarray
    view_b: np.ndarray
    degrees: np.ndarray
    expanded_rows: np.ndarray


def _finite_float(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _strict_config(config, kind):
    if isinstance(config, kind):
        result = config
    elif type(config) is dict:
        names = {field.name for field in fields(kind)}
        if set(config) != names:
            raise ValueError(f"{kind.__name__} must contain exactly {sorted(names)}")
        try:
            result = kind(**config)
        except TypeError as exc:
            raise ValueError(f"Invalid {kind.__name__}") from exc
    else:
        raise ValueError(f"Expected {kind.__name__} or its exact dictionary schema")
    if type(result.schema_version) is not int or result.schema_version != 1:
        raise ValueError("Unsupported batch-decoding schema version")
    return result


def validate_alignment_config(config=DistributionAlignmentConfig()):
    """Return a validated immutable alignment configuration."""
    result = _strict_config(config, DistributionAlignmentConfig)
    unknown = _finite_float(result.official_unknown_prior, "official_unknown_prior")
    known = _finite_float(result.official_known_prior, "official_known_prior")
    strength = _finite_float(result.strength, "strength")
    temperature = _finite_float(result.temperature, "temperature")
    epsilon = _finite_float(result.epsilon, "epsilon")
    tolerance = _finite_float(result.tolerance, "tolerance")
    if not 0 < unknown < 1 or not 0 < known < 1:
        raise ValueError("Official class priors must be strictly between zero and one")
    if not 0 <= strength <= 1:
        raise ValueError("Alignment strength must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("Alignment temperature must be positive")
    if not 0 < epsilon < 1:
        raise ValueError("Alignment epsilon must be in (0, 1)")
    if type(result.max_iterations) is not int or result.max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if tolerance <= 0:
        raise ValueError("Alignment tolerance must be positive")
    return result


def validate_powered_ratio_config(config=PoweredRatioAlignmentConfig()):
    """Return a strictly validated one-step alignment configuration."""
    result = _strict_config(config, PoweredRatioAlignmentConfig)
    unknown = _finite_float(result.design_unknown_prior,
                            "design_unknown_prior")
    known = _finite_float(result.design_known_prior,
                          "design_known_prior")
    unknown_strength = _finite_float(result.unknown_strength,
                                     "unknown_strength")
    known_strength = _finite_float(result.known_strength, "known_strength")
    epsilon = _finite_float(result.epsilon, "epsilon")
    if not 0 < unknown < 1 or not 0 < known < 1:
        raise ValueError("Official class priors must be strictly between zero and one")
    if not 0 <= unknown_strength <= 1 or not 0 <= known_strength <= 1:
        raise ValueError("Powered-ratio strengths must be in [0, 1]")
    if not 0 < epsilon < 1:
        raise ValueError("Powered-ratio epsilon must be in (0, 1)")
    return result


def validate_expansion_config(config=QueryExpansionConfig()):
    """Return a validated immutable graph/expansion configuration."""
    result = _strict_config(config, QueryExpansionConfig)
    floor = _finite_float(result.fused_cosine_floor, "fused_cosine_floor")
    beta = _finite_float(result.beta, "beta")
    if type(result.k) is not int or result.k < 1:
        raise ValueError("k must be a positive integer")
    if not -1 <= floor <= 1:
        raise ValueError("fused_cosine_floor must be in [-1, 1]")
    if not 0 <= beta <= 1:
        raise ValueError("Expansion beta must be in [0, 1]")
    if type(result.block_size) is not int or result.block_size < 1:
        raise ValueError("block_size must be a positive integer")
    return result


def _probability_inputs(probabilities, valid):
    try:
        original = np.asarray(probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a real floating matrix") from exc
    mask = np.asarray(valid)
    if (original.ndim != 2 or original.shape[0] < 1 or original.shape[1] < 2
            or original.dtype.kind != "f"):
        raise ValueError("Probabilities must be a nonempty real floating matrix")
    if mask.dtype != np.bool_ or mask.shape != (len(original),):
        raise ValueError("Validity must be an aligned boolean vector")
    source = np.array(original, copy=True, order="C")
    values = np.asarray(original, dtype=np.float64, order="C")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Every probability must be finite and in [0, 1]")
    sums = values.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-6):
        raise ValueError("Every probability row must sum to one")
    # Fit from normalized rows but preserve the caller's numeric values for all
    # invalid rows and for the strength-zero control.
    normalized = values / sums
    return source, values, normalized, mask


def _official_prior(columns, config):
    result = np.full(columns, config.official_known_prior, dtype=np.float64)
    result[0] = config.official_unknown_prior
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Official unknown/known priors disagree with the class count")
    return result


def _design_prior(columns, config):
    result = np.full(columns, config.design_known_prior, dtype=np.float64)
    result[0] = config.design_unknown_prior
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Design unknown/known priors disagree with the class count")
    return result


def _row_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    result /= result.sum(axis=1, keepdims=True)
    return result


def powered_ratio_alignment(probabilities, valid,
                            config=PoweredRatioAlignmentConfig()):
    """Apply the preregistered one-step, label-free S014 correction.

    ``observed`` is the valid-query mean.  Every column uses its absolute
    official/observed marginal ratio, with separate exponents for unknown and
    known columns.  The preregistered unknown exponent zero gives an exact
    direct factor of one.  The common known-mass factor is intentionally kept:
    it is part of the sealed method and can move the open-set gate after final
    row normalization.
    """
    config = validate_powered_ratio_config(config)
    source, values, normalized, mask = _probability_inputs(probabilities, valid)
    design = _design_prior(values.shape[1], config)
    count = int(mask.sum())
    if count:
        observed = normalized[mask].mean(axis=0)
    else:
        observed = design.copy()
    known_mass = float(observed[1:].sum())
    if known_mass <= config.epsilon:
        observed_known = np.zeros(values.shape[1] - 1, dtype=np.float64)
    else:
        observed_known = observed[1:] / known_mass
    design_known = design[1:] / design[1:].sum()

    ratios = np.empty(values.shape[1], dtype=np.float64)
    ratios[0] = (config.design_unknown_prior
                 / max(float(observed[0]), config.epsilon))
    ratios[1:] = design[1:] / np.maximum(observed[1:], config.epsilon)
    strengths = np.full(values.shape[1], config.known_strength,
                        dtype=np.float64)
    strengths[0] = config.unknown_strength
    factors = np.power(ratios, strengths)

    if count == 0 or (config.unknown_strength == 0.0
                      and config.known_strength == 0.0):
        output = source.copy()
    else:
        output = values.copy()
        weighted = normalized[mask] * factors[None, :]
        denominators = weighted.sum(axis=1, keepdims=True)
        if (not np.isfinite(weighted).all() or not np.isfinite(denominators).all()
                or (denominators <= 0).any()):
            raise RuntimeError("Powered-ratio alignment produced invalid weights")
        output[mask] = weighted / denominators
    # Upstream invalid embeddings have deterministic unknown probabilities.
    # Enforce that release invariant even if a caller supplies another simplex.
    output[~mask] = 0.0
    output[~mask, 0] = 1.0
    adjusted = output[mask].mean(axis=0) if count else design.copy()
    if (not np.isfinite(output).all()
            or not np.allclose(output.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise RuntimeError("Powered-ratio alignment produced invalid probability rows")
    return PoweredRatioAlignmentResult(
        probabilities=output,
        observed_prior=observed.copy(),
        design_prior=design,
        observed_known_conditional=observed_known,
        design_known_conditional=design_known,
        prior_ratios=ratios,
        factors=factors,
        adjusted_prior=adjusted,
        unknown_factor=float(factors[0]),
        unknown_probability_may_change_via_row_normalization=(
            config.unknown_strength == 0.0 and config.known_strength != 0.0),
        valid_rows=count,
    )


# The short deployment-facing name intentionally resolves to the preregistered
# one-step policy.  ``soft_distribution_alignment`` below is the IPFP primitive.
distribution_alignment = powered_ratio_alignment


def soft_distribution_alignment(probabilities, valid, config=DistributionAlignmentConfig()):
    """Align valid rows to a softened official marginal without hard quotas.

    The target marginal is ``(1-strength) * observed + strength * official``.
    Invalid rows are excluded from every fitted quantity and copied unchanged.
    The solver is an iterative proportional fit in log space.  Its returned
    dual vector uses column zero as a deterministic gauge anchor.
    """
    config = validate_alignment_config(config)
    source, values, normalized, mask = _probability_inputs(probabilities, valid)
    official = _official_prior(values.shape[1], config)
    count = int(mask.sum())

    # An all-invalid batch contains no observable marginal.  The official prior
    # is a finite neutral diagnostic; the actual probability rows stay fixed.
    observed = normalized[mask].mean(axis=0) if count else official.copy()
    target = ((1.0 - config.strength) * observed
              + config.strength * official)
    target /= target.sum()

    if config.strength == 0.0 or count == 0:
        result = source.copy()
        return DistributionAlignmentResult(
            probabilities=result,
            observed_prior=observed.copy(),
            official_prior=official,
            target_prior=target,
            achieved_prior=observed.copy(),
            duals=np.zeros(values.shape[1], dtype=np.float64),
            residual_trace=np.empty(0, dtype=np.float64),
            residual=0.0,
            iterations=0,
            converged=True,
            valid_rows=count,
        )

    active = normalized[mask]
    base_logits = np.log(np.maximum(active, config.epsilon)) / config.temperature
    duals = np.zeros(values.shape[1], dtype=np.float64)
    trace = []
    aligned = None
    achieved = None
    for iteration in range(1, config.max_iterations + 1):
        aligned = _row_softmax(base_logits + duals[None, :])
        achieved = aligned.mean(axis=0)
        residual = float(np.max(np.abs(achieved - target)))
        trace.append(residual)
        if residual <= config.tolerance:
            break
        # The positive official component makes every target reachable.  A
        # floor also keeps the update finite if exp underflows temporarily.
        duals += np.log(np.maximum(target, config.epsilon)) \
            - np.log(np.maximum(achieved, config.epsilon))
        duals -= duals[0]
    else:
        iteration = config.max_iterations

    # Recompute when the last loop body updated the duals.  This ensures the
    # reported probabilities, marginal and residual describe the same dual.
    if trace[-1] > config.tolerance and iteration == config.max_iterations:
        aligned = _row_softmax(base_logits + duals[None, :])
        achieved = aligned.mean(axis=0)
        final_residual = float(np.max(np.abs(achieved - target)))
        if final_residual != trace[-1]:
            trace.append(final_residual)
    else:
        final_residual = trace[-1]

    output = values.copy()
    output[mask] = aligned
    if (not np.isfinite(output).all()
            or not np.allclose(output.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise RuntimeError("Distribution alignment produced invalid probability rows")
    return DistributionAlignmentResult(
        probabilities=output,
        observed_prior=observed.copy(),
        official_prior=official,
        target_prior=target,
        achieved_prior=achieved,
        duals=duals.copy(),
        residual_trace=np.asarray(trace, dtype=np.float64),
        residual=final_residual,
        iterations=iteration,
        converged=final_residual <= config.tolerance,
        valid_rows=count,
    )


def _embedding_inputs(view_a, view_b, valid):
    try:
        originals = np.asarray(view_a), np.asarray(view_b)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding views must be real floating matrices") from exc
    mask = np.asarray(valid)
    if (any(value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1
            or value.dtype.kind != "f" for value in originals)
            or originals[0].shape[0] != originals[1].shape[0]):
        raise ValueError("Embedding views must be aligned nonempty floating matrices")
    if mask.dtype != np.bool_ or mask.shape != (len(originals[0]),):
        raise ValueError("Validity must be an aligned boolean vector")
    normalized = []
    for value in originals:
        safe = np.zeros(value.shape, dtype=np.float64)
        active = np.asarray(value[mask], dtype=np.float64)
        if not np.isfinite(active).all():
            raise ValueError("Valid embedding rows must be finite")
        norms = np.linalg.norm(active, axis=1)
        if (norms <= 1e-12).any() or not np.isfinite(norms).all():
            raise ValueError("Valid embedding rows must have positive finite norm")
        safe[mask] = active / norms[:, None]
        normalized.append(safe)
    return originals, normalized[0], normalized[1], mask


def _topk_cutoffs(values, valid, k, block_size):
    size = len(values)
    cutoffs = np.full(size, np.inf, dtype=np.float64)
    candidates = int(valid.sum()) - 1
    if candidates <= 0:
        return cutoffs
    selected = min(k, candidates)
    columns = np.flatnonzero(valid)
    for start in range(0, size, block_size):
        stop = min(start + block_size, size)
        rows = np.arange(start, stop)
        active_rows = rows[valid[rows]]
        if not len(active_rows):
            continue
        similarities = np.clip(values[active_rows] @ values[columns].T, -1.0, 1.0)
        positions = np.searchsorted(columns, active_rows)
        similarities[np.arange(len(active_rows)), positions] = -np.inf
        cutoffs[active_rows] = np.partition(similarities, -selected, axis=1)[:, -selected]
    return cutoffs


def build_dual_view_reciprocal_graph(view_a, view_b, valid,
                                     config=QueryExpansionConfig()):
    """Build a blockwise mutual-top-k intersection from two cosine views.

    Membership at a kth-place tie is inclusive.  This makes the graph
    permutation equivariant even when more than ``k`` items share the cutoff.
    Every returned edge is undirected, unique, ordered ``i < j``, and passes
    mutual top-k independently in both views plus the fused-cosine floor.
    """
    config = validate_expansion_config(config)
    _, left, right, mask = _embedding_inputs(view_a, view_b, valid)
    size = len(left)
    cutoff_left = _topk_cutoffs(left, mask, config.k, config.block_size)
    cutoff_right = _topk_cutoffs(right, mask, config.k, config.block_size)
    all_columns = np.arange(size)
    edge_parts = []
    score_parts = []
    for start in range(0, size, config.block_size):
        stop = min(start + config.block_size, size)
        rows = np.arange(start, stop)
        cosine_left = np.clip(left[rows] @ left.T, -1.0, 1.0)
        cosine_right = np.clip(right[rows] @ right.T, -1.0, 1.0)
        fused = 0.5 * (cosine_left + cosine_right)
        accepted = (mask[rows, None] & mask[None, :]
                    & (all_columns[None, :] > rows[:, None])
                    & (cosine_left >= cutoff_left[rows, None])
                    & (cosine_left >= cutoff_left[None, :])
                    & (cosine_right >= cutoff_right[rows, None])
                    & (cosine_right >= cutoff_right[None, :])
                    & (fused >= config.fused_cosine_floor))
        local_row, column = np.nonzero(accepted)
        if len(local_row):
            edge_parts.append(np.column_stack((rows[local_row], column)))
            score_parts.append(fused[local_row, column])
    edges = (np.concatenate(edge_parts).astype(np.int64, copy=False)
             if edge_parts else np.empty((0, 2), dtype=np.int64))
    scores = (np.concatenate(score_parts).astype(np.float64, copy=False)
              if score_parts else np.empty(0, dtype=np.float64))
    degrees = np.bincount(edges.ravel(), minlength=size).astype(np.int64)
    return ReciprocalGraph(
        size=size,
        edges=edges,
        fused_cosine=scores,
        view_a_cutoff=cutoff_left,
        view_b_cutoff=cutoff_right,
        degrees=degrees,
    )


def _validate_graph(graph, size, valid):
    if not isinstance(graph, ReciprocalGraph) or type(graph.size) is not int or graph.size != size:
        raise ValueError("Reciprocal graph size differs from the embedding batch")
    edges = np.asarray(graph.edges)
    scores = np.asarray(graph.fused_cosine)
    degrees = np.asarray(graph.degrees)
    if (edges.dtype != np.int64 or edges.ndim != 2 or edges.shape[1:] != (2,)
            or scores.dtype != np.float64 or scores.shape != (len(edges),)
            or degrees.dtype != np.int64 or degrees.shape != (size,)
            or not np.isfinite(scores).all()):
        raise ValueError("Malformed reciprocal graph arrays")
    if len(edges):
        if ((edges < 0).any() or (edges >= size).any()
                or (edges[:, 0] >= edges[:, 1]).any()
                or not valid[edges].all()
                or len(np.unique(edges[:, 0] * size + edges[:, 1])) != len(edges)):
            raise ValueError("Reciprocal graph contains an invalid, repeated or self edge")
    expected = np.bincount(edges.ravel(), minlength=size).astype(np.int64)
    if not np.array_equal(degrees, expected):
        raise ValueError("Reciprocal graph degrees disagree with its edges")
    return edges, degrees


def uniform_query_expansion(view_a, view_b, valid, graph,
                            config=QueryExpansionConfig()):
    """Apply one simultaneous, uniform-neighbor update to both query views.

    For rows with neighbors the pre-normalization update is
    ``(1-beta) * query + beta * mean(neighbors)``.  The graph is never rebuilt
    from expanded values, so propagation is exactly one pass.  Invalid and
    isolated rows remain byte-for-byte equal to their inputs.
    """
    config = validate_expansion_config(config)
    originals, left, right, mask = _embedding_inputs(view_a, view_b, valid)
    edges, degrees = _validate_graph(graph, len(left), mask)
    if config.beta == 0.0:
        return QueryExpansionResult(
            view_a=originals[0].copy(),
            view_b=originals[1].copy(),
            degrees=degrees.copy(),
            expanded_rows=np.zeros(len(left), dtype=bool),
        )

    outputs = []
    expanded = mask & (degrees > 0)
    for original, normalized in zip(originals, (left, right), strict=True):
        neighbor_sum = np.zeros_like(normalized)
        if len(edges):
            np.add.at(neighbor_sum, edges[:, 0], normalized[edges[:, 1]])
            np.add.at(neighbor_sum, edges[:, 1], normalized[edges[:, 0]])
        changed = np.asarray(original).copy()
        if expanded.any():
            mean = neighbor_sum[expanded] / degrees[expanded, None]
            mixed = ((1.0 - config.beta) * normalized[expanded]
                     + config.beta * mean)
            norms = np.linalg.norm(mixed, axis=1)
            if (norms <= 1e-12).any() or not np.isfinite(norms).all():
                raise ValueError("Query expansion produced a zero or nonfinite vector")
            # Preserve the source floating dtype expected by the encoder/scorer.
            changed[expanded] = (mixed / norms[:, None]).astype(changed.dtype)
        outputs.append(changed)
    return QueryExpansionResult(
        view_a=outputs[0],
        view_b=outputs[1],
        degrees=degrees.copy(),
        expanded_rows=expanded,
    )
