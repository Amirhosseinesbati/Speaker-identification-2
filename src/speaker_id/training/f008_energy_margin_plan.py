"""Pure, deterministic margin planning for F008 energy outlier exposure.

The F008 loss needs two fixed energy boundaries before its tail-training
updates begin.  This module deliberately has no Torch, audio, model, file, or
tracking dependency: callers pass already materialized calibration energies and
persist the returned JSON-safe plan in their own protocol seal.

Energy follows F008's configured convention, ``-T * logsumexp(logits / T)``.
Lower values therefore mean that a sample lies closer to at least one known
speaker weight; higher values mean it is more out-of-set.  Only the known
energy upper quantile determines a boundary.  Unknown energies are retained
solely as diagnostics, so a diverse unknown cohort is never treated as one
shared class or allowed to move the known acceptance boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any


F008_ENERGY_MARGIN_PLAN_SCHEMA = "f008-energy-margin-plan-v1"
F008_ENERGY_ORIENTATION = "lower_energy_is_more_known__higher_energy_is_more_unknown"
F008_ENERGY_QUANTILE_METHOD = "linear"


def build_energy_margin_plan(
    known_energies: Iterable[Real],
    unknown_energies: Iterable[Real],
    energy_margin_config: Mapping[str, object],
) -> dict[str, Any]:
    """Build a JSON-safe, digest-bound F008 energy-margin plan.

    ``energy_margin_config`` is the ``energy_margin`` object from
    ``campp_f008_unknown_oe.json``.  The planner consumes its exact fields
    ``known_maximum_quantile`` and ``declared_minimum_energy_gap``.  Both
    vectors must be nonempty finite scalar iterables; only their values, not
    labels or audio identities, are present here.

    The known maximum is the deterministic linear quantile, equivalent to
    NumPy's ``quantile(..., method="linear")``.  The unknown minimum is
    *always* the known maximum plus the declared positive gap.  Unknown
    energies appear in diagnostics and do not affect either boundary.

    The caller is responsible for writing this returned plan once and
    rechecking ``plan_sha256`` before any tail update or outer evaluation.
    """
    quantile, gap = _parse_config(energy_margin_config)
    known = _finite_sorted_vector("known_energies", known_energies)
    unknown = _finite_sorted_vector("unknown_energies", unknown_energies)

    maximum_known = _linear_quantile(known, quantile)
    minimum_unknown = _finite_sum("minimum_unknown_energy", maximum_known, gap)
    if minimum_unknown <= maximum_known:
        # The finite positive gap check makes this unreachable for ordinary
        # numbers.  Retain it so a future numeric change cannot emit an
        # overlapping open-set margin.
        raise ValueError(
            "minimum_unknown_energy must be strictly greater than maximum_known_energy"
        )

    body: dict[str, Any] = {
        "schema_version": F008_ENERGY_MARGIN_PLAN_SCHEMA,
        "energy_orientation": F008_ENERGY_ORIENTATION,
        "quantile_method": F008_ENERGY_QUANTILE_METHOD,
        "known_maximum_quantile": quantile,
        "declared_minimum_energy_gap": gap,
        "maximum_known_energy": maximum_known,
        "minimum_unknown_energy": minimum_unknown,
        "known_margin_relation": "known_energy_less_than_or_equal_to_maximum_known_energy",
        "unknown_margin_relation": "unknown_energy_greater_than_or_equal_to_minimum_unknown_energy",
        "unknown_energy_role": "diagnostics_only_does_not_set_margins",
        "known_diagnostics": _diagnostics(known, maximum_known, relation="less_than_or_equal"),
        "unknown_diagnostics": _diagnostics(
            unknown, minimum_unknown, relation="greater_than_or_equal"
        ),
    }
    return {**body, "plan_sha256": _canonical_sha256(body)}


def verify_energy_margin_plan(plan: Mapping[str, object]) -> dict[str, Any]:
    """Validate the shape and digest of a previously materialized plan.

    This does not recreate the plan from source energies, which keeps it safe
    for a worker that must use one already sealed calibration result.
    """
    if not isinstance(plan, Mapping):
        raise TypeError("energy margin plan must be a mapping")
    required = {
        "schema_version",
        "energy_orientation",
        "quantile_method",
        "known_maximum_quantile",
        "declared_minimum_energy_gap",
        "maximum_known_energy",
        "minimum_unknown_energy",
        "known_margin_relation",
        "unknown_margin_relation",
        "unknown_energy_role",
        "known_diagnostics",
        "unknown_diagnostics",
        "plan_sha256",
    }
    if set(plan) != required:
        raise ValueError("energy margin plan schema changed")
    if (
        plan["schema_version"] != F008_ENERGY_MARGIN_PLAN_SCHEMA
        or plan["energy_orientation"] != F008_ENERGY_ORIENTATION
        or plan["quantile_method"] != F008_ENERGY_QUANTILE_METHOD
        or plan["known_margin_relation"]
        != "known_energy_less_than_or_equal_to_maximum_known_energy"
        or plan["unknown_margin_relation"]
        != "unknown_energy_greater_than_or_equal_to_minimum_unknown_energy"
        or plan["unknown_energy_role"] != "diagnostics_only_does_not_set_margins"
    ):
        raise ValueError("energy margin plan identity changed")

    quantile = _strict_probability("known_maximum_quantile", plan["known_maximum_quantile"])
    gap = _positive_real("declared_minimum_energy_gap", plan["declared_minimum_energy_gap"])
    maximum_known = _finite_real("maximum_known_energy", plan["maximum_known_energy"])
    minimum_unknown = _finite_real("minimum_unknown_energy", plan["minimum_unknown_energy"])
    expected_minimum = _finite_sum("minimum_unknown_energy", maximum_known, gap)
    if minimum_unknown != expected_minimum or minimum_unknown <= maximum_known:
        raise ValueError("energy margin plan boundaries are inconsistent")
    _validate_diagnostics("known_diagnostics", plan["known_diagnostics"], maximum_known)
    _validate_diagnostics("unknown_diagnostics", plan["unknown_diagnostics"], minimum_unknown)

    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected_digest = _canonical_sha256(body)
    if plan["plan_sha256"] != expected_digest:
        raise ValueError("energy margin plan digest changed")
    # Return a fresh ordinary dict, not the caller's Mapping subtype.
    return dict(plan)


def _parse_config(config: Mapping[str, object]) -> tuple[float, float]:
    if not isinstance(config, Mapping):
        raise TypeError("energy_margin_config must be a mapping")
    required = {"known_maximum_quantile", "declared_minimum_energy_gap"}
    missing = required - set(config)
    if missing:
        raise ValueError(
            "energy_margin_config is missing " + ", ".join(sorted(missing))
        )
    return (
        _strict_probability("known_maximum_quantile", config["known_maximum_quantile"]),
        _positive_real("declared_minimum_energy_gap", config["declared_minimum_energy_gap"]),
    )


def _finite_sorted_vector(name: str, values: Iterable[Real]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable of finite real values")
    result = tuple(_finite_real(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(result))


def _linear_quantile(values: tuple[float, ...], quantile: float) -> float:
    """Return the order-invariant linear sample quantile for sorted values."""
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    result = values[lower] + fraction * (values[upper] - values[lower])
    return _finite_real("maximum_known_energy", result)


def _diagnostics(
    values: tuple[float, ...], boundary: float, *, relation: str
) -> dict[str, float | int | str]:
    if relation == "less_than_or_equal":
        satisfied = sum(value <= boundary for value in values)
    elif relation == "greater_than_or_equal":
        satisfied = sum(value >= boundary for value in values)
    else:  # Defensive internal check, never user controlled.
        raise RuntimeError("unsupported energy diagnostic relation")
    return {
        "count": len(values),
        "minimum": values[0],
        "mean": _finite_real("energy diagnostic mean", math.fsum(values) / len(values)),
        "maximum": values[-1],
        "boundary_satisfied_fraction": satisfied / len(values),
    }


def _validate_diagnostics(name: str, value: object, boundary: float) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "count", "minimum", "mean", "maximum", "boundary_satisfied_fraction"
    }:
        raise ValueError(f"{name} schema changed")
    count = value["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{name}.count must be a positive integer")
    minimum = _finite_real(f"{name}.minimum", value["minimum"])
    mean = _finite_real(f"{name}.mean", value["mean"])
    maximum = _finite_real(f"{name}.maximum", value["maximum"])
    fraction = _bounded_probability(f"{name}.boundary_satisfied_fraction", value["boundary_satisfied_fraction"])
    if minimum > mean or mean > maximum or not math.isfinite(boundary):
        raise ValueError(f"{name} summary is inconsistent")
    # The original individual values are intentionally absent from the seal,
    # so the summary cannot rederive the fraction.  Its boundedness and the
    # plan digest still make accidental alteration fail closed.
    del fraction


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _finite_sum(name: str, first: float, second: float) -> float:
    return _finite_real(name, first + second)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return 0.0 if result == 0.0 else result


def _positive_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _strict_probability(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def _bounded_probability(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result
