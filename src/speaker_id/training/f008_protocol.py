"""Pure role and unknown-exposure planning primitives for F008.

F008 may train on the ``unknown_development`` population already isolated by
the C002 calibration-role protocol.  This module turns those role rows into a
small authenticated population receipt and derives each unknown exposure from
the step counter.  It deliberately has no audio, Torch, tracking, or server
dependency.

Unknown examples are sampled by content group and never receive a classifier
target.  This keeps duplicate-rich groups from gaining extra probability and
prevents callers from silently treating every unknown recording as one
speaker identity.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


ROLE_POOL_SCHEMA = "f008-c002-role-pools-v1"
UNKNOWN_EXPOSURE_SCHEMA = "f008-unknown-exposure-plan-v1"
UNKNOWN_SAMPLING = "counter_hash_uniform_group_without_replacement_then_uniform_file_v1"

_ROLE_FLAGS = {
    "known_enrollment": (True, True, False, False),
    "unknown_development": (True, False, False, False),
    "known_calibration_query": (False, False, True, False),
    "unknown_calibration_query": (False, False, True, False),
    "outer_validation": (False, False, False, True),
    "training_excluded": (False, False, False, False),
}
_FLAG_FIELDS = (
    "encoder_fit_allowed",
    "enrollment_allowed",
    "calibration_query",
    "outer_evaluation_included",
)
_POOL_FIELDS = frozenset({
    "schema_version",
    "outer_fold",
    "scope",
    "source_role_rows",
    "known_rows",
    "unknown_rows",
    "known_rows_sha256",
    "unknown_rows_sha256",
    "known_group_ids",
    "unknown_group_ids",
    "calibration_group_ids",
    "outer_group_ids",
    "excluded_group_ids",
    "sampling_unit",
    "group_disjointness",
    "signature",
})
_PLAN_FIELDS = frozenset({
    "schema_version",
    "role_pool_signature",
    "outer_fold",
    "step",
    "seed",
    "samples_per_step",
    "sampling",
    "rows",
    "signature",
})


def canonical(value: object) -> bytes:
    """Return F008's finite, deterministic JSON encoding."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"F008 {label} must be an integer")
    return value


def _row_fold(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value == value.strip() and value.isascii() and value.isdigit():
        parsed = int(value)
        if value == str(parsed):
            return parsed
    raise ValueError("F008 role outer_fold must be a canonical nonnegative integer")


def _flag(value: object, label: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str) and value == value.strip():
        normalized = value.lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"F008 role flag {label} must be boolean")


def _text(value: object, label: str) -> str:
    _require(isinstance(value, str) and value != "" and value == value.strip(),
             f"F008 role {label} must be a nonempty string")
    return value


def _normalized_fold_rows(roles: Sequence[Mapping[str, Any]], outer_fold: int) -> list[dict[str, Any]]:
    _integer(outer_fold, "outer fold")
    _require(isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)) and roles,
             "F008 requires C002-style role rows")
    selected: list[dict[str, Any]] = []
    required = {"outer_fold", "audio_file", "speaker_id", "group_id", "role", *_FLAG_FIELDS}
    for source in roles:
        _require(isinstance(source, Mapping), "F008 role rows must be objects")
        _require("outer_fold" in source, "F008 role row lacks outer_fold")
        if _row_fold(source["outer_fold"]) != outer_fold:
            continue
        _require(required <= set(source), "F008 selected role row is incomplete")
        name = _text(source["audio_file"], "audio_file")
        group = _text(source["group_id"], "group_id")
        role = _text(source["role"], "role")
        _require(role in _ROLE_FLAGS, "F008 encountered an unknown C002 role")
        flags = tuple(_flag(source[field], field) for field in _FLAG_FIELDS)
        _require(flags == _ROLE_FLAGS[role], "F008 role permissions differ from C002 semantics")

        eligible: bool | None = None
        if "source_train_eligible" in source:
            eligible = _flag(source["source_train_eligible"], "source_train_eligible")
            if role in {"known_enrollment", "unknown_development",
                        "known_calibration_query", "unknown_calibration_query"}:
                _require(eligible, "F008 permitted inner role must be training eligible")
            if role == "training_excluded":
                _require(not eligible, "F008 training_excluded role cannot be training eligible")

        normalized: dict[str, Any] = {
            "audio_file": name,
            "group_id": group,
            "role": role,
            **dict(zip(_FLAG_FIELDS, flags, strict=True)),
        }
        if eligible is not None:
            normalized["source_train_eligible"] = eligible

        # Speaker identity is materialized only for encoder-fit rows.  In
        # particular, this protocol can validate and plan before outer truth
        # is exposed by a later experiment state machine.
        if flags[0]:
            speaker = _text(source["speaker_id"], "fit speaker_id")
            if role == "unknown_development":
                _require(speaker == "unknown", "F008 unknown fit row changed identity")
            else:
                _require(speaker != "unknown", "F008 known fit row changed identity")
            normalized["speaker_id"] = speaker
        selected.append(normalized)

    _require(selected, "F008 outer fold has no role rows")
    names = [row["audio_file"] for row in selected]
    _require(len(names) == len(set(names)), "F008 outer fold has duplicate audio role rows")
    return selected


def role_pools(roles: Sequence[Mapping[str, Any]], outer_fold: int) -> dict[str, Any]:
    """Derive authenticated known and unknown encoder-fit populations.

    The input may contain role rows for several outer folds.  Only the selected
    fold is inspected.  Calibration and outer speaker labels are deliberately
    not read.
    """
    rows = _normalized_fold_rows(roles, outer_fold)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)

    fit_groups = {row["group_id"] for row in rows if row["encoder_fit_allowed"]}
    calibration_groups = {row["group_id"] for row in rows if row["calibration_query"]}
    outer_groups = {row["group_id"] for row in rows if row["outer_evaluation_included"]}
    unknown_fit_groups = {
        row["group_id"] for row in rows if row["role"] == "unknown_development"
    }
    _require(
        not unknown_fit_groups & (calibration_groups | outer_groups),
        "F008 unknown encoder-fit groups overlap calibration or outer roles",
    )
    _require(
        not fit_groups & calibration_groups
        and not fit_groups & outer_groups
        and not calibration_groups & outer_groups,
        "F008 fit, calibration and outer groups must be pairwise disjoint",
    )

    for group, members in by_group.items():
        assignments = {
            tuple(row[field] for field in _FLAG_FIELDS) for row in members
        }
        _require(len(assignments) == 1, "F008 content group crosses role permissions")
        fitting = any(row["encoder_fit_allowed"] for row in members)
        if fitting:
            labels = {row["speaker_id"] for row in members}
            roles_in_group = {row["role"] for row in members}
            _require(
                len(labels) == 1 and len(roles_in_group) == 1,
                f"F008 fit group {group} has conflicting labels or roles",
            )

    known = sorted(
        ({key: row[key] for key in ("audio_file", "speaker_id", "group_id", "role")}
         for row in rows if row["role"] == "known_enrollment"),
        key=lambda row: row["audio_file"],
    )
    unknown = sorted(
        ({key: row[key] for key in ("audio_file", "speaker_id", "group_id", "role")}
         for row in rows if row["role"] == "unknown_development"),
        key=lambda row: row["audio_file"],
    )
    _require(known, "F008 requires known encoder-fit rows")
    _require(unknown, "F008 requires unknown_development encoder-fit rows")
    known_group_ids = sorted({row["group_id"] for row in known})
    unknown_group_ids = sorted({row["group_id"] for row in unknown})
    _require(not set(known_group_ids) & set(unknown_group_ids),
             "F008 known and unknown fit groups overlap")

    excluded_groups = {
        row["group_id"] for row in rows if row["role"] == "training_excluded"
    }
    body = {
        "schema_version": ROLE_POOL_SCHEMA,
        "outer_fold": outer_fold,
        "scope": "c002_encoder_fit_allowed_rows_only",
        "source_role_rows": len(rows),
        "known_rows": known,
        "unknown_rows": unknown,
        "known_rows_sha256": _sha256(known),
        "unknown_rows_sha256": _sha256(unknown),
        "known_group_ids": known_group_ids,
        "unknown_group_ids": unknown_group_ids,
        "calibration_group_ids": sorted(calibration_groups),
        "outer_group_ids": sorted(outer_groups),
        "excluded_group_ids": sorted(excluded_groups),
        "sampling_unit": "unknown_content_group_then_file",
        "group_disjointness": {
            "unknown_fit_vs_calibration": True,
            "unknown_fit_vs_outer": True,
            "fit_vs_calibration": True,
            "fit_vs_outer": True,
            "calibration_vs_outer": True,
        },
    }
    return {**body, "signature": _sha256(body)}


def _validated_pools(pools: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(pools, Mapping) and set(pools) == _POOL_FIELDS,
             "F008 role-pool receipt schema changed")
    body = {key: pools[key] for key in pools if key != "signature"}
    _require(
        pools["schema_version"] == ROLE_POOL_SCHEMA
        and pools["scope"] == "c002_encoder_fit_allowed_rows_only"
        and pools["sampling_unit"] == "unknown_content_group_then_file"
        and pools["group_disjointness"] == {
            "unknown_fit_vs_calibration": True,
            "unknown_fit_vs_outer": True,
            "fit_vs_calibration": True,
            "fit_vs_outer": True,
            "calibration_vs_outer": True,
        }
        and pools["signature"] == _sha256(body),
        "F008 role-pool receipt identity changed",
    )
    _integer(pools["outer_fold"], "role-pool outer fold")
    _integer(pools["source_role_rows"], "source role row count", minimum=1)
    for kind, expected_role, expected_unknown in (
        ("known", "known_enrollment", False),
        ("unknown", "unknown_development", True),
    ):
        values = pools[f"{kind}_rows"]
        _require(isinstance(values, list) and values, f"F008 {kind} role pool is empty")
        _require(values == sorted(values, key=lambda row: row.get("audio_file", "")),
                 f"F008 {kind} role pool order changed")
        names: list[str] = []
        for row in values:
            _require(isinstance(row, dict)
                     and set(row) == {"audio_file", "speaker_id", "group_id", "role"},
                     f"F008 {kind} role-pool row schema changed")
            names.append(_text(row["audio_file"], f"{kind} audio_file"))
            _text(row["group_id"], f"{kind} group_id")
            speaker = _text(row["speaker_id"], f"{kind} speaker_id")
            _require(row["role"] == expected_role and ((speaker == "unknown") is expected_unknown),
                     f"F008 {kind} role-pool identity changed")
        _require(len(names) == len(set(names)), f"F008 {kind} role pool has duplicate files")
        _require(pools[f"{kind}_rows_sha256"] == _sha256(values),
                 f"F008 {kind} role-pool hash changed")
        groups = pools[f"{kind}_group_ids"]
        _require(
            isinstance(groups, list) and groups == sorted(set(groups))
            and set(groups) == {row["group_id"] for row in values},
            f"F008 {kind} role-pool groups changed",
        )
    _require(not set(pools["known_group_ids"]) & set(pools["unknown_group_ids"]),
             "F008 known and unknown role-pool groups overlap")
    for field in ("calibration_group_ids", "outer_group_ids", "excluded_group_ids"):
        values = pools[field]
        _require(isinstance(values, list) and values == sorted(set(values))
                 and all(isinstance(value, str) and value for value in values),
                 f"F008 {field} changed")
    fit = set(pools["known_group_ids"]) | set(pools["unknown_group_ids"])
    calibration, outer = set(pools["calibration_group_ids"]), set(pools["outer_group_ids"])
    _require(not fit & calibration and not fit & outer and not calibration & outer,
             "F008 role-pool groups are no longer disjoint")
    return dict(pools)


def _rank(*values: object) -> bytes:
    return hashlib.sha256(canonical(values)).digest()


def unknown_exposure_plan(
    pools: Mapping[str, Any],
    step: int,
    *,
    seed: int,
    samples_per_step: int,
) -> dict[str, Any]:
    """Return one resume-stable, group-balanced unknown exposure plan."""
    pools = _validated_pools(pools)
    _integer(step, "plan step")
    _integer(seed, "sampling seed")
    _integer(samples_per_step, "unknown samples per step", minimum=1)
    group_ids = pools["unknown_group_ids"]
    _require(samples_per_step <= len(group_ids),
             "F008 unknown sampling forbids content-group replacement")

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pools["unknown_rows"]:
        members[row["group_id"]].append(row)
    group_order = sorted(
        group_ids,
        key=lambda group: (
            _rank(seed, "F008-unknown-group-v1", pools["signature"], step, group),
            group,
        ),
    )[:samples_per_step]
    rows = []
    for slot, group in enumerate(group_order):
        choices = sorted(
            members[group],
            key=lambda row: (
                _rank(seed, "F008-unknown-file-v1", pools["signature"], step,
                      slot, group, row["audio_file"]),
                row["audio_file"],
            ),
        )
        selected = choices[0]
        crop_seed = int.from_bytes(
            _rank(seed, "F008-unknown-crop-v1", pools["signature"], step,
                  slot, group, selected["audio_file"])[:8],
            "little",
        ) & ((1 << 63) - 1)
        rows.append({
            "slot": slot,
            "audio_file": selected["audio_file"],
            "group_id": group,
            "crop_seed": crop_seed,
            "stream": "unknown_oe",
        })
    body = {
        "schema_version": UNKNOWN_EXPOSURE_SCHEMA,
        "role_pool_signature": pools["signature"],
        "outer_fold": pools["outer_fold"],
        "step": step,
        "seed": seed,
        "samples_per_step": samples_per_step,
        "sampling": UNKNOWN_SAMPLING,
        "rows": rows,
    }
    return {**body, "signature": _sha256(body)}


def validate_unknown_exposure_plan(
    plan: Mapping[str, Any],
    pools: Mapping[str, Any],
) -> None:
    """Reject a mutated or cross-fold plan before a worker can consume it."""
    pools = _validated_pools(pools)
    _require(isinstance(plan, Mapping) and set(plan) == _PLAN_FIELDS,
             "F008 unknown exposure plan schema changed")
    body = {key: plan[key] for key in plan if key != "signature"}
    _require(
        plan["schema_version"] == UNKNOWN_EXPOSURE_SCHEMA
        and plan["role_pool_signature"] == pools["signature"]
        and plan["outer_fold"] == pools["outer_fold"]
        and plan["sampling"] == UNKNOWN_SAMPLING
        and plan["signature"] == _sha256(body),
        "F008 unknown exposure plan identity changed",
    )
    expected = unknown_exposure_plan(
        pools,
        _integer(plan["step"], "plan step"),
        seed=_integer(plan["seed"], "sampling seed"),
        samples_per_step=_integer(plan["samples_per_step"], "unknown samples per step", minimum=1),
    )
    _require(dict(plan) == expected, "F008 unknown exposure plan is not counter-derived")


def unknown_plan_range_sha256(
    pools: Mapping[str, Any],
    start: int,
    stop: int,
    *,
    seed: int,
    samples_per_step: int,
) -> str:
    """Hash an exact half-open range of independently derived step plans."""
    pools = _validated_pools(pools)
    _integer(start, "plan range start")
    _integer(stop, "plan range stop", minimum=1)
    _integer(seed, "sampling seed")
    _integer(samples_per_step, "unknown samples per step", minimum=1)
    _require(start < stop, "F008 plan range must be nonempty and increasing")
    digest = hashlib.sha256()
    for step in range(start, stop):
        plan = unknown_exposure_plan(
            pools, step, seed=seed, samples_per_step=samples_per_step,
        )
        digest.update(canonical({"step": step, "plan": plan}))
    return digest.hexdigest()
