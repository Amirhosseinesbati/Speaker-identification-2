"""Focused CUDA tail worker for F008 open-set outlier exposure.

F008 is deliberately a small sibling of F005.  It starts every arm from an
authenticated F005 shared-head checkpoint, replays F005's *known* tail plans,
and adds an identity-free loss only on separately planned unknown rows.  The
module owns local checkpoints and JSONL history only: it neither chooses an
arm nor talks to MLflow.

The streaming implementation deserves one explicit note.  F005 uses eight
four-pair microbatches for a 32-pair optimizer update.  F008's OE terms are
summed over their 64 paired views and divided by 64, rather than taking eight
means and adding them.  This makes the gradient exactly equal to one OE loss
over the full intended batch while retaining F005's memory behaviour.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import time
from numbers import Real
from typing import Any, Callable

import numpy as np

from speaker_id.adaptation.open_set_oe import (
    energy_separation_oe_loss,
    raw_cosine_logits,
    uniform_oe_loss,
)
from speaker_id.adaptation.paired_views import paired_waveform_views
from speaker_id.training.f005_contract import ADVANCED_DIMENSION
from speaker_id.training.f005_runner import plan_range_sha256, training_step_plan
from speaker_id.training.f005_worker import (
    _WaveformCache,
    _atomic_checkpoint,
    _load_training_state,
    _padded_view,
    _present,
    _trim_history,
    _validate_training_state_structure,
    recover_fixed_partial,
    state_dict_sha256,
)
from speaker_id.training.f007_worker import (
    _load_authenticated_shared_head,
    _make_components,
    _set_tail_schedule,
    f007_dual_aam_task,
)
from speaker_id.training.f008_config import config_signature, validate_f008_config
from speaker_id.training.f008_preflight import verify_preflight_receipt
from speaker_id.training.f008_protocol import (
    role_pools,
    unknown_exposure_plan,
    unknown_plan_range_sha256,
    validate_unknown_exposure_plan,
)
from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_total_steps


F008_TAIL_CHECKPOINT_SCHEMA = "f008-open-set-oe-tail-checkpoint-v1"
F008_GRADIENT_PROBE_SCHEMA = "f008-open-set-oe-gradient-probe-v1"
F008_RUNTIME_RECEIPT_SCHEMA = "f008-runtime-receipt-v1"
F008_TRAINABLE_ARM_IDS = ("energy_005", "uniform_005")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in _SHA256_CHARACTERS for character in value),
             f"F008 {label} must be a lowercase SHA-256")
    return value


def _f008_config(config_or_contract: Mapping[str, object]) -> dict[str, Any]:
    _require(isinstance(config_or_contract, Mapping), "F008 config/contract must be an object")
    candidate = config_or_contract.get("config", config_or_contract)
    _require(isinstance(candidate, Mapping), "F008 configuration is missing")
    config = dict(candidate)
    validate_f008_config(config)
    return config


def _f008_signature(config_or_contract: Mapping[str, object]) -> str:
    config = _f008_config(config_or_contract)
    supplied = config_or_contract.get("signature") if isinstance(config_or_contract, Mapping) else None
    if supplied is None:
        return config_signature(config)
    return _sha256(supplied, "experiment signature")


def _runtime_binding(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object],
) -> tuple[dict[str, Any], str, str, str]:
    """Return the small, immutable runtime binding without reading audio or Torch.

    F008 deliberately inherits F005's authorized Vast marker.  This is a
    narrow identity check, not a second readiness or audio audit: the F005
    source/role receipts have already been authenticated before a tail can be
    reached.
    """
    config = _f008_config(f008_config_or_contract)
    f008_signature = _f008_signature(f008_config_or_contract)
    _require(
        f008_signature == config_signature(config),
        "F008 runtime requires a configuration-matching experiment signature",
    )
    _require(isinstance(f005_contract, Mapping), "F008 runtime requires an F005 contract")
    f005_signature = _sha256(f005_contract.get("signature"), "F005 experiment signature")
    _require(
        f005_signature == config["source_f005"]["experiment_signature"],
        "F008 runtime F005 signature differs from the declared source",
    )
    f005_config = f005_contract.get("config")
    readiness = f005_contract.get("readiness")
    _require(isinstance(f005_config, Mapping) and isinstance(readiness, Mapping),
             "F008 runtime requires F005 config and readiness evidence")
    expected_instance = readiness.get("config", {}).get("expected_vast_instance_id")
    _require(
        (type(expected_instance) is int and expected_instance > 0)
        or (isinstance(expected_instance, str) and expected_instance.isdecimal()),
        "F008 runtime F005 expected Vast instance is invalid",
    )
    expected_instance_text = str(expected_instance)
    execution = config["execution"]
    _require(
        f005_config.get("execution") == execution
        and f005_config.get("device") == config["device"] == "cuda"
        and f005_config.get("cpu_threads") == config["cpu_threads"] == 4,
        "F008 runtime execution differs from the authenticated F005 source",
    )
    return config, f008_signature, f005_signature, expected_instance_text


def attest_f008_runtime(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object],
) -> dict[str, Any]:
    """Perform F008's one targeted CUDA check and issue a reusable receipt.

    The launcher calls this exactly once for a logical F008 run.  It must run
    before a model is constructed: CUBLAS and numerical thread variables are
    checked before importing Torch.  It intentionally does not re-read the
    manifest or hash audio; those costly source checks belong to F005's sealed
    readiness evidence.
    """
    config, f008_signature, f005_signature, expected_instance = _runtime_binding(
        f008_config_or_contract, f005_contract,
    )
    execution = config["execution"]
    expected_threads = execution["thread_environment"]
    _require(
        os.environ.get("VAST_INSTANCE_ID") == expected_instance,
        "F008 runtime requires F005's authorized Vast instance marker",
    )
    # These checks intentionally precede the local Torch import.
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == execution["cublas_workspace_config"] == ":4096:8",
        "F008 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import",
    )
    _require(
        all(os.environ.get(key) == value for key, value in expected_threads.items()),
        "F008 numerical thread environment differs from the execution contract",
    )
    _require(
        config["training"]["mixed_precision"] is False
        and execution["tensor_dtype"] == "float32"
        and execution["no_cpu_fallback"] is True
        and execution["deterministic_algorithms"] == "enforce_error",
        "F008 FP32 deterministic execution policy changed",
    )
    import torch

    _require(config["device"] == "cuda" and torch.cuda.is_available(),
             "F008 requires CUDA and forbids CPU fallback")
    gpu_name = torch.cuda.get_device_name(0)
    _require(execution["gpu_name_contains"] in gpu_name,
             "F008 requires the authorized RTX 3090")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    _require(
        torch.get_default_dtype() == torch.float32
        and torch.get_num_threads() == config["cpu_threads"]
        and torch.are_deterministic_algorithms_enabled() is True
        and torch.backends.cudnn.benchmark is False,
        "F008 runtime could not enforce the declared FP32 deterministic settings",
    )
    body = {
        "schema_version": F008_RUNTIME_RECEIPT_SCHEMA,
        "f008_signature": f008_signature,
        "source_f005_signature": f005_signature,
        "expected_vast_instance_id": expected_instance,
        "vast_instance_id": expected_instance,
        "execution": execution,
        "device": "cuda",
        "cuda_available": True,
        "gpu_name": gpu_name,
        "cpu_threads": config["cpu_threads"],
        "torch_default_dtype": "float32",
        "torch_num_threads": config["cpu_threads"],
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cublas_checked_before_torch_import": True,
        "thread_environment_checked_before_torch_import": True,
        "checked_once_per_logical_run": True,
    }
    return {**body, "receipt_sha256": _sha(body)}


def validate_f008_runtime_receipt(
    receipt: Mapping[str, object], f008_config_or_contract: Mapping[str, object],
    f005_contract: Mapping[str, object],
) -> dict[str, Any]:
    """Validate the one-time F008 runtime receipt without rebuilding a model."""
    config, f008_signature, f005_signature, expected_instance = _runtime_binding(
        f008_config_or_contract, f005_contract,
    )
    _require(isinstance(receipt, Mapping), "F008 worker requires a runtime receipt")
    value = dict(receipt)
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version", "f008_signature", "source_f005_signature",
        "expected_vast_instance_id", "vast_instance_id", "execution", "device",
        "cuda_available", "gpu_name", "cpu_threads", "torch_default_dtype",
        "torch_num_threads", "deterministic_algorithms_enabled", "cudnn_benchmark",
        "cublas_checked_before_torch_import", "thread_environment_checked_before_torch_import",
        "checked_once_per_logical_run",
    }
    _require(set(body) == required and set(value) == required | {"receipt_sha256"},
             "F008 runtime receipt schema changed")
    _require(
        value.get("receipt_sha256") == _sha(body)
        and body.get("schema_version") == F008_RUNTIME_RECEIPT_SCHEMA
        and body.get("f008_signature") == f008_signature
        and body.get("source_f005_signature") == f005_signature
        and body.get("expected_vast_instance_id") == expected_instance
        and body.get("vast_instance_id") == expected_instance
        and body.get("execution") == config["execution"]
        and body.get("device") == "cuda"
        and body.get("cuda_available") is True
        and isinstance(body.get("gpu_name"), str)
        and config["execution"]["gpu_name_contains"] in body["gpu_name"]
        and body.get("cpu_threads") == config["cpu_threads"]
        and body.get("torch_default_dtype") == "float32"
        and body.get("torch_num_threads") == config["cpu_threads"]
        and body.get("deterministic_algorithms_enabled") is True
        and body.get("cudnn_benchmark") is False
        and body.get("cublas_checked_before_torch_import") is True
        and body.get("thread_environment_checked_before_torch_import") is True
        and body.get("checked_once_per_logical_run") is True,
        "F008 runtime receipt is invalid or belongs to another run",
    )
    # These are inexpensive launch-marker checks, not a repeated CUDA or data
    # audit.  They stop a receipt from being replayed by a differently launched
    # process before it constructs a live model.
    _require(os.environ.get("VAST_INSTANCE_ID") == expected_instance,
             "F008 runtime receipt is being replayed on another Vast instance")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == config["execution"]["cublas_workspace_config"],
             "F008 runtime receipt is being replayed without its CUBLAS setting")
    _require(
        all(os.environ.get(key) == item
            for key, item in config["execution"]["thread_environment"].items()),
        "F008 runtime receipt is being replayed with another thread environment",
    )
    return value


def f008_arm(config_or_contract: Mapping[str, object], arm_id: str) -> dict[str, Any]:
    """Return one declared trainable F008 arm and reject the reused control."""
    config = _f008_config(config_or_contract)
    _require(isinstance(arm_id, str), "F008 arm id must be a string")
    arms = config["arms"]
    _require(isinstance(arms, list) and tuple(
        item.get("id") for item in arms if isinstance(item, Mapping)
    ) == ("control_f005",) + F008_TRAINABLE_ARM_IDS, "F008 arm order changed")
    arm = next((dict(item) for item in arms if item["id"] == arm_id), None)
    expected = {
        "energy_005": {"id": "energy_005", "kind": "energy_margin_outlier_exposure", "lambda": 0.05},
        "uniform_005": {"id": "uniform_005", "kind": "uniform_outlier_exposure", "lambda": 0.05},
    }
    _require(arm_id in expected and arm == expected[arm_id],
             "F008 worker can train only one declared OE arm")
    return arm


def _tail_bounds(f005_fit: Mapping[str, object], f008_config: Mapping[str, object]) -> tuple[int, int]:
    _require(isinstance(f005_fit, Mapping), "F008 requires an F005 fit configuration")
    head_steps = f005_fit.get("adaptation_schedule", {}).get("head_only_steps")
    total = adaptation_total_steps(dict(f005_fit))
    training = f008_config["training"]
    _require(head_steps == 600 and total == 1100
             and training["tail_steps"] == total - head_steps == 500,
             "F008 is pinned to F005's completed 600+500 schedule")
    _require(f005_fit.get("batch_pairs") == training["known_batch_pairs"] == 32
             and training["unknown_batch_pairs"] == 32
             and f005_fit.get("microbatch_pairs") == training["microbatch_pairs"] == 4,
             "F008 paired batch protocol differs from F005")
    return int(head_steps), int(total)


def _validated_preflight(
    receipt: Mapping[str, object], *, f008_signature: str, f005_signature: str,
    source_receipt: Mapping[str, object], outer_fold: int, role_pool: Mapping[str, object],
    shared_head_sha256: str,
) -> dict[str, Any]:
    """Check that the immutable source-energy receipt belongs to this tail."""
    preflight = verify_preflight_receipt(receipt)
    _require(
        preflight["f008_signature"] == f008_signature
        and preflight["f005_signature"] == f005_signature
        and preflight["f005_source_receipt_sha256"] == _sha(dict(source_receipt))
        and preflight["outer_fold"] == outer_fold
        and preflight["role_pool_signature"] == role_pool["signature"]
        and preflight["shared_head_checkpoint_sha256"] == shared_head_sha256,
        "F008 preflight receipt belongs to another source, fold, role pool or head",
    )
    _require(
        preflight["known_energy_views"] == 2 * len(role_pool["known_rows"])
        and preflight["unknown_energy_views"] == 2 * len(role_pool["unknown_rows"]),
        "F008 preflight energy counts differ from the sealed role population",
    )
    return preflight


def f008_tail_identity(
    f008_config_or_contract: Mapping[str, object],
    f005_contract: Mapping[str, object],
    outer_fold: int,
    arm: Mapping[str, object],
    *,
    shared_head_checkpoint_sha256: str,
    known_tail_plan_sha256: str,
    unknown_plan_sha256: str,
    role_pool: Mapping[str, object],
    preflight_receipt: Mapping[str, object],
    source_receipt: Mapping[str, object],
    runtime_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the complete immutable identity for one F008 tail checkpoint."""
    config = _f008_config(f008_config_or_contract)
    signature = _f008_signature(f008_config_or_contract)
    _require(type(outer_fold) is int and outer_fold in config["fold_ids"],
             "F008 outer fold is not configured")
    expected_arm = f008_arm(config, arm.get("id") if isinstance(arm, Mapping) else "")
    _require(dict(arm) == expected_arm, "F008 arm identity is malformed")
    _require(isinstance(f005_contract, Mapping), "F008 requires an F005 contract")
    f005_signature = _sha256(f005_contract.get("signature"), "F005 experiment signature")
    _require(f005_signature == config["source_f005"]["experiment_signature"],
             "F008 F005 source signature differs from configuration")
    fit = f005_contract.get("config", {}).get("fit")
    _require(isinstance(fit, Mapping), "F008 F005 fit configuration is missing")
    head_steps, total = _tail_bounds(fit, config)
    validate_unknown_exposure_plan(
        unknown_exposure_plan(role_pool, head_steps, seed=config["seed"], samples_per_step=32),
        role_pool,
    )
    for label, value in (
        ("shared-head checkpoint", shared_head_checkpoint_sha256),
        ("known F005 tail plan", known_tail_plan_sha256),
        ("unknown exposure plan", unknown_plan_sha256),
        ("runtime receipt", runtime_receipt_sha256),
    ):
        _sha256(value, label)
    _require(isinstance(role_pool, Mapping) and role_pool.get("outer_fold") == outer_fold,
             "F008 role pool is missing or belongs to another fold")
    preflight = _validated_preflight(
        preflight_receipt, f008_signature=signature, f005_signature=f005_signature,
        source_receipt=source_receipt, outer_fold=outer_fold, role_pool=role_pool,
        shared_head_sha256=shared_head_checkpoint_sha256,
    )
    margin = preflight["energy_margin_plan"]
    body = {
        "schema_version": 1,
        "f008_signature": signature,
        "source_f005_signature": f005_signature,
        "f005_source_receipt_sha256": _sha(dict(source_receipt)),
        "outer_fold": outer_fold,
        "arm": expected_arm,
        "shared_head_checkpoint_sha256": shared_head_checkpoint_sha256,
        "f005_tail_plan_sha256": known_tail_plan_sha256,
        "unknown_role_pool_signature": role_pool["signature"],
        "unknown_plan_sha256": unknown_plan_sha256,
        "unknown_plan_step_range": {"start": head_steps, "stop": total},
        "unknown_sampling_seed": config["seed"],
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "energy_margin_plan_sha256": margin["plan_sha256"],
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "objective": (
            "mean_short_long_aam_plus_energy_separation_oe"
            if expected_arm["id"] == "energy_005"
            else "mean_short_long_aam_plus_uniform_oe"
        ),
        "oe_full_batch_views": 2 * config["training"]["unknown_batch_pairs"],
        "embedding_dimension": ADVANCED_DIMENSION,
        "checkpoint_scope": "server_only_until_promotion",
    }
    return {**body, "signature": _sha(body)}


def f008_tail_checkpoint_metadata(
    identity: Mapping[str, object], f005_fit: Mapping[str, object], completed_steps: int,
) -> dict[str, Any]:
    """Return resume-safe checkpoint metadata for an exact F008 tail boundary."""
    total = adaptation_total_steps(dict(f005_fit))
    head_steps = f005_fit["adaptation_schedule"]["head_only_steps"]
    _require(type(completed_steps) is int and head_steps <= completed_steps <= total,
             "F008 tail checkpoint step is outside the F005 tail range")
    _require(identity.get("embedding_dimension") == ADVANCED_DIMENSION
             and identity.get("oe_full_batch_views") == 64
             and isinstance(identity.get("runtime_receipt_sha256"), str),
             "F008 checkpoint identity is malformed")
    _sha256(identity["runtime_receipt_sha256"], "checkpoint runtime receipt")
    body = {
        "format_version": 1,
        "checkpoint_schema": F008_TAIL_CHECKPOINT_SCHEMA,
        "stage": "tail",  # Keep F005's structural optimizer validator applicable.
        "f008_stage": "open_set_oe_tail",
        "f008_signature": identity["f008_signature"],
        "source_f005_signature": identity["source_f005_signature"],
        "f005_source_receipt_sha256": identity["f005_source_receipt_sha256"],
        "arm_signature": identity["signature"],
        "outer_fold": identity["outer_fold"],
        "arm_id": identity["arm"]["id"],
        "arm_kind": identity["arm"]["kind"],
        "oe_lambda": identity["arm"]["lambda"],
        "shared_head_checkpoint_sha256": identity["shared_head_checkpoint_sha256"],
        "f005_tail_plan_sha256": identity["f005_tail_plan_sha256"],
        "unknown_role_pool_signature": identity["unknown_role_pool_signature"],
        "unknown_plan_sha256": identity["unknown_plan_sha256"],
        "unknown_plan_step_range": identity["unknown_plan_step_range"],
        "unknown_sampling_seed": identity["unknown_sampling_seed"],
        "preflight_receipt_sha256": identity["preflight_receipt_sha256"],
        "energy_margin_plan_sha256": identity["energy_margin_plan_sha256"],
        "runtime_receipt_sha256": identity["runtime_receipt_sha256"],
        "objective": identity["objective"],
        "oe_full_batch_views": identity["oe_full_batch_views"],
        "embedding_dimension": ADVANCED_DIMENSION,
        "completed_steps": completed_steps,
        "total_steps": total,
        "schedule_state": adaptation_checkpoint_state(dict(f005_fit), completed_steps),
        "oe_averaged_over_full_32_pair_batch": True,
        "unknown_aam_target_assigned": False,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    return {**body, "metadata_sha256": _sha(body)}


def validate_f008_resume_payload(
    payload: Mapping[str, object], identity: Mapping[str, object], f005_fit: Mapping[str, object],
) -> dict[str, Any]:
    """Reject a resume checkpoint unless every source/protocol seal matches."""
    required = {"metadata", "encoder", "head", "optimizer", "torch_rng", "cuda_rng"}
    _require(isinstance(payload, Mapping) and set(payload) == required,
             "F008 checkpoint payload fields differ from the fixed format")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping) and type(metadata.get("completed_steps")) is int,
             "F008 checkpoint metadata is incomplete")
    expected = f008_tail_checkpoint_metadata(identity, f005_fit, metadata["completed_steps"])
    _require(dict(metadata) == expected,
             "F008 resume checkpoint belongs to another arm, fold, source, plan, preflight or step")
    return expected


def _validate_cosine_logits(logits: object, *, name: str) -> None:
    import torch

    _require(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
             and logits.ndim == 2 and logits.shape[0] > 0
             and logits.shape[1] == 446 and logits.device.type in {"cpu", "cuda"}
             and bool(torch.isfinite(logits.detach()).all().item())
             and float(logits.detach().min().cpu()) >= -1.00001
             and float(logits.detach().max().cpu()) <= 1.00001,
             f"F008 {name} must be finite FP32 [rows,446] cosine logits")


def _energy_settings(
    energy_margin_config: Mapping[str, object], energy_margin_plan: Mapping[str, object],
) -> tuple[float, float, float, float, float]:
    _require(isinstance(energy_margin_config, Mapping) and isinstance(energy_margin_plan, Mapping),
             "F008 energy configuration is missing")
    temperature = energy_margin_config.get("energy_temperature")
    softplus = energy_margin_config.get("softplus_temperature")
    known_weight = energy_margin_config.get("known_energy_weight")
    unknown_weight = energy_margin_config.get("unknown_energy_weight")
    maximum_known = energy_margin_plan.get("maximum_known_energy")
    minimum_unknown = energy_margin_plan.get("minimum_unknown_energy")
    _require(temperature == 0.05 and softplus == 0.05
             and known_weight == unknown_weight == 1.0
             and isinstance(maximum_known, (int, float)) and not isinstance(maximum_known, bool)
             and isinstance(minimum_unknown, (int, float)) and not isinstance(minimum_unknown, bool)
             and float(minimum_unknown) > float(maximum_known),
             "F008 sealed energy settings changed")
    return (float(temperature), float(softplus), float(maximum_known),
            float(minimum_unknown), float(known_weight))


def f008_oe_loss(
    known_cosine_logits,
    unknown_cosine_logits,
    arm: Mapping[str, object],
    *,
    energy_margin_config: Mapping[str, object],
    energy_margin_plan: Mapping[str, object],
) -> tuple["Any", dict[str, Any]]:
    """Calculate exactly one declared OE objective across complete view batches.

    ``known_cosine_logits`` and ``unknown_cosine_logits`` must each contain
    both paired views.  Uniform OE deliberately ignores known logits; they are
    accepted as ``None`` to make it impossible for a caller to accidentally
    give unknown rows an AAM target merely to manufacture a known branch.
    """
    arm_id = arm.get("id") if isinstance(arm, Mapping) else None
    _require(arm_id in F008_TRAINABLE_ARM_IDS, "F008 OE arm is invalid")
    _validate_cosine_logits(unknown_cosine_logits, name="unknown OE logits")
    temperature, softplus, maximum_known, minimum_unknown, known_weight = _energy_settings(
        energy_margin_config, energy_margin_plan,
    )
    if arm_id == "energy_005":
        _validate_cosine_logits(known_cosine_logits, name="known energy logits")
        loss, diagnostics = energy_separation_oe_loss(
            known_cosine_logits, unknown_cosine_logits,
            energy_temperature=temperature, maximum_known_energy=maximum_known,
            minimum_unknown_energy=minimum_unknown, softplus_temperature=softplus,
        )
        _require(known_weight == 1.0, "F008 known energy weight changed")
        return loss, dict(diagnostics)
    _require(known_cosine_logits is None, "F008 uniform OE does not consume known cosine logits")
    loss, diagnostics = uniform_oe_loss(unknown_cosine_logits, temperature=temperature)
    return loss, dict(diagnostics)


def f008_combined_objective(
    known_short_h,
    known_long_h,
    known_targets,
    unknown_short_h,
    unknown_long_h,
    head,
    arm: Mapping[str, object],
    *,
    normalization_pairs: int,
    energy_margin_config: Mapping[str, object],
    energy_margin_plan: Mapping[str, object],
) -> tuple["Any", dict[str, Any]]:
    """Reference full-batch objective used to test the streaming update path."""
    import torch

    task, task_diagnostics = f007_dual_aam_task(
        known_short_h, known_long_h, known_targets, head,
        normalization_pairs=normalization_pairs,
    )
    unknown_logits = raw_cosine_logits(torch.cat((unknown_short_h, unknown_long_h), dim=0), head.weight)
    if arm["id"] == "energy_005":
        known_logits = raw_cosine_logits(torch.cat((known_short_h, known_long_h), dim=0), head.weight)
    else:
        known_logits = None
    oe, oe_diagnostics = f008_oe_loss(
        known_logits, unknown_logits, arm,
        energy_margin_config=energy_margin_config, energy_margin_plan=energy_margin_plan,
    )
    total = task + float(arm["lambda"]) * oe
    return total, {
        "task": task_diagnostics,
        "oe": oe_diagnostics,
        "task_loss": float(task.detach().cpu()),
        "oe_loss": float(oe.detach().cpu()),
        "combined_loss": float(total.detach().cpu()),
        "unknown_aam_target_assigned": False,
        "oe_views": int(unknown_logits.shape[0]),
    }


def _streaming_oe_contribution(
    cosine_logits,
    arm: Mapping[str, object],
    *,
    stream: str,
    total_unknown_views: int,
    energy_margin_config: Mapping[str, object],
    energy_margin_plan: Mapping[str, object],
) -> tuple["Any", dict[str, float | int]]:
    """Return one microbatch's *sum-normalized* OE contribution.

    Summing all returned tensors for a step produces exactly
    :func:`f008_oe_loss` on all 64 unknown views (and 64 known views for the
    energy arm).  The denominator is never the local four-pair microbatch.
    """
    import torch
    from torch.nn import functional as F

    _validate_cosine_logits(cosine_logits, name=f"{stream} streaming OE logits")
    _require(stream in {"known", "unknown"}, "F008 OE stream is invalid")
    _require(type(total_unknown_views) is int and total_unknown_views == 64,
             "F008 OE must be normalized over exactly 32 paired unknown examples")
    arm_id = arm.get("id") if isinstance(arm, Mapping) else None
    temperature, softplus, maximum_known, minimum_unknown, known_weight = _energy_settings(
        energy_margin_config, energy_margin_plan,
    )
    if arm_id == "energy_005":
        energy = -temperature * torch.logsumexp(cosine_logits / temperature, dim=1)
        if stream == "known":
            penalty = F.softplus((energy - maximum_known) / softplus)
            contribution = 0.5 * known_weight * penalty.sum() / total_unknown_views
        else:
            penalty = F.softplus((minimum_unknown - energy) / softplus)
            contribution = 0.5 * penalty.sum() / total_unknown_views
        return contribution, {
            "rows": int(cosine_logits.shape[0]),
            "energy_sum": float(energy.detach().sum().cpu()),
            "penalty_sum": float(penalty.detach().sum().cpu()),
            "contribution": float(contribution.detach().cpu()),
        }
    _require(arm_id == "uniform_005" and stream == "unknown",
             "F008 uniform OE is unknown-only")
    per_row = -torch.log_softmax(cosine_logits / temperature, dim=1).mean(dim=1)
    contribution = per_row.sum() / total_unknown_views
    return contribution, {
        "rows": int(cosine_logits.shape[0]),
        "uniform_cross_entropy_sum": float(per_row.detach().sum().cpu()),
        "contribution": float(contribution.detach().cpu()),
    }


def _unknown_microbatch(
    f005_contract: Mapping[str, object], plan_rows: Sequence[Mapping[str, object]], root: Path,
    waveform_cache: _WaveformCache | None = None, io_stats: dict[str, float | int] | None = None,
) -> dict[str, "Any"]:
    """Decode paired unknown views without manufacturing a classifier target."""
    import torch
    from speaker_id.models.campp import make_fbank, read_mono

    _require(plan_rows, "F008 unknown microbatch is empty")
    views = f005_contract["config"]["views"]
    short_samples = round(views["short_seconds"] * views["sample_rate"])
    long_samples = round(views["long_seconds"] * views["sample_rate"])
    data = Path(root) / f005_contract["readiness"]["config"]["data_dir"]
    short_features, long_features, real_short, real_long, eligible = [], [], [], [], []
    for item in plan_rows:
        _require(isinstance(item, Mapping) and set(item) == {
            "slot", "audio_file", "group_id", "crop_seed", "stream"
        } and item["stream"] == "unknown_oe" and "target" not in item,
                 "F008 unknown plan row may not contain an AAM target")
        name = item["audio_file"]
        _require(isinstance(name, str) and name, "F008 unknown audio filename is invalid")
        signal = None if waveform_cache is None else waveform_cache.get(name)
        if signal is None:
            started = time.monotonic()
            signal = read_mono(data / name, sample_rate=views["sample_rate"])
            signal.setflags(write=False)
            if waveform_cache is not None:
                waveform_cache.put(name, signal)
            if io_stats is not None:
                io_stats["decode_seconds"] += time.monotonic() - started
                io_stats["decode_misses"] += 1
        elif io_stats is not None:
            io_stats["cache_hits"] += 1
        short, long, metadata = paired_waveform_views(
            signal, rng=np.random.default_rng(item["crop_seed"]),
            short_seconds=views["short_seconds"], long_seconds=views["long_seconds"],
            sample_rate=views["sample_rate"],
        )
        short_features.append(make_fbank(_padded_view(short, short_samples)))
        long_features.append(make_fbank(_padded_view(long, long_samples)))
        real_short.append(metadata["student"]["real_samples"])
        real_long.append(metadata["teacher"]["real_samples"])
        eligible.append(metadata["consistency_mask"])
    result = {
        "short": torch.stack(short_features), "long": torch.stack(long_features),
        "student_real_samples": torch.tensor(real_short, dtype=torch.int64),
        "teacher_real_samples": torch.tensor(real_long, dtype=torch.int64),
        "eligible": torch.tensor(eligible, dtype=torch.bool),
    }
    _require("targets" not in result, "F008 unknown batch unexpectedly has AAM targets")
    return result


def _known_cpu_batches(
    f005_contract: Mapping[str, object], root: Path, outer_fold: int, step: int,
    cache: _WaveformCache, io_stats: dict[str, float | int],
) -> list[dict[str, Any]]:
    from speaker_id.training.f005_worker import _microbatch

    fit = f005_contract["config"]["fit"]
    plan = training_step_plan(dict(f005_contract), outer_fold, step)
    _require(len(plan) == 32 and all(type(row.get("target")) is int and 0 <= row["target"] < 446
                                     for row in plan),
             "F008 known plan is not the exact 32-pair F005 AAM plan")
    micro = fit["microbatch_pairs"]
    return [
        _microbatch(dict(f005_contract), plan[start:start + micro], Path(root), cache, io_stats)
        for start in range(0, len(plan), micro)
    ]


def _unknown_cpu_batches(
    f005_contract: Mapping[str, object], root: Path, pools: Mapping[str, object], step: int,
    *, seed: int, cache: _WaveformCache, io_stats: dict[str, float | int],
) -> list[dict[str, Any]]:
    fit = f005_contract["config"]["fit"]
    plan = unknown_exposure_plan(pools, step, seed=seed, samples_per_step=32)
    validate_unknown_exposure_plan(plan, pools)
    rows = plan["rows"]
    _require(len(rows) == 32 and all("target" not in row for row in rows),
             "F008 unknown exposure plan must contain 32 target-free rows")
    micro = fit["microbatch_pairs"]
    return [
        _unknown_microbatch(f005_contract, rows[start:start + micro], Path(root), cache, io_stats)
        for start in range(0, len(rows), micro)
    ]


def _move_to_cuda(cpu_batch: Mapping[str, object]) -> dict[str, Any]:
    return {key: value.to("cuda", non_blocking=False) for key, value in cpu_batch.items()}


def _assert_embeddings(short_h, long_h, rows: int) -> None:
    import torch

    _require(short_h.shape == long_h.shape == (rows, ADVANCED_DIMENSION)
             and short_h.dtype == torch.float32 and long_h.dtype == torch.float32,
             "F008 advanced encoder output is not FP32 [batch,192]")


def _grad_norm(parameters: Sequence[object]) -> float:
    import torch

    squares = []
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            squares.append(gradient.detach().float().square().sum())
    if not squares:
        return 0.0
    result = torch.sqrt(torch.stack(squares).sum())
    _require(bool(torch.isfinite(result).item()), "F008 gradient norm is nonfinite")
    return float(result.cpu())


def _backward_f008_step(
    f005_contract: Mapping[str, object], root: Path, outer_fold: int, step: int,
    arm: Mapping[str, object], pools: Mapping[str, object], encoder, head,
    *, energy_margin_config: Mapping[str, object], energy_margin_plan: Mapping[str, object],
    unknown_sampling_seed: int, cache: _WaveformCache, io_stats: dict[str, float | int],
) -> dict[str, float | int]:
    """Accumulate one exact full-batch F008 gradient, without an optimizer step."""
    fit = f005_contract["config"]["fit"]
    known_batches = _known_cpu_batches(f005_contract, root, outer_fold, step, cache, io_stats)
    unknown_batches = _unknown_cpu_batches(
        f005_contract, root, pools, step, seed=unknown_sampling_seed, cache=cache, io_stats=io_stats,
    )
    _require(len(known_batches) == len(unknown_batches) == 8,
             "F008 must use eight paired four-example microbatches per stream")
    aggregate: defaultdict[str, float] = defaultdict(float)
    total_views = 64
    for cpu_batch in known_batches:
        batch = _move_to_cuda(cpu_batch)
        rows = len(batch["targets"])
        short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
        _assert_embeddings(short_h, long_h, rows)
        task, diagnostics = f007_dual_aam_task(
            short_h, long_h, batch["targets"], head, normalization_pairs=32,
        )
        loss = task
        for key, value in diagnostics.items():
            aggregate[f"known_{key}"] += float(value)
        if arm["id"] == "energy_005":
            import torch
            cosine = raw_cosine_logits(torch.cat((short_h, long_h), dim=0), head.weight)
            contribution, details = _streaming_oe_contribution(
                cosine, arm, stream="known", total_unknown_views=total_views,
                energy_margin_config=energy_margin_config, energy_margin_plan=energy_margin_plan,
            )
            loss = loss + float(arm["lambda"]) * contribution
            aggregate["known_oe_contribution"] += details["contribution"]
            aggregate["known_energy_sum"] += details["energy_sum"]
            aggregate["known_oe_penalty_sum"] += details["penalty_sum"]
            aggregate["known_oe_rows"] += details["rows"]
        loss.backward()
        aggregate["task_loss"] += float(task.detach().cpu())
    for cpu_batch in unknown_batches:
        batch = _move_to_cuda(cpu_batch)
        _require("targets" not in batch, "F008 unknown batch cannot enter AAM")
        rows = len(batch["short"])
        short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
        _assert_embeddings(short_h, long_h, rows)
        import torch
        cosine = raw_cosine_logits(torch.cat((short_h, long_h), dim=0), head.weight)
        contribution, details = _streaming_oe_contribution(
            cosine, arm, stream="unknown", total_unknown_views=total_views,
            energy_margin_config=energy_margin_config, energy_margin_plan=energy_margin_plan,
        )
        weighted = float(arm["lambda"]) * contribution
        weighted.backward()
        aggregate["unknown_oe_contribution"] += details["contribution"]
        aggregate["unknown_oe_rows"] += details["rows"]
        if arm["id"] == "energy_005":
            aggregate["unknown_energy_sum"] += details["energy_sum"]
            aggregate["unknown_oe_penalty_sum"] += details["penalty_sum"]
        else:
            aggregate["unknown_uniform_cross_entropy_sum"] += details["uniform_cross_entropy_sum"]
    _require(aggregate["unknown_oe_rows"] == 64
             and (arm["id"] != "energy_005" or aggregate["known_oe_rows"] == 64),
             "F008 OE did not use both short and long views of the full batch")
    aggregate["oe_loss"] = aggregate["known_oe_contribution"] + aggregate["unknown_oe_contribution"]
    aggregate["combined_loss"] = aggregate["task_loss"] + float(arm["lambda"]) * aggregate["oe_loss"]
    aggregate["unknown_aam_target_assigned"] = 0.0
    aggregate["oe_full_batch_views"] = float(total_views)
    return dict(aggregate)


def _probe_receipt(identity: Mapping[str, object], *, step: int, encoder, head,
                   update_metrics: Mapping[str, float | int]) -> dict[str, Any]:
    """Seal raw head/encoder gradient norms after a backward-only update."""
    encoder_norm = _grad_norm([parameter for parameter in encoder.parameters() if parameter.requires_grad])
    head_norm = _grad_norm(list(head.parameters()))
    total_norm = float((encoder_norm ** 2 + head_norm ** 2) ** 0.5)
    body = {
        "schema_version": F008_GRADIENT_PROBE_SCHEMA,
        "arm_signature": identity["signature"],
        "outer_fold": identity["outer_fold"],
        "runtime_receipt_sha256": identity["runtime_receipt_sha256"],
        "probe_step": step,
        "optimizer_steps_persisted": 0,
        "source_f005_optimizer_and_rng_restored_before_probe": True,
        "raw_encoder_gradient_norm": encoder_norm,
        "raw_head_gradient_norm": head_norm,
        "raw_total_gradient_norm": total_norm,
        "gradient_clip_norm": 5.0,
        "clip_would_apply": total_norm > 5.0,
        "task_loss": float(update_metrics["task_loss"]),
        "oe_loss": float(update_metrics["oe_loss"]),
        "combined_loss": float(update_metrics["combined_loss"]),
        "unknown_aam_target_assigned": False,
        "oe_full_batch_views": int(update_metrics["oe_full_batch_views"]),
    }
    return {**body, "probe_sha256": _sha(body)}


def _read_gradient_probe(path: Path, identity: Mapping[str, object]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F008 gradient probe receipt is unreadable") from error
    _require(isinstance(value, Mapping), "F008 gradient probe receipt must be an object")
    body = {key: item for key, item in value.items() if key != "probe_sha256"}
    _require(value.get("schema_version") == F008_GRADIENT_PROBE_SCHEMA
             and value.get("probe_sha256") == _sha(body)
             and value.get("arm_signature") == identity["signature"]
             and value.get("outer_fold") == identity["outer_fold"]
             and value.get("runtime_receipt_sha256") == identity["runtime_receipt_sha256"]
             and value.get("optimizer_steps_persisted") == 0
             and value.get("source_f005_optimizer_and_rng_restored_before_probe") is True
             and value.get("unknown_aam_target_assigned") is False
             and value.get("oe_full_batch_views") == 64,
             "F008 gradient probe receipt differs from this tail source")
    return dict(value)


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Create an immutable local receipt; checkpoints themselves remain mutable."""
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(),
             f"F008 immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    _require(not partial.exists() and not partial.is_symlink(),
             f"F008 immutable receipt has a pending partial: {partial}")
    encoded = _canonical(dict(value)) + b"\n"
    with partial.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(partial, path)
    except FileExistsError:
        raise FileExistsError(f"F008 immutable receipt raced with another writer: {path}") from None
    finally:
        if partial.exists() or partial.is_symlink():
            partial.unlink()


def _scalar_step_event(step: int, metrics: Mapping[str, object]) -> dict[str, float | int | str]:
    """Return the JSON-safe scalar event exposed to an optional live callback."""
    _require(type(step) is int and step > 0 and isinstance(metrics, Mapping),
             "F008 step callback event is malformed")
    event: dict[str, float | int | str] = {"step": step, "phase": "tail"}
    for key, value in metrics.items():
        _require(isinstance(key, str) and key.startswith("fit/"),
                 "F008 step callback may expose only fit metrics")
        _require(isinstance(value, Real) and not isinstance(value, bool)
                 and math.isfinite(float(value)),
                 "F008 step callback metric must be a finite scalar")
        event[key] = float(value)
    return event


def _run_tail_updates(
    f005_contract: Mapping[str, object], root: Path, outer_fold: int, arm: Mapping[str, object],
    pools: Mapping[str, object], encoder, head, optimizer, *, energy_margin_config: Mapping[str, object],
    energy_margin_plan: Mapping[str, object], start_step: int, stop_step: int,
    unknown_sampling_seed: int, history_path: Path, save_checkpoint,
    on_step: Callable[[dict[str, float | int | str]], None] | None = None,
) -> dict[str, Any]:
    """Run F008's 500-step tail with optional scalar-only live callbacks.

    The worker itself has no tracker dependency.  ``on_step`` is called only
    after the matching JSONL event has been appended, so a launcher can emit
    live MLflow metrics without letting MLflow become part of the training
    implementation or checkpoint format.
    """
    import torch

    _require(on_step is None or callable(on_step), "F008 on_step must be callable")
    fit = f005_contract["config"]["fit"]
    cache = _WaveformCache(fit["waveform_cache_max_bytes"])
    io_stats: dict[str, float | int] = {"decode_seconds": 0.0, "decode_misses": 0, "cache_hits": 0}
    started = time.monotonic()
    for step in range(start_step, stop_step):
        scheduled = _set_tail_schedule(encoder, head, optimizer, dict(fit), step)
        optimizer.zero_grad(set_to_none=True)
        aggregate = _backward_f008_step(
            f005_contract, root, outer_fold, step, arm, pools, encoder, head,
            energy_margin_config=energy_margin_config, energy_margin_plan=energy_margin_plan,
            unknown_sampling_seed=unknown_sampling_seed, cache=cache, io_stats=io_stats,
        )
        parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
        head_parameters = list(head.parameters())
        encoder_gradient_norm = _grad_norm(parameters)
        head_gradient_norm = _grad_norm(head_parameters)
        parameters += head_parameters
        gradient = torch.nn.utils.clip_grad_norm_(parameters, fit["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient).item()):
            raise FloatingPointError(f"F008 nonfinite gradient at step {step}; optimizer not advanced")
        total_gradient_norm = float(gradient.detach().cpu())
        optimizer.step()
        count = float(fit["batch_pairs"])
        metrics = {
            "fit/task_loss": aggregate["task_loss"],
            "fit/oe_loss": aggregate["oe_loss"],
            "fit/combined_loss": aggregate["combined_loss"],
            "fit/oe_lambda": float(arm["lambda"]),
            "fit/short_aam": aggregate["known_short_aam_sum"] / count,
            "fit/long_aam": aggregate["known_long_aam_sum"] / count,
            "fit/short_accuracy": aggregate["known_short_correct"] / count,
            "fit/long_accuracy": aggregate["known_long_correct"] / count,
            "fit/oe_full_batch_views": aggregate["oe_full_batch_views"],
            "fit/unknown_aam_target_assigned": aggregate["unknown_aam_target_assigned"],
            "fit/encoder_gradient_norm_preclip": encoder_gradient_norm,
            "fit/head_gradient_norm_preclip": head_gradient_norm,
            "fit/total_gradient_norm_preclip": total_gradient_norm,
            # Retain the original key for existing live dashboards; it is the
            # same pre-clipping total returned by PyTorch.
            "fit/gradient_norm": total_gradient_norm,
            "fit/encoder_lr": float(scheduled["encoder_lr"]),
            "fit/head_lr": float(scheduled["head_lr"]),
            "fit/margin": float(scheduled["margin"]),
            "fit/gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "fit/io_decode_seconds": float(io_stats["decode_seconds"]),
            "fit/io_cache_hits": float(io_stats["cache_hits"]),
            "fit/io_decode_misses": float(io_stats["decode_misses"]),
            "fit/elapsed_seconds": time.monotonic() - started,
        }
        if arm["id"] == "energy_005":
            metrics.update({
                "fit/known_energy": aggregate["known_energy_sum"] / 64.0,
                "fit/unknown_energy": aggregate["unknown_energy_sum"] / 64.0,
                "fit/known_energy_penalty": aggregate["known_oe_penalty_sum"] / 64.0,
                "fit/unknown_energy_penalty": aggregate["unknown_oe_penalty_sum"] / 64.0,
            })
        else:
            metrics["fit/unknown_uniform_cross_entropy"] = (
                aggregate["unknown_uniform_cross_entropy_sum"] / 64.0
            )
        checkpoint_due = ((step + 1) % fit["checkpoint_every_steps"] == 0
                          or step + 1 == stop_step)
        event = _scalar_step_event(step + 1, metrics)
        with Path(history_path).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            if checkpoint_due:
                handle.flush()
                os.fsync(handle.fileno())
        if on_step is not None:
            # A fresh dict prevents external tracking code from mutating the
            # in-worker event used by subsequent checkpoint/report logic.
            on_step(dict(event))
        if checkpoint_due:
            save_checkpoint(step + 1)
    return {
        "elapsed_seconds": time.monotonic() - started,
        "waveforms_cached": len(cache),
        "waveform_cache_bytes": cache.bytes,
        "waveform_cache_max_bytes": cache.maximum_bytes,
        **io_stats,
    }


def _build_identity_and_source(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object], root: Path,
    outer_fold: int, arm_id: str, shared_head_checkpoint: Path,
    source_receipt: Mapping[str, object], preflight_receipt: Mapping[str, object],
    *, runtime_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, int, int, dict[str, Any]]:
    """Authenticate the one F005 fork and derive F008's immutable plans once."""
    _sha256(runtime_receipt_sha256, "runtime receipt")
    config = _f008_config(f008_config_or_contract)
    arm = f008_arm(config, arm_id)
    f005_signature = _sha256(f005_contract.get("signature"), "F005 experiment signature")
    _require(f005_signature == config["source_f005"]["experiment_signature"],
             "F008 F005 source differs from its configured source")
    source_state = source_receipt.get("experiment_state") if isinstance(source_receipt, Mapping) else None
    _require(isinstance(source_state, Mapping)
             and source_state.get("status") == "complete"
             and source_state.get("experiment_signature") == f005_signature,
             "F008 F005 source receipt does not attest the bridged historic contract")
    fit = f005_contract.get("config", {}).get("fit")
    _require(isinstance(fit, Mapping), "F008 F005 fit configuration is absent")
    start, stop = _tail_bounds(fit, config)
    pools = role_pools(f005_contract.get("roles", ()), outer_fold)
    known_plan = plan_range_sha256(dict(f005_contract), outer_fold, start, stop)
    unknown_plan = unknown_plan_range_sha256(
        pools, start, stop, seed=config["seed"], samples_per_step=32,
    )
    shared_payload, shared_sha = _load_authenticated_shared_head(
        dict(f005_contract), dict(source_receipt), Path(root), outer_fold, Path(shared_head_checkpoint),
    )
    identity = f008_tail_identity(
        f008_config_or_contract, f005_contract, outer_fold, arm,
        shared_head_checkpoint_sha256=shared_sha, known_tail_plan_sha256=known_plan,
        unknown_plan_sha256=unknown_plan, role_pool=pools,
        preflight_receipt=preflight_receipt, source_receipt=source_receipt,
        runtime_receipt_sha256=runtime_receipt_sha256,
    )
    return config, arm, pools, shared_payload, shared_sha, start, stop, identity


def _probe_f008_tail_gradients_validated(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object], root: Path,
    outer_fold: int, arm_id: str, shared_head_checkpoint: Path,
    source_receipt: Mapping[str, object], preflight_receipt: Mapping[str, object],
    *, runtime_receipt_sha256: str,
) -> dict[str, Any]:
    """Run one real F008 backward pass and persist zero optimizer updates.

    The returned receipt contains no parameter values.  It is intentionally a
    separate model allocation from the later tail run so restoring F005's
    optimizer and RNG state into the real fork remains exact.
    """
    import torch

    values = _build_identity_and_source(
        f008_config_or_contract, f005_contract, root, outer_fold, arm_id,
        shared_head_checkpoint, source_receipt, preflight_receipt,
        runtime_receipt_sha256=runtime_receipt_sha256,
    )
    config, arm, pools, shared_payload, _shared_sha, start, _stop, identity = values
    encoder, head, optimizer, _trainable, _seed = _make_components(dict(f005_contract), Path(root), outer_fold)
    try:
        metadata = shared_payload["metadata"]
        _validate_training_state_structure(shared_payload, encoder, head, optimizer, metadata)
        _load_training_state(shared_payload, encoder, head, optimizer, metadata)
        before_encoder, before_head = state_dict_sha256(encoder.state_dict()), state_dict_sha256(head.state_dict())
        _set_tail_schedule(encoder, head, optimizer, f005_contract["config"]["fit"], start)
        optimizer.zero_grad(set_to_none=True)
        cache = _WaveformCache(f005_contract["config"]["fit"]["waveform_cache_max_bytes"])
        io_stats: dict[str, float | int] = {"decode_seconds": 0.0, "decode_misses": 0, "cache_hits": 0}
        metrics = _backward_f008_step(
            f005_contract, Path(root), outer_fold, start, arm, pools, encoder, head,
            energy_margin_config=config["energy_margin"],
            energy_margin_plan=verify_preflight_receipt(preflight_receipt)["energy_margin_plan"],
            unknown_sampling_seed=config["seed"], cache=cache, io_stats=io_stats,
        )
        _require(before_encoder == state_dict_sha256(encoder.state_dict())
                 and before_head == state_dict_sha256(head.state_dict()),
                 "F008 zero-optimizer probe changed model parameters")
        return _probe_receipt(identity, step=start, encoder=encoder, head=head,
                              update_metrics=metrics)
    finally:
        del encoder, head, optimizer
        torch.cuda.empty_cache()


def probe_f008_tail_gradients(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object], root: Path,
    outer_fold: int, arm_id: str, shared_head_checkpoint: Path,
    source_receipt: Mapping[str, object], preflight_receipt: Mapping[str, object],
    *, runtime_receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Run one F008 backward-only update after validating its runtime receipt.

    The receipt is required before the private worker imports Torch or builds a
    model.  The returned receipt contains no parameter values and persists no
    optimizer update.
    """
    runtime = validate_f008_runtime_receipt(
        runtime_receipt, f008_config_or_contract, f005_contract,
    )
    return _probe_f008_tail_gradients_validated(
        f008_config_or_contract, f005_contract, root, outer_fold, arm_id,
        shared_head_checkpoint, source_receipt, preflight_receipt,
        runtime_receipt_sha256=runtime["receipt_sha256"],
    )


def load_authenticated_f008_tail(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object], root: Path,
    outer_fold: int, arm_id: str, shared_head_checkpoint: Path,
    source_receipt: Mapping[str, object], preflight_receipt: Mapping[str, object],
    checkpoint: Path,
    *, runtime_receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Build an authenticated completed F008 encoder/head for a future extractor.

    First the exact F005 shared-head payload restores the original optimizer
    and RNG state, then the F008 checkpoint is validated and restored.  The
    caller owns the returned live CUDA modules and must release them.  No cache
    or parameter values are copied outside the server by this loader.
    """
    runtime_receipt = validate_f008_runtime_receipt(
        runtime_receipt, f008_config_or_contract, f005_contract,
    )
    import torch

    values = _build_identity_and_source(
        f008_config_or_contract, f005_contract, root, outer_fold, arm_id,
        shared_head_checkpoint, source_receipt, preflight_receipt,
        runtime_receipt_sha256=runtime_receipt["receipt_sha256"],
    )
    config, arm, _pools, shared_payload, _shared_sha, _start, stop, identity = values
    del config, arm
    encoder, head, optimizer, trainable, seed = _make_components(dict(f005_contract), Path(root), outer_fold)
    try:
        shared_metadata = shared_payload["metadata"]
        _validate_training_state_structure(shared_payload, encoder, head, optimizer, shared_metadata)
        _load_training_state(shared_payload, encoder, head, optimizer, shared_metadata)
        _require(Path(checkpoint).is_file() and not Path(checkpoint).is_symlink(),
                 "F008 completed checkpoint must be a regular file")
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        metadata = validate_f008_resume_payload(payload, identity, f005_contract["config"]["fit"])
        _require(metadata["completed_steps"] == stop,
                 "F008 extractor accepts only a completed 500-step tail checkpoint")
        _validate_training_state_structure(payload, encoder, head, optimizer, metadata)
        _load_training_state(payload, encoder, head, optimizer, metadata)
        encoder.eval(); head.eval()
        return {
            "encoder": encoder, "head": head, "optimizer": optimizer,
            "identity": identity, "metadata": metadata,
            "checkpoint_sha256": _sha_file(Path(checkpoint)),
            "source_f005_optimizer_and_rng_restored_before_tail": True,
            "initialization_seed": seed, "trainable": trainable,
        }
    except Exception:
        del encoder, head, optimizer
        torch.cuda.empty_cache()
        raise


def fit_f008_tail(
    f008_config_or_contract: Mapping[str, object], f005_contract: Mapping[str, object], root: Path,
    outer_fold: int, arm_id: str, shared_head_checkpoint: Path,
    source_receipt: Mapping[str, object], preflight_receipt: Mapping[str, object], output: Path,
    *, runtime_receipt: Mapping[str, object], resume: bool = False,
    on_step: Callable[[dict[str, float | int | str]], None] | None = None,
) -> dict[str, Any]:
    """Train/resume one F008 tail after a one-time runtime attestation.

    ``on_step`` is an optional scalar-only callback for a launcher to publish
    live metrics.  It is deliberately injected rather than importing MLflow
    into this worker.
    """
    runtime_receipt_value = validate_f008_runtime_receipt(
        runtime_receipt, f008_config_or_contract, f005_contract,
    )
    import torch

    values = _build_identity_and_source(
        f008_config_or_contract, f005_contract, root, outer_fold, arm_id,
        shared_head_checkpoint, source_receipt, preflight_receipt,
        runtime_receipt_sha256=runtime_receipt_value["receipt_sha256"],
    )
    config, arm, pools, shared_payload, shared_sha, start, stop, identity = values
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, history = output / "last.pt", output / "fit_history.jsonl"
    probe_path = output / "zero_optimizer_gradient_probe.json"
    checkpoint_present = _present(checkpoint) or _present(checkpoint.with_suffix(checkpoint.suffix + ".partial"))
    if checkpoint_present and not resume:
        raise FileExistsError("Existing F008 tail checkpoint requires explicit resume")
    if probe_path.exists() or probe_path.is_symlink():
        probe = _read_gradient_probe(probe_path, identity)
    else:
        probe = _probe_f008_tail_gradients_validated(
            f008_config_or_contract, f005_contract, root, outer_fold, arm_id,
            shared_head_checkpoint, source_receipt, preflight_receipt,
            runtime_receipt_sha256=runtime_receipt_value["receipt_sha256"],
        )
        _write_new_json(probe_path, probe)
        probe = _read_gradient_probe(probe_path, identity)
    encoder, head, optimizer, trainable, seed = _make_components(dict(f005_contract), Path(root), outer_fold)
    try:
        source_metadata = shared_payload["metadata"]
        _validate_training_state_structure(shared_payload, encoder, head, optimizer, source_metadata)
        _load_training_state(shared_payload, encoder, head, optimizer, source_metadata)
        fork_encoder = state_dict_sha256(encoder.state_dict())
        fork_head = state_dict_sha256(head.state_dict())

        def validate_checkpoint(candidate: Path) -> dict[str, Any]:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            metadata = validate_f008_resume_payload(payload, identity, f005_contract["config"]["fit"])
            _validate_training_state_structure(payload, encoder, head, optimizer, metadata)
            return metadata

        if checkpoint_present:
            recover_fixed_partial(checkpoint, validate_checkpoint, derived=True)
        current = start
        if _present(checkpoint):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = validate_f008_resume_payload(payload, identity, f005_contract["config"]["fit"])
            _load_training_state(payload, encoder, head, optimizer, metadata)
            current = metadata["completed_steps"]
        _trim_history(history, current, first_step=start + 1)

        def save(completed: int) -> None:
            _atomic_checkpoint({
                "metadata": f008_tail_checkpoint_metadata(identity, f005_contract["config"]["fit"], completed),
                "encoder": encoder.state_dict(), "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }, checkpoint)

        runtime = _run_tail_updates(
            f005_contract, Path(root), outer_fold, arm, pools, encoder, head, optimizer,
            energy_margin_config=config["energy_margin"],
            energy_margin_plan=verify_preflight_receipt(preflight_receipt)["energy_margin_plan"],
            start_step=current, stop_step=stop, unknown_sampling_seed=config["seed"],
            history_path=history, save_checkpoint=save, on_step=on_step,
        )
        report = {
            "status": "complete", "stage": "open_set_oe_tail", "outer_fold": outer_fold,
            "arm": arm, "unit_signature": identity["signature"],
            "source_f005_signature": f005_contract["signature"],
            "f005_tail_plan_sha256": identity["f005_tail_plan_sha256"],
            "unknown_plan_sha256": identity["unknown_plan_sha256"],
            "preflight_receipt_sha256": identity["preflight_receipt_sha256"],
            "energy_margin_plan_sha256": identity["energy_margin_plan_sha256"],
            "runtime_receipt_sha256": identity["runtime_receipt_sha256"],
            "shared_head_checkpoint_sha256": shared_sha,
            "fork_encoder_sha256": fork_encoder, "fork_head_sha256": fork_head,
            "initialization_seed": seed,
            "source_f005_optimizer_and_rng_restored_before_tail": True,
            "zero_optimizer_gradient_probe_path": str(probe_path),
            "zero_optimizer_gradient_probe_sha256": probe["probe_sha256"],
            "completed_steps": stop,
            "schedule_state": adaptation_checkpoint_state(f005_contract["config"]["fit"], stop),
            "checkpoint_mlflow_uploaded": False, "checkpoint_local_transfer": False,
            "unknown_aam_target_assigned": False,
            **trainable, **runtime,
        }
        return {"report": report, "checkpoint": checkpoint, "probe": probe, "identity": identity}
    finally:
        del encoder, head, optimizer
        torch.cuda.empty_cache()
