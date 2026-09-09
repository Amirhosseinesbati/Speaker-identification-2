"""L2-SP-only tail worker for F007.

F007 is intentionally a sibling of the completed F005 experiment.  It reuses
an authenticated F005 shared-head checkpoint, but never mutates F005 outputs
or re-runs its shared-head phase.  Each new arm performs only the 500 tail
updates and adds L2-SP exactly once after all task microbatches have
contributed their gradients for an optimizer update.

The module contains no MLflow client and never uploads a checkpoint, waveform,
or embedding.  Its JSON receipts contain provenance and parameter metadata,
never parameter values.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from speaker_id.adaptation.l2sp import (
    build_l2sp_anchor,
    l2sp_gradient_norm_ratio,
    l2sp_penalty,
)
from speaker_id.training.f005_contract import ADVANCED_DIMENSION
from speaker_id.training.f005_runner import (
    plan_range_sha256,
    shared_head_identity,
    validate_shared_head_payload,
)
from speaker_id.training.f005_worker import (
    _WaveformCache,
    _atomic_checkpoint,
    _load_training_state,
    _microbatch,
    _present,
    _trim_history,
    _validate_training_state_structure,
    recover_fixed_partial,
    state_dict_sha256,
)
from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_step, adaptation_total_steps


F007_TAIL_CHECKPOINT_SCHEMA = "f007-l2sp-tail-checkpoint-v1"
F007_PROBE_SCHEMA = "f007-l2sp-virtual-probe-v1"
F007_NEW_ARM_IDS = ("l2sp_001", "l2sp_01")
_SHA256 = frozenset("0123456789abcdef")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
    _require(
        isinstance(value, str) and len(value) == 64
        and all(character in _SHA256 for character in value),
        f"F007 {label} must be a lowercase SHA-256",
    )
    return value


def _f007_config(contract_or_config: dict) -> dict:
    _require(isinstance(contract_or_config, dict), "F007 contract/config must be an object")
    config = contract_or_config.get("config", contract_or_config)
    _require(isinstance(config, dict) and config.get("experiment_code") == "F007",
             "F007 worker requires an F007 configuration")
    return config


def _f007_signature(contract_or_config: dict) -> str:
    config = _f007_config(contract_or_config)
    signature = contract_or_config.get("signature") if isinstance(contract_or_config, dict) else None
    if signature is None:
        return _sha(config)
    return _sha256(signature, "experiment signature")


def _arm(config: dict, arm_id: str) -> dict:
    _require(isinstance(arm_id, str), "F007 arm id must be a string")
    arms = config.get("arms")
    _require(isinstance(arms, list) and tuple(arm.get("id") for arm in arms if isinstance(arm, dict))
             == ("control_f005",) + F007_NEW_ARM_IDS,
             "F007 arm order changed")
    selected = next((arm for arm in arms if arm["id"] == arm_id), None)
    _require(isinstance(selected, dict) and set(selected) == {"id", "kind", "lambda"},
             "F007 arm schema changed")
    _require(arm_id in F007_NEW_ARM_IDS and selected["kind"] == "l2sp_anchored_tail",
             "F007 worker can train only a declared new L2-SP arm")
    _require(type(selected["lambda"]) in (int, float) and not isinstance(selected["lambda"], bool)
             and selected["lambda"] in (0.01, 0.1),
             "F007 L2-SP lambda changed")
    return selected


def _source_shared_sha(source_receipt: dict, outer: int) -> str:
    _require(isinstance(source_receipt, dict)
             and source_receipt.get("schema_version") == "f007-f005-source-receipt-v1",
             "F007 requires the authenticated F005 source receipt")
    folds = source_receipt.get("folds")
    _require(isinstance(folds, list), "F007 F005 source receipt lacks folds")
    candidates = [row for row in folds if isinstance(row, dict) and row.get("outer_fold") == outer]
    _require(len(candidates) == 1, "F007 F005 source receipt fold is ambiguous")
    try:
        value = candidates[0]["shared_head"]["checkpoint"]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("F007 F005 source receipt lacks the shared-head hash") from error
    return _sha256(value, "F005 shared-head checkpoint hash")


def batchnorm_affine_parameter_names(encoder: object) -> frozenset[str]:
    """Return the exact BatchNorm weight/bias names, including frozen ones.

    F007 keeps BatchNorm affine parameters task-trainable where F005 made them
    trainable, but excludes them from the L2-SP anchor.  Enumerating module
    parameters rather than matching textual names prevents a CAM++ rename from
    silently changing the regularized set.
    """
    import torch

    modules = getattr(encoder, "named_modules", None)
    if not callable(modules):
        raise TypeError("F007 encoder must provide named_modules()")
    names: set[str] = set()
    for module_name, module in modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        direct = module.named_parameters(recurse=False)
        for parameter_name, parameter in direct:
            if parameter is None:
                continue
            _require(parameter_name in {"weight", "bias"},
                     "F007 BatchNorm unexpectedly exposes a non-affine parameter")
            complete = f"{module_name}.{parameter_name}" if module_name else parameter_name
            _require(complete not in names, "F007 BatchNorm affine name is duplicated")
            names.add(complete)
    return frozenset(names)


def _anchor_receipt(anchor, excluded: frozenset[str]) -> dict:
    """Add the explicit BN exclusion list to L2-SP's safe base receipt."""
    return {
        **anchor.receipt,
        "excluded_batchnorm_affine_names": sorted(excluded),
    }


def f007_dual_aam_task(
    short_h,
    long_h,
    targets,
    head,
    *,
    normalization_pairs: int,
) -> tuple["Any", dict[str, float | int]]:
    """Return only F005's dual-AAM task term, with no consistency objective."""
    import torch
    from torch.nn import functional as F

    _require(type(normalization_pairs) is int and normalization_pairs > 0,
             "F007 task normalization pair count must be positive")
    _require(
        isinstance(short_h, torch.Tensor) and isinstance(long_h, torch.Tensor)
        and short_h.dtype == torch.float32 and long_h.dtype == torch.float32
        and short_h.ndim == 2 and short_h.shape == long_h.shape
        and short_h.shape[1] == ADVANCED_DIMENSION
        and isinstance(targets, torch.Tensor) and targets.dtype == torch.int64
        and targets.shape == (short_h.shape[0],) and targets.device == short_h.device
        and 0 < len(targets) <= normalization_pairs,
        "F007 task requires matching FP32 [batch,192] embeddings and known targets",
    )
    short_logits, long_logits = head(short_h, targets), head(long_h, targets)
    short_sum = F.cross_entropy(short_logits, targets, reduction="sum")
    long_sum = F.cross_entropy(long_logits, targets, reduction="sum")
    loss = (short_sum + long_sum) / (2 * normalization_pairs)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("F007 task loss is nonfinite")
    return loss, {
        "task_loss": float(loss.detach().cpu()),
        "short_aam_sum": float(short_sum.detach().cpu()),
        "long_aam_sum": float(long_sum.detach().cpu()),
        "short_correct": int((short_logits.argmax(1) == targets).sum().detach().cpu()),
        "long_correct": int((long_logits.argmax(1) == targets).sum().detach().cpu()),
        "rows": int(targets.numel()),
    }


def attest_f007_runtime(f007_contract: dict) -> dict:
    """Do the one targeted CUDA/runtime check and return a reusable receipt.

    The F007 orchestrator calls this once per logical run.  Tail units receive
    the receipt and validate only its immutable identity instead of repeating
    device/package/data checks for every arm.
    """
    # The one-time runtime receipt is meaningful only for a fully authenticated
    # F007 contract, never for a caller-supplied lookalike config.
    from speaker_id.training.f007_contract import validate_f007_contract

    validate_f007_contract(f007_contract)
    config = _f007_config(f007_contract)
    execution = config.get("execution")
    _require(isinstance(execution, dict), "F007 execution policy is missing")
    expected_threads = execution.get("thread_environment")
    _require(isinstance(expected_threads, dict) and expected_threads == {
        "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    }, "F007 numerical thread environment changed")
    _require(
        execution.get("cublas_workspace_config") == ":4096:8"
        and execution.get("gpu_name_contains") == "RTX 3090"
        and execution.get("tensor_dtype") == "float32"
        and execution.get("no_cpu_fallback") is True
        and execution.get("deterministic_algorithms") == "enforce_error",
        "F007 deterministic CUDA policy changed",
    )
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
             "F007 requires CUBLAS_WORKSPACE_CONFIG before Torch import")
    _require(all(os.environ.get(key) == value for key, value in expected_threads.items()),
             "F007 numerical thread environment differs from the launcher contract")
    import torch

    _require(config.get("device") == "cuda" and torch.cuda.is_available(),
             "F007 requires CUDA and forbids CPU fallback")
    name = torch.cuda.get_device_name(0)
    _require("RTX 3090" in name, "F007 requires the authorized RTX 3090")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    body = {
        "schema_version": "f007-runtime-receipt-v1",
        "f007_signature": _f007_signature(f007_contract),
        "execution": execution,
        "device": "cuda",
        "gpu_name": name,
        "cuda_available": True,
        "cpu_threads": config["cpu_threads"],
        "checked_once_per_logical_run": True,
    }
    return {**body, "receipt_sha256": _sha(body)}


def _validate_runtime_receipt(receipt: dict, f007_contract: dict) -> None:
    _require(isinstance(receipt, dict), "F007 worker requires a runtime receipt")
    from speaker_id.training.f007_contract import validate_f007_contract

    validate_f007_contract(f007_contract)
    config = _f007_config(f007_contract)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(
        receipt.get("schema_version") == "f007-runtime-receipt-v1"
        and receipt.get("receipt_sha256") == _sha(body)
        and receipt.get("f007_signature") == _f007_signature(f007_contract)
        and receipt.get("execution") == config["execution"]
        and receipt.get("device") == "cuda"
        and receipt.get("cuda_available") is True
        and receipt.get("cpu_threads") == config["cpu_threads"]
        and isinstance(receipt.get("gpu_name"), str)
        and config["execution"]["gpu_name_contains"] in receipt["gpu_name"]
        and receipt.get("checked_once_per_logical_run") is True,
        "F007 runtime receipt is invalid or belongs to another run",
    )


def f007_tail_identity(
    f007_contract: dict,
    f005_contract: dict,
    outer: int,
    arm: dict,
    *,
    shared_head_checkpoint_sha256: str,
    tail_plan_sha256: str,
    anchor_receipt: dict,
) -> dict:
    """Build the immutable identity for a new F007 tail checkpoint."""
    config = _f007_config(f007_contract)
    _require(outer in config.get("fold_ids", ()), "F007 outer fold is not configured")
    _require(arm == _arm(config, arm.get("id")), "F007 arm identity is malformed")
    _sha256(shared_head_checkpoint_sha256, "shared-head checkpoint hash")
    _sha256(tail_plan_sha256, "tail plan hash")
    _require(isinstance(f005_contract, dict), "F007 requires an in-memory F005 contract")
    f005_signature = _sha256(f005_contract.get("signature"), "F005 experiment signature")
    anchor_digest = _sha(anchor_receipt)
    body = {
        "schema_version": 1,
        "f007_signature": _f007_signature(f007_contract),
        "source_f005_signature": f005_signature,
        "outer_fold": outer,
        "arm_id": arm["id"],
        "l2sp_lambda": float(arm["lambda"]),
        "shared_head_checkpoint_sha256": shared_head_checkpoint_sha256,
        "tail_plan_sha256": tail_plan_sha256,
        "anchor_receipt_sha256": anchor_digest,
        "anchor_scope": "trainable_encoder_parameters_excluding_batchnorm_affine",
        "objective": "dual_aam_plus_l2sp_once_per_optimizer_step",
        "embedding_dimension": ADVANCED_DIMENSION,
        "checkpoint_scope": "server_only_until_promotion",
    }
    return {**body, "signature": _sha(body)}


def f007_tail_checkpoint_metadata(identity: dict, f005_fit: dict, completed_steps: int) -> dict:
    """Return fully deterministic metadata for one F007 checkpoint boundary."""
    total = adaptation_total_steps(f005_fit)
    head_steps = f005_fit["adaptation_schedule"]["head_only_steps"]
    _require(type(completed_steps) is int and head_steps <= completed_steps <= total,
             "F007 tail checkpoint step is outside the F005 tail range")
    _require(identity.get("embedding_dimension") == ADVANCED_DIMENSION,
             "F007 identity does not bind the advanced CAM++ endpoint")
    body = {
        "format_version": 1,
        "checkpoint_schema": F007_TAIL_CHECKPOINT_SCHEMA,
        "stage": "tail",  # Keeps F005's structural optimizer validator applicable.
        "f007_stage": "l2sp_tail",
        "f007_signature": identity["f007_signature"],
        "source_f005_signature": identity["source_f005_signature"],
        "arm_signature": identity["signature"],
        "outer_fold": identity["outer_fold"],
        "arm_id": identity["arm_id"],
        "l2sp_lambda": identity["l2sp_lambda"],
        "shared_head_checkpoint_sha256": identity["shared_head_checkpoint_sha256"],
        "tail_plan_sha256": identity["tail_plan_sha256"],
        "anchor_receipt_sha256": identity["anchor_receipt_sha256"],
        "anchor_scope": identity["anchor_scope"],
        "objective": identity["objective"],
        "embedding_dimension": ADVANCED_DIMENSION,
        "completed_steps": completed_steps,
        "total_steps": total,
        "schedule_state": adaptation_checkpoint_state(f005_fit, completed_steps),
        "l2sp_applied_once_per_optimizer_step": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    return {**body, "metadata_sha256": _sha(body)}


def validate_f007_resume_payload(payload: dict, identity: dict, f005_fit: dict) -> dict:
    """Reject a cross-arm/fold/source resume before live state is mutated."""
    required = {"metadata", "encoder", "head", "optimizer", "torch_rng", "cuda_rng"}
    _require(isinstance(payload, dict) and set(payload) == required,
             "F007 checkpoint payload fields differ from the fixed format")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict) and type(metadata.get("completed_steps")) is int,
             "F007 checkpoint metadata is incomplete")
    expected = f007_tail_checkpoint_metadata(identity, f005_fit, metadata["completed_steps"])
    _require(metadata == expected,
             "F007 resume checkpoint belongs to another arm, fold, source, plan, anchor or step")
    return metadata


def _make_components(f005_contract: dict, root: Path, outer: int) -> tuple[object, object, object, dict, int]:
    """Make F005-compatible model/optimizer objects without a second audit."""
    import torch
    from speaker_id.candidates.campp_advanced import load_advanced
    from speaker_id.training.fit import AAMHead, set_trainable_tail

    config, fit = f005_contract["config"], f005_contract["config"]["fit"]
    model = f005_contract["advanced_model"]
    _require(model.get("embedding_dim") == ADVANCED_DIMENSION,
             "F007 requires the F005 advanced 192D endpoint")
    encoder = load_advanced(model, Path(root), "cuda")
    trainable = set_trainable_tail(encoder, fit["trainable_prefixes"])
    pairing = shared_head_identity(f005_contract, outer, plan_sha256="0" * 64)["pairing_signature"]
    seed = int(pairing[:16], 16) % (2 ** 63 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = AAMHead(embedding_dim=ADVANCED_DIMENSION, classes=446,
                   margin=fit["margin"], scale=fit["scale"]).to("cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": fit["encoder_lr"]},
        {"params": head.parameters(), "lr": fit["head_lr"]},
    ], weight_decay=fit["weight_decay"])
    return encoder, head, optimizer, trainable, seed


def _load_authenticated_shared_head(
    f005_contract: dict,
    source_receipt: dict,
    root: Path,
    outer: int,
    shared_head_checkpoint: Path,
) -> tuple[dict, str]:
    """Read one source checkpoint after matching its cached F007 receipt."""
    import torch

    checkpoint = Path(shared_head_checkpoint)
    _require(checkpoint.is_file() and not checkpoint.is_symlink(),
             "F007 shared-head checkpoint must be a regular file")
    actual_sha = _sha_file(checkpoint)
    _require(actual_sha == _source_shared_sha(source_receipt, outer),
             "F007 shared-head checkpoint bytes differ from the attested F005 source")
    fit = f005_contract["config"]["fit"]
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    identity = shared_head_identity(
        f005_contract, outer,
        plan_sha256=plan_range_sha256(f005_contract, outer, 0, head_steps),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = validate_shared_head_payload(payload, identity, fit)
    _require(metadata["completed_steps"] == head_steps
             and metadata["byte_identical_fork_source"] is True,
             "F007 requires the completed F005 shared-head fork source")
    return payload, actual_sha


def _set_tail_schedule(encoder, head, optimizer, f005_fit: dict, step: int) -> dict:
    from speaker_id.training.fit import freeze_batchnorm

    scheduled = adaptation_step(f005_fit, step)
    _require(scheduled["phase"] == "tail", "F007 may run only the F005 tail phase")
    optimizer.param_groups[0]["lr"] = scheduled["encoder_lr"]
    optimizer.param_groups[1]["lr"] = scheduled["head_lr"]
    head.margin = scheduled["margin"]
    encoder.train()
    freeze_batchnorm(encoder)
    head.train()
    return scheduled


def _cpu_batches(f005_contract: dict, root: Path, outer: int, step: int,
                 waveform_cache: _WaveformCache, io_stats: dict) -> list[dict]:
    from speaker_id.training.f005_runner import training_step_plan

    fit = f005_contract["config"]["fit"]
    plan = training_step_plan(f005_contract, outer, step)
    micro = fit["microbatch_pairs"]
    return [
        _microbatch(f005_contract, plan[start:start + micro], Path(root), waveform_cache, io_stats)
        for start in range(0, len(plan), micro)
    ]


def _task_batch_loss(encoder, head, cpu_batch: dict, *, normalization_pairs: int):
    import torch

    batch = {key: value.to("cuda") for key, value in cpu_batch.items()}
    short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
    _require(short_h.shape == (len(batch["targets"]), ADVANCED_DIMENSION)
             and long_h.shape == short_h.shape
             and short_h.dtype == torch.float32 and long_h.dtype == torch.float32,
             "F007 advanced encoder output is not FP32 [batch,192]")
    return f007_dual_aam_task(
        short_h, long_h, batch["targets"], head,
        normalization_pairs=normalization_pairs,
    )


def _virtual_probe(
    f005_contract: dict,
    root: Path,
    outer: int,
    shared_payload: dict,
    arm: dict,
    anchor_receipt: dict,
) -> dict:
    """Measure the pre-registered post-step-600 L2-SP gradient ratio.

    This function builds a separate model, performs one task-only virtual
    update at global step 600, and computes the ratio on global step 601.  No
    checkpoint, optimizer state, or model used by real training is touched.
    """
    import torch

    fit = f005_contract["config"]["fit"]
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    _require(head_steps == 600, "F007 virtual probe requires F005 tail to start at step 600")
    encoder, head, optimizer, _, _ = _make_components(f005_contract, root, outer)
    try:
        # The existing F005 structural check is pure and validates all tensor,
        # optimizer, and RNG fields before the source state is loaded.
        metadata = shared_payload["metadata"]
        _validate_training_state_structure(shared_payload, encoder, head, optimizer, metadata)
        _load_training_state(shared_payload, encoder, head, optimizer, metadata)
        excluded = batchnorm_affine_parameter_names(encoder)
        anchor = build_l2sp_anchor(encoder, exclude_names=excluded)
        _require(_anchor_receipt(anchor, excluded) == anchor_receipt,
                 "F007 virtual probe anchor differs from the tail anchor")
        io_stats = {"decode_seconds": 0.0, "decode_misses": 0, "cache_hits": 0}
        cache = _WaveformCache(fit["waveform_cache_max_bytes"])

        # Global update 600: task only; L2-SP is deliberately absent.
        _set_tail_schedule(encoder, head, optimizer, fit, head_steps)
        optimizer.zero_grad(set_to_none=True)
        for cpu_batch in _cpu_batches(f005_contract, root, outer, head_steps, cache, io_stats):
            task_loss, _ = _task_batch_loss(
                encoder, head, cpu_batch, normalization_pairs=fit["batch_pairs"],
            )
            task_loss.backward()
        parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
        gradient = torch.nn.utils.clip_grad_norm_(parameters, fit["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient).item()):
            raise FloatingPointError("F007 virtual step 600 task gradient is nonfinite")
        optimizer.step()

        # Global update 601: retain the graph only for diagnostic gradients.
        measurement_step = head_steps + 1
        _set_tail_schedule(encoder, head, optimizer, fit, measurement_step)
        losses = []
        for cpu_batch in _cpu_batches(f005_contract, root, outer, measurement_step, cache, io_stats):
            task_loss, _ = _task_batch_loss(
                encoder, head, cpu_batch, normalization_pairs=fit["batch_pairs"],
            )
            losses.append(task_loss)
        task = sum(losses[1:], losses[0]) if len(losses) > 1 else losses[0]
        penalty = l2sp_penalty(encoder, anchor)
        ratio = l2sp_gradient_norm_ratio(
            task, penalty, encoder, anchor, lambda_sp=float(arm["lambda"]),
        )
        _require(ratio["ratio_status"] == "finite" and ratio["task_to_weighted_l2sp_ratio"] is not None,
                 "F007 virtual L2-SP ratio unexpectedly has zero weighted gradient")
        body = {
            "schema_version": F007_PROBE_SCHEMA,
            "outer_fold": outer,
            "arm_id": arm["id"],
            "lambda": float(arm["lambda"]),
            "kind": "virtual_task_step_then_gradient_ratio",
            "task_only_virtual_step": head_steps,
            "measurement_step": measurement_step,
            "optimizer_steps_persisted": 0,
            "anchor_receipt_sha256": _sha(anchor_receipt),
            "ratio": ratio,
            "selection_effect": "none_predeclared_lambda_only",
        }
        return {**body, "probe_sha256": _sha(body)}
    finally:
        del encoder, head, optimizer
        torch.cuda.empty_cache()


def _write_new_json(path: Path, value: dict) -> None:
    """Create an immutable JSON receipt without accepting an overwrite."""
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(),
             f"F007 immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    _require(not partial.exists() and not partial.is_symlink(),
             f"F007 immutable receipt has a pending partial: {partial}")
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(partial, path)
    except FileExistsError:
        raise FileExistsError(f"F007 immutable receipt raced with another writer: {path}") from None
    finally:
        if partial.exists() or partial.is_symlink():
            partial.unlink()


def _read_probe(path: Path, *, outer: int, arm: dict, anchor_receipt: dict) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F007 virtual probe receipt is unreadable") from error
    _require(isinstance(value, dict), "F007 virtual probe receipt must be an object")
    body = {key: item for key, item in value.items() if key != "probe_sha256"}
    _require(
        value.get("schema_version") == F007_PROBE_SCHEMA
        and value.get("probe_sha256") == _sha(body)
        and value.get("outer_fold") == outer
        and value.get("arm_id") == arm["id"]
        and value.get("lambda") == float(arm["lambda"])
        and value.get("task_only_virtual_step") == 600
        and value.get("measurement_step") == 601
        and value.get("optimizer_steps_persisted") == 0
        and value.get("anchor_receipt_sha256") == _sha(anchor_receipt),
        "F007 virtual probe receipt differs from this tail source",
    )
    return value


def _run_tail_updates(
    f005_contract: dict,
    root: Path,
    outer: int,
    arm: dict,
    encoder,
    head,
    optimizer,
    anchor,
    start_step: int,
    stop_step: int,
    history_path: Path,
    tracker,
    save_checkpoint,
) -> dict:
    """Run tail updates; L2-SP's backward happens once after microbatches."""
    import torch

    fit = f005_contract["config"]["fit"]
    cache = _WaveformCache(fit["waveform_cache_max_bytes"])
    io_stats = {"decode_seconds": 0.0, "decode_misses": 0, "cache_hits": 0}
    started = time.monotonic()
    lambda_sp = float(arm["lambda"])
    for step in range(start_step, stop_step):
        scheduled = _set_tail_schedule(encoder, head, optimizer, fit, step)
        optimizer.zero_grad(set_to_none=True)
        aggregate: defaultdict[str, float] = defaultdict(float)
        for cpu_batch in _cpu_batches(f005_contract, root, outer, step, cache, io_stats):
            task_loss, diagnostics = _task_batch_loss(
                encoder, head, cpu_batch, normalization_pairs=fit["batch_pairs"],
            )
            task_loss.backward()
            for key, value in diagnostics.items():
                aggregate[key] += float(value)

        # This is intentionally outside the microbatch loop.  It gives the
        # exact gradient of lambda * 0.5 * sum((p - p0)^2) once per optimizer
        # step, before clipping and `optimizer.step()`.
        unweighted_penalty = l2sp_penalty(encoder, anchor)
        weighted_penalty = lambda_sp * unweighted_penalty
        if not bool(torch.isfinite(weighted_penalty.detach()).item()):
            raise FloatingPointError(f"F007 nonfinite L2-SP loss at step {step}")
        weighted_penalty.backward()
        parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
        gradient = torch.nn.utils.clip_grad_norm_(parameters, fit["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient).item()):
            raise FloatingPointError(f"F007 nonfinite gradient at step {step}; optimizer not advanced")
        optimizer.step()

        count = float(fit["batch_pairs"])
        metrics = {
            "fit/task_loss": aggregate["task_loss"],
            "fit/short_aam": aggregate["short_aam_sum"] / count,
            "fit/long_aam": aggregate["long_aam_sum"] / count,
            "fit/l2sp_unweighted": float(unweighted_penalty.detach().cpu()),
            "fit/l2sp_weighted": float(weighted_penalty.detach().cpu()),
            "fit/l2sp_lambda": lambda_sp,
            "fit/l2sp_application_count": 1.0,
            "fit/short_accuracy": aggregate["short_correct"] / count,
            "fit/long_accuracy": aggregate["long_correct"] / count,
            "fit/gradient_norm": float(gradient.detach().cpu()),
            "fit/encoder_lr": float(scheduled["encoder_lr"]),
            "fit/head_lr": float(scheduled["head_lr"]),
            "fit/margin": float(scheduled["margin"]),
            "fit/gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "fit/io_decode_seconds": io_stats["decode_seconds"],
            "fit/io_cache_hits": io_stats["cache_hits"],
            "fit/io_decode_misses": io_stats["decode_misses"],
            "fit/elapsed_seconds": time.monotonic() - started,
        }
        checkpoint_due = ((step + 1) % fit["checkpoint_every_steps"] == 0
                          or step + 1 == stop_step)
        with Path(history_path).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"step": step + 1, "phase": "tail", **metrics},
                                    allow_nan=False) + "\n")
            if checkpoint_due:
                handle.flush()
                os.fsync(handle.fileno())
        if tracker is not None:
            tracker.log_metrics(metrics, step=step + 1,
                                sync=(step + 1) % 10 == 0, strict=False)
        if checkpoint_due:
            save_checkpoint(step + 1)
    return {
        "elapsed_seconds": time.monotonic() - started,
        "waveforms_cached": len(cache),
        "waveform_cache_bytes": cache.bytes,
        "waveform_cache_max_bytes": cache.maximum_bytes,
        **io_stats,
    }


def fit_l2sp_tail(
    f007_contract: dict,
    f005_contract: dict,
    root: Path,
    outer: int,
    arm_id: str,
    shared_head_checkpoint: Path,
    source_receipt: dict,
    runtime_receipt: dict,
    output: Path,
    tracker=None,
    *,
    resume: bool = False,
) -> dict:
    """Train or resume one new F007 L2-SP tail from F005's shared head.

    `runtime_receipt` is intentionally required: full CUDA/venv/device health
    is checked once by the orchestrator rather than redundantly for all four
    GPU units.  Source checkpoint bytes remain bound to the cached F005 source
    receipt for this particular fold.
    """
    import torch
    from speaker_id.training.f007_contract import validate_f007_contract

    validate_f007_contract(f007_contract)
    config = _f007_config(f007_contract)
    _validate_runtime_receipt(runtime_receipt, f007_contract)
    arm = _arm(config, arm_id)
    fit = f005_contract["config"]["fit"]
    head_steps, total = fit["adaptation_schedule"]["head_only_steps"], adaptation_total_steps(fit)
    _require(head_steps == 600 and total == 1100,
             "F007 is pinned to the completed F005 600+500 schedule")
    _require(
        source_receipt == f007_contract.get("source_f005_receipt"),
        "F007 passed F005 source receipt differs from its authenticated contract",
    )
    _require(f005_contract.get("signature") == source_receipt.get("experiment_state", {}).get("experiment_signature"),
             "F007 current F005 contract differs from the attested source experiment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    shared_payload, shared_sha = _load_authenticated_shared_head(
        f005_contract, source_receipt, root, outer, shared_head_checkpoint,
    )
    # Build an anchor once from a fresh, authenticated source state.  The
    # actual training model is created *after* the optional virtual probe so
    # restoring F005's saved RNG state remains exact for the real first update.
    bootstrap_encoder, bootstrap_head, bootstrap_optimizer, _, _ = _make_components(
        f005_contract, root, outer,
    )
    try:
        _validate_training_state_structure(
            shared_payload, bootstrap_encoder, bootstrap_head, bootstrap_optimizer,
            shared_payload["metadata"],
        )
        _load_training_state(
            shared_payload, bootstrap_encoder, bootstrap_head, bootstrap_optimizer,
            shared_payload["metadata"],
        )
        excluded = batchnorm_affine_parameter_names(bootstrap_encoder)
        bootstrap_anchor = build_l2sp_anchor(bootstrap_encoder, exclude_names=excluded)
        anchor_receipt = _anchor_receipt(bootstrap_anchor, excluded)
    finally:
        del bootstrap_encoder, bootstrap_head, bootstrap_optimizer
        torch.cuda.empty_cache()

    tail_plan = plan_range_sha256(f005_contract, outer, head_steps, total)
    identity = f007_tail_identity(
        f007_contract, f005_contract, outer, arm,
        shared_head_checkpoint_sha256=shared_sha,
        tail_plan_sha256=tail_plan,
        anchor_receipt=anchor_receipt,
    )
    checkpoint, history = output / "last.pt", output / "fit_history.jsonl"
    probe_path = output / "virtual_l2sp_probe.json"
    checkpoint_present = _present(checkpoint) or _present(checkpoint.with_suffix(checkpoint.suffix + ".partial"))
    if checkpoint_present and not resume:
        raise FileExistsError("Existing F007 tail checkpoint requires explicit resume")
    if probe_path.exists() or probe_path.is_symlink():
        probe = _read_probe(probe_path, outer=outer, arm=arm, anchor_receipt=anchor_receipt)
    else:
        probe = _virtual_probe(
            f005_contract, root, outer, shared_payload, arm, anchor_receipt,
        )
        _write_new_json(probe_path, probe)
        probe = _read_probe(probe_path, outer=outer, arm=arm, anchor_receipt=anchor_receipt)

    encoder, head, optimizer, trainable, seed = _make_components(f005_contract, root, outer)
    try:
        _validate_training_state_structure(shared_payload, encoder, head, optimizer, shared_payload["metadata"])
        _load_training_state(shared_payload, encoder, head, optimizer, shared_payload["metadata"])
        excluded = batchnorm_affine_parameter_names(encoder)
        anchor = build_l2sp_anchor(encoder, exclude_names=excluded)
        live_receipt = _anchor_receipt(anchor, excluded)
        _require(live_receipt == anchor_receipt, "F007 live anchor differs from its bootstrap receipt")
        fork_encoder = state_dict_sha256(encoder.state_dict())
        fork_head = state_dict_sha256(head.state_dict())

        def validate_checkpoint(candidate: Path) -> dict:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            metadata = validate_f007_resume_payload(payload, identity, fit)
            _validate_training_state_structure(payload, encoder, head, optimizer, metadata)
            return metadata

        if checkpoint_present:
            recover_fixed_partial(checkpoint, validate_checkpoint, derived=True)
        start = head_steps
        if _present(checkpoint):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = validate_f007_resume_payload(payload, identity, fit)
            _load_training_state(payload, encoder, head, optimizer, metadata)
            start = metadata["completed_steps"]
        _trim_history(history, start, first_step=head_steps + 1)

        def save(completed: int) -> None:
            _atomic_checkpoint({
                "metadata": f007_tail_checkpoint_metadata(identity, fit, completed),
                "encoder": encoder.state_dict(), "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }, checkpoint)

        runtime = _run_tail_updates(
            f005_contract, root, outer, arm, encoder, head, optimizer, anchor,
            start, total, history, tracker, save,
        )
        report = {
            "status": "complete",
            "stage": "l2sp_tail",
            "outer_fold": outer,
            "arm_id": arm_id,
            "l2sp_lambda": float(arm["lambda"]),
            "unit_signature": identity["signature"],
            "source_f005_signature": f005_contract["signature"],
            "tail_plan_sha256": tail_plan,
            "shared_head_checkpoint_sha256": shared_sha,
            "fork_encoder_sha256": fork_encoder,
            "fork_head_sha256": fork_head,
            "initialization_seed": seed,
            "anchor_receipt": anchor_receipt,
            "anchor_receipt_sha256": _sha(anchor_receipt),
            "l2sp_applied_once_per_optimizer_step": True,
            "dynamic_consistency": False,
            "raw_h_mse": False,
            "virtual_probe_path": str(probe_path),
            "virtual_probe_sha256": probe["probe_sha256"],
            "completed_steps": total,
            "schedule_state": adaptation_checkpoint_state(fit, total),
            "checkpoint_mlflow_uploaded": False,
            "checkpoint_local_transfer": False,
            **trainable,
            **runtime,
        }
        if tracker is not None:
            tracker.add_artifact(history, "training/fit_history.jsonl")
            tracker.add_artifact(probe_path, "receipts/virtual_l2sp_probe.json")
        return {"report": report, "checkpoint": checkpoint, "probe": probe}
    finally:
        del encoder, head, optimizer
        torch.cuda.empty_cache()
