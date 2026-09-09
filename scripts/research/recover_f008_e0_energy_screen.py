"""Recover the failed F008 E0 screen without mutating its original evidence.

The original E0 directory is read-only evidence.  This launcher creates its
only output below ``<failed-e0-directory>/recovery_v1``.  It authenticates and
reuses fold 0's completed checkpoint/cache without constructing a model or
opening audio, seals fold 0 again with the patched duration canonicalization,
then trains/extracts only fold 1 under a new, linked MLflow recovery run.

``--verify-only`` performs no CUDA attestation, model construction, audio
access, extraction callback, MLflow request, or filesystem mutation.  It
still validates the completed fold-0 cache and checkpoint bytes so an operator
can fail closed before requesting GPU work.  ``--execute`` performs a fresh
runtime attestation and requires its receipt SHA-256 to be exactly equal to
the failed screen's saved runtime receipt before it creates recovery evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.research import run_f008_energy_screen as e0


RECOVERY_VERSION = "recovery_v1"
RECOVERY_SCHEMA = "f008-e0-recovery-v1"
RECOVERY_FOLD_ARTIFACT_SOURCE = {"0": "failed_e0_parent", "1": "recovery_root"}
RECOVERY_FOLD_ARTIFACT_RELATIVE = {
    "0": "fold_0/energy_005",
    "1": "fold_1/energy_005",
}
_SHA256 = frozenset("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in _SHA256 for item in value)


def _regular_below(root: Path, relative: str, label: str) -> Path:
    """Resolve one regular non-symlink file below a confined evidence root."""
    base = Path(root).resolve(strict=True)
    candidate = base / PurePosixPath(relative)
    resolved = candidate.resolve(strict=True)
    _require(
        not candidate.is_symlink() and resolved.is_relative_to(base)
        and resolved.is_file() and not resolved.is_symlink(),
        f"F008 recovery {label} path escapes its evidence root",
    )
    return resolved


def _regular_directory_below(root: Path, relative: str, label: str) -> Path:
    base = Path(root).resolve(strict=True)
    candidate = base / PurePosixPath(relative)
    resolved = candidate.resolve(strict=True)
    _require(
        not candidate.is_symlink() and resolved.is_relative_to(base)
        and resolved.is_dir() and not resolved.is_symlink(),
        f"F008 recovery {label} path escapes its evidence root",
    )
    return resolved


def _read_json_below(root: Path, relative: str, label: str) -> tuple[dict[str, Any], Path]:
    path = _regular_below(root, relative, label)
    return e0._read_json(path, label), path


@dataclass(frozen=True)
class FailedE0Evidence:
    """Compact, JSON-safe identity of a failed E0 root that stays immutable."""

    root: Path
    parent_run_id: str
    failure_sha256: str
    runtime_summary_sha256: str
    previous_runtime_receipt_sha256: str


@dataclass(frozen=True)
class ReusedFold0:
    """Authenticated in-memory cache and compact source receipts for fold 0."""

    cache: dict[str, Any]
    checkpoint_receipt: dict[str, Any]
    checkpoint_path: Path
    cache_directory: Path
    tail_identity: dict[str, Any]
    checkpoint_relative_path: str
    cache_relative_path: str


def _validate_previous_runtime_summary(
        runtime: Mapping[str, object], *, f008_signature: str,
        f005_signature: str, execution: Mapping[str, object],
) -> str:
    """Validate the persisted redacted E0 runtime summary without CUDA access."""
    required = {
        "schema_version", "receipt_sha256", "f008_signature", "source_f005_signature",
        "expected_vast_instance_id", "vast_instance_id", "device", "gpu_name",
        "cuda_available", "cpu_threads", "checked_once_per_logical_run", "execution",
    }
    _require(isinstance(runtime, Mapping) and set(runtime) == required,
             "F008 recovery failed E0 runtime summary schema changed")
    receipt_sha = runtime.get("receipt_sha256")
    _require(
        _is_sha256(receipt_sha)
        and runtime.get("f008_signature") == f008_signature
        and runtime.get("source_f005_signature") == f005_signature
        and runtime.get("execution") == execution
        and runtime.get("device") == "cuda"
        and runtime.get("cuda_available") is True
        and isinstance(runtime.get("gpu_name"), str)
        and execution.get("gpu_name_contains") in runtime["gpu_name"]
        and runtime.get("checked_once_per_logical_run") is True,
        "F008 recovery failed E0 runtime summary does not match the pinned execution",
    )
    return str(receipt_sha)


def validate_failed_e0_evidence(
        failed_root: Path, *, f008_signature: str, f005_signature: str,
        execution: Mapping[str, object],
) -> FailedE0Evidence:
    """Validate only durable failed-run metadata; never amend the failed root."""
    root = Path(failed_root).resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "F008 recovery failed E0 root is unavailable")
    failure, failure_path = _read_json_below(root, "failure.json", "failed E0 receipt")
    _require(
        failure.get("status") == "failed" and failure.get("screen") == "E0"
        and failure.get("outer_evaluation_called") is False
        and failure.get("promotion_allowed") is False,
        "F008 recovery input is not an immutable failed E0 screen",
    )
    runtime, runtime_path = _read_json_below(root, "runtime_receipt.json", "failed E0 runtime receipt")
    previous_sha = _validate_previous_runtime_summary(
        runtime, f008_signature=f008_signature, f005_signature=f005_signature,
        execution=execution,
    )
    tracking, _tracking_path = _read_json_below(root, "tracking/run_state.json", "failed E0 tracking state")
    parent_run_id = tracking.get("run_id")
    _require(isinstance(parent_run_id, str) and parent_run_id,
             "F008 recovery failed E0 tracking state has no parent run ID")
    return FailedE0Evidence(
        root=root,
        parent_run_id=parent_run_id,
        failure_sha256=_sha256_file(failure_path),
        runtime_summary_sha256=_sha256_file(runtime_path),
        previous_runtime_receipt_sha256=previous_sha,
    )


def _validate_fold0_tail_binding(
        tail: Mapping[str, object], *, f008_signature: str,
        f005_signature: str, source_receipt_sha256: str,
        previous_runtime_receipt_sha256: str,
) -> None:
    arm = tail.get("arm") if isinstance(tail, Mapping) else None
    _require(
        isinstance(tail, Mapping)
        and tail.get("f008_signature") == f008_signature
        and tail.get("source_f005_signature") == f005_signature
        and tail.get("f005_source_receipt_sha256") == source_receipt_sha256
        and tail.get("outer_fold") == 0
        and isinstance(arm, Mapping) and arm.get("id") == "energy_005"
        and tail.get("runtime_receipt_sha256") == previous_runtime_receipt_sha256,
        "F008 recovery fold 0 tail belongs to another source, arm, or runtime receipt",
    )


def verify_reusable_fold0(
        failed: FailedE0Evidence, *, f008_signature: str,
        f005_contract: Mapping[str, object], source_receipt: Mapping[str, object],
) -> ReusedFold0:
    """Authenticate fold 0 bytes/cache without model loading or audio extraction."""
    from speaker_id.training.f008_extraction import (
        build_f008_advanced_cache_plan,
        load_f008_advanced_cache,
        validate_f008_tail_checkpoint_receipt,
        verify_validated_f008_tail_checkpoint_bytes,
    )

    checkpoint_payload, _checkpoint_receipt_path = _read_json_below(
        failed.root, "fold_0/energy_005/checkpoint_receipt.json", "fold 0 checkpoint receipt",
    )
    checkpoint = validate_f008_tail_checkpoint_receipt(checkpoint_payload)
    checkpoint_path = _regular_below(
        failed.root, "fold_0/energy_005/tail/last.pt", "fold 0 checkpoint",
    )
    verify_validated_f008_tail_checkpoint_bytes(checkpoint_path, checkpoint)
    tail = checkpoint["tail_identity"]
    _validate_fold0_tail_binding(
        tail, f008_signature=f008_signature,
        f005_signature=str(f005_contract["signature"]),
        source_receipt_sha256=e0._canonical_sha256(dict(source_receipt)),
        previous_runtime_receipt_sha256=failed.previous_runtime_receipt_sha256,
    )
    fold_directory = _regular_directory_below(
        failed.root, "fold_0/energy_005", "fold 0 evidence directory",
    )
    cache_directory = _regular_directory_below(
        failed.root, "fold_0/energy_005/full_scoring/energy_005", "fold 0 cache directory",
    )
    stored_identity, _identity_path = _read_json_below(
        failed.root, "fold_0/energy_005/full_scoring/energy_005/cache_identity.json",
        "fold 0 cache identity",
    )
    _read_json_below(
        failed.root, "fold_0/energy_005/full_scoring/energy_005/cache_receipt.json",
        "fold 0 cache receipt",
    )
    plan = build_f008_advanced_cache_plan(
        tail, output_directory=fold_directory,
        relative_cache_directory="full_scoring/energy_005", checkpoint_receipt=checkpoint,
        manifest=f005_contract["manifest"], inference=stored_identity.get("inference"),
    )
    _require(plan.identity == stored_identity,
             "F008 recovery fold 0 cache identity differs from its completed checkpoint plan")
    cache = load_f008_advanced_cache(plan)
    _require(
        cache["identity"] == stored_identity
        and cache["identity"]["checkpoint"] == checkpoint
        and cache["receipt"]["identity"] == stored_identity
        and cache["receipt"]["completed"] is True,
        "F008 recovery fold 0 cache receipt no longer authenticates the cache",
    )
    return ReusedFold0(
        cache=cache, checkpoint_receipt=checkpoint, checkpoint_path=checkpoint_path,
        cache_directory=cache_directory, tail_identity=dict(tail),
        checkpoint_relative_path="fold_0/energy_005/tail/last.pt",
        cache_relative_path="fold_0/energy_005/full_scoring/energy_005",
    )


def _recovery_root(failed: FailedE0Evidence, *, create: bool) -> Path:
    output = failed.root / RECOVERY_VERSION
    if create:
        _require(not output.exists() and not output.is_symlink(),
                 "F008 recovery_v1 already exists; recovery evidence is immutable")
        output.mkdir(parents=False, exist_ok=False)
    else:
        _require(not output.exists() and not output.is_symlink(),
                 "F008 recovery_v1 already exists; verify a new failed root instead")
    return output


def recovery_declaration(
        failed: FailedE0Evidence, *, fresh_runtime_receipt_sha256: str | None,
        fold0: ReusedFold0 | None = None,
) -> dict[str, object]:
    """Return the deterministic metadata bridge that outer evaluation rechecks."""
    _require(fresh_runtime_receipt_sha256 is None or _is_sha256(fresh_runtime_receipt_sha256),
             "F008 recovery fresh runtime receipt SHA is invalid")
    values: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA,
        "version": RECOVERY_VERSION,
        "failed_e0_directory_name": failed.root.name,
        "failed_e0_parent_run_id": failed.parent_run_id,
        "failed_e0_failure_sha256": failed.failure_sha256,
        "failed_e0_runtime_summary_sha256": failed.runtime_summary_sha256,
        "failed_e0_runtime_receipt_sha256": failed.previous_runtime_receipt_sha256,
        "fresh_runtime_receipt_sha256": fresh_runtime_receipt_sha256,
        "fold_artifact_source_by_outer": dict(RECOVERY_FOLD_ARTIFACT_SOURCE),
        "fold_artifact_relative_path_by_outer": dict(RECOVERY_FOLD_ARTIFACT_RELATIVE),
        "failed_root_mutated": False,
        "fold_0_training_executed": False,
        "fold_0_audio_extraction_executed": False,
        # Fold 1 has not run when this declaration is first persisted in the
        # fold-0 report and parent-run provenance.  The completed screen gets
        # a separately materialized declaration after the fresh fold succeeds.
        "fold_1_training_executed": False,
        "fold_1_audio_extraction_executed": False,
        "fold_1_training_planned": True,
        "fold_1_audio_extraction_planned": True,
    }
    if fold0 is not None:
        values["fold_0_reused_checkpoint_receipt_sha256"] = fold0.checkpoint_receipt["signature"]
        values["fold_0_reused_cache_identity_signature"] = fold0.cache["identity"]["signature"]
        values["fold_0_reused_cache_receipt_sha256"] = fold0.cache["receipt"]["receipt_sha256"]
        values["fold_0_checkpoint_relative_path"] = fold0.checkpoint_relative_path
        values["fold_0_cache_relative_path"] = fold0.cache_relative_path
    return values


def _fold_report(
        *, outer: int, checkpoint_receipt: Mapping[str, object], cache: Mapping[str, object],
        binding: Mapping[str, object], frozen_receipt: Mapping[str, object],
        inner: Mapping[str, object], recovery: Mapping[str, object],
        fit: Mapping[str, object], gradient_probe: Mapping[str, object] | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "status": "complete",
        "screen": {
            "id": "E0", "active_arms": ["control_f005", "energy_005"],
            "deferred_uniform": True, "outer_evaluation_called": False,
            "outer_labels_used_for_selection": False,
            "selection_or_promotion_allowed": False,
        },
        "outer_fold": outer,
        "fit": dict(fit),
        "checkpoint_receipt": dict(checkpoint_receipt),
        "cache": {
            "identity_signature": cache["identity"]["signature"],
            "receipt_sha256": cache["receipt"]["receipt_sha256"],
            "row_count": int(len(cache["embeddings"])),
            "server_only": True,
            "mlflow_upload_allowed": False,
            "local_transfer_allowed": False,
        },
        "source_control_binding": dict(binding),
        "frozen_c002_cache": dict(frozen_receipt),
        "inner_calibration": dict(inner),
        "recovery": dict(recovery),
        "outer_truth_read": False,
        "model_weights_uploaded": False,
        "optimizer_state_uploaded": False,
        "embeddings_uploaded": False,
        "raw_audio_uploaded": False,
    }
    if gradient_probe is not None:
        report["gradient_probe"] = dict(gradient_probe)
    return report


def _seal_pretruth(
        *, recovery_fold_directory: Path, outer: int,
        f005_contract: Mapping[str, object], public_embeddings, frozen_advanced_embeddings,
        control_embeddings, energy_embeddings, valid, source_receipt: Mapping[str, object],
        control_binding: Mapping[str, object], scoring_spec: Mapping[str, object],
) -> tuple[dict[str, object], Path]:
    from speaker_id.training.f008_scoring import prepare_and_seal_pretruth, reload_pretruth_seal

    policy_path = recovery_fold_directory / "scoring" / "pretruth_e0_energy_only_seal.json"
    _require(not policy_path.exists() and not policy_path.is_symlink(),
             "F008 recovery refuses to replace a pretruth seal")
    prepared = prepare_and_seal_pretruth(
        f005_contract, outer, public_embeddings=public_embeddings,
        frozen_advanced_embeddings=frozen_advanced_embeddings,
        f005_control_embeddings=control_embeddings,
        f005_source_receipt=source_receipt, f005_control_binding=control_binding,
        f008_advanced_embeddings_by_arm={
            "control_f005": control_embeddings,
            "energy_005": energy_embeddings,
        },
        valid=valid, scoring_spec=scoring_spec, policy_seal_path=policy_path,
    )
    reloaded = reload_pretruth_seal(
        policy_path, f005_contract, outer, scoring_spec=scoring_spec,
        f005_source_receipt=source_receipt, f005_control_binding=control_binding,
    )
    _require(reloaded == prepared["policy_reload"],
             "F008 recovery pretruth seal changed after its first disk reload")
    return reloaded, policy_path


def _reused_fold0_pretruth(
        *, recovery_root: Path, reused: ReusedFold0,
        f005_contract: Mapping[str, object], source_receipt: Mapping[str, object],
        source_root: Path, public_embeddings, frozen_advanced_embeddings, frozen_valid,
        frozen_receipt: Mapping[str, object], scoring_spec: Mapping[str, object],
        recovery: Mapping[str, object],
) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    """Seal fold 0 from authenticated arrays; deliberately no tail/audio call."""
    import numpy as np

    from speaker_id.training.f008_extraction import load_reused_f005_control_cache
    from speaker_id.training.f008_scoring import bind_authenticated_f005_control_embeddings

    fold_directory = recovery_root / "fold_0" / "energy_005"
    fold_directory.mkdir(parents=True, exist_ok=False)
    control = load_reused_f005_control_cache(
        reused.tail_identity, source_receipt, f005_run_directory=source_root,
        manifest=f005_contract["manifest"], outer_fold=0,
    )
    _require(
        np.array_equal(frozen_valid, reused.cache["valid"])
        and np.array_equal(frozen_valid, control["valid"]),
        "F008 recovery fold 0 frozen/control/energy validity masks differ",
    )
    binding = bind_authenticated_f005_control_embeddings(
        source_receipt, 0, embeddings=control["embeddings"], valid=control["valid"],
    )
    policy_reload, policy_path = _seal_pretruth(
        recovery_fold_directory=fold_directory, outer=0, f005_contract=f005_contract,
        public_embeddings=public_embeddings, frozen_advanced_embeddings=frozen_advanced_embeddings,
        control_embeddings=control["embeddings"], energy_embeddings=reused.cache["embeddings"],
        valid=frozen_valid, source_receipt=source_receipt, control_binding=binding,
        scoring_spec=scoring_spec,
    )
    inner = e0._inner_summary(policy_reload, outer=0)
    report = _fold_report(
        outer=0, checkpoint_receipt=reused.checkpoint_receipt, cache=reused.cache,
        binding=binding, frozen_receipt=frozen_receipt, inner=inner, recovery={
            **dict(recovery), "mode": "reused_completed_failed_e0_cache",
            "training_executed": False, "audio_extraction_executed": False,
        },
        fit={
            "status": "reused_completed_tail", "training_executed": False,
            "audio_extraction_executed": False,
            "checkpoint_relative_to_failed_e0": reused.checkpoint_relative_path,
            "cache_relative_to_failed_e0": reused.cache_relative_path,
        },
        gradient_probe=None,
    )
    report_path = fold_directory / "fold_report.json"
    e0._write_json(report_path, report)
    return report, policy_path, binding, inner, policy_reload


def _fresh_fold1(
        *, recovery_root: Path, parent, binding_state: Path, config: Mapping[str, object],
        f008_signature: str, f005_contract: Mapping[str, object], source_receipt: Mapping[str, object],
        source_root: Path, preflight: Mapping[int, Mapping[str, object]], runtime_receipt: Mapping[str, object],
        public_embeddings, frozen_advanced_embeddings, frozen_valid, frozen_receipt: Mapping[str, object],
        scoring_spec: Mapping[str, object], recovery: Mapping[str, object],
):
    """Train/extract precisely fold 1, the only fresh CUDA work in recovery_v1."""
    import numpy as np

    from speaker_id.tracking import DurableMLflowRun
    from speaker_id.training.f008_extraction import (
        load_reused_f005_control_cache,
        validated_f008_tail_checkpoint_receipt,
    )
    from speaker_id.training.f008_scoring import bind_authenticated_f005_control_embeddings
    from speaker_id.training.f008_worker import fit_f008_tail, load_authenticated_f008_tail

    outer = 1
    fold_directory = recovery_root / "fold_1" / "energy_005"
    fold_directory.mkdir(parents=True, exist_ok=False)
    child = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=fold_directory / "tracking", binding=e0._binding(binding_state),
        parent_run_id=parent.run_id, run_name="F008-E0-recovery_v1-energy_005-fold1",
        config={
            "recovery_version": RECOVERY_VERSION, "recovery": dict(recovery),
            "screen": {"id": "E0", "deferred_uniform": True,
                       "outer_evaluation_called": False, "promotion_allowed": False},
            "outer_fold": outer, "active_arm": "energy_005",
            "f008_config_signature": f008_signature,
            "f005_contract_signature": f005_contract["signature"],
            "preflight_receipt_sha256": preflight[outer]["receipt_sha256"],
            "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
            "fold_0_training_executed": False,
        },
        input_paths={
            "launcher": Path(__file__), "worker": ROOT / "src/speaker_id/training/f008_worker.py",
            "extraction": ROOT / "src/speaker_id/training/f008_extraction.py",
            "scoring": ROOT / "src/speaker_id/training/f008_scoring.py",
        },
        run_kind="f008_e0_recovery_v1_fold1_energy_tail", training_started=True,
    )
    try:
        child.flush(strict=True)
        shared_head = e0._source_shared_head(source_root, source_receipt, outer)

        def on_step(event: Mapping[str, object]) -> None:
            _require(type(event.get("step")) is int and event.get("phase") == "tail",
                     "F008 recovery worker emitted a malformed live metric event")
            metrics = {
                str(name): float(value)
                for name, value in event.items()
                if name not in {"step", "phase"}
                and type(value) in (int, float) and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            child.log_metrics(metrics, step=event["step"], sync=False)
            if event["step"] % 25 == 0:
                child.flush(strict=False)

        fit = fit_f008_tail(
            config, f005_contract, ROOT, outer, "energy_005", shared_head,
            source_receipt, preflight[outer], fold_directory / "tail",
            runtime_receipt=runtime_receipt, resume=False, on_step=on_step,
        )
        e0._log_probe_metrics(child, fit["probe"], outer)
        history_path = fold_directory / "tail" / "fit_history.jsonl"
        if not (child.directory / "events.jsonl").read_text(encoding="utf-8").strip():
            e0._log_history_metrics(child, history_path)
        loaded = load_authenticated_f008_tail(
            config, f005_contract, ROOT, outer, "energy_005", shared_head,
            source_receipt, preflight[outer], fit["checkpoint"], runtime_receipt=runtime_receipt,
        )
        try:
            _require(loaded["identity"] == fit["identity"],
                     "F008 recovery fold 1 completed-tail identity changed")
            checkpoint_receipt = validated_f008_tail_checkpoint_receipt(
                loaded["identity"], checkpoint_path=fit["checkpoint"],
                checkpoint_metadata=loaded["metadata"],
            )
            checkpoint_receipt_path = fold_directory / "checkpoint_receipt.json"
            e0._write_json(checkpoint_receipt_path, checkpoint_receipt)
            cache = e0._extract_energy_cache(
                loaded_tail=loaded, checkpoint=fit["checkpoint"], checkpoint_receipt=checkpoint_receipt,
                f005_contract=f005_contract, fold_directory=fold_directory,
                runtime_receipt=runtime_receipt, tracker=child,
            )
        finally:
            for name in ("encoder", "head", "optimizer"):
                if name in loaded:
                    del loaded[name]
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        control = load_reused_f005_control_cache(
            fit["identity"], source_receipt, f005_run_directory=source_root,
            manifest=f005_contract["manifest"], outer_fold=outer,
        )
        _require(
            np.array_equal(frozen_valid, cache["valid"])
            and np.array_equal(frozen_valid, control["valid"]),
            "F008 recovery fold 1 frozen/control/energy validity masks differ",
        )
        control_binding = bind_authenticated_f005_control_embeddings(
            source_receipt, outer, embeddings=control["embeddings"], valid=control["valid"],
        )
        policy_reload, policy_path = _seal_pretruth(
            recovery_fold_directory=fold_directory, outer=outer, f005_contract=f005_contract,
            public_embeddings=public_embeddings, frozen_advanced_embeddings=frozen_advanced_embeddings,
            control_embeddings=control["embeddings"], energy_embeddings=cache["embeddings"],
            valid=frozen_valid, source_receipt=source_receipt, control_binding=control_binding,
            scoring_spec=scoring_spec,
        )
        inner = e0._inner_summary(policy_reload, outer=outer)
        e0._log_inner_metrics(child, inner, outer)
        report = _fold_report(
            outer=outer, checkpoint_receipt=checkpoint_receipt, cache=cache,
            binding=control_binding, frozen_receipt=frozen_receipt, inner=inner,
            recovery={**dict(recovery), "mode": "fresh_fold_1_only",
                      "training_executed": True, "audio_extraction_executed": True},
            fit=fit["report"], gradient_probe=fit["probe"],
        )
        report_path = fold_directory / "fold_report.json"
        e0._write_json(report_path, report)
        for source, relative in (
            (fold_directory / "tail" / "zero_optimizer_gradient_probe.json", "probe/gradient_probe.json"),
            (history_path, "training/fit_history.jsonl"),
            (checkpoint_receipt_path, "checkpoint/checkpoint_receipt.json"),
            (fold_directory / "full_scoring" / "energy_005" / "cache_receipt.json", "cache/cache_receipt.json"),
            (policy_path, "scoring/pretruth_e0_energy_only_seal.json"),
            (report_path, "fold_report.json"),
        ):
            e0._safe_add_metadata_artifact(child, source, relative)
        child.write_report(report, markdown=(
            "# F008 E0 recovery_v1 fold 1\n\n"
            "Only fold 1 was trained and extracted afresh. Fold 0 remains an authenticated, "
            "server-only reuse from the immutable failed E0 root. No outer evaluation or promotion occurred.\n"
        ))
        child.flush(strict=True)
        child.verify_artifacts()
        child.finish("FINISHED", strict=True)
        child.verify_remote_metadata()
        return {
            "report": report, "policy_path": policy_path, "policy_reload": policy_reload,
            "inner": inner,
            "child_run_id": child.run_id, "checkpoint_receipt": checkpoint_receipt,
            "cache": cache, "gradient_probe_sha256": fit["probe"]["probe_sha256"],
        }
    except BaseException as error:
        failure = {
            "status": "failed", "screen": "E0", "recovery_version": RECOVERY_VERSION,
            "outer_fold": outer, "error_type": type(error).__name__,
            "error": child.redactor.text(str(error)), "outer_evaluation_called": False,
            "model_weights_uploaded": False, "optimizer_state_uploaded": False,
            "embeddings_uploaded": False, "raw_audio_uploaded": False,
        }
        failure_path = fold_directory / "e0_recovery_failure.json"
        if not failure_path.exists():
            e0._write_json(failure_path, failure)
        try:
            e0._safe_add_metadata_artifact(child, failure_path, "e0_recovery_failure.json")
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


def _recovery_provenance(
        failed: FailedE0Evidence, recovery: Mapping[str, object],
        fold0: ReusedFold0, *, f008_signature: str, f005_signature: str,
) -> dict[str, object]:
    return {
        "schema_version": RECOVERY_SCHEMA,
        "recovery": dict(recovery),
        "f008_config_signature": f008_signature,
        "f005_contract_signature": f005_signature,
        "failed_e0": {
            "directory_name": failed.root.name,
            "parent_run_id": failed.parent_run_id,
            "failure_sha256": failed.failure_sha256,
            "runtime_summary_sha256": failed.runtime_summary_sha256,
            "runtime_receipt_sha256": failed.previous_runtime_receipt_sha256,
        },
        "fold_0_reuse": {
            "checkpoint_receipt_sha256": fold0.checkpoint_receipt["signature"],
            "checkpoint_bytes_sha256": fold0.checkpoint_receipt["checkpoint_sha256"],
            "cache_identity_signature": fold0.cache["identity"]["signature"],
            "cache_receipt_sha256": fold0.cache["receipt"]["receipt_sha256"],
            "checkpoint_relative_to_failed_e0": fold0.checkpoint_relative_path,
            "cache_relative_to_failed_e0": fold0.cache_relative_path,
            "training_executed": False,
            "audio_extraction_executed": False,
        },
        "raw_audio_uploaded": False,
        "embeddings_uploaded": False,
        "model_weights_uploaded": False,
        "optimizer_state_uploaded": False,
        "local_model_transfer": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/campp_f008_unknown_oe.json"))
    parser.add_argument("--binding-state", type=Path,
                        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"))
    parser.add_argument("--preflight-directory", type=Path,
                        default=e0.DEFAULT_PREFLIGHT_DIRECTORY)
    parser.add_argument("--failed-e0-directory", type=Path, required=True,
                        help="Failed F008_E0_ENERGY_SCREEN_* directory below F008 output_root.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true",
                      help="Authenticate recovery inputs without CUDA, MLflow, audio, or writes.")
    mode.add_argument("--execute", action="store_true",
                      help="Create recovery_v1, reuse fold 0, and train/extract only fold 1.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_path = e0._confined(args.config, "configs/train", must_exist=True)
    from speaker_id.training.f008_config import config_signature, load_f008_config

    config = load_f008_config(config_path)
    signature = config_signature(config)
    active_spec = e0.reduced_energy_screen_spec(config)
    failed_directory = e0._confined(args.failed_e0_directory, str(config["output_root"]), must_exist=True)
    _require(failed_directory.name != RECOVERY_VERSION,
             "F008 recovery input must be the failed E0 root, not recovery_v1")

    f005_config_path = e0._confined(Path(config["source_f005"]["config_path"]),
                                     "configs/train", must_exist=True)
    _require(e0._sha256_file(f005_config_path) == config["source_f005"]["config_sha256"],
             "F008 recovery pinned F005 config bytes changed")
    from speaker_id.training.f005_contract import load_f005_contract
    from speaker_id.training.f007_source import load_f005_source_receipt
    from speaker_id.training.f008_preflight import (
        authenticated_f005_source_contract,
        validate_f005_control_selection,
    )

    current_f005 = load_f005_contract(f005_config_path, ROOT, verify_audio=False,
                                      verify_sources=False)
    source_root = Path(config["source_f005"]["run_dir"])
    f005_contract, source_bridge = authenticated_f005_source_contract(
        current_f005, source_root, ROOT, config["source_f005"],
    )
    source_receipt = load_f005_source_receipt(source_root)
    _require(source_receipt["experiment_state"]["experiment_signature"] == f005_contract["signature"],
             "F008 recovery source receipt does not match the bridged F005 contract")
    control_seals = validate_f005_control_selection(source_root, config["source_f005"])
    preflight_directory = e0._confined(args.preflight_directory, str(config["output_root"]), must_exist=True)
    preflight = e0._preflight_receipts(
        preflight_directory, config_signature=signature,
        f005_signature=f005_contract["signature"], source_receipt=source_receipt,
        f005_contract=f005_contract, source_root=source_root,
        fold_ids=list(config["fold_ids"]),
    )
    failed = validate_failed_e0_evidence(
        failed_directory, f008_signature=signature,
        f005_signature=f005_contract["signature"], execution=config["execution"],
    )

    # No function above (and no verify-only path below) opens audio, creates a
    # model, invokes CUDA, or contacts MLflow.  It authenticates checkpoint
    # bytes and cache files only through their existing receipts.
    if args.verify_only:
        reused = verify_reusable_fold0(
            failed, f008_signature=signature, f005_contract=f005_contract,
            source_receipt=source_receipt,
        )
        _require(reused.tail_identity["preflight_receipt_sha256"] == preflight[0]["receipt_sha256"],
                 "F008 recovery fold 0 checkpoint does not bind the pinned preflight receipt")
        _recovery_root(failed, create=False)
        print(json.dumps({
            "status": "f008_e0_recovery_v1_verified_no_cuda_no_mlflow_no_audio_no_writes",
            "failed_e0_directory": str(failed.root),
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "previous_runtime_receipt_sha256": failed.previous_runtime_receipt_sha256,
            "fresh_runtime_attestation_required_before_execute": True,
            "fold_0": {
                "checkpoint_receipt_sha256": reused.checkpoint_receipt["signature"],
                "cache_identity_signature": reused.cache["identity"]["signature"],
                "cache_receipt_sha256": reused.cache["receipt"]["receipt_sha256"],
                "training_executed": False,
                "audio_extraction_executed": False,
            },
            "fold_1": {"training_will_execute_only_with_execute": True},
            "outer_evaluation_called": False,
            "promotion_allowed": False,
        }, ensure_ascii=False, indent=2), flush=True)
        return

    # The fresh CUDA receipt is the first execute-only action and must be an
    # exact hash match before we read/reuse cache arrays, create a spool, or
    # create any recovery_v1 path.
    from speaker_id.training.f008_worker import attest_f008_runtime

    runtime_receipt = attest_f008_runtime(config, f005_contract)
    _require(runtime_receipt["receipt_sha256"] == failed.previous_runtime_receipt_sha256,
             "F008 recovery fresh runtime receipt SHA differs from the failed E0 receipt")
    reused = verify_reusable_fold0(
        failed, f008_signature=signature, f005_contract=f005_contract,
        source_receipt=source_receipt,
    )
    _require(reused.tail_identity["preflight_receipt_sha256"] == preflight[0]["receipt_sha256"],
             "F008 recovery fold 0 checkpoint does not bind the pinned preflight receipt")
    binding_path = e0._confined(args.binding_state, "artifacts/infrastructure", must_exist=True)
    recovery_root = _recovery_root(failed, create=True)
    recovery = recovery_declaration(
        failed, fresh_runtime_receipt_sha256=runtime_receipt["receipt_sha256"], fold0=reused,
    )

    from speaker_id.evaluation.scoring_bridge import _load_c002_identity_cache
    from speaker_id.tracking import DurableMLflowRun

    c002_root = Path(f005_contract["config"]["source_c002b"]["run_dir"])
    frozen_vectors, frozen_valid, frozen_receipt = _load_c002_identity_cache(
        c002_root, f005_contract["manifest"],
    )
    resolved = {
        "recovery": recovery,
        "f008_config": config,
        "f008_config_signature": signature,
        "f005_contract_signature": f005_contract["signature"],
        "current_reconstructed_f005_signature": current_f005["signature"],
        "source_contract_bridge": source_bridge,
        "control_selection_seals": control_seals,
        "preflight_receipt_sha256_by_outer": {
            str(outer): receipt["receipt_sha256"] for outer, receipt in preflight.items()
        },
        "verify_audio": False,
        "fold_0_reused_without_training_or_audio_extraction": True,
        "fold_1_fresh_training_and_audio_extraction_only": True,
        "mlflow_forbidden_payloads_uploaded": False,
    }
    parent = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=recovery_root / "tracking", binding=e0._binding(binding_path),
        parent_run_id=failed.parent_run_id,
        run_name="F008-E0-recovery_v1-fold0-reuse-fold1-fresh-calibration-screen",
        config=resolved,
        input_paths={
            "f008_config": config_path, "f005_config": f005_config_path,
            "launcher": Path(__file__), "source_launcher": ROOT / "scripts/research/run_f008_energy_screen.py",
            "worker": ROOT / "src/speaker_id/training/f008_worker.py",
            "extraction": ROOT / "src/speaker_id/training/f008_extraction.py",
            "scoring": ROOT / "src/speaker_id/training/f008_scoring.py",
            "evaluator": ROOT / "src/speaker_id/training/f008_e0_evaluation.py",
        },
        run_kind="f008_e0_recovery_v1_energy_only_calibration_screen", training_started=True,
    )
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        runtime_path = recovery_root / "runtime_receipt.json"
        provenance_path = recovery_root / "recovery_provenance.json"
        bridge_path = recovery_root / "source_contract_bridge.json"
        seals_path = recovery_root / "f005_control_selection_seals.json"
        e0._write_json(runtime_path, e0._runtime_summary(runtime_receipt))
        e0._write_json(provenance_path, _recovery_provenance(
            failed, recovery, reused, f008_signature=signature,
            f005_signature=f005_contract["signature"],
        ))
        e0._write_json(bridge_path, source_bridge)
        e0._write_json(seals_path, control_seals)
        for source, relative in (
            (runtime_path, "runtime/runtime_receipt.json"),
            (provenance_path, "provenance/recovery_provenance.json"),
            (bridge_path, "provenance/source_contract_bridge.json"),
            (seals_path, "provenance/f005_control_selection_seals.json"),
        ):
            e0._safe_add_metadata_artifact(parent, source, relative)
        for outer, receipt in preflight.items():
            path = recovery_root / "preflight" / f"fold_{outer}_energy_margin_receipt.json"
            e0._write_json(path, receipt)
            e0._safe_add_metadata_artifact(parent, path, f"preflight/{path.name}")
        parent.log_metrics({
            "recovery/fold_0/training_executed": 0.0,
            "recovery/fold_0/audio_extraction_executed": 0.0,
            "recovery/fold_1/training_planned": 1.0,
            "recovery/runtime_sha_exact_match": 1.0,
        }, step=0, sync=False)
        parent.flush(strict=True)

        fold0_report, fold0_policy_path, fold0_binding, fold0_inner, fold0_policy_reload = _reused_fold0_pretruth(
            recovery_root=recovery_root, reused=reused, f005_contract=f005_contract,
            source_receipt=source_receipt, source_root=source_root,
            public_embeddings=frozen_vectors["public"],
            frozen_advanced_embeddings=frozen_vectors["advanced"], frozen_valid=frozen_valid,
            frozen_receipt=frozen_receipt, scoring_spec=active_spec, recovery=recovery,
        )
        fold0_report_path = recovery_root / "fold_0" / "energy_005" / "fold_report.json"
        e0._safe_add_metadata_artifact(parent, fold0_policy_path,
                                       "fold_0/scoring/pretruth_e0_energy_only_seal.json")
        e0._safe_add_metadata_artifact(parent, fold0_report_path, "fold_0/fold_report.json")
        e0._log_inner_metrics(parent, fold0_inner, 0)
        parent.flush(strict=True)

        fold1 = _fresh_fold1(
            recovery_root=recovery_root, parent=parent, binding_state=binding_path,
            config=config, f008_signature=signature, f005_contract=f005_contract,
            source_receipt=source_receipt, source_root=source_root, preflight=preflight,
            runtime_receipt=runtime_receipt, public_embeddings=frozen_vectors["public"],
            frozen_advanced_embeddings=frozen_vectors["advanced"], frozen_valid=frozen_valid,
            frozen_receipt=frozen_receipt, scoring_spec=active_spec, recovery=recovery,
        )
        fold1_report_path = recovery_root / "fold_1" / "energy_005" / "fold_report.json"
        completed_recovery = {
            **recovery,
            "fold_1_training_executed": True,
            "fold_1_audio_extraction_executed": True,
        }
        screen = {
            "status": "complete",
            "screen": {
                "id": "E0", "active_arms": active_spec["arm_ids"],
                "deferred_uniform": True, "outer_evaluation_called": False,
                "outer_labels_used_for_selection": False,
                "selection_or_promotion_allowed": False,
                "interpretation": "calibration_only_causal_screen_not_oof_or_leaderboard_claim",
            },
            "recovery": completed_recovery,
            "recovery_tracking_run_id": parent.run_id,
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "runtime": e0._runtime_summary(runtime_receipt),
            "preflight_receipt_sha256_by_outer": {
                str(outer): receipt["receipt_sha256"] for outer, receipt in preflight.items()
            },
            "folds": [
                {
                    "outer_fold": 0, "child_run_id": None,
                    "checkpoint_receipt_sha256": reused.checkpoint_receipt["signature"],
                    "cache_receipt_sha256": reused.cache["receipt"]["receipt_sha256"],
                    "pretruth_seal_sha256": fold0_policy_reload["seal_sha256"],
                    "pretruth_seal_file_sha256": e0._sha256_file(fold0_policy_path),
                    "fold_report_sha256": e0._sha256_file(fold0_report_path),
                    "inner_calibration": fold0_inner,
                    "training_executed": False, "audio_extraction_executed": False,
                },
                {
                    "outer_fold": 1, "child_run_id": fold1["child_run_id"],
                    "gradient_probe_sha256": fold1["gradient_probe_sha256"],
                    "checkpoint_receipt_sha256": fold1["checkpoint_receipt"]["signature"],
                    "cache_receipt_sha256": fold1["cache"]["receipt"]["receipt_sha256"],
                    "pretruth_seal_sha256": fold1["policy_reload"]["seal_sha256"],
                    "pretruth_seal_file_sha256": e0._sha256_file(fold1["policy_path"]),
                    "fold_report_sha256": e0._sha256_file(fold1_report_path),
                    "inner_calibration": fold1["inner"],
                    "training_executed": True, "audio_extraction_executed": True,
                },
            ],
            "uniform_next_step_allowed_only_after_review": True,
            "outer_evaluation_called": False,
            "promotion_decision": "forbidden_for_E0_calibration_only_screen",
            "raw_audio_uploaded": False,
            "embeddings_uploaded": False,
            "model_weights_uploaded": False,
            "optimizer_state_uploaded": False,
            "local_model_transfer": False,
        }
        screen_path = recovery_root / "screen_report.json"
        e0._write_json(screen_path, screen)
        e0._safe_add_metadata_artifact(parent, screen_path, "screen_report.json")
        parent.write_report(screen, markdown=(
            "# F008 E0 recovery_v1: fold 0 reuse, fold 1 fresh\n\n"
            "Fold 0 checkpoint/cache were authenticated in the immutable failed E0 root and reused "
            "without training or audio extraction. Only fold 1 was trained/extracted again. Both policies "
            "were sealed on calibration roles only; outer evaluation and promotion remain forbidden.\n"
        ))
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.finish("FINISHED", strict=True)
        parent.verify_remote_metadata()
        print(json.dumps({
            "status": "complete", "run_id": parent.run_id, "output": str(recovery_root),
            "recovery_version": RECOVERY_VERSION, "active_arms": active_spec["arm_ids"],
            "outer_evaluation_called": False, "promotion_allowed": False,
            "fold_0_training_executed": False, "fold_0_audio_extraction_executed": False,
            "fold_1_training_executed": True, "fold_1_audio_extraction_executed": True,
        }, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        failure = {
            "status": "failed", "screen": "E0", "recovery_version": RECOVERY_VERSION,
            "error_type": type(error).__name__, "error": parent.redactor.text(str(error)),
            "outer_evaluation_called": False, "promotion_allowed": False,
            "model_weights_uploaded": False, "optimizer_state_uploaded": False,
            "embeddings_uploaded": False, "raw_audio_uploaded": False,
        }
        failure_path = recovery_root / "failure.json"
        if not failure_path.exists():
            e0._write_json(failure_path, failure)
        try:
            e0._safe_add_metadata_artifact(parent, failure_path, "failure.json")
            parent.write_report(failure)
            parent.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
