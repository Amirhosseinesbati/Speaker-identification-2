"""Read-only, one-shot outer evaluation for a completed F008 E0 screen.

E0 trains only the energy arm and seals its calibration policies without
touching outer labels.  This module is the deliberately separate follow-up
stage: it authenticates the completed screen, reloads the server-only caches,
recomputes the sealed pretruth score bundles with guarded outer rows, reloads
*all* fold seals, and only then materializes outer labels for one evaluation.

It never constructs a model, opens audio, changes a configuration, retrains,
or transfers an artifact.  Callers should upload only its compact metadata
report to MLflow; cache/checkpoint and per-file evaluation receipts stay on the
server.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training.f008_scoring import (
    ACTIVE_ARM_EVALUATION_SCHEMA,
    bind_authenticated_f005_control_embeddings,
    evaluate_outer_active_arms_once,
    normalize_scoring_spec,
    normalize_outer_duration,
    rebuild_and_validate_pretruth_bundle,
    reload_all_pretruth_seals,
    scoring_spec_from_f008_config,
)


E0_SCREEN_ID = "E0"
E0_ACTIVE_ARMS = ("control_f005", "energy_005")
E0_OUTER_REPORT_SCHEMA = "f008-e0-post-screen-outer-report-v1"
E0_RECOVERY_VERSION = "recovery_v1"
E0_RECOVERY_SCHEMA = "f008-e0-recovery-v1"
E0_RECOVERY_FOLD_ARTIFACT_SOURCE = {"0": "failed_e0_parent", "1": "recovery_root"}
E0_RECOVERY_FOLD_ARTIFACT_RELATIVE = {
    "0": "fold_0/energy_005",
    "1": "fold_1/energy_005",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"F008 E0 {label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F008 E0 {label} is unreadable") from error
    _require(isinstance(value, dict), f"F008 E0 {label} must be a JSON object")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically create output evidence and refuse to replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F008 E0 refuses to replace evidence: {path}")
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"F008 E0 refuses to replace evidence: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _regular_below(root: Path, relative: str, label: str) -> Path:
    base = Path(root).resolve(strict=True)
    candidate = base / PurePosixPath(relative)
    resolved = candidate.resolve(strict=True)
    _require(
        not candidate.is_symlink() and resolved.is_relative_to(base)
        and resolved.is_file() and not resolved.is_symlink(),
        f"F008 E0 {label} path escapes its screen directory",
    )
    return resolved


def _regular_directory_below(root: Path, relative: str, label: str) -> Path:
    base = Path(root).resolve(strict=True)
    candidate = base / PurePosixPath(relative)
    resolved = candidate.resolve(strict=True)
    _require(
        not candidate.is_symlink() and resolved.is_relative_to(base)
        and resolved.is_dir() and not resolved.is_symlink(),
        f"F008 E0 {label} path escapes its screen directory",
    )
    return resolved


def e0_active_scoring_spec(config: Mapping[str, object]) -> dict[str, object]:
    """Project immutable F008 config into E0's two actually available arms."""
    full = scoring_spec_from_f008_config(config)
    return normalize_scoring_spec({
        **full,
        "arm_ids": list(E0_ACTIVE_ARMS),
        "control_arm_id": E0_ACTIVE_ARMS[0],
        "arm_tie_order": list(E0_ACTIVE_ARMS),
    })


def _validated_recovery_declaration(screen: Mapping[str, object]) -> dict[str, object] | None:
    """Validate the fixed recovery_v1 bridge without accepting supplied paths."""
    recovery = screen.get("recovery")
    if recovery is None:
        return None
    _require(isinstance(recovery, Mapping), "F008 E0 recovery declaration is malformed")
    required_sha = (
        "failed_e0_failure_sha256", "failed_e0_runtime_summary_sha256",
        "failed_e0_runtime_receipt_sha256", "fresh_runtime_receipt_sha256",
        "fold_0_reused_checkpoint_receipt_sha256",
        "fold_0_reused_cache_identity_signature",
        "fold_0_reused_cache_receipt_sha256",
    )
    _require(
        recovery.get("schema_version") == E0_RECOVERY_SCHEMA
        and recovery.get("version") == E0_RECOVERY_VERSION
        and isinstance(recovery.get("failed_e0_directory_name"), str)
        and bool(recovery["failed_e0_directory_name"])
        and isinstance(recovery.get("failed_e0_parent_run_id"), str)
        and bool(recovery["failed_e0_parent_run_id"])
        and all(_is_sha256(recovery.get(key)) for key in required_sha)
        and recovery.get("fresh_runtime_receipt_sha256")
        == recovery.get("failed_e0_runtime_receipt_sha256")
        and recovery.get("fold_artifact_source_by_outer")
        == E0_RECOVERY_FOLD_ARTIFACT_SOURCE
        and recovery.get("fold_artifact_relative_path_by_outer")
        == E0_RECOVERY_FOLD_ARTIFACT_RELATIVE
        and recovery.get("failed_root_mutated") is False
        and recovery.get("fold_0_training_executed") is False
        and recovery.get("fold_0_audio_extraction_executed") is False
        and recovery.get("fold_1_training_executed") is True
        and recovery.get("fold_1_audio_extraction_executed") is True
        and isinstance(screen.get("recovery_tracking_run_id"), str)
        and bool(screen["recovery_tracking_run_id"]),
        "F008 E0 recovery declaration is not the fixed authenticated recovery_v1 bridge",
    )
    return dict(recovery)


def _resolve_fold_artifact_directory(
        e0_directory: Path, *, outer: int,
        recovery: Mapping[str, object] | None,
) -> Path:
    """Resolve fixed cache/checkpoint roots for standard or recovery E0 evidence."""
    root = Path(e0_directory).resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "F008 E0 screen directory is unavailable")
    if recovery is None:
        return _regular_directory_below(root, f"fold_{outer}/energy_005", "energy fold directory")
    _require(
        root.name == E0_RECOVERY_VERSION
        and root.parent.is_dir() and not root.parent.is_symlink()
        and root.parent.name == recovery["failed_e0_directory_name"],
        "F008 E0 recovery root is not attached to its immutable failed E0 parent",
    )
    _require(str(outer) in E0_RECOVERY_FOLD_ARTIFACT_SOURCE,
             "F008 E0 recovery has no fixed artifact root for this fold")
    source = E0_RECOVERY_FOLD_ARTIFACT_SOURCE[str(outer)]
    relative = E0_RECOVERY_FOLD_ARTIFACT_RELATIVE[str(outer)]
    base = root.parent if source == "failed_e0_parent" else root
    return _regular_directory_below(base, relative, "recovery energy fold directory")


def _validated_recovery_runtime_hashes(
        e0_directory: Path, *, recovery: Mapping[str, object],
        f008_signature: str, f005_signature: str,
) -> tuple[str, str]:
    """Read both durable recovery runtime receipts before cache reuse.

    The recovery declaration is not treated as the source of truth here: the
    failed parent and fresh recovery summaries are read from their fixed,
    confined locations.  Their receipt hashes are then compared with the
    declaration before either cached fold can be scored.
    """
    root = Path(e0_directory).resolve(strict=True)
    _require(
        root.name == E0_RECOVERY_VERSION
        and root.parent.is_dir() and not root.parent.is_symlink()
        and root.parent.name == recovery.get("failed_e0_directory_name"),
        "F008 E0 recovery runtime roots are not attached to the declared failed parent",
    )
    failed_path = _regular_below(root.parent, "runtime_receipt.json", "failed runtime receipt")
    fresh_path = _regular_below(root, "runtime_receipt.json", "fresh recovery runtime receipt")
    failed = _read_json(failed_path, "failed runtime receipt")
    fresh = _read_json(fresh_path, "fresh recovery runtime receipt")
    failed_hash = failed.get("receipt_sha256")
    fresh_hash = fresh.get("receipt_sha256")
    _require(
        failed.get("schema_version") == "f008-runtime-receipt-v1"
        and fresh.get("schema_version") == "f008-runtime-receipt-v1"
        and failed.get("f008_signature") == f008_signature
        and fresh.get("f008_signature") == f008_signature
        and failed.get("source_f005_signature") == f005_signature
        and fresh.get("source_f005_signature") == f005_signature
        and _is_sha256(failed_hash) and _is_sha256(fresh_hash)
        and failed_hash == recovery.get("failed_e0_runtime_receipt_sha256")
        and fresh_hash == recovery.get("fresh_runtime_receipt_sha256")
        and _sha256_file(failed_path) == recovery.get("failed_e0_runtime_summary_sha256"),
        "F008 E0 recovery runtime receipts differ from the immutable recovery declaration",
    )
    return str(failed_hash), str(fresh_hash)


def _validate_recovery_cached_tail_runtime(
        *, recovery: Mapping[str, object] | None,
        recovery_runtime_hashes: tuple[str, str] | None,
        tail: Mapping[str, object],
) -> None:
    """Bind a cached tail to both independently read recovery runtimes."""
    if recovery is None:
        _require(recovery_runtime_hashes is None,
                 "F008 E0 non-recovery cache unexpectedly has recovery runtime evidence")
        return
    _require(recovery_runtime_hashes is not None,
             "F008 E0 recovery cache lacks independently read runtime evidence")
    failed_runtime_hash, fresh_runtime_hash = recovery_runtime_hashes
    tail_runtime_hash = tail.get("runtime_receipt_sha256")
    _require(
        _is_sha256(tail_runtime_hash)
        and tail_runtime_hash == failed_runtime_hash
        and tail_runtime_hash == fresh_runtime_hash
        and tail_runtime_hash == recovery.get("failed_e0_runtime_receipt_sha256")
        and tail_runtime_hash == recovery.get("fresh_runtime_receipt_sha256"),
        "F008 E0 recovery cached tail belongs to another failed or fresh runtime receipt",
    )


def _validate_recovery_fold0_reuse(
        *, recovery: Mapping[str, object] | None, outer: int,
        checkpoint: Mapping[str, object], cache: Mapping[str, object],
        fold_summary: Mapping[str, object],
) -> None:
    """Bind recovery's declared fold-0 reuse values to validated source/cache bytes."""
    if recovery is None or outer != 0:
        return
    identity = cache.get("identity") if isinstance(cache, Mapping) else None
    receipt = cache.get("receipt") if isinstance(cache, Mapping) else None
    checkpoint_signature = checkpoint.get("signature")
    cache_identity_signature = identity.get("signature") if isinstance(identity, Mapping) else None
    cache_receipt_sha256 = receipt.get("receipt_sha256") if isinstance(receipt, Mapping) else None
    _require(
        _is_sha256(checkpoint_signature)
        and _is_sha256(cache_identity_signature)
        and _is_sha256(cache_receipt_sha256)
        and recovery.get("fold_0_reused_checkpoint_receipt_sha256") == checkpoint_signature
        and recovery.get("fold_0_reused_cache_identity_signature") == cache_identity_signature
        and recovery.get("fold_0_reused_cache_receipt_sha256") == cache_receipt_sha256
        and fold_summary.get("checkpoint_receipt_sha256") == checkpoint_signature
        and fold_summary.get("cache_receipt_sha256") == cache_receipt_sha256,
        "F008 E0 recovery fold 0 reuse declaration differs from validated checkpoint, cache, or fold summary",
    )


def validate_completed_e0_screen(
        screen: Mapping[str, object], *, f008_signature: str,
        f005_signature: str, fold_ids: Sequence[int],
) -> dict[str, object]:
    """Reject an incomplete/calibration-only screen before cache inspection."""
    _require(
        isinstance(screen, Mapping) and screen.get("status") == "complete"
        and screen.get("f008_config_signature") == f008_signature
        and screen.get("f005_contract_signature") == f005_signature,
        "F008 E0 screen is incomplete or belongs to another configuration/source",
    )
    declaration = screen.get("screen")
    _require(
        isinstance(declaration, Mapping)
        and declaration.get("id") == E0_SCREEN_ID
        and declaration.get("active_arms") == list(E0_ACTIVE_ARMS)
        and declaration.get("deferred_uniform") is True
        and declaration.get("outer_evaluation_called") is False
        and declaration.get("outer_labels_used_for_selection") is False
        and declaration.get("selection_or_promotion_allowed") is False
        and screen.get("outer_evaluation_called") is False
        and screen.get("promotion_decision") == "forbidden_for_E0_calibration_only_screen",
        "F008 E0 screen declaration is not the required calibration-only screen",
    )
    recovery = _validated_recovery_declaration(screen)
    folds = screen.get("folds")
    expected = list(fold_ids)
    _require(
        isinstance(folds, list) and len(folds) == len(expected),
        "F008 E0 screen has an incomplete fold inventory",
    )
    by_outer: dict[int, dict[str, object]] = {}
    for row in folds:
        _require(isinstance(row, Mapping) and type(row.get("outer_fold")) is int,
                 "F008 E0 fold summary is malformed")
        outer = int(row["outer_fold"])
        _require(outer in expected and outer not in by_outer
                 and _is_sha256(row.get("checkpoint_receipt_sha256"))
                 and _is_sha256(row.get("cache_receipt_sha256"))
                 and _is_sha256(row.get("pretruth_seal_sha256")),
                 "F008 E0 fold summary lacks immutable evidence")
        if recovery is not None:
            _require(
                _is_sha256(row.get("fold_report_sha256"))
                and _is_sha256(row.get("pretruth_seal_file_sha256"))
                and row.get("training_executed") is (outer == 1)
                and row.get("audio_extraction_executed") is (outer == 1),
                "F008 E0 recovery fold summary lacks its recovery-bound evidence",
            )
        by_outer[outer] = dict(row)
    _require(set(by_outer) == set(expected), "F008 E0 screen folds differ from config")
    return {"screen": dict(screen), "folds_by_outer": by_outer, "recovery": recovery}


@dataclass(frozen=True)
class E0PreparedFold:
    """Only compact post-rebuild evidence; embeddings stay out of this object."""

    outer_fold: int
    pretruth: dict[str, object]
    policy_path: Path
    f005_control_binding: dict[str, object]
    checkpoint_receipt_sha256: str
    cache_identity_signature: str
    cache_receipt_sha256: str


def _validate_fold_report(
        report: Mapping[str, object], *, outer: int,
        screen_fold: Mapping[str, object], f008_signature: str,
) -> None:
    declaration = report.get("screen") if isinstance(report, Mapping) else None
    cache = report.get("cache") if isinstance(report, Mapping) else None
    _require(
        isinstance(report, Mapping) and report.get("status") == "complete"
        and report.get("outer_fold") == outer
        and isinstance(declaration, Mapping)
        and declaration.get("id") == E0_SCREEN_ID
        and declaration.get("active_arms") == list(E0_ACTIVE_ARMS)
        and declaration.get("outer_evaluation_called") is False
        and declaration.get("selection_or_promotion_allowed") is False
        and isinstance(cache, Mapping)
        and cache.get("server_only") is True
        and cache.get("mlflow_upload_allowed") is False
        and cache.get("local_transfer_allowed") is False
        and _is_sha256(cache.get("identity_signature"))
        and _is_sha256(cache.get("receipt_sha256"))
        and isinstance(report.get("source_control_binding"), Mapping)
        and _is_sha256(report["source_control_binding"].get("binding_sha256"))
        and isinstance(report.get("inner_calibration"), Mapping)
        and report.get("outer_truth_read") is False
        and report.get("model_weights_uploaded") is False
        and report.get("optimizer_state_uploaded") is False
        and report.get("embeddings_uploaded") is False
        and report.get("raw_audio_uploaded") is False,
        "F008 E0 fold report is not a completed server-only calibration screen",
    )
    _require(
        cache["receipt_sha256"] == screen_fold.get("cache_receipt_sha256"),
        "F008 E0 screen and fold report name different energy-cache receipts",
    )
    # The actual cache receipt is compared after its per-file authentication.
    _require(_is_sha256(f008_signature), "F008 signature is malformed")


def _load_energy_cache_and_control(
        *, e0_directory: Path, artifact_fold_directory: Path, outer: int, f008_signature: str,
        f005_contract: Mapping[str, object], source_receipt: Mapping[str, object],
        source_root: Path, frozen_valid: np.ndarray,
        fold_summary: Mapping[str, object], frozen_receipt: Mapping[str, object],
        recovery: Mapping[str, object] | None,
        recovery_runtime_hashes: tuple[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object],
           dict[str, object], Path, dict[str, object]]:
    """Authenticate E0's persisted energy cache and F005 control cache in place."""
    from speaker_id.training.f008_extraction import (
        build_f008_advanced_cache_plan,
        load_f008_advanced_cache,
        load_reused_f005_control_cache,
        validate_f008_tail_checkpoint_receipt,
    )

    root = Path(e0_directory).resolve(strict=True)
    fold_root = Path(artifact_fold_directory).resolve(strict=True)
    _require(fold_root.is_dir() and not fold_root.is_symlink(),
             "F008 E0 energy cache/checkpoint fold directory is unavailable")
    report_path = _regular_below(
        root, f"fold_{outer}/energy_005/fold_report.json", "fold report",
    )
    report = _read_json(report_path, "fold report")
    _validate_fold_report(report, outer=outer, screen_fold=fold_summary,
                          f008_signature=f008_signature)
    checkpoint = validate_f008_tail_checkpoint_receipt(_read_json(
        _regular_below(fold_root, "checkpoint_receipt.json", "checkpoint receipt"),
        "checkpoint receipt",
    ))
    cache_relative = "full_scoring/energy_005"
    stored_identity = _read_json(
        _regular_below(fold_root, f"{cache_relative}/cache_identity.json", "energy cache identity"),
        "energy cache identity",
    )
    _require(
        stored_identity.get("checkpoint") == checkpoint
        and isinstance(stored_identity.get("tail_identity"), Mapping),
        "F008 E0 energy cache is not bound to its completed checkpoint",
    )
    tail = stored_identity["tail_identity"]
    _require(
        tail.get("f008_signature") == f008_signature
        and tail.get("source_f005_signature") == f005_contract.get("signature")
        and tail.get("f005_source_receipt_sha256") == _sha(dict(source_receipt))
        and tail.get("outer_fold") == outer
        and isinstance(tail.get("arm"), Mapping)
        and tail["arm"].get("id") == "energy_005",
        "F008 E0 energy cache belongs to another arm, fold, or source",
    )
    _validate_recovery_cached_tail_runtime(
        recovery=recovery, recovery_runtime_hashes=recovery_runtime_hashes, tail=tail,
    )
    plan = build_f008_advanced_cache_plan(
        tail, output_directory=fold_root, relative_cache_directory="full_scoring/energy_005",
        checkpoint_receipt=checkpoint, manifest=f005_contract["manifest"],
        inference=stored_identity.get("inference"),
    )
    _require(plan.identity == stored_identity,
             "F008 E0 energy cache plan differs from its stored identity")
    cache = load_f008_advanced_cache(plan)
    _require(
        cache["identity"]["signature"] == report["cache"]["identity_signature"]
        and cache["receipt"]["receipt_sha256"] == report["cache"]["receipt_sha256"]
        and cache["receipt"]["receipt_sha256"] == fold_summary["cache_receipt_sha256"]
        and checkpoint["signature"] == fold_summary["checkpoint_receipt_sha256"],
        "F008 E0 fold/report cache or checkpoint evidence changed",
    )
    _validate_recovery_fold0_reuse(
        recovery=recovery, outer=outer, checkpoint=checkpoint, cache=cache,
        fold_summary=fold_summary,
    )
    control = load_reused_f005_control_cache(
        tail, source_receipt, f005_run_directory=source_root,
        manifest=f005_contract["manifest"], outer_fold=outer,
    )
    _require(
        np.array_equal(frozen_valid, cache["valid"])
        and np.array_equal(frozen_valid, control["valid"]),
        "F008 E0 frozen/control/energy validity masks differ",
    )
    binding = bind_authenticated_f005_control_embeddings(
        source_receipt, outer, embeddings=control["embeddings"], valid=control["valid"],
    )
    _require(
        binding == report["source_control_binding"]
        and report.get("frozen_c002_cache") == frozen_receipt,
        "F008 E0 F005 control binding or frozen source receipt changed",
    )
    policy_path = _regular_below(
        root, f"fold_{outer}/energy_005/scoring/pretruth_e0_energy_only_seal.json",
        "pretruth policy seal",
    )
    if "fold_report_sha256" in fold_summary:
        _require(
            _sha256_file(report_path) == fold_summary["fold_report_sha256"]
            and _sha256_file(policy_path) == fold_summary["pretruth_seal_file_sha256"],
            "F008 E0 recovery report or pretruth seal bytes changed",
        )
    return (
        np.asarray(cache["embeddings"], dtype=np.float32),
        np.asarray(control["embeddings"], dtype=np.float32),
        binding, checkpoint, cache, policy_path, report,
    )


def rebuild_e0_pretruth_bundles(
        *, e0_directory: Path, config: Mapping[str, object],
        f008_signature: str, f005_contract: Mapping[str, object],
        source_receipt: Mapping[str, object], source_root: Path,
        screen: Mapping[str, object],
) -> dict[int, E0PreparedFold]:
    """Recompute E0's two-arm score bundles and match every immutable seal.

    All cache access is server-local.  The scorer guards outer rows, so this
    function never consumes an outer speaker label.
    """
    fold_ids = list(config["fold_ids"])
    validated_screen = validate_completed_e0_screen(
        screen, f008_signature=f008_signature,
        f005_signature=str(f005_contract["signature"]), fold_ids=fold_ids,
    )
    active_spec = e0_active_scoring_spec(config)
    root = Path(e0_directory).resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "F008 E0 screen directory is unavailable")
    recovery_runtime_hashes = (
        _validated_recovery_runtime_hashes(
            root, recovery=validated_screen["recovery"], f008_signature=f008_signature,
            f005_signature=str(f005_contract["signature"]),
        ) if validated_screen["recovery"] is not None else None
    )

    from speaker_id.evaluation.scoring_bridge import _load_c002_identity_cache

    c002_root = Path(f005_contract["config"]["source_c002b"]["run_dir"])
    frozen_vectors, frozen_valid, frozen_receipt = _load_c002_identity_cache(
        c002_root, f005_contract["manifest"],
    )
    public = frozen_vectors["public"]
    frozen_advanced = frozen_vectors["advanced"]
    prepared: dict[int, E0PreparedFold] = {}
    for outer in fold_ids:
        artifact_fold_directory = _resolve_fold_artifact_directory(
            root, outer=outer, recovery=validated_screen["recovery"],
        )
        energy, control, binding, checkpoint, cache, policy_path, report = _load_energy_cache_and_control(
            e0_directory=root, artifact_fold_directory=artifact_fold_directory,
            outer=outer, f008_signature=f008_signature,
            f005_contract=f005_contract, source_receipt=source_receipt,
            source_root=source_root, frozen_valid=frozen_valid,
            fold_summary=validated_screen["folds_by_outer"][outer],
            frozen_receipt=frozen_receipt,
            recovery=validated_screen["recovery"],
            recovery_runtime_hashes=recovery_runtime_hashes,
        )
        rebuilt = rebuild_and_validate_pretruth_bundle(
            f005_contract, outer, public_embeddings=public,
            frozen_advanced_embeddings=frozen_advanced,
            f005_control_embeddings=control, f005_source_receipt=source_receipt,
            f005_control_binding=binding,
            f008_advanced_embeddings_by_arm={
                "control_f005": control,
                "energy_005": energy,
            },
            valid=frozen_valid, scoring_spec=active_spec,
            policy_seal_path=policy_path,
        )
        _require(
            rebuilt["outer_truth_read"] is False
            and rebuilt["policy_reload"]["seal_sha256"]
            == validated_screen["folds_by_outer"][outer]["pretruth_seal_sha256"]
            and rebuilt["policy_reload"]["seal"]["scoring_spec"] == active_spec,
            "F008 E0 recomputed pretruth evidence differs from the screen seal",
        )
        prepared[outer] = E0PreparedFold(
            outer_fold=outer, pretruth=rebuilt, policy_path=policy_path,
            f005_control_binding=binding,
            checkpoint_receipt_sha256=checkpoint["signature"],
            cache_identity_signature=cache["identity"]["signature"],
            cache_receipt_sha256=cache["receipt"]["receipt_sha256"],
        )
    _require(set(prepared) == set(fold_ids), "F008 E0 prepared folds are incomplete")
    return prepared


def materialize_outer_truth_rows(
        contract: Mapping[str, object], outer: int, policy_seal: Mapping[str, object],
) -> list[dict[str, object]]:
    """Materialize the minimum ordered outer truth rows *after* all seals reload."""
    manifest = contract.get("manifest")
    folds = contract.get("folds")
    roles = contract.get("roles")
    metadata = policy_seal.get("outer_public_metadata") if isinstance(policy_seal, Mapping) else None
    _require(
        isinstance(manifest, list) and isinstance(folds, list) and isinstance(roles, list)
        and isinstance(metadata, list),
        "F008 E0 outer truth inputs are incomplete",
    )
    manifest_by_name = {row.get("audio_file"): row for row in manifest if isinstance(row, Mapping)}
    folds_by_name = {row.get("audio_file"): row for row in folds if isinstance(row, Mapping)}
    roles_by_name = {
        row.get("audio_file"): row for row in roles
        if isinstance(row, Mapping) and row.get("outer_fold") == outer
    }
    _require(
        len(manifest_by_name) == len(manifest) and len(folds_by_name) == len(folds)
        and len(roles_by_name) == len(manifest),
        "F008 E0 manifest/fold/role identities are malformed",
    )
    rows: list[dict[str, object]] = []
    for public in metadata:
        _require(isinstance(public, Mapping) and isinstance(public.get("audio_file"), str),
                 "F008 E0 sealed outer metadata are malformed")
        name = public["audio_file"]
        source, fold, role = manifest_by_name.get(name), folds_by_name.get(name), roles_by_name.get(name)
        _require(isinstance(source, Mapping) and isinstance(fold, Mapping) and isinstance(role, Mapping),
                 "F008 E0 outer truth differs from its sealed public identity")
        duration = normalize_outer_duration(source.get("duration_seconds"))
        _require(
            truth(role.get("outer_evaluation_included"))
            and fold.get("group_id") == public.get("group_id")
            and duration == public.get("duration_seconds")
            and truth(source.get("has_nonzero_signal")) == public.get("has_nonzero_signal"),
            "F008 E0 outer truth differs from its sealed public identity",
        )
        speaker = source.get("speaker_id")
        _require(isinstance(speaker, str) and speaker in contract.get("labels", ()),
                 "F008 E0 outer speaker label is outside the fixed label map")
        rows.append({
            "audio_file": name, "speaker_id": speaker, "group_id": fold["group_id"],
            "duration_seconds": duration,
            "has_nonzero_signal": source["has_nonzero_signal"],
        })
    return rows


def _compact_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _index_predictions(rows: object, *, expected: set[str], labels: set[str], name: str) -> dict[str, str]:
    _require(isinstance(rows, list), f"F008 E0 {name} predictions must be a list")
    indexed: dict[str, str] = {}
    for row in rows:
        _require(
            isinstance(row, Mapping) and isinstance(row.get("audio_file"), str)
            and row["audio_file"] not in indexed and row.get("speaker_id") in labels,
            f"F008 E0 {name} predictions are malformed",
        )
        indexed[row["audio_file"]] = row["speaker_id"]
    _require(set(indexed) == expected, f"F008 E0 {name} predictions lack outer coverage")
    return indexed


def _metrics_match(recorded: object, calculated: Mapping[str, object], name: str) -> None:
    _require(isinstance(recorded, Mapping) and set(recorded) == set(calculated),
             f"F008 E0 {name} recorded metric schema changed")
    _require(recorded == calculated, f"F008 E0 {name} recorded metrics differ from predictions")


def _outer_rows_for_receipt(
        receipt: Mapping[str, object], *, outer: int,
        contract: Mapping[str, object],
) -> list[dict[str, object]]:
    references = receipt.get("outer_reference")
    _require(isinstance(references, list), "F008 E0 evaluation has no outer reference")
    manifest = {row.get("audio_file"): row for row in contract["manifest"]}
    folds = {row.get("audio_file"): row for row in contract["folds"]}
    roles = {
        row.get("audio_file"): row for row in contract["roles"]
        if row.get("outer_fold") == outer
    }
    names: set[str] = set()
    output: list[dict[str, object]] = []
    for row in references:
        _require(isinstance(row, Mapping) and isinstance(row.get("audio_file"), str),
                 "F008 E0 outer reference row is malformed")
        name = row["audio_file"]
        source, fold, role = manifest.get(name), folds.get(name), roles.get(name)
        _require(isinstance(source, Mapping) and isinstance(fold, Mapping) and isinstance(role, Mapping),
                 "F008 E0 outer reference differs from immutable contract")
        duration = normalize_outer_duration(source.get("duration_seconds"))
        _require(
            name not in names and truth(role.get("outer_evaluation_included"))
            and row.get("speaker_id") == source.get("speaker_id")
            and row.get("group_id") == fold.get("group_id")
            and row.get("duration_seconds") == duration
            and truth(row.get("has_nonzero_signal")) == truth(source.get("has_nonzero_signal")),
            "F008 E0 outer reference differs from immutable contract",
        )
        names.add(name)
        output.append(dict(row))
    _require(bool(output), "F008 E0 outer reference is empty")
    return output


def aggregate_e0_outer_evaluations(
        contract: Mapping[str, object], receipts_by_outer: Mapping[int, Mapping[str, object]],
        *, active_arms: Sequence[str] = E0_ACTIVE_ARMS,
) -> dict[str, object]:
    """Pool both immutable fold receipts and calculate control→energy deltas."""
    labels = contract.get("labels")
    fold_ids = contract.get("config", {}).get("fold_ids") if isinstance(contract.get("config"), Mapping) else None
    arms = list(active_arms)
    _require(
        isinstance(labels, list) and len(labels) == 447 and labels[0] == "unknown"
        and isinstance(fold_ids, list) and set(receipts_by_outer) == set(fold_ids)
        and arms == list(E0_ACTIVE_ARMS),
        "F008 E0 aggregate needs the fixed full label map and both E0 arms",
    )
    references: list[dict[str, object]] = []
    predicted_rows: dict[str, list[dict[str, str]]] = {arm: [] for arm in arms}
    fold_metrics: dict[str, dict[str, dict[str, object]]] = {}
    receipt_hashes: dict[str, str] = {}
    selected_by_outer: dict[str, str] = {}
    seen: set[str] = set()
    for outer in fold_ids:
        receipt = receipts_by_outer[outer]
        _require(isinstance(receipt, Mapping), "F008 E0 outer receipt is malformed")
        body = {key: value for key, value in receipt.items() if key != "evaluation_sha256"}
        _require(
            receipt.get("schema_version") == ACTIVE_ARM_EVALUATION_SCHEMA
            and receipt.get("experiment_signature") == contract.get("signature")
            and receipt.get("outer_fold") == outer
            and receipt.get("active_arm_ids") == arms
            and receipt.get("one_shot_outer_evaluation") is True
            and receipt.get("evaluation_sha256") == _sha(body)
            and isinstance(receipt.get("predictions"), Mapping)
            and set(receipt["predictions"]) == set(arms)
            and isinstance(receipt.get("metrics"), Mapping)
            and set(receipt["metrics"]) == set(arms),
            "F008 E0 outer receipt identity or arm inventory changed",
        )
        outer_reference = _outer_rows_for_receipt(receipt, outer=outer, contract=contract)
        names = {row["audio_file"] for row in outer_reference}
        _require(not (seen & names), "F008 E0 outer fold references overlap")
        seen.update(names)
        references.extend(outer_reference)
        fold_metrics[str(outer)] = {}
        for arm in arms:
            indexed = _index_predictions(
                receipt["predictions"][arm], expected=names, labels=set(labels),
                name=f"fold {outer} {arm}",
            )
            rows = [{"audio_file": row["audio_file"], "speaker_id": indexed[row["audio_file"]]}
                    for row in outer_reference]
            calculated = score_predictions(outer_reference, rows, labels)
            _metrics_match(receipt["metrics"][arm], calculated, f"fold {outer} {arm}")
            predicted_rows[arm].extend(rows)
            fold_metrics[str(outer)][arm] = _compact_metrics(calculated)
        receipt_hashes[str(outer)] = str(receipt["evaluation_sha256"])
        selected = receipt.get("selected_arm")
        _require(selected in arms, "F008 E0 receipt selected a non-E0 arm")
        selected_by_outer[str(outer)] = selected

    manifest_names = [row.get("audio_file") for row in contract["manifest"]]
    _require(len(manifest_names) == len(set(manifest_names)) and seen == set(manifest_names),
             "F008 E0 outer fold union is not full OOF coverage")
    reference_by_name = {row["audio_file"]: row for row in references}
    reference = [reference_by_name[name] for name in manifest_names]
    metrics: dict[str, dict[str, object]] = {}
    predictions_by_name: dict[str, dict[str, str]] = {}
    for arm in arms:
        prediction = _index_predictions(
            predicted_rows[arm], expected=set(manifest_names), labels=set(labels), name=arm,
        )
        predictions_by_name[arm] = prediction
        metrics[arm] = score_predictions(
            reference,
            [{"audio_file": row["audio_file"], "speaker_id": prediction[row["audio_file"]]}
             for row in reference], labels,
        )
    control, energy = metrics[E0_ACTIVE_ARMS[0]], metrics[E0_ACTIVE_ARMS[1]]
    directional = {
        key: {
            "control_f005": int(control["errors"][key]),
            "energy_005": int(energy["errors"][key]),
            "delta_energy_minus_control": int(energy["errors"][key]) - int(control["errors"][key]),
        }
        for key in ("known_to_unknown", "unknown_to_known", "known_to_other_known")
    }
    control_class = {row["speaker_id"]: row for row in control["per_class"]}
    energy_class = {row["speaker_id"]: row for row in energy["per_class"]}
    class_deltas: list[dict[str, object]] = []
    for label in labels:
        before, after = control_class[label], energy_class[label]
        matching = [row for row in reference if row["speaker_id"] == label]
        class_deltas.append({
            "speaker_id": label,
            "support": int(before["support"]),
            "control_f005_f1": float(before["f1"]),
            "energy_005_f1": float(after["f1"]),
            "f1_delta_energy_minus_control": float(after["f1"] - before["f1"]),
            "control_f005_true_positive": int(before["true_positive"]),
            "energy_005_true_positive": int(after["true_positive"]),
            "corrected": sum(
                predictions_by_name[E0_ACTIVE_ARMS[0]][row["audio_file"]] != label
                and predictions_by_name[E0_ACTIVE_ARMS[1]][row["audio_file"]] == label
                for row in matching
            ),
            "regressed": sum(
                predictions_by_name[E0_ACTIVE_ARMS[0]][row["audio_file"]] == label
                and predictions_by_name[E0_ACTIVE_ARMS[1]][row["audio_file"]] != label
                for row in matching
            ),
        })
    return {
        "schema_version": E0_OUTER_REPORT_SCHEMA,
        "experiment_signature": contract["signature"],
        "stage": "E0_post_screen_outer_evaluation",
        "active_arm_ids": arms,
        "selected_arm_by_outer_fold": selected_by_outer,
        "outer_evaluation_receipt_sha256_by_outer": receipt_hashes,
        "metrics": {arm: _compact_metrics(metrics[arm]) for arm in arms},
        "fold_metrics": fold_metrics,
        "deltas": {
            "macro_f1_energy_minus_control": float(energy["macro_f1"] - control["macro_f1"]),
            "accuracy_energy_minus_control": float(energy["accuracy"] - control["accuracy"]),
            "directional_errors": directional,
        },
        "class_f1_deltas_energy_minus_control": class_deltas,
        "outer_truth_first_access_stage": "after_all_disk_reloaded_pretruth_seals",
        "one_shot_outer_evaluation": True,
        "selection_or_promotion_allowed": False,
        "promotion_decision": "forbidden_for_e0_post_screen_comparison_only",
        "raw_audio_uploaded": False,
        "embeddings_uploaded": False,
        "model_weights_uploaded": False,
        "optimizer_state_uploaded": False,
        "local_model_transfer": False,
    }


def prepared_e0_metadata(prepared: Mapping[int, E0PreparedFold]) -> dict[str, object]:
    """Return only MLflow-safe rebuild identities; never score arrays/caches."""
    result: dict[str, object] = {}
    for outer, item in sorted(prepared.items()):
        seal = item.pretruth["policy_reload"]["seal"]
        result[str(outer)] = {
            "policy_path": str(item.policy_path),
            "policy_file_sha256": item.pretruth["policy_reload"]["file_sha256"],
            "policy_seal_sha256": item.pretruth["policy_reload"]["seal_sha256"],
            "selected_arm": item.pretruth["selected_arm"],
            "f005_control_binding_sha256": item.f005_control_binding["binding_sha256"],
            "checkpoint_receipt_sha256": item.checkpoint_receipt_sha256,
            "cache_identity_signature": item.cache_identity_signature,
            "cache_receipt_sha256": item.cache_receipt_sha256,
            "arm_policy_score_evidence_sha256": {
                arm: seal["arm_policies"][arm]["score_evidence"]["receipt_sha256"]
                for arm in E0_ACTIVE_ARMS
            },
            "outer_truth_read_during_rebuild": False,
        }
    return result


def evaluate_rebuilt_e0_outer(
        *, contract: Mapping[str, object], config: Mapping[str, object],
        source_receipt: Mapping[str, object], prepared: Mapping[int, E0PreparedFold],
        output_directory: Path,
        outer_truth_provider: Callable[[Mapping[str, object], int, Mapping[str, object]], list[dict[str, object]]]
        = materialize_outer_truth_rows,
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    """Reload all fold seals, then evaluate control and energy once per fold.

    The provider is deliberately a callable rather than pre-materialized rows:
    it cannot access an outer ``speaker_id`` until this function has completed
    the all-fold seal reload below.
    """
    fold_ids = list(config["fold_ids"])
    spec = e0_active_scoring_spec(config)
    _require(
        isinstance(prepared, Mapping) and set(prepared) == set(fold_ids)
        and callable(outer_truth_provider),
        "F008 E0 outer evaluation needs every rebuilt fold and a truth provider",
    )
    for outer in fold_ids:
        item = prepared[outer]
        _require(
            isinstance(item, E0PreparedFold) and item.outer_fold == outer
            and item.pretruth.get("outer_truth_read") is False
            and item.pretruth.get("policy_reload", {}).get("seal", {}).get("scoring_spec") == spec,
            "F008 E0 rebuilt fold is not a sealed two-arm pretruth bundle",
        )
    output = Path(output_directory)
    _require(output.is_dir() and not output.is_symlink(),
             "F008 E0 outer evaluation output directory is unavailable")

    # This completes the all-fold disk reload before a truth provider can be
    # invoked.  The scorer repeats the check immediately before each one-shot
    # evaluation, which also catches any post-reload seal mutation.
    all_reloads = reload_all_pretruth_seals(
        contract, scoring_spec=spec,
        policy_paths_by_outer={outer: prepared[outer].policy_path for outer in fold_ids},
        f005_source_receipt=source_receipt,
        f005_control_bindings_by_outer={
            outer: prepared[outer].f005_control_binding for outer in fold_ids
        },
    )
    receipts: dict[int, dict[str, object]] = {}
    for outer in fold_ids:
        item = prepared[outer]
        path = output / f"fold_{outer}_outer_evaluation.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"F008 E0 outer fold was already evaluated once: {path}")
        rows = outer_truth_provider(contract, outer, item.pretruth["policy_reload"]["seal"])
        receipts[outer] = evaluate_outer_active_arms_once(
            contract, item.pretruth, all_reloads, rows, list(contract["labels"]),
            scoring_spec=spec, arm_ids=list(E0_ACTIVE_ARMS), evaluation_path=path,
        )
    report = aggregate_e0_outer_evaluations(contract, receipts)
    _write_new_json(output / "outer_evaluation_report.json", report)
    return report, receipts
