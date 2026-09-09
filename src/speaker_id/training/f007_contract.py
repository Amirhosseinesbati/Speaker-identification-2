"""Fail-closed protocol primitives for the F007 L2-SP experiment.

F007 is intentionally a sibling of F005.  This module owns only immutable
protocol evidence and the small resumable state machine needed to protect the
evaluation order.  It does not import Torch, decode audio, open an MLflow
client, inspect F005 caches, or materialize an outer label.

The public sequence is deliberately narrow:

* build a contract from the preregistered F007 configuration and a previously
  authenticated F005 source receipt;
* write and disk-reload a known-query arm-selection seal for every outer fold;
* write and disk-reload an independent open-set policy seal for every fold;
* register both sets of immutable receipts in the F007 state; and only then
  call :func:`unlock_outer_truth`.

``unlock_outer_truth`` is the sole state transition that exposes a capability
to an outer-evaluation loader.  A scoring implementation should require that
capability before it calls any function which reads outer speaker labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import uuid


F007_CONFIG_SHA256 = "2fa575df3f724e89e8922995335285d335ad63e02f97bfa0862c073f1cc53c30"
F007_CONTRACT_SCHEMA = "f007-l2sp-contract-v1"
F007_STATE_SCHEMA = "f007-l2sp-experiment-state-v1"
F007_ARM_SELECTION_SEAL_SCHEMA = "f007-known-query-arm-selection-v1"
F007_POLICY_SEAL_SCHEMA = "f007-role-safe-open-set-policy-v1"

ARM_IDS = ("control_f005", "l2sp_001", "l2sp_01")
POLICY_COMPARATORS = ("frozen_same_protocol", "reused_control", "selected_arm")
_POLICY_FIXED_SOURCES = {
    "frozen_same_protocol": "c002b_frozen",
    "reused_control": "control_f005",
}
_SELECTION_ORDER = (
    "macro_f1_observed_known_labels",
    "top1_accuracy",
    "fixed_arm_tie_order",
)
_STATE_FIELDS = frozenset({
    "schema_version", "experiment_signature", "contract_sha256",
    "source_f005_receipt_sha256", "output_directory", "status", "phase",
    "tail_units", "known_scores", "arm_seals", "full_scoring",
    "policy_seals", "parity", "outer_truth_materialized",
    "outer_truth_access", "outer_evaluations", "aggregate", "tracking",
    "resume_supported", "server_only_artifacts",
    "mlflow_forbidden_payloads_uploaded",
})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PERMIT_SENTINEL = object()


def canonical(value: object) -> bytes:
    """Return the canonical finite-JSON representation used by F007 receipts."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value),
        f"F007 {label} must be a lowercase SHA-256",
    )
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    _require(type(value) is int and (minimum is None or value >= minimum),
             f"F007 {label} must be an integer")
    return value


def _finite_probability(value: object, label: str) -> float:
    _require(type(value) in (int, float) and not isinstance(value, bool),
             f"F007 {label} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0,
             f"F007 {label} must be a finite probability")
    return result


def _relative_receipt_path(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"F007 {label} path is missing")
    parsed = PurePosixPath(value)
    _require(
        not parsed.is_absolute() and parsed.as_posix() == value
        and ".." not in parsed.parts and "\\" not in value and ":" not in value,
        f"F007 {label} path must be a safe relative POSIX path",
    )
    return value


def _clone_json(value: object) -> Any:
    """Copy only canonical JSON so callers cannot mutate a returned contract."""
    return json.loads(canonical(value).decode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"F007 {label} must be a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F007 {label} must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), f"F007 {label} must contain a JSON object")
    return value, payload


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Write one receipt exactly once; a pre-existing path is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F007 refuses to replace sealed evidence: {path}")
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
            raise FileExistsError(f"F007 refuses to replace sealed evidence: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _confined_path(output_directory: Path, path: Path, label: str) -> Path:
    """Resolve an existing receipt only when it remains inside the run output."""
    output = Path(output_directory).resolve()
    candidate = Path(path).resolve(strict=True)
    _require(candidate.is_relative_to(output) and candidate != output,
             f"F007 {label} must stay below the configured output directory")
    current = candidate.parent
    while current != output:
        _require(not current.is_symlink(), f"F007 {label} may not cross a symlink")
        current = current.parent
    return candidate


def validate_f007_config(config: dict[str, Any]) -> None:
    """Accept precisely the preregistered F007 L2-SP protocol.

    F007 has a small fixed GPU budget.  A changed lambda, arm, scoring role,
    retention rule, or promotion gate is a distinct experiment rather than a
    silent mutation of the one which may later be compared against C002b.
    """
    _require(isinstance(config, dict), "F007 configuration must be a JSON object")
    try:
        digest = canonical_sha256(config)
    except (TypeError, ValueError) as error:
        raise ValueError("F007 configuration must contain finite canonical JSON") from error
    _require(digest == F007_CONFIG_SHA256,
             "F007 requires its exact preregistered configuration")

    _require(
        config.get("schema_version") == 1
        and config.get("experiment_code") == "F007"
        and config.get("device") == "cuda"
        and config.get("cpu_threads") == 4
        and config.get("fold_ids") == [0, 1],
        "F007 execution identity changed",
    )
    source = config.get("source_f005")
    _require(
        isinstance(source, dict)
        and source.get("reuse_control") is True
        and source.get("reuse_shared_head") is True
        and source.get("required_selected_arm_by_outer_fold") == {"0": "control", "1": "control"}
        and _sha256(source.get("config_sha256"), "source F005 config hash")
        and isinstance(source.get("parent_run_id"), str) and source["parent_run_id"],
        "F007 must reuse the authenticated F005 control and shared heads",
    )
    _require(config.get("execution") == {
        "gpu_name_contains": "RTX 3090", "tensor_dtype": "float32",
        "no_cpu_fallback": True, "deterministic_algorithms": "enforce_error",
        "cublas_workspace_config": ":4096:8",
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
    }, "F007 deterministic CUDA execution policy changed")
    _require(
        [(row.get("id"), row.get("kind"), row.get("lambda"))
         for row in config.get("arms", [])]
        == [
            ("control_f005", "reused_f005_control", 0.0),
            ("l2sp_001", "l2sp_anchored_tail", 0.01),
            ("l2sp_01", "l2sp_anchored_tail", 0.1),
        ], "F007 arm identity/order changed")
    _require(config.get("l2sp") == {
        "formula": "lambda_times_half_sum_squared_distance",
        "anchor_source": "authenticated_f005_shared_head_encoder_before_tail",
        "anchor_scope": "trainable_encoder_parameters_excluding_batchnorm_affine",
        "aam_head_included": False, "batchnorm_affine_in_l2sp": False,
        "batchnorm_affine_remains_task_trainable": True,
        "probe": {
            "kind": "virtual_task_step_then_gradient_ratio",
            "task_only_virtual_step": 600, "measurement_step": 601,
            "optimizer_steps_persisted": 0, "selects_lambda": False,
            "expected_lambda_ratio": 10.0,
        },
    }, "F007 L2-SP definition changed")
    _require(config.get("training") == {
        "reuse_f005_fit_rows": True, "reuse_f005_nested_crop_plan": True,
        "reuse_f005_tail_steps": 500, "reuse_f005_optimizer_and_schedule": True,
        "dynamic_consistency": False, "raw_h_mse": False,
        "tail_checkpoint_every_steps": 100, "new_gpu_tail_jobs": 4,
    }, "F007 F005-tail reuse policy changed")
    _require(config.get("selection") == {
        "known_query_scope": "original_group_disjoint_calibration_query_rows_only",
        "metric": "known_query_macro_f1_over_observed_labels_then_top1_accuracy",
        "arm_tie_order": list(ARM_IDS),
        "unknown_calibration_hidden_until_arm_sealed": True,
        "outer_labels_forbidden_until_all_policies_sealed": True,
        "refit_after_selection": False,
    }, "F007 known-only arm-selection policy changed")
    _require(config.get("scoring") == {
        "protocol": "c002b_family_disjoint_roles_v1",
        "reference_method": "max_reference",
        "fusion": "same_reference_sqrt_weighted_encoder_concatenation",
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
        "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
        "margin_weights": [0.0, 0.5], "threshold_candidates": 201,
        "probability_temperature": 0.05,
        "historical_c002b_fixed_policy": "diagnostic_only_never_selectable",
    }, "F007 role-safe scoring policy changed")
    promotion = config.get("promotion")
    _require(
        isinstance(promotion, dict)
        and promotion.get("goal_target_oof_macro_f1") == 0.965
        and promotion.get("minimum_selected_delta_vs_reused_control") == 0.003
        and promotion.get("minimum_selected_delta_vs_c002b") == 0.003
        and promotion.get("require_exact_cpu_cuda_prediction_parity") is True
        and promotion.get("all_conditions_required") is True
        and promotion.get("otherwise") == "retain_c002b",
        "F007 promotion guards changed",
    )
    _require(config.get("mlflow") == {
        "experiment_id": "1",
        "upload": "configs_source_snapshot_receipts_seals_metrics_reports_only_no_audio_embeddings_weights_optimizer_or_credentials",
    }, "F007 MLflow boundary changed")


def _validate_source_file_entry(value: object, label: str) -> dict[str, str]:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"},
             f"F007 F005 source {label} receipt shape changed")
    return {
        "path": _relative_receipt_path(value["path"], f"F005 source {label}"),
        "sha256": _sha256(value["sha256"], f"F005 source {label} hash"),
    }


def validate_f005_source_receipt(receipt: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Validate the compact source receipt returned by ``f007_source``.

    This intentionally does not reopen F005 checkpoints/caches.  Those bytes
    were already verified by :func:`speaker_id.training.f007_source.load_f005_source_receipt`.
    The F007 contract only freezes the returned receipt and its F005 parent
    identity before any new tail can begin.
    """
    from speaker_id.training.f007_source import F007_F005_SOURCE_RECEIPT_SCHEMA

    _require(isinstance(receipt, dict), "F007 F005 source receipt must be a JSON object")
    required = {"schema_version", "source_run_directory", "experiment_state", "fold_ids", "folds"}
    _require(set(receipt) == required, "F007 F005 source receipt schema changed")
    source = config["source_f005"]
    _require(
        receipt["schema_version"] == F007_F005_SOURCE_RECEIPT_SCHEMA
        and isinstance(receipt["source_run_directory"], str)
        and receipt["source_run_directory"] == source["run_dir"],
        "F007 F005 source receipt is from a different run directory",
    )
    state = receipt["experiment_state"]
    _require(
        isinstance(state, dict)
        and set(state) == {"path", "sha256", "status", "experiment_signature"}
        and _relative_receipt_path(state["path"], "F005 experiment state") == "experiment_state.json"
        and _sha256(state["sha256"], "F005 experiment state hash")
        and state["status"] == "complete"
        and _sha256(state["experiment_signature"], "F005 experiment signature"),
        "F007 F005 source state receipt is invalid",
    )
    configured_folds = config["fold_ids"]
    _require(receipt["fold_ids"] == configured_folds and isinstance(receipt["folds"], list)
             and len(receipt["folds"]) == len(configured_folds),
             "F007 F005 source fold identity changed")
    result_folds: list[dict[str, Any]] = []
    for expected_outer, entry in zip(configured_folds, receipt["folds"], strict=True):
        _require(isinstance(entry, dict) and set(entry) == {
            "outer_fold", "shared_head", "control_tail", "full_scoring_control",
        } and entry["outer_fold"] == expected_outer,
                 "F007 F005 source fold receipt is malformed or reordered")
        shared = entry["shared_head"]
        _require(isinstance(shared, dict) and set(shared) == {"checkpoint", "unit_report"},
                 "F007 F005 shared-head receipt changed")
        shared_checkpoint = _validate_source_file_entry(shared["checkpoint"], "shared-head checkpoint")
        shared_report = _validate_source_file_entry(shared["unit_report"], "shared-head report")
        control = entry["control_tail"]
        _require(isinstance(control, dict) and set(control) == {
            "checkpoint", "unit_report", "shared_head_checkpoint_sha256",
        }, "F007 F005 control-tail receipt changed")
        control_checkpoint = _validate_source_file_entry(control["checkpoint"], "control-tail checkpoint")
        control_report = _validate_source_file_entry(control["unit_report"], "control-tail report")
        _require(
            _sha256(control["shared_head_checkpoint_sha256"], "control-tail shared-head hash")
            == shared_checkpoint["sha256"],
            "F007 F005 control tail is not bound to its shared head",
        )
        full = entry["full_scoring_control"]
        _require(isinstance(full, dict) and set(full) == {"identity", "receipt"},
                 "F007 F005 full-scoring control receipt changed")
        identity = full["identity"]
        _require(isinstance(identity, dict) and set(identity) == {"path", "sha256", "signature"},
                 "F007 F005 control cache identity receipt changed")
        identity_normalized = {
            **_validate_source_file_entry(
                {"path": identity["path"], "sha256": identity["sha256"]},
                "control cache identity",
            ),
            "signature": _sha256(identity["signature"], "control cache identity signature"),
        }
        cache_receipt = full["receipt"]
        _require(isinstance(cache_receipt, dict) and set(cache_receipt) == {
            "path", "sha256", "receipt_sha256", "file_count",
        }, "F007 F005 control cache receipt changed")
        cache_normalized = {
            **_validate_source_file_entry(
                {"path": cache_receipt["path"], "sha256": cache_receipt["sha256"]},
                "control cache receipt",
            ),
            "receipt_sha256": _sha256(cache_receipt["receipt_sha256"], "control cache receipt self hash"),
            "file_count": _integer(cache_receipt["file_count"], "control cache file count", minimum=1),
        }
        result_folds.append({
            "outer_fold": expected_outer,
            "shared_head": {"checkpoint": shared_checkpoint, "unit_report": shared_report},
            "control_tail": {
                "checkpoint": control_checkpoint, "unit_report": control_report,
                "shared_head_checkpoint_sha256": shared_checkpoint["sha256"],
            },
            "full_scoring_control": {"identity": identity_normalized, "receipt": cache_normalized},
        })
    return {
        "schema_version": receipt["schema_version"],
        "source_run_directory": receipt["source_run_directory"],
        "experiment_state": {
            "path": "experiment_state.json", "sha256": state["sha256"],
            "status": "complete", "experiment_signature": state["experiment_signature"],
        },
        "fold_ids": list(configured_folds), "folds": result_folds,
    }


def build_f007_contract(config: dict[str, Any], f005_source_receipt: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact F007 protocol to an authenticated, completed F005 run."""
    validate_f007_config(config)
    source = validate_f005_source_receipt(f005_source_receipt, config)
    config_copy = _clone_json(config)
    body = {
        "schema_version": F007_CONTRACT_SCHEMA,
        "config": config_copy,
        "config_sha256": canonical_sha256(config_copy),
        "source_f005_receipt": source,
        "source_f005_receipt_sha256": canonical_sha256(source),
        "fold_ids": list(config_copy["fold_ids"]),
        "arm_ids": list(ARM_IDS),
        "policy_comparators": list(POLICY_COMPARATORS),
        "selection_tie_order": list(config_copy["selection"]["arm_tie_order"]),
    }
    return {**body, "signature": canonical_sha256(body)}


def validate_f007_contract(contract: dict[str, Any]) -> None:
    """Reject a contract whose configuration, source receipt, or signature drifted."""
    _require(isinstance(contract, dict), "F007 contract must be a JSON object")
    required = {
        "schema_version", "config", "config_sha256", "source_f005_receipt",
        "source_f005_receipt_sha256", "fold_ids", "arm_ids", "policy_comparators",
        "selection_tie_order", "signature",
    }
    _require(set(contract) == required, "F007 contract schema changed")
    validate_f007_config(contract["config"])
    _require(
        contract["schema_version"] == F007_CONTRACT_SCHEMA
        and contract["config_sha256"] == canonical_sha256(contract["config"])
        and contract["fold_ids"] == contract["config"]["fold_ids"]
        and contract["arm_ids"] == list(ARM_IDS)
        and contract["policy_comparators"] == list(POLICY_COMPARATORS)
        and contract["selection_tie_order"] == list(ARM_IDS)
        and contract["source_f005_receipt_sha256"]
        == canonical_sha256(contract["source_f005_receipt"]),
        "F007 contract identity changed",
    )
    normalized_source = validate_f005_source_receipt(
        contract["source_f005_receipt"], contract["config"],
    )
    _require(normalized_source == contract["source_f005_receipt"],
             "F007 contract source receipt is not canonical")
    body = {key: value for key, value in contract.items() if key != "signature"}
    _require(contract["signature"] == canonical_sha256(body),
             "F007 contract signature changed")


def _contract_sha256(contract: dict[str, Any]) -> str:
    validate_f007_contract(contract)
    return canonical_sha256(contract)


def fresh_f007_state(contract: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    """Create an empty F007 state with no authority to read outer truth."""
    validate_f007_contract(contract)
    output = Path(output_directory).resolve()
    _require(output.name != "", "F007 output directory is invalid")
    return {
        "schema_version": F007_STATE_SCHEMA,
        "experiment_signature": contract["signature"],
        "contract_sha256": _contract_sha256(contract),
        "source_f005_receipt_sha256": contract["source_f005_receipt_sha256"],
        "output_directory": str(output),
        "status": "initialized", "phase": "initialized",
        "tail_units": {}, "known_scores": {}, "arm_seals": {},
        "full_scoring": {}, "policy_seals": {}, "parity": None,
        "outer_truth_materialized": False, "outer_truth_access": None,
        "outer_evaluations": {}, "aggregate": None,
        "tracking": {"parent": None, "children": {}},
        "resume_supported": True,
        "server_only_artifacts": {
            "checkpoints": True, "embedding_caches": True,
            "optimizer_state": True,
        },
        "mlflow_forbidden_payloads_uploaded": False,
    }


def _forbid_pretruth_payloads(value: object) -> None:
    """Reject accidental outer-label materialization before the gate opens."""
    forbidden = {"outer_rows", "outer_labels", "outer_truth_rows", "outer_truth_labels"}
    if isinstance(value, dict):
        for key, item in value.items():
            _require(str(key).lower() not in forbidden,
                     "F007 state contains an outer-truth payload before policy sealing")
            _forbid_pretruth_payloads(item)
    elif isinstance(value, list):
        for item in value:
            _forbid_pretruth_payloads(item)


def _state_arm_record(record: object, contract: dict[str, Any], output: Path,
                      outer: int) -> dict[str, Any]:
    _require(isinstance(record, dict) and set(record) == {
        "path", "file_sha256", "seal_sha256", "selected_arm", "disk_reloaded",
    }, "F007 state arm-seal record schema changed")
    path = _confined_path(output, Path(record["path"]), "arm-selection seal")
    reloaded = reload_arm_selection_seal(path, contract, outer)
    expected = {
        "path": str(path), "file_sha256": reloaded["file_sha256"],
        "seal_sha256": reloaded["seal_sha256"],
        "selected_arm": reloaded["seal"]["selected_arm"], "disk_reloaded": True,
    }
    _require(record == expected, "F007 state arm-seal record no longer matches disk")
    return expected


def _state_arm_reload(state: dict[str, Any], contract: dict[str, Any], output: Path,
                      outer: int) -> dict[str, Any]:
    """Recover the full reload object from its compact, authenticated state row."""
    record = _state_arm_record(state["arm_seals"].get(str(outer)), contract, output, outer)
    reloaded = reload_arm_selection_seal(Path(record["path"]), contract, outer)
    _require(
        reloaded["file_sha256"] == record["file_sha256"]
        and reloaded["seal_sha256"] == record["seal_sha256"]
        and reloaded["seal"]["selected_arm"] == record["selected_arm"],
        "F007 state arm-seal reload differs from its compact record",
    )
    return reloaded


def _state_policy_record(state: dict[str, Any], record: object, contract: dict[str, Any],
                         output: Path, outer: int) -> dict[str, Any]:
    _require(isinstance(record, dict) and set(record) == {
        "path", "file_sha256", "seal_sha256", "selected_arm",
        "arm_selection_seal_sha256", "disk_reloaded",
    }, "F007 state policy-seal record schema changed")
    arm = _state_arm_reload(state, contract, output, outer)
    path = _confined_path(output, Path(record["path"]), "policy seal")
    reloaded = reload_policy_seal(path, contract, outer, arm)
    expected = {
        "path": str(path), "file_sha256": reloaded["file_sha256"],
        "seal_sha256": reloaded["seal_sha256"],
        "selected_arm": reloaded["seal"]["selected_arm"],
        "arm_selection_seal_sha256": reloaded["seal"]["arm_selection_seal_sha256"],
        "disk_reloaded": True,
    }
    _require(record == expected, "F007 state policy-seal record no longer matches disk")
    return expected


def _policy_digest(state: dict[str, Any], contract: dict[str, Any], output: Path) -> str:
    records = {}
    for outer in contract["fold_ids"]:
        record = _state_policy_record(
            state, state["policy_seals"].get(str(outer)), contract, output, outer,
        )
        records[str(outer)] = {
            "arm_selection_seal_sha256": record["arm_selection_seal_sha256"],
            "policy_seal_sha256": record["seal_sha256"],
        }
    return canonical_sha256(records)


def validate_f007_state(state: dict[str, Any], contract: dict[str, Any],
                        output_directory: Path, *, verify_seals: bool = True) -> None:
    """Validate state identity and enforce the policy-before-outer-truth invariant."""
    validate_f007_contract(contract)
    _require(isinstance(state, dict) and set(state) == _STATE_FIELDS,
             "F007 state schema changed")
    output = Path(output_directory).resolve()
    _require(
        state["schema_version"] == F007_STATE_SCHEMA
        and state["experiment_signature"] == contract["signature"]
        and state["contract_sha256"] == _contract_sha256(contract)
        and state["source_f005_receipt_sha256"] == contract["source_f005_receipt_sha256"]
        and Path(state["output_directory"]).resolve() == output,
        "F007 state belongs to a different experiment or output directory",
    )
    _require(state["status"] in {"initialized", "running", "complete", "failed"}
             and isinstance(state["phase"], str) and state["phase"],
             "F007 state status/phase is invalid")
    _require(
        all(isinstance(state[name], dict) for name in (
            "tail_units", "known_scores", "arm_seals", "full_scoring",
            "policy_seals", "outer_evaluations",
        ))
        and isinstance(state["tracking"], dict)
        and state["tracking"].get("children") is not None
        and isinstance(state["tracking"].get("children"), dict)
        and type(state["resume_supported"]) is bool
        and state["resume_supported"] is True
        and state["server_only_artifacts"] == {
            "checkpoints": True, "embedding_caches": True, "optimizer_state": True,
        }
        and state["mlflow_forbidden_payloads_uploaded"] is False,
        "F007 state retention/tracking boundary changed",
    )
    _require(type(state["outer_truth_materialized"]) is bool,
             "F007 outer-truth materialization flag is invalid")
    configured_keys = {str(outer) for outer in contract["fold_ids"]}
    _require(set(state["arm_seals"]).issubset(configured_keys)
             and set(state["policy_seals"]).issubset(configured_keys),
             "F007 state contains an unconfigured outer fold")
    if not state["outer_truth_materialized"]:
        _require(state["outer_truth_access"] is None and not state["outer_evaluations"],
                 "F007 outer evaluation exists before outer truth was unlocked")
        _forbid_pretruth_payloads({
            key: value for key, value in state.items()
            if key not in {"outer_truth_materialized", "outer_truth_access", "outer_evaluations"}
        })
    else:
        _require(isinstance(state["outer_truth_access"], dict) and set(state["outer_truth_access"]) == {
            "schema_version", "policy_digest", "fold_ids",
        } and state["outer_truth_access"]["schema_version"] == 1
                 and state["outer_truth_access"]["fold_ids"] == contract["fold_ids"]
                 and _sha256(state["outer_truth_access"]["policy_digest"], "outer-truth policy digest"),
                 "F007 outer-truth access record is malformed")
        _require(set(state["policy_seals"]) == configured_keys,
                 "F007 outer truth was unlocked before every fold policy was sealed")
    if verify_seals:
        for outer in contract["fold_ids"]:
            if str(outer) in state["arm_seals"]:
                _state_arm_record(state["arm_seals"][str(outer)], contract, output, outer)
            if str(outer) in state["policy_seals"]:
                _state_policy_record(state, state["policy_seals"][str(outer)], contract, output, outer)
        if state["outer_truth_materialized"]:
            _require(state["outer_truth_access"]["policy_digest"]
                     == _policy_digest(state, contract, output),
                     "F007 outer-truth access is not bound to current policy seals")


def write_f007_state(path: Path, state: dict[str, Any], contract: dict[str, Any],
                     output_directory: Path) -> None:
    """Atomically replace the mutable state after validating its immutable bindings."""
    validate_f007_state(state, contract, output_directory)
    path = Path(path)
    output = Path(output_directory).resolve()
    candidate = path.resolve() if path.exists() else path.absolute()
    _require(candidate.is_relative_to(output),
             "F007 experiment state must stay in the configured output directory")
    _require(not path.is_symlink(), "F007 experiment state may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_f007_state(path: Path, contract: dict[str, Any], output_directory: Path,
                    *, verify_seals: bool = True) -> dict[str, Any]:
    value, _ = _regular_json(path, "experiment state")
    validate_f007_state(value, contract, output_directory, verify_seals=verify_seals)
    return value


def select_f007_arm(arm_metrics: Mapping[str, Mapping[str, object]],
                    contract: dict[str, Any]) -> tuple[str, dict[str, dict[str, float]]]:
    """Recompute the only permitted known-query selection decision."""
    validate_f007_contract(contract)
    _require(isinstance(arm_metrics, Mapping) and set(arm_metrics) == set(ARM_IDS),
             "F007 arm metrics must contain exactly the preregistered arms")
    normalized: dict[str, dict[str, float]] = {}
    for arm in ARM_IDS:
        metric = arm_metrics[arm]
        _require(isinstance(metric, Mapping) and set(metric) == {
            "macro_f1_observed_known_labels", "top1_accuracy",
        }, "F007 arm metrics schema changed")
        normalized[arm] = {
            "macro_f1_observed_known_labels": _finite_probability(
                metric["macro_f1_observed_known_labels"], f"{arm} known macro-F1"),
            "top1_accuracy": _finite_probability(metric["top1_accuracy"], f"{arm} known top-1"),
        }
    tie_order = tuple(contract["selection_tie_order"])
    winner = max(
        tie_order,
        key=lambda arm: (
            normalized[arm]["macro_f1_observed_known_labels"],
            normalized[arm]["top1_accuracy"], -tie_order.index(arm),
        ),
    )
    return winner, normalized


def _known_score_receipts(value: object) -> dict[str, dict[str, object]]:
    _require(isinstance(value, Mapping) and set(value) == set(ARM_IDS),
             "F007 known-score receipts must contain exactly the preregistered arms")
    result: dict[str, dict[str, object]] = {}
    for arm in ARM_IDS:
        entry = value[arm]
        _require(isinstance(entry, Mapping) and set(entry) == {"sha256", "rows"},
                 "F007 known-score receipt schema changed")
        result[arm] = {
            "sha256": _sha256(entry["sha256"], f"{arm} known-score receipt"),
            "rows": _integer(entry["rows"], f"{arm} known-score rows", minimum=1),
        }
    return result


def arm_selection_seal(contract: dict[str, Any], outer: int,
                       arm_metrics: Mapping[str, Mapping[str, object]],
                       known_score_receipts: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    """Build an immutable known-only decision with no unknown/outer access."""
    validate_f007_contract(contract)
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 outer fold is not configured")
    selected_arm, metrics = select_f007_arm(arm_metrics, contract)
    receipts = _known_score_receipts(known_score_receipts)
    body = {
        "schema_version": F007_ARM_SELECTION_SEAL_SCHEMA,
        "experiment_signature": contract["signature"], "outer_fold": outer,
        "arm_ids": list(ARM_IDS), "selection_order": list(_SELECTION_ORDER),
        "selection_config_sha256": canonical_sha256(contract["config"]["selection"]),
        "known_score_receipts": receipts, "arm_metrics": metrics,
        "selected_arm": selected_arm,
        "scientific_conclusion": (
            "reused_control_selected_on_known_queries"
            if selected_arm == "control_f005" else "l2sp_candidate_selected_on_known_queries"
        ),
        "unknown_calibration_materialized": False,
        "unknown_similarity_computed": False,
        "outer_rows_or_labels_read": False,
        "refit_after_selection": False,
    }
    return {**body, "seal_sha256": canonical_sha256(body)}


def _validate_arm_seal(seal: dict[str, Any], contract: dict[str, Any], outer: int) -> None:
    required = {
        "schema_version", "experiment_signature", "outer_fold", "arm_ids",
        "selection_order", "selection_config_sha256", "known_score_receipts",
        "arm_metrics", "selected_arm", "scientific_conclusion",
        "unknown_calibration_materialized", "unknown_similarity_computed",
        "outer_rows_or_labels_read", "refit_after_selection", "seal_sha256",
    }
    _require(set(seal) == required, "F007 arm-selection seal schema changed")
    _require(
        seal["schema_version"] == F007_ARM_SELECTION_SEAL_SCHEMA
        and seal["experiment_signature"] == contract["signature"]
        and seal["outer_fold"] == outer
        and seal["arm_ids"] == list(ARM_IDS)
        and seal["selection_order"] == list(_SELECTION_ORDER)
        and seal["selection_config_sha256"] == canonical_sha256(contract["config"]["selection"])
        and seal["unknown_calibration_materialized"] is False
        and seal["unknown_similarity_computed"] is False
        and seal["outer_rows_or_labels_read"] is False
        and seal["refit_after_selection"] is False,
        "F007 arm-selection seal identity or anti-leak claim is invalid",
    )
    selected, metrics = select_f007_arm(seal["arm_metrics"], contract)
    _require(_known_score_receipts(seal["known_score_receipts"])
             == seal["known_score_receipts"], "F007 arm-selection receipt is not canonical")
    expected_conclusion = (
        "reused_control_selected_on_known_queries"
        if selected == "control_f005" else "l2sp_candidate_selected_on_known_queries"
    )
    _require(
        seal["selected_arm"] == selected
        and seal["scientific_conclusion"] == expected_conclusion
        and metrics == seal["arm_metrics"],
        "F007 sealed arm is not the recomputed known-query winner",
    )
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    _require(seal["seal_sha256"] == canonical_sha256(body),
             "F007 arm-selection seal hash changed")


def write_arm_selection_seal(path: Path, contract: dict[str, Any], outer: int,
                             arm_metrics: Mapping[str, Mapping[str, object]],
                             known_score_receipts: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    """Write then immediately authenticate a one-shot F007 arm-selection seal."""
    seal = arm_selection_seal(contract, outer, arm_metrics, known_score_receipts)
    _write_new_json(path, seal)
    reloaded = reload_arm_selection_seal(path, contract, outer)
    _require(reloaded["seal"] == seal, "F007 arm-selection seal changed during write/reload")
    return reloaded


def reload_arm_selection_seal(path: Path, contract: dict[str, Any], outer: int) -> dict[str, Any]:
    """Disk-reload and recompute a F007 known-only arm decision."""
    validate_f007_contract(contract)
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 outer fold is not configured")
    seal, payload = _regular_json(path, "arm-selection seal")
    _validate_arm_seal(seal, contract, outer)
    return {
        "kind": "f007_arm_selection_disk_reload", "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(), "seal_sha256": seal["seal_sha256"],
        "disk_reloaded": True, "seal": seal,
    }


def _verify_arm_reload(reload: Mapping[str, Any], contract: dict[str, Any], outer: int) -> dict[str, Any]:
    _require(isinstance(reload, Mapping) and reload.get("kind") == "f007_arm_selection_disk_reload"
             and reload.get("disk_reloaded") is True and isinstance(reload.get("path"), str),
             "F007 policy sealing requires a disk-reloaded arm-selection seal")
    current = reload_arm_selection_seal(Path(reload["path"]), contract, outer)
    _require(
        current["file_sha256"] == reload.get("file_sha256")
        and current["seal_sha256"] == reload.get("seal_sha256")
        and current["seal"] == reload.get("seal"),
        "F007 arm-selection reload changed before policy sealing",
    )
    return current


def _normalise_policies(value: object, contract: dict[str, Any], selected_arm: str) -> dict[str, dict[str, object]]:
    _require(isinstance(value, Mapping) and set(value) == set(POLICY_COMPARATORS),
             "F007 policy seal must contain every fixed comparator")
    scoring = contract["config"]["scoring"]
    result: dict[str, dict[str, object]] = {}
    for comparator in POLICY_COMPARATORS:
        entry = value[comparator]
        _require(isinstance(entry, Mapping) and set(entry) == {
            "model_source", "alpha", "unknown_weight", "margin_weight",
            "threshold", "calibration_receipt_sha256",
        }, "F007 comparator policy schema changed")
        expected_source = _POLICY_FIXED_SOURCES.get(comparator, selected_arm)
        _require(entry["model_source"] == expected_source,
                 "F007 comparator policy is bound to the wrong model source")
        alpha = _finite_probability(entry["alpha"], f"{comparator} alpha")
        unknown_weight = _finite_probability(entry["unknown_weight"], f"{comparator} unknown weight")
        margin_weight = _finite_probability(entry["margin_weight"], f"{comparator} margin weight")
        threshold = _finite_probability(entry["threshold"], f"{comparator} threshold")
        _require(alpha in scoring["alphas"] and unknown_weight in scoring["unknown_weights"]
                 and margin_weight in scoring["margin_weights"],
                 "F007 comparator policy is outside the preregistered scoring grid")
        result[comparator] = {
            "model_source": expected_source, "alpha": alpha,
            "unknown_weight": unknown_weight, "margin_weight": margin_weight,
            "threshold": threshold,
            "calibration_receipt_sha256": _sha256(
                entry["calibration_receipt_sha256"], f"{comparator} calibration receipt"),
        }
    return result


def policy_seal(contract: dict[str, Any], outer: int,
                arm_selection_reload: Mapping[str, Any],
                policies: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    """Build a role-safe policy receipt bound to a reloaded arm-selection seal."""
    validate_f007_contract(contract)
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 outer fold is not configured")
    arm = _verify_arm_reload(arm_selection_reload, contract, outer)
    selected = arm["seal"]["selected_arm"]
    normalized = _normalise_policies(policies, contract, selected)
    body = {
        "schema_version": F007_POLICY_SEAL_SCHEMA,
        "experiment_signature": contract["signature"], "outer_fold": outer,
        "arm_selection_seal_sha256": arm["seal_sha256"],
        "arm_selection_file_sha256": arm["file_sha256"], "selected_arm": selected,
        "policy_comparators": list(POLICY_COMPARATORS),
        "scoring_config_sha256": canonical_sha256(contract["config"]["scoring"]),
        "policies": normalized,
        "calibration_scope": "independent_group_disjoint_known_and_unknown_queries_only",
        "historical_c002b_fixed_policy": "diagnostic_only_never_selectable",
        "outer_rows_or_labels_read": False,
    }
    return {**body, "seal_sha256": canonical_sha256(body)}


def _validate_policy_seal(seal: dict[str, Any], contract: dict[str, Any], outer: int,
                          arm: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "experiment_signature", "outer_fold",
        "arm_selection_seal_sha256", "arm_selection_file_sha256", "selected_arm",
        "policy_comparators", "scoring_config_sha256", "policies",
        "calibration_scope", "historical_c002b_fixed_policy",
        "outer_rows_or_labels_read", "seal_sha256",
    }
    _require(set(seal) == required, "F007 policy seal schema changed")
    _require(
        seal["schema_version"] == F007_POLICY_SEAL_SCHEMA
        and seal["experiment_signature"] == contract["signature"]
        and seal["outer_fold"] == outer
        and seal["arm_selection_seal_sha256"] == arm["seal_sha256"]
        and seal["arm_selection_file_sha256"] == arm["file_sha256"]
        and seal["selected_arm"] == arm["seal"]["selected_arm"]
        and seal["policy_comparators"] == list(POLICY_COMPARATORS)
        and seal["scoring_config_sha256"] == canonical_sha256(contract["config"]["scoring"])
        and seal["calibration_scope"] == "independent_group_disjoint_known_and_unknown_queries_only"
        and seal["historical_c002b_fixed_policy"] == "diagnostic_only_never_selectable"
        and seal["outer_rows_or_labels_read"] is False,
        "F007 policy seal identity or anti-leak claim is invalid",
    )
    normalized = _normalise_policies(seal["policies"], contract, seal["selected_arm"])
    _require(normalized == seal["policies"], "F007 policy seal is not canonical")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    _require(seal["seal_sha256"] == canonical_sha256(body),
             "F007 policy seal hash changed")


def write_policy_seal(path: Path, contract: dict[str, Any], outer: int,
                      arm_selection_reload: Mapping[str, Any],
                      policies: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    """Write then disk-reload a one-shot role-safe F007 policy seal."""
    seal = policy_seal(contract, outer, arm_selection_reload, policies)
    _write_new_json(path, seal)
    reloaded = reload_policy_seal(path, contract, outer, arm_selection_reload)
    _require(reloaded["seal"] == seal, "F007 policy seal changed during write/reload")
    return reloaded


def reload_policy_seal(path: Path, contract: dict[str, Any], outer: int,
                       arm_selection_reload: Mapping[str, Any]) -> dict[str, Any]:
    """Disk-reload a policy only after authenticating its arm-selection parent."""
    validate_f007_contract(contract)
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 outer fold is not configured")
    arm = _verify_arm_reload(arm_selection_reload, contract, outer)
    seal, payload = _regular_json(path, "policy seal")
    _validate_policy_seal(seal, contract, outer, arm)
    return {
        "kind": "f007_policy_disk_reload", "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(), "seal_sha256": seal["seal_sha256"],
        "disk_reloaded": True, "seal": seal,
    }


def register_arm_selection_seal(state: dict[str, Any], contract: dict[str, Any],
                                output_directory: Path,
                                arm_selection_reload: Mapping[str, Any]) -> dict[str, Any]:
    """Record a disk-reloaded arm seal without allowing replacement or mutation."""
    validate_f007_state(state, contract, output_directory, verify_seals=True)
    _require(state["outer_truth_materialized"] is False,
             "F007 cannot change an arm seal after outer truth was unlocked")
    outer = arm_selection_reload.get("seal", {}).get("outer_fold") if isinstance(arm_selection_reload, Mapping) else None
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 arm-selection reload has no configured outer fold")
    reloaded = _verify_arm_reload(arm_selection_reload, contract, outer)
    path = _confined_path(output_directory, Path(reloaded["path"]), "arm-selection seal")
    record = {
        "path": str(path), "file_sha256": reloaded["file_sha256"],
        "seal_sha256": reloaded["seal_sha256"],
        "selected_arm": reloaded["seal"]["selected_arm"], "disk_reloaded": True,
    }
    existing = state["arm_seals"].get(str(outer))
    _require(existing is None or existing == record,
             "F007 refuses to replace a registered arm-selection seal")
    state["arm_seals"][str(outer)] = record
    return record


def register_policy_seal(state: dict[str, Any], contract: dict[str, Any],
                         output_directory: Path,
                         policy_reload: Mapping[str, Any]) -> dict[str, Any]:
    """Record a policy seal only if its reloaded arm parent is already registered."""
    validate_f007_state(state, contract, output_directory, verify_seals=True)
    _require(state["outer_truth_materialized"] is False,
             "F007 cannot change a policy seal after outer truth was unlocked")
    outer = policy_reload.get("seal", {}).get("outer_fold") if isinstance(policy_reload, Mapping) else None
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 policy reload has no configured outer fold")
    arm_reload = _state_arm_reload(state, contract, Path(output_directory).resolve(), outer)
    reloaded = reload_policy_seal(
        Path(policy_reload.get("path", "")), contract, outer, arm_reload,
    )
    _require(
        policy_reload.get("kind") == "f007_policy_disk_reload"
        and policy_reload.get("disk_reloaded") is True
        and policy_reload.get("file_sha256") == reloaded["file_sha256"]
        and policy_reload.get("seal_sha256") == reloaded["seal_sha256"]
        and policy_reload.get("seal") == reloaded["seal"],
        "F007 policy reload changed before state registration",
    )
    path = _confined_path(output_directory, Path(reloaded["path"]), "policy seal")
    record = {
        "path": str(path), "file_sha256": reloaded["file_sha256"],
        "seal_sha256": reloaded["seal_sha256"],
        "selected_arm": reloaded["seal"]["selected_arm"],
        "arm_selection_seal_sha256": reloaded["seal"]["arm_selection_seal_sha256"],
        "disk_reloaded": True,
    }
    existing = state["policy_seals"].get(str(outer))
    _require(existing is None or existing == record,
             "F007 refuses to replace a registered policy seal")
    state["policy_seals"][str(outer)] = record
    return record


@dataclass(frozen=True)
class OuterTruthPermit:
    """Opaque capability issued only after every F007 policy seal is authenticated."""

    experiment_signature: str
    policy_digest: str
    fold_ids: tuple[int, ...]
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _PERMIT_SENTINEL:
            raise TypeError("F007 outer-truth permits may only be issued by unlock_outer_truth")


def unlock_outer_truth(state: dict[str, Any], contract: dict[str, Any],
                       output_directory: Path) -> OuterTruthPermit:
    """Irreversibly open the one-shot outer-evaluation gate after both policy seals.

    The caller must persist ``state`` immediately before any outer label is
    materialized.  A crash after this transition intentionally fails closed:
    the evaluator must use its one-shot receipt rather than recompute a new
    outer-label-dependent decision.
    """
    validate_f007_state(state, contract, output_directory, verify_seals=True)
    _require(state["outer_truth_materialized"] is False,
             "F007 outer truth was already unlocked")
    expected = {str(outer) for outer in contract["fold_ids"]}
    _require(set(state["policy_seals"]) == expected,
             "F007 all fold policies must be sealed before outer truth")
    digest = _policy_digest(state, contract, Path(output_directory).resolve())
    state["outer_truth_materialized"] = True
    state["outer_truth_access"] = {
        "schema_version": 1, "policy_digest": digest,
        "fold_ids": list(contract["fold_ids"]),
    }
    state["status"] = "running"
    state["phase"] = "one_shot_outer_evaluation"
    return OuterTruthPermit(
        experiment_signature=contract["signature"], policy_digest=digest,
        fold_ids=tuple(contract["fold_ids"]), _sentinel=_PERMIT_SENTINEL,
    )


def require_outer_truth_permit(permit: OuterTruthPermit, state: dict[str, Any],
                               contract: dict[str, Any], output_directory: Path,
                               outer: int) -> None:
    """Check the capability immediately before a scorer reads an outer label."""
    _require(isinstance(permit, OuterTruthPermit), "F007 outer truth requires a permit")
    validate_f007_state(state, contract, output_directory, verify_seals=True)
    _require(type(outer) is int and outer in contract["fold_ids"],
             "F007 outer fold is not configured")
    access = state["outer_truth_access"]
    _require(
        state["outer_truth_materialized"] is True
        and permit.experiment_signature == contract["signature"]
        and permit.policy_digest == access["policy_digest"]
        and permit.fold_ids == tuple(contract["fold_ids"]),
        "F007 outer-truth permit is not bound to the sealed policy state",
    )
