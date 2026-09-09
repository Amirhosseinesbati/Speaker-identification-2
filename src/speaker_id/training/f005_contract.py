"""Strict, data-aware contract for the F005 paired advanced-CAM++ experiment.

F005 deliberately does not reuse the historical 512-dimensional training
loader.  Its trainable endpoint is the pinned 192-dimensional advanced model;
the public 512-dimensional endpoint remains an immutable C002b scoring input.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

from speaker_id.candidates.campp_advanced import validate_advanced_config
from speaker_id.data.splits import truth
from speaker_id.models.campp import file_sha256
from speaker_id.training.contracts import load_contract


CONFIG_SHA256 = "657a7310a4b8a27b11f8a8094d7b378631b77881ad090e088094d415ed0e325e"
ADVANCED_WEIGHTS_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
PUBLIC_DIMENSION = 512
ADVANCED_DIMENSION = 192
ARM_IDS = ("control", "treatment_mse0", "treatment_mse01", "treatment_mse05")
TREATMENT_IDS = ARM_IDS[1:]
EXPECTED_FIT_ROWS = {0: 662, 1: 667}
EXPECTED_KNOWN_QUERY_LABELS = {0: 443, 1: 445}
EXPECTED_ABSENT_KNOWN_LABELS = {0: 3, 1: 1}


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _relative_project_path(value: str, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix + "/"):
        return False
    parsed = PurePosixPath(value)
    return not parsed.is_absolute() and parsed.as_posix() == value and ".." not in parsed.parts and "\\" not in value and ":" not in value


def validate_f005_config(config: dict) -> None:
    """Accept exactly the preregistered F005 protocol, including retention.

    A canonical digest is intentional here.  Adding a seemingly harmless arm,
    changing a source receipt, or allowing weights into MLflow creates a new
    experiment code rather than silently changing F005 after results exist.
    """
    if not isinstance(config, dict):
        raise ValueError("F005 configuration must be a JSON object")
    try:
        digest = hashlib.sha256(canonical(config)).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("F005 configuration must contain finite canonical JSON") from error
    if digest != CONFIG_SHA256:
        raise ValueError("F005 requires its exact preregistered configuration")

    if config["experiment_code"] != "F005" or config["device"] != "cuda" or config["tracking_required"] is not True:
        raise ValueError("F005 execution identity changed")
    if config["execution"] != {
        "gpu_name_contains": "RTX 3090", "tensor_dtype": "float32", "no_cpu_fallback": True,
        "deterministic_algorithms": "enforce_error", "cublas_workspace_config": ":4096:8",
        "thread_environment": {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
    }:
        raise ValueError("F005 deterministic CUDA execution policy changed")
    if config["advanced_model_config"] != "configs/model/campp_advanced.json":
        raise ValueError("F005 must tune the pinned advanced endpoint")
    arms = config["arms"]
    if tuple(arm["id"] for arm in arms) != ARM_IDS:
        raise ValueError("F005 arm identity/order changed")
    if [(arm["cosine_gamma"], arm["raw_h_mse_lambda"]) for arm in arms] != [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.1), (0.5, 0.5)
    ]:
        raise ValueError("F005 control/treatment coefficients changed")
    if config["views"] != {
        "algorithm": "nested_real_waveform_crops_v1", "sample_rate": 16000,
        "short_seconds": 3.0, "long_seconds": 8.0, "pad_after_crop": True,
        "same_plan_for_all_arms": True,
    }:
        raise ValueError("F005 paired-view protocol changed")
    fit = config["fit"]
    if (fit["mixed_precision"] is not False or fit["freeze_batchnorm"] is not True
            or fit["classification"] != "mean_short_long_aam"
            or fit["consistency_phase"] != "tail_only"
            or fit["consistency_ramp_tail_steps"] != 100
            or fit["raw_h_definition"] != "final_campp_192d_output_before_l2_normalization"
            or fit["long_consistency_target"] != "dynamic_same_step_stop_gradient"
            or fit["long_aam_updates_encoder"] is not True
            or fit["shared_head_stage"] != "run_once_per_outer_then_fork_byte_identical_checkpoint_to_all_four_tails"
            or fit["adaptation_schedule"] != {
                "scheme": "head_warmup_tail_v1", "head_only_steps": 600,
                "margin_ramp_tail_steps": 100, "encoder_lr_warmup_tail_steps": 50,
                "head_lr_warmup_start_factor": 0.1,
            }
            or fit["epochs"] * fit["steps_per_epoch"] != 500
            or fit["batch_pairs"] % fit["microbatch_pairs"] != 0
            or fit["waveform_cache_max_bytes"] != 1073741824):
        raise ValueError("F005 must preserve its FP32 F004 schedule and complete paired updates")
    selection = config["arm_selection"]
    if (selection["arm_training_scope"] != "all_known_encoder_fit_allowed_rows_once"
            or selection["query_scope"] != "original_disjoint_known_calibration_query_rows_only"
            or selection["reference_scope"] != "all_eligible_outer_training_known_references_with_whole_query_group_exclusion"
            or selection["arm_tie_order"] != list(ARM_IDS)
            or selection["control_is_selectable"] is not True
            or selection["unknown_calibration_hidden_until_arm_sealed"] is not True
            or selection["refit_after_selection"] is not False
            or selection["expected_observed_known_labels_by_outer_fold"] != {"0": 443, "1": 445}
            or selection["expected_absent_known_labels_by_outer_fold"] != {"0": 3, "1": 1}):
        raise ValueError("F005 fit-all/calibration-only selection protocol changed")
    if config["deployment_selection"] != {
        "scope": "defined_only_after_separate_pooled_oof_promotion_decision",
        "per_outer_arm_selection_must_not_be_globally_pooled": True,
    }:
        raise ValueError("F005 per-outer selection cannot define deployment before OOF promotion")
    scoring = config["scoring"]
    if (scoring["protocol"] != "heldout_reference_scores_original_disjoint_roles"
            or scoring["no_exact_c002b_reproduction_claim"] is not True
            or scoring["no_outer_tuning"] is not True):
        raise ValueError("F005 adapted scoring must use its authoritative heldout-reference protocol")
    bootstrap = config["bootstrap"]
    if (bootstrap["kind"] != "paired_whole_content_group_unstratified"
            or bootstrap["mixed_label_groups_preserved"] is not True
            or bootstrap["true_class_purity_assumed"] is not False):
        raise ValueError("F005 bootstrap must preserve complete mixed-label groups")
    if config["mlflow"]["experiment_id"] != "1" or set(config["mlflow"]["forbidden"]) != {
        "raw_audio", "embeddings", "model_weights", "optimizer_state", "credentials"
    }:
        raise ValueError("F005 MLflow boundary changed")
    if (config["retention"]["local_transfer"] != "promotion_only"
            or "intermediate_checkpoints" not in config["retention"]["server_only"]):
        raise ValueError("F005 promotion-only retention changed")
    promotion = config["promotion"]
    if (promotion["goal_target_oof_macro_f1"] != 0.965
            or promotion["incumbent_promotion_is_distinct_from_goal_completion"] is not True
            or promotion["minimum_treatment_delta_vs_fresh_control"] != 0.003
            or promotion["minimum_selected_delta_vs_c002b"] != 0.003
            or promotion["minimum_control_delta_vs_c002b_when_selected"] != 0.003
            or promotion["minimum_accuracy_delta_vs_c002b"] != 0.0
            or promotion["minimum_short_known_top1_delta_vs_c002b"] != 0.0
            or promotion["maximum_unknown_to_known_increase_vs_c002b"] != 2
            or promotion["maximum_known_to_other_known_increase_vs_c002b"] != 0
            or promotion["all_conditions_required"] is not True):
        raise ValueError("F005 promotion guards changed")


def _validate_role_scopes(readiness: dict, config: dict) -> dict:
    labels = readiness["labels"]
    known = set(labels[1:])
    summaries = []
    for outer in config["fold_ids"]:
        rows = [row for row in readiness["roles"] if int(row["outer_fold"]) == outer]
        fit = [row for row in rows if truth(row["encoder_fit_allowed"])]
        known_queries = [row for row in rows if truth(row["calibration_query"]) and row["speaker_id"] != "unknown"]
        unknown_queries = [row for row in rows if truth(row["calibration_query"]) and row["speaker_id"] == "unknown"]
        enrollment = [row for row in rows if truth(row["enrollment_allowed"])]
        if (len(fit) != EXPECTED_FIT_ROWS[outer] or any(row["speaker_id"] == "unknown" for row in fit)
                or set(row["speaker_id"] for row in fit) != known):
            raise ValueError("F005 fit-all rows differ from the authoritative known-only role scope")
        observed = set(row["speaker_id"] for row in known_queries)
        if (len(observed) != EXPECTED_KNOWN_QUERY_LABELS[outer]
                or len(known - observed) != EXPECTED_ABSENT_KNOWN_LABELS[outer]):
            raise ValueError("F005 known calibration label coverage changed")
        if set(row["speaker_id"] for row in enrollment) != known or not unknown_queries:
            raise ValueError("F005 C002b calibration/enrollment support is incomplete")
        fit_groups = {row["group_id"] for row in fit}
        query_groups = {row["group_id"] for row in known_queries + unknown_queries}
        if fit_groups & query_groups:
            raise ValueError("F005 calibration query content leaked into encoder fitting")
        summaries.append({
            "outer_fold": outer,
            "fit_rows": len(fit),
            "fit_known_labels": len({row["speaker_id"] for row in fit}),
            "known_query_rows": len(known_queries),
            "known_query_labels": len(observed),
            "absent_known_query_labels": sorted(known - observed),
            "unknown_query_rows": len(unknown_queries),
            "enrollment_rows": len(enrollment),
            "fit_query_groups_disjoint": True,
        })
    return {"folds": summaries}


def verify_f005_sources(contract: dict, root: Path) -> dict:
    """Verify pinned advanced weights and immutable C002b receipts before CUDA."""
    root = Path(root).resolve()
    model = contract["advanced_model"]
    weights = (root / model["weights_path"]).resolve()
    if (not weights.is_file() or not weights.is_relative_to(root / "artifacts/models")
            or weights.stat().st_size != model["weights_bytes"]
            or file_sha256(weights) != ADVANCED_WEIGHTS_SHA256):
        raise ValueError("F005 advanced192 checkpoint bytes differ from the pinned source")
    source = contract["config"]["source_c002b"]
    run = Path(source["run_dir"])
    if not run.is_absolute() or not run.is_dir():
        raise ValueError("F005 C002b source run is unavailable")
    verified = {}
    for relative, expected in source["artifacts"].items():
        if not _relative_project_path("artifacts/training/" + relative, "artifacts/training"):
            raise ValueError("F005 C002b receipt contains an unsafe artifact path")
        path = run / PurePosixPath(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"F005 C002b source hash mismatch: {relative}")
        verified[relative] = expected
    return {
        "advanced_dimension": ADVANCED_DIMENSION,
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256,
        "public_dimension": PUBLIC_DIMENSION,
        "public_endpoint_trainable": False,
        "c002b_artifacts": verified,
    }


def load_f005_contract(config_path: Path, root: Path, *, verify_audio: bool = False,
                       verify_sources: bool = False) -> dict:
    """Load F005 without constructing Torch, a model, or an MLflow run."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_f005_config(config)
    readiness_path = (root / config["readiness_config"]).resolve()
    readiness = load_contract(readiness_path, root, verify_audio=verify_audio)
    if (readiness["config"]["experiment_code"] != "B002-C002"
            or readiness["model"]["embedding_dim"] != PUBLIC_DIMENSION
            or readiness["config"]["fold_ids"] != config["fold_ids"]
            or len(readiness["manifest"]) != config["expected_source_files"]
            or len(readiness["labels"]) != config["evaluation_classes"]):
        raise ValueError("F005 readiness is not the pinned C002 data/public endpoint")
    model_path = (root / config["advanced_model_config"]).resolve()
    advanced = json.loads(model_path.read_text(encoding="utf-8"))
    validate_advanced_config(advanced)
    if advanced["embedding_dim"] != ADVANCED_DIMENSION or advanced["weights_sha256"] != ADVANCED_WEIGHTS_SHA256:
        raise ValueError("F005 trainable endpoint must be advanced CAM++ 192D")
    role_summary = _validate_role_scopes(readiness, config)
    code_paths = [
        root / "src/speaker_id/adaptation/paired_views.py",
        root / "src/speaker_id/adaptation/consistency.py",
        root / "src/speaker_id/candidates/campp_advanced.py",
        root / "src/speaker_id/evaluation/group_bootstrap.py",
        root / "src/speaker_id/training/f005_contract.py",
        root / "src/speaker_id/training/f005_experiment.py",
        root / "src/speaker_id/training/f005_runner.py",
        root / "src/speaker_id/training/f005_scoring.py",
        root / "src/speaker_id/training/f005_worker.py",
        root / "src/speaker_id/training/fit.py",
        root / "src/speaker_id/training/schedules.py",
        root / "scripts/train_f005.py",
        root / "scripts/run_f005_experiment.py",
        root / "scripts/infra/run_f005.sh",
        root / "scripts/infra/install_f005_supervisor.sh",
        root / "scripts/infra/with_project_env.py",
    ]
    missing = [path for path in code_paths if not path.is_file()]
    if missing:
        raise ValueError("F005 implementation is incomplete: " + ", ".join(path.name for path in missing))
    source_entries = list((root / "src").rglob("*"))
    source_symlinks = [path for path in source_entries if path.is_symlink()]
    if source_symlinks:
        raise ValueError("F005 source tree may not contain symlinks")
    source_files = sorted(
        path for path in source_entries
        if path.is_file()
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and not any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                    for part in path.relative_to(root / "src").parts)
    )
    if not source_files:
        raise ValueError("F005 source tree is empty")
    source_tree = [
        {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        for path in source_files
    ]
    identity = {
        "schema_version": 1,
        "experiment": config,
        "config_sha256": file_sha256(config_path),
        "readiness_signature": readiness["signature"],
        "readiness_input_hashes": readiness["input_hashes"],
        "advanced_model_config_sha256": file_sha256(model_path),
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256,
        "trainable_embedding_dimension": ADVANCED_DIMENSION,
        "frozen_public_embedding_dimension": PUBLIC_DIMENSION,
        "code_hashes": {path.relative_to(root).as_posix(): file_sha256(path) for path in code_paths},
        "src_tree_file_count": len(source_tree),
        "src_tree_sha256": hashlib.sha256(canonical(source_tree)).hexdigest(),
    }
    contract = {
        "config": config,
        "readiness": readiness,
        "advanced_model": advanced,
        "role_summary": role_summary,
        "identity": identity,
        "manifest": readiness["manifest"],
        "folds": readiness["folds"],
        "roles": readiness["roles"],
        "labels": readiness["labels"],
    }
    contract["signature"] = hashlib.sha256(canonical(identity)).hexdigest()
    contract["source_verification"] = verify_f005_sources(contract, root) if verify_sources else None
    return contract


def require_execution_environment(contract: dict) -> dict:
    """Check server markers without importing Torch; the worker checks CUDA next."""
    config = contract["config"]
    expected = str(contract["readiness"]["config"]["expected_vast_instance_id"])
    if os.environ.get("VAST_INSTANCE_ID") != expected:
        raise RuntimeError("F005 requires the C002-authorized Vast instance marker")
    return {
        "vast_instance_id": expected,
        "device": config["device"],
        "cpu_threads": config["cpu_threads"],
        "trainable_endpoint": "advanced_campp_192d",
        "frozen_endpoint": "public_campp_512d",
    }
