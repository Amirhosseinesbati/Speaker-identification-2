"""Fixed, auditable configuration surface for F008 unknown outlier exposure."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


F008_CONFIG_SCHEMA = 1
F008_ARM_IDS = ("control_f005", "energy_005", "uniform_005")


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def config_signature(config: Mapping[str, object]) -> str:
    validate_f008_config(config)
    return hashlib.sha256(canonical(config)).hexdigest()


def load_f008_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("F008 config must contain a JSON object")
    validate_f008_config(value)
    return deepcopy(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"F008 {label} must be a lowercase SHA-256")


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    _require(type(value) in (int, float) and not isinstance(value, bool),
             f"F008 {label} must be a finite number")
    result = float(value)
    _require(math.isfinite(result) and (result > 0.0 if positive else True),
             f"F008 {label} must be a finite{' positive' if positive else ''} number")
    return result


def validate_f008_config(config: Mapping[str, object]) -> None:
    """Reject any unreviewed F008 change before source probing or tail updates."""
    required = {
        "schema_version", "experiment_code", "run_name", "hypothesis", "source_f005",
        "readiness_config", "advanced_model_config", "output_root", "seed", "fold_ids",
        "expected_source_files", "known_classes", "evaluation_classes", "expected_role_counts",
        "device", "cpu_threads", "execution", "training", "energy_margin", "arms",
        "selection", "scoring", "promotion", "mlflow", "retention",
    }
    _require(isinstance(config, Mapping) and set(config) == required,
             "F008 config fields changed")
    _require(config["schema_version"] == F008_CONFIG_SCHEMA and config["experiment_code"] == "F008"
             and isinstance(config["run_name"], str) and config["run_name"].startswith("F008-")
             and isinstance(config["hypothesis"], str) and bool(config["hypothesis"]),
             "F008 identity is invalid")
    _require(config["readiness_config"] == "configs/train/campp_coverage_c002.json"
             and config["advanced_model_config"] == "configs/model/campp_advanced.json"
             and config["output_root"] == "artifacts/training/f008_unknown_oe"
             and config["fold_ids"] == [0, 1] and type(config["seed"]) is int
             and config["seed"] == 20260909 and config["expected_source_files"] == 4529
             and config["known_classes"] == 446 and config["evaluation_classes"] == 447
             and config["device"] == "cuda" and config["cpu_threads"] == 4,
             "F008 data/model binding changed")
    _validate_source(config["source_f005"])
    _validate_role_counts(config["expected_role_counts"])
    _validate_execution(config["execution"])
    _validate_training(config["training"])
    _validate_energy_margin(config["energy_margin"])
    _validate_arms(config["arms"])
    _validate_selection(config["selection"])
    _validate_scoring(config["scoring"])
    _validate_promotion(config["promotion"])
    _require(config["mlflow"] == {
        "experiment_id": "1",
        "upload": "resolved_config_source_snapshot_receipts_seals_metrics_reports_only_no_audio_embeddings_weights_optimizer_or_credentials",
    }, "F008 MLflow boundary changed")
    _require(config["retention"] == {
        "local_transfer": "promotion_only",
        "server_only": ["embedding_caches", "intermediate_checkpoints", "optimizer_states", "failed_arms", "run_directory"],
    }, "F008 retention boundary changed")


def _validate_source(value: object) -> None:
    _require(isinstance(value, Mapping) and set(value) == {
        "run_dir", "parent_run_id", "config_path", "config_sha256", "experiment_signature",
        "resolved_config_sha256", "source_snapshot_relative_path", "source_snapshot_sha256",
        "required_selected_arm_by_outer_fold", "arm_selection_seals", "reuse_shared_head",
        "reuse_control_tail_as_comparator_only",
    }, "F008 F005 source binding changed")
    _require(value["run_dir"] == "/workspace/Speaker-identification-2-c002/artifacts/training/f005_consistency/F005_supervised_primary"
             and value["parent_run_id"] == "5ad78a9dba1f4ab4a32783e445d91681"
             and value["config_path"] == "configs/train/campp_f005_consistency.json"
             and value["experiment_signature"] == "1dff7bf656fa9ba12afb9c8b3c3523a489609a402ff1f6e91d484ac1f592b3ab"
             and value["source_snapshot_relative_path"] == "tracking/parent/artifacts/source_snapshot.zip"
             and value["required_selected_arm_by_outer_fold"] == {"0": "control", "1": "control"}
             and value["reuse_shared_head"] is True
             and value["reuse_control_tail_as_comparator_only"] is True,
             "F008 F005 source values changed")
    _sha256(value["config_sha256"], "F005 config checksum")
    _sha256(value["experiment_signature"], "F005 experiment signature")
    _sha256(value["resolved_config_sha256"], "F005 resolved config checksum")
    _sha256(value["source_snapshot_sha256"], "F005 source snapshot checksum")
    seals = value["arm_selection_seals"]
    _require(isinstance(seals, Mapping) and set(seals) == {"0", "1"},
             "F008 F005 arm-selection seals changed")
    for fold, expected in (
        ("0", {"file_sha256": "fa087e98f84ca76a65cc472f58663a87456bec83b76afdb4e6429babb10b9546",
               "seal_sha256": "2423ab94a04cc6230d9ba6db24baa1fcdcc000224ff59884a6ec6bb7cf381021",
               "selected_arm": "control"}),
        ("1", {"file_sha256": "e0bec08e310a307802dfc6cb08e24138d4a60c867532b194b72a2c19166fbe69",
               "seal_sha256": "e2fa8c6b34c2f2749897995a6c6617decb97999e7e789eed5b386535d77d140c",
               "selected_arm": "control"}),
    ):
        _require(seals[fold] == expected, "F008 F005 arm-selection seals changed")


def _validate_role_counts(value: object) -> None:
    expected = {
        "0": {"known_fit_rows": 662, "known_fit_groups": 662, "unknown_fit_rows": 555,
              "unknown_fit_groups": 555, "known_calibration_rows": 443,
              "unknown_calibration_rows": 556},
        "1": {"known_fit_rows": 667, "known_fit_groups": 667, "unknown_fit_rows": 556,
              "unknown_fit_groups": 556, "known_calibration_rows": 445,
              "unknown_calibration_rows": 556},
    }
    _require(value == expected, "F008 role-count baseline changed")


def _validate_execution(value: object) -> None:
    _require(value == {
        "gpu_name_contains": "RTX 3090", "tensor_dtype": "float32", "no_cpu_fallback": True,
        "deterministic_algorithms": "enforce_error", "cublas_workspace_config": ":4096:8",
        "thread_environment": {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
    }, "F008 execution policy changed")


def _validate_training(value: object) -> None:
    _require(isinstance(value, Mapping) and value == {
        "source": "authenticated_f005_shared_head_before_tail",
        "known_plan": "exact_f005_tail_known_speaker_file_and_nested_crop_plan",
        "known_objective": "mean_short_long_aam", "tail_steps": 500,
        "known_batch_pairs": 32, "unknown_batch_pairs": 32, "microbatch_pairs": 4,
        "unknown_sampling": "step_counter_seeded_uniform_permitted_unknown_file_with_paired_nested_crops_v1",
        "unknown_scope": "unknown_encoder_fit_allowed_only_group_disjoint_from_calibration_and_outer",
        "freeze_batchnorm": True, "gradient_clip_norm": 5.0, "checkpoint_every_steps": 100,
        "checkpoint_scope": "server_only_until_promotion", "mixed_precision": False,
    }, "F008 training protocol changed")


def _validate_energy_margin(value: object) -> None:
    _require(isinstance(value, Mapping) and set(value) == {
        "formula", "energy_definition", "energy_temperature", "softplus_temperature",
        "known_maximum_quantile", "declared_minimum_energy_gap", "unknown_source_distribution",
        "margin_fit_scope", "margin_values_sealed_before_tail_training",
        "known_energy_weight", "unknown_energy_weight",
    }, "F008 energy-margin fields changed")
    _require(value["formula"] == "0.5_times_mean_softplus((E_known-maximum_known_energy)/softplus_temperature)_plus_0.5_times_mean_softplus((minimum_unknown_energy-E_unknown)/softplus_temperature)"
             and value["energy_definition"] == "negative_temperature_logsumexp_of_pre_margin_unscaled_cosine_logits"
             and value["unknown_source_distribution"] == "diagnostic_only; the unknown lower margin is the fitted known_upper_quantile plus the declared positive gap"
             and value["margin_fit_scope"] == "all_permitted_known_and_unknown_encoder_fit_rows_with_authenticated_f005_shared_head_only"
             and value["margin_values_sealed_before_tail_training"] is True,
             "F008 energy-margin semantics changed")
    _require(_finite(value["energy_temperature"], "energy temperature", positive=True) == 0.05
             and _finite(value["softplus_temperature"], "softplus temperature", positive=True) == 0.05
             and _finite(value["known_maximum_quantile"], "known energy quantile", positive=True) == 0.95
             and _finite(value["declared_minimum_energy_gap"], "energy gap", positive=True) == 0.02
             and _finite(value["known_energy_weight"], "known energy weight", positive=True) == 1.0
             and _finite(value["unknown_energy_weight"], "unknown energy weight", positive=True) == 1.0,
             "F008 energy-margin values changed")


def _validate_arms(value: object) -> None:
    _require(value == [
        {"id": "control_f005", "kind": "reused_f005_dual_aam_control", "lambda": 0.0},
        {"id": "energy_005", "kind": "energy_margin_outlier_exposure", "lambda": 0.05},
        {"id": "uniform_005", "kind": "uniform_outlier_exposure", "lambda": 0.05},
    ], "F008 arms changed")


def _validate_selection(value: object) -> None:
    _require(value == {
        "calibration_scope": "original_group_disjoint_known_and_unknown_calibration_query_rows_only",
        "primary_metric": "macro_f1_447", "known_preservation_metric": "known_query_macro_f1_over_observed_labels",
        "maximum_known_preservation_decline_vs_control": 0.001,
        "tie_order": list(F008_ARM_IDS), "outer_labels_forbidden_until_all_arm_policies_sealed": True,
        "refit_after_selection": False,
    }, "F008 selection policy changed")


def _validate_scoring(value: object) -> None:
    _require(value == {
        "protocol": "c002b_family_disjoint_roles_v1", "reference_method": "max_reference",
        "fusion": "same_reference_sqrt_weighted_encoder_concatenation",
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
        "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0], "margin_weights": [0.0, 0.5],
        "threshold_candidates": 201, "probability_temperature": 0.05,
        "historical_c002b_fixed_policy": "diagnostic_only_never_selectable",
    }, "F008 scoring policy changed")


def _validate_promotion(value: object) -> None:
    _require(value == {
        "goal_target_oof_macro_f1": 0.965, "minimum_selected_delta_vs_reused_control": 0.003,
        "minimum_selected_delta_vs_c002b": 0.003, "minimum_accuracy_delta_vs_c002b": 0.0,
        "minimum_short_known_top1_delta_vs_c002b": 0.0, "minimum_each_fold_delta_vs_reused_control": -0.001,
        "minimum_each_fold_delta_vs_c002b": -0.001, "maximum_unknown_to_known_increase_vs_c002b": 2,
        "maximum_known_to_other_known_increase_vs_c002b": 0,
        "minimum_group_bootstrap_lower_bound_vs_reused_control": -0.0005,
        "minimum_group_bootstrap_lower_bound_vs_c002b": -0.0005,
        "require_exact_cpu_cuda_prediction_parity": True, "all_conditions_required": True,
        "otherwise": "retain_c002b",
    }, "F008 promotion rules changed")
