"""Tracked local postprocessing of immutable public CAM++ embedding caches.

This launcher neither loads audio nor forwards or fits an encoder. Historical
remote training guards stay unchanged; only this new cached-scoring experiment
has the user's explicit local-execution authorization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import socket
import time
import uuid

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.packaging.selected_sources import load_sources
from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
from speaker_id.tracking.snapshot import git_provenance
from speaker_id.training.adaptation_comparison import paired_diagnostics
from speaker_id.training.fusion_suite import verify_predictions
from speaker_id.training.runner import write_csv, write_json


PACKAGE_PATH = "configs/package/campp_selected.json"
PACKAGE_SHA = "1f0281a98f60ccfd7c309a67a2dec3936b1c787f3f8b26d43b53f743eb17ae2b"
AUDIT_PATH = "artifacts/infrastructure/S008_verification/verification.json"
AUDIT_SHA = "5b4f08c1dc5084f892826b3161f2b7bc5132c1e01c9c9039f42c8fd5c0719857"
SOURCE_RUN = "artifacts/training/S008_20260907T233351Z_b35f3c28"
SOURCE_PARENT = "63f25c9baebf4ec0a0907254ac8fce01"
SOURCE_CHILD = "7f8e37733bee40798a36d49e62960bde"
SOURCE_COMMIT = "17f92bd83123834cc7af04f57d411b803f3ce5c6"
SOURCE_REPORT_SHA = "b8058fba1a8967daf4f332fd60455f60c683441c46aefc118db31a66bf024036"
BASELINE_MACRO_F1 = 0.9565282892229405
BASELINE_THRESHOLD_ATOL = 1e-6
BASELINE_PROBABILITY_ATOL = 1e-5
LIMITATIONS = [
    "Repeated development on these folds is not an untouched test or hidden leaderboard estimate.",
    "All postprocessing policy and gate selection uses group-excluded inner queries only.",
    "No encoder updates, audio extraction, training-data relabeling or outer-fold fitting is allowed.",
    "Per-family OOF comparisons are descriptive; the combined policy is selected on inner data.",
    "Quality-gate meta folds hold out logistic coefficient fitting only; their features share permitted outer-training references. This is not fully nested end-to-end score generation.",
    "P002 remains immutable. This experiment does not rebuild or replace a submission package.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    original = root / value
    _require(not original.is_symlink(), "Input cannot be a symlink")
    path = original.resolve()
    _require(path.is_relative_to(root) and path.is_file()
             and not any(parent.is_symlink() for parent in original.parents if parent.is_relative_to(root)),
             "Input must be an existing regular file inside this project")
    return path


def validate_config(config: dict) -> None:
    from speaker_id.postprocessing.scoring import CANDIDATES
    required = {"schema_version", "experiment_code", "run_name", "output_root",
                "source_package_config", "source_package_config_sha256",
                "source_verification", "source_verification_sha256", "device",
                "cpu_threads", "probability_temperature", "candidates", "selection_policy",
                "baseline_numerical_policy"}
    _require(isinstance(config, dict) and required.issubset(config), "Incomplete S011 configuration")
    _require(config["schema_version"] == 1 and config["experiment_code"] == "S011"
             and config["run_name"] == "S011-campp-local-postprocessing"
             and config["output_root"] == "artifacts/training", "Unexpected local scoring experiment identity")
    _require(config["source_package_config"] == PACKAGE_PATH
             and config["source_package_config_sha256"] == PACKAGE_SHA
             and config["source_verification"] == AUDIT_PATH
             and config["source_verification_sha256"] == AUDIT_SHA,
             "S011 requires the pinned P002/S008 public512+advanced192 source")
    _require(config["device"] == "cuda" and type(config["cpu_threads"]) is int
             and 1 <= config["cpu_threads"] <= 16
             and config["probability_temperature"] == .05,
             "S011 requires its explicitly authorized local CUDA numerical policy")
    _require(config["candidates"] == list(CANDIDATES), "Candidate configuration differs from committed scoring policies")
    _require(isinstance(config["selection_policy"], str) and config["selection_policy"].strip(),
             "The inner-only policy-selection rule must be recorded")
    numerical = config["baseline_numerical_policy"]
    _require(isinstance(numerical, dict)
             and numerical.get("threshold_atol") == BASELINE_THRESHOLD_ATOL
             and numerical.get("probability_atol") == BASELINE_PROBABILITY_ATOL
             and numerical.get("relative_tolerance") == 0.0
             and isinstance(numerical.get("rationale"), str) and numerical["rationale"].strip(),
             "Use the preregistered Linux-to-Windows FP32 tolerance; no automatic relaxation")


def _source_checks(root: Path, config: dict) -> dict:
    package_path = _path(root, config["source_package_config"])
    audit_path = _path(root, config["source_verification"])
    _require(file_sha256(package_path) == PACKAGE_SHA and file_sha256(audit_path) == AUDIT_SHA,
             "Pinned source package or verification receipt changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("status") == "verified" and audit.get("parent_run_id") == SOURCE_PARENT
             and audit.get("children", {}).get("S008c") == SOURCE_CHILD
             and audit.get("git_commit") == SOURCE_COMMIT
             and audit.get("all_four_mlflow_finished") is True,
             "The source audit does not identify the fully completed S008 procedure")
    report_path = _path(root, SOURCE_RUN + "/S008c/experiment_report.json")
    _require(file_sha256(report_path) == SOURCE_REPORT_SHA, "The selected historical S008c report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report["oof"]["macro_f1"] == BASELINE_MACRO_F1, "The historical baseline score differs")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    _require(package["selection"]["family"] == "public_advanced", "Only the two unchanged public encoders are allowed")
    return {"package": package, "audit": audit, "report": report,
            "paths": {"source_package_config": package_path, "source_verification": audit_path,
                      "source_report": report_path}}


def validate_inputs(root: Path, config_path: Path) -> dict:
    """Validate immutable metadata only; do not start runs or score recordings."""
    config_path = _path(root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    checks = _source_checks(root, config)
    return {"status": "validated_no_experiment_started", "experiment": "S011",
            "source_package_sha256": file_sha256(checks["paths"]["source_package_config"]),
            "source_verification_sha256": file_sha256(checks["paths"]["source_verification"]),
            "candidate_count": len(config["candidates"]), "execution_device": config["device"],
            "encoder_updates": 0, "audio_extraction": False}


def _supporting_inputs(root: Path, config: dict) -> dict[str, Path]:
    """Allowlist curated research and sanitized readiness; never raw tool output."""
    paths = {
        "research_notes": _path(root, config["research_notes"]),
        "research_manifest": _path(root, "artifacts/infrastructure/S011_preparation/research/manifest.json"),
        "local_cuda_readiness": _path(root, config["local_readiness"]),
        "mlflow_readiness": _path(root, "artifacts/infrastructure/S011_preparation/mlflow_readonly_connectivity.json"),
        "local_install_verification": _path(root, "artifacts/infrastructure/S011_preparation/local_install_verification.json"),
        "protocol_audit": _path(root, "artifacts/infrastructure/S011_preparation/protocol_audit.md"),
    }
    research = json.loads(paths["research_manifest"].read_text(encoding="utf-8"))
    _require(research.get("status") == "complete"
             and research.get("notes", {}).get("sha256") == file_sha256(paths["research_notes"])
             and research["notes"]["bytes"] == paths["research_notes"].stat().st_size
             and _path(root, research["notes"]["path"]) == paths["research_notes"],
             "The curated research note differs from its completed manifest")
    cuda = json.loads(paths["local_cuda_readiness"].read_text(encoding="utf-8"))
    tracking = json.loads(paths["mlflow_readiness"].read_text(encoding="utf-8"))
    install = json.loads(paths["local_install_verification"].read_text(encoding="utf-8"))
    _require(cuda.get("status") == "passed" and cuda.get("cuda_available") is True
             and cuda.get("torch_cuda_build") == "12.8", "Local CUDA readiness did not pass")
    _require(tracking.get("status") == "passed" and tracking.get("experiment_id") == "1"
             and tracking.get("ownership_scope_matches") is True, "The owned MLflow readiness did not pass")
    _require(install.get("status") == "passed"
             and install.get("receipts", {}).get("local_cuda_readiness.json") == file_sha256(paths["local_cuda_readiness"])
             and install["receipts"].get("mlflow_readonly_connectivity.json") == file_sha256(paths["mlflow_readiness"]),
             "Installation verification differs from its CUDA/MLflow receipts")
    return paths


def _execution_environment(config: dict) -> dict:
    import torch
    _require(torch.cuda.is_available(), "The local CUDA environment is unavailable; no CPU fallback is automatic")
    _require(torch.version.cuda is not None and torch.version.cuda.startswith("12.8"),
             "Local torch must use the requested CUDA 12.8 build")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    properties = torch.cuda.get_device_properties(0)
    return {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": np.__version__,
            "device": "cuda:0", "device_name": properties.name, "device_memory_bytes": properties.total_memory,
            "tf32_enabled": False, "cpu_threads": config["cpu_threads"],
            "execution_scope": "user_authorized_local_cached_postprocessing", "encoder_updates": 0,
            "raw_audio_read": False, "encoder_forward_calls": 0}


def _seal_sources(output: Path, sources: dict) -> dict:
    contract = sources["contract"]
    vectors, valid = sources["vectors"], sources["valid"]
    _require(set(vectors) == {"public", "advanced"}
             and vectors["public"].shape == (4529, 512)
             and vectors["advanced"].shape == (4529, 192)
             and valid.shape == (4529,) and valid.dtype == np.bool_
             and int((~valid).sum()) == 89, "Unexpected source vector identities or zero-signal policy")
    path = output / "verified_source_arrays.npz"
    np.savez_compressed(path, public=vectors["public"], advanced=vectors["advanced"], valid=valid,
                        audio_files=np.asarray([row["audio_file"] for row in contract["manifest"]]),
                        labels=np.asarray(contract["labels"]))
    arrays = {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                    "array_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
              for name, array in {**vectors, "valid": valid}.items()}
    receipt = {"schema_version": 1, "status": "verified_original_sources_consolidated",
               "file": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size,
               "arrays": arrays, "data_input_hashes": contract["input_hashes"],
               "source_provenance_sha256": file_sha256(output / "source_provenance.json"),
               "source_package_sha256": PACKAGE_SHA, "source_audit_sha256": AUDIT_SHA,
               "loader_sha256": file_sha256(Path(__file__).resolve().parents[1] / "packaging/selected_sources.py")}
    write_json(output / "verified_source_arrays.json", receipt)
    for array in (*vectors.values(), valid):
        array.flags.writeable = False
    return receipt


def _result_predictions(contract: dict, prepared: dict, result: dict) -> tuple[list, dict]:
    scores = prepared["scores_by_alpha"][0.0]
    indices = scores["outer_indices"]
    labels = contract["labels"]
    probabilities = np.asarray(result["probabilities"])
    _require(probabilities.shape == (len(indices), len(labels)) and np.isfinite(probabilities).all()
             and np.all(probabilities >= 0)
             and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10),
             "A policy did not return valid original-order 447-class probabilities")
    invalid = ~scores["outer_valid"]
    _require(np.all(probabilities[invalid, 0] == 1) and np.all(probabilities[invalid, 1:] == 0),
             "Invalid outer rows must preserve the exact unknown fallback")
    references = [contract["manifest"][int(i)] for i in indices]
    predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(index)]}
                   for row, index in zip(references, probabilities.argmax(axis=1))]
    return predictions, score_predictions(references, predictions, labels)


def _compare_calibration(observed: dict, expected: dict) -> float:
    """Fixed before execution: only threshold float drift may be tolerated."""
    _require(set(observed) == set(expected) and "threshold" in expected,
             "Baseline calibration fields differ")
    _require(all(observed[key] == value for key, value in expected.items() if key != "threshold"),
             "Baseline coefficients or curve inner F1 differ")
    difference = abs(float(observed["threshold"]) - float(expected["threshold"]))
    _require(math.isfinite(difference) and difference <= BASELINE_THRESHOLD_ATOL,
             "Baseline threshold drift exceeds the fixed 1e-6 absolute tolerance")
    return difference


def _compare_alpha_search(observed: dict, expected: dict) -> dict:
    _require(isinstance(observed, dict) and set(observed) == set(expected),
             "The baseline five-alpha candidate set differs")
    maximum, rows = 0., 0
    for alpha, actual in observed.items():
        original = expected[alpha]
        _require(set(actual) == set(original)
                 and actual["advanced_weight"] == original["advanced_weight"]
                 and len(actual["curve"]) == len(original["curve"]),
                 "Baseline candidate identity or complete curve length differs")
        maximum = max(maximum, _compare_calibration(actual["selected"], original["selected"]))
        for actual_row, original_row in zip(actual["curve"], original["curve"]):
            maximum = max(maximum, _compare_calibration(actual_row, original_row))
            rows += 1
    return {"all_five_alpha_calibration_curves_match_with_fixed_threshold_tolerance": True,
            "curve_rows_verified": rows, "curve_threshold_max_abs_difference": maximum,
            "curve_inner_f1_and_coefficients_exact": True}


def _verify_baseline_fold(root: Path, contract: dict, prepared: dict, result: dict, outer: int) -> dict:
    reference = root / SOURCE_RUN / "S008c"
    predictions, metrics = _result_predictions(contract, prepared, result)
    check = verify_predictions(reference / f"fold_{outer}/predictions.csv", predictions)
    historical = json.loads((reference / f"fold_{outer}/evaluation.json").read_text(encoding="utf-8"))
    _require(metrics == historical["outer"], "Baseline predictions must reproduce all original metrics exactly")
    selected_drift = _compare_calibration(result["calibration"], historical["calibration"])
    _require(result["policy"]["advanced_weight"] == historical["policy"]["advanced_weight"],
             "Recomputed baseline selected a different alpha")
    _require(result["inner_macro_f1_447"] == historical["calibration"]["inner_macro_f1_447"],
             "Recomputed baseline inner criterion differs")
    alpha_path = reference / f"fold_{outer}/inner_alpha_calibration.json"
    alpha_evidence = json.loads(alpha_path.read_text(encoding="utf-8"))
    search_check = _compare_alpha_search(result.get("baseline_alpha_candidates"), alpha_evidence["candidates"])
    historical_arrays = reference / f"fold_{outer}/outer_probabilities.npz"
    with np.load(historical_arrays, allow_pickle=False) as saved:
        observed = np.asarray(result["probabilities"])
        _require(saved["probabilities"].shape == observed.shape
                 and np.array_equal(saved["audio_files"], np.asarray([row["audio_file"] for row in predictions]))
                 and np.array_equal(saved["labels"], np.asarray(contract["labels"])),
                 "Baseline probability row/label mapping differs")
        error = float(np.max(np.abs(saved["probabilities"] - observed)))
        _require(np.allclose(saved["probabilities"], observed, rtol=0, atol=BASELINE_PROBABILITY_ATOL),
                 "Baseline probabilities differ beyond the fixed 1e-5 absolute tolerance")
    return {**check, **search_check, "selected_alpha_matches": True, "inner_criterion_matches": True,
            "all_447_class_metrics_exact": True,
            "selected_threshold_max_abs_difference": selected_drift,
            "threshold_atol": BASELINE_THRESHOLD_ATOL,
            "source_alpha_calibration_sha256": file_sha256(alpha_path),
            "probability_max_abs_difference": error, "probability_atol": BASELINE_PROBABILITY_ATOL,
            "relative_tolerance": 0.0,
            "source_probability_sha256": file_sha256(historical_arrays)}


def _persist_fold(directory: Path, tracker, contract: dict, prepared: dict,
                  result: dict, outer: int, baseline_check: dict | None = None) -> tuple[list, dict]:
    directory.mkdir(parents=True, exist_ok=False)
    predictions, metrics = _result_predictions(contract, prepared, result)
    base = prepared["scores_by_alpha"][0.0]
    scores = result["scores"]
    calibration = result["calibration"]
    report = {"outer_fold": outer, "outer": metrics, "policy": result["policy"],
              "calibration": calibration, "inner_macro_f1_447": result["inner_macro_f1_447"],
              "source_reproduction": baseline_check,
              "inner_query_files": len(base["calibration_indices"]), "outer_files": len(base["outer_indices"]),
              "threshold": calibration["threshold"], "threshold_axis_label": "Inner-selected postprocessing rejection score",
              "scoring_provenance": scores.get("provenance", base["provenance"]),
              "probability_semantics": "normalized decision scores, not calibrated posteriors"}
    write_json(directory / "evaluation.json", report)
    write_json(directory / "selected_policy.json", {"policy": result["policy"], "calibration": calibration})
    write_json(directory / "calibration.json", {"selected": calibration, "curve": result["curves"]})
    if "baseline_alpha_candidates" in result:
        write_json(directory / "inner_alpha_calibration.json", {"candidates": result["baseline_alpha_candidates"]})
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "per_class.csv", metrics["per_class"])
    np.savez_compressed(directory / "outer_probabilities.npz", probabilities=result["probabilities"],
                        audio_files=np.asarray([row["audio_file"] for row in predictions]),
                        labels=np.asarray(contract["labels"]))
    score_arrays = {key: value for key, value in scores.items()
                    if isinstance(value, np.ndarray) and value.dtype.kind != "O"}
    np.savez_compressed(directory / "selected_score_arrays.npz", **score_arrays)
    support = {key: value for key, value in base["reference_counts"].items() if isinstance(value, np.ndarray)}
    np.savez_compressed(directory / "reference_support.npz", **support,
                        calibration_indices=base["calibration_indices"], outer_indices=base["outer_indices"],
                        reference_indices=np.asarray(base["provenance"]["reference_indices"], dtype=np.int64),
                        known_labels=np.asarray(base["known_labels"]))
    from speaker_id.training.plots import evaluation_plots
    curve = result["curves"]
    if isinstance(curve, list):
        curve = [row for row in curve if all(row.get(key) == calibration.get(key)
                                            for key in ("unknown_weight", "margin_weight"))]
    _require(isinstance(curve, list) and curve, "Selected calibration curve is required for diagnostic plots")
    evaluation_plots(directory, report, curve)
    for path in sorted(directory.iterdir()):
        if path.is_file():
            tracker.add_artifact(path, f"fold_{outer}/" + path.name)
    tracker.log_metrics({f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"],
                         f"fold_{outer}/outer_accuracy": metrics["accuracy"],
                         f"fold_{outer}/inner_macro_f1_447": result["inner_macro_f1_447"],
                         **{f"fold_{outer}/{key}": value for key, value in metrics["errors"].items()},
                         **{f"fold_{outer}/selected_{key}": value for key, value in calibration.items()
                            if type(value) in (int, float) and math.isfinite(value)}}, sync=True, strict=True)
    return predictions, report


def _save_policy_search(output: Path, outer: int, search: dict, parent) -> None:
    directory = output / f"policy_search/fold_{outer}"
    directory.mkdir(parents=True, exist_ok=False)
    curves = search["curves"]
    values = np.asarray(curves["values"])
    _require(values.ndim == 2 and values.shape[1] == len(curves["columns"])
             and np.isfinite(values).all(), "All inner calibration curves must be finite and dimensionally labeled")
    np.savez_compressed(directory / "all_inner_curves.npz", values=values)
    write_json(directory / "all_inner_curves.json", {"columns": curves["columns"],
               "candidate_ids": curves["candidate_ids"], "row_count": len(values),
               "array_sha256": file_sha256(directory / "all_inner_curves.npz")})
    write_json(directory / "candidate_summary.json", {"candidates": search["candidate_summary"],
               "selected_overall": search["overall"]["policy"],
               "selected_families": {name: result["policy"] for name, result in search["families"].items()}})
    for path in sorted(directory.iterdir()):
        parent.add_artifact(path, f"policy_search/fold_{outer}/" + path.name)
    parent.flush(strict=True)


def _finish_child(directory, tracker, contract, predictions, reports, family, baseline_predictions):
    metrics = score_predictions(contract["manifest"], predictions, contract["labels"])
    paired = paired_diagnostics(contract["manifest"], baseline_predictions, predictions, contract["labels"])
    report = {"recipe": family, "oof": metrics, "folds": reports, "paired_against_S008c": paired,
              "oof_macro_f1_delta": metrics["macro_f1"] - BASELINE_MACRO_F1,
              "selection_scope": "group-excluded inner queries only", "encoder_updates": 0,
              "limitations": LIMITATIONS}
    write_json(directory / "experiment_report.json", report)
    write_csv(directory / "oof_predictions.csv", predictions)
    write_csv(directory / "oof_per_class.csv", metrics["per_class"])
    for name in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
        tracker.add_artifact(directory / name)
    tracker.log_metrics({"oof/macro_f1_447": metrics["macro_f1"], "oof/accuracy": metrics["accuracy"],
                         "oof/macro_f1_delta_vs_S008c": report["oof_macro_f1_delta"],
                         **{"oof/" + key: value for key, value in metrics["errors"].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# S011 {family}\n\nOOF Macro-F1: {metrics['macro_f1']:.9f}; delta from exact S008c: {report['oof_macro_f1_delta']:+.9f}.\n\nPostprocessing and gate selection use inner queries only. Both folds cover all original 4529 rows and 447 labels, including zero-signal unknown fallbacks. No encoder training or audio extraction.\n")
    tracker.finish("FINISHED", strict=True)
    return report


def execute_suite(root: Path, config_path: Path, binding_path: Path) -> dict:
    """Execute only the new explicitly authorized local cached-scoring suite."""
    from speaker_id.postprocessing.scoring import prepare_fold, select_baseline, evaluate_policies
    root = root.resolve()
    config_path, binding_path = _path(root, config_path), _path(root, binding_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    checked = _source_checks(root, config)
    supporting = _supporting_inputs(root, config)
    source_state = git_provenance(root)
    _require(source_state["src_dirty"] is False and re.fullmatch(r"[a-f0-9]{40}", source_state["git_commit"] or ""),
             "Commit reviewed source before a tracked local postprocessing run")
    execution = _execution_environment(config)
    print(json.dumps({"stage": "source_verification", "status": "started"}), flush=True)
    sources = load_sources(root, checked["package"], verify_audio=False)
    contract = sources["contract"]
    _require(len(contract["manifest"]) == 4529 and len(contract["labels"]) == 447,
             "Original source row or label contract differs")
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    binding.validate()
    _require(binding.experiment_id == "1", "Use only the owned existing project experiment 1")
    output = root / config["output_root"] / ("S011_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    effective_limitations = list(dict.fromkeys(LIMITATIONS + config.get("limitations", [])))
    candidate_count = sum(len(candidate.get("alphas", [0., .25, .5, .75, 1.])) for candidate in config["candidates"])
    resolved = {"suite": config, "source": {"run": SOURCE_RUN, "parent_run_id": SOURCE_PARENT,
                "child_run_id": SOURCE_CHILD, "git_commit": SOURCE_COMMIT,
                "recipe_report_sha256": SOURCE_REPORT_SHA}, "execution": execution,
                "data_input_hashes": contract["input_hashes"], "limitations": effective_limitations,
                "finite_search": {"method_settings": len(config["candidates"]),
                    "method_alpha_settings_per_fold": candidate_count, "baseline_selector_entries_per_fold": 1,
                    "total_selection_entries_per_fold": candidate_count + 1, "outer_folds": 2,
                    "known_classes": 446, "evaluation_classes": 447, "original_rows": 4529},
                "supporting_evidence": {name: {"path": path.relative_to(root).as_posix(),
                                              "sha256": file_sha256(path)} for name, path in supporting.items()}}
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "source_provenance.json", sources["proof"])
    array_receipt = _seal_sources(output, sources)
    input_paths = {"suite_config": config_path, "launcher": root / "scripts/score_postprocessing.py",
                   **checked["paths"], **supporting, **{key: root / contract["config"][key]
                                         for key in ("manifest", "folds", "roles", "label_map", "model_config")}}
    common = {"project_root": root, "binding": binding, "input_paths": input_paths,
              "run_kind": "local_frozen_cache_postprocessing", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=config["run_name"], config=resolved, **common)
    active_children, child_records = [], {}
    try:
        for name in ("source_provenance.json", "verified_source_arrays.json", "verified_source_arrays.npz"):
            parent.add_artifact(output / name)
        parent.add_artifact(config_path, "input_configs/suite_config.json")
        parent.add_artifact(input_paths["launcher"], "input_configs/launcher.py")
        for name in ("source_package_config", "source_verification", "source_report"):
            parent.add_artifact(checked["paths"][name], "source_inputs/" + name + ".json")
        for name, path in supporting.items():
            parent.add_artifact(path, "preparation/" + name + path.suffix)
        # Historical byte receipts are verified locally; these checks prove the
        # current tracking destination is live before any scoring computation.
        parent.flush(strict=True)
        initial_tracking = {"artifacts": parent.verify_artifacts(), "metadata": parent.verify_remote_metadata()}
        write_json(output / "initial_tracking_verification.json", initial_tracking)
        parent.add_artifact(output / "initial_tracking_verification.json")
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
                   "stage": "baseline_reproduction", "git_commit": source_state["git_commit"]})
        print(json.dumps({"stage": "source_verification", "status": "passed", "output": str(output),
                          "parent_run_id": parent.run_id}), flush=True)
        prepared, baseline, checks, baseline_predictions = {}, {}, {}, []
        for outer in (0, 1):
            print(json.dumps({"stage": "baseline_reproduction", "fold": outer, "status": "started"}), flush=True)
            prepared[outer] = prepare_fold(sources["vectors"]["public"], sources["vectors"]["advanced"],
                sources["valid"], contract["manifest"], contract["folds"], contract["labels"], outer)
            baseline[outer] = select_baseline(prepared[outer])
            checks[str(outer)] = _verify_baseline_fold(root, contract, prepared[outer], baseline[outer], outer)
            rows, _ = _result_predictions(contract, prepared[outer], baseline[outer])
            baseline_predictions.extend(rows)
            write_json(output / "baseline_control_checks.json", checks)
            parent.add_artifact(output / "baseline_control_checks.json")
            parent.log_metrics({"baseline/completed_verified_folds": outer + 1}, step=outer, sync=True, strict=True)
        pooled_baseline = score_predictions(contract["manifest"], baseline_predictions, contract["labels"])
        _require(pooled_baseline == checked["report"]["oof"] and pooled_baseline["macro_f1"] == BASELINE_MACRO_F1,
                 "Both folds must reproduce the complete exact S008c pooled metrics")
        checks["pooled"] = {**verify_predictions(root / SOURCE_RUN / "S008c/oof_predictions.csv", baseline_predictions),
                             "exact_pooled_metrics": True, "macro_f1": BASELINE_MACRO_F1}
        write_json(output / "baseline_control_checks.json", checks)
        parent.add_artifact(output / "baseline_control_checks.json")
        parent.flush(strict=True)

        def begin_child(family):
            _require(re.fullmatch(r"[a-z][a-z0-9_]*", family) is not None, "Unsafe policy family identifier")
            directory = output / family
            directory.mkdir(exist_ok=False)
            child = DurableMLflowRun.prepare(spool_dir=directory / "tracking", parent_run_id=parent.run_id,
                run_name="S011-" + family, config={**resolved, "recipe": family}, **common)
            active_children.append(child)
            child.add_artifact(output / "baseline_control_checks.json")
            child.flush(strict=True)
            child_records[family] = {"tracker": child, "directory": directory, "predictions": [], "folds": []}
            return child_records[family]

        record = begin_child("baseline")
        for outer in (0, 1):
            rows, report = _persist_fold(record["directory"] / f"fold_{outer}", record["tracker"], contract,
                prepared[outer], baseline[outer], outer, checks[str(outer)])
            record["predictions"].extend(rows)
            record["folds"].append(report)
        reports = {"baseline": _finish_child(record["directory"], record["tracker"], contract,
                   record["predictions"], record["folds"], "baseline", baseline_predictions)}
        active_children.remove(record["tracker"])

        expected_families = None
        for outer in (0, 1):
            write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
                       "stage": "inner_policy_search", "outer_fold": outer, "baseline_both_folds_predictions_metrics_exact": True})
            sequence = [0]
            last_progress_flush = [time.monotonic()]
            def progress(info):
                sequence[0] += 1
                print(json.dumps({"stage": "inner_policy_search", "fold": outer, **info}), flush=True)
                numeric = {f"fold_{outer}/search/{key}": value for key, value in info.items()
                           if type(value) in (int, float) and math.isfinite(value)}
                if numeric:
                    parent.log_metrics(numeric, step=sequence[0], sync=False)
                if sequence[0] % 8 == 0 or time.monotonic() - last_progress_flush[0] >= 30:
                    parent.flush(strict=True)
                    last_progress_flush[0] = time.monotonic()
            search = evaluate_policies(prepared[outer], device=config["device"], on_progress=progress)
            _require(len(search["candidate_summary"]) == candidate_count + 1,
                     "The full finite candidate search did not complete")
            families = dict(search["families"])
            _require("overall" not in families, "Reserved combined-selector identifier cannot be duplicated")
            if "baseline" in families:
                repeated = families.pop("baseline")
                _require(repeated["policy"] == baseline[outer]["policy"]
                         and repeated["calibration"] == baseline[outer]["calibration"]
                         and np.array_equal(repeated["probabilities"], baseline[outer]["probabilities"]),
                         "Policy search changed the already-verified baseline")
            if expected_families is None:
                expected_families = set(families)
                for family in [*families, "overall"]:
                    begin_child(family)
            else:
                _require(set(families) == expected_families, "Both folds must evaluate the same committed policy families")
            _save_policy_search(output, outer, search, parent)
            for family, result in {**families, "overall": search["overall"]}.items():
                record = child_records[family]
                rows, report = _persist_fold(record["directory"] / f"fold_{outer}", record["tracker"],
                                             contract, prepared[outer], result, outer)
                record["predictions"].extend(rows)
                record["folds"].append(report)
        for family in [*sorted(expected_families), "overall"]:
            record = child_records[family]
            reports[family] = _finish_child(record["directory"], record["tracker"], contract,
                record["predictions"], record["folds"], family, baseline_predictions)
            active_children.remove(record["tracker"])
        # Detect any accidental mutation of the read-only source arrays.
        for name, array in {**sources["vectors"], "valid": sources["valid"]}.items():
            _require(hashlib.sha256(array.tobytes()).hexdigest() == array_receipt["arrays"][name]["array_sha256"],
                     "Postprocessing mutated an original public embedding cache")
        import torch
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": reports,
                  "baseline_control_checks": checks, "selection_policy": config["selection_policy"],
                  "elapsed_seconds": time.monotonic() - started, "encoder_updates": 0,
                  "source_arrays_unchanged": True, "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "children": {key: value["tracker"].run_id for key, value in child_records.items()},
                  "finite_search": resolved["finite_search"], "limitations": effective_limitations}
        write_json(output / "experiment_report.json", report)
        parent.add_artifact(output / "experiment_report.json")
        parent.add_artifact(output / "resolved_config.json")
        for family, item in reports.items():
            parent.log_metrics({family + "/oof_macro_f1_447": item["oof"]["macro_f1"],
                                family + "/oof_delta": item["oof_macro_f1_delta"]}, sync=False)
        parent.write_report(report, markdown="# S011 local cached postprocessing\n\nBoth S008c baseline folds reproduced all predictions and metrics exactly before new policy evaluation; preregistered absolute thresholds bound platform numerical drift. Every family and the combined selector use only group-excluded inner queries for policy and gate selection. No audio extraction or encoder fitting occurred; source caches and P002 remain unchanged. Quality-gate meta folds hold out logistic coefficient fitting only and share permitted reference features; they are not fully nested end-to-end score generation.\n\n" + "\n".join(
            f"- {family}: OOF Macro-F1 {item['oof']['macro_f1']:.9f}; delta {item['oof_macro_f1_delta']:+.9f}."
            for family, item in reports.items()) + "\n")
        parent.finish("FINISHED", strict=True)
        roundtrip = {}
        for family, record in child_records.items():
            tracker = record["tracker"]
            roundtrip[family] = {"artifacts": tracker.verify_artifacts(), "metadata": tracker.verify_remote_metadata()}
        roundtrip["parent"] = {"artifacts": parent.verify_artifacts(), "metadata": parent.verify_remote_metadata()}
        write_json(output / "tracking_roundtrip_verification.json", {"status": "passed", "runs": roundtrip})
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id,
                   "children": report["children"], "all_mlflow_finished_and_verified": True,
                   "git_commit": source_state["git_commit"]})
        return {"output": str(output), "parent_run_id": parent.run_id,
                "results": {family: item["oof"]["macro_f1"] for family, item in reports.items()},
                "all_mlflow_finished_and_verified": True, "new_submission_built": False}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id,
                   "error_type": type(error).__name__, "error": parent.redactor.text(str(error)),
                   "elapsed_seconds": time.monotonic() - started}
        write_json(output / "failure.json", failure)
        for child in active_children:
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        if parent.state.get("remote_status") != "FINISHED":
            parent.write_report(failure)
            parent.finish("FAILED", strict=False)
        else:
            failure["computation_finished_tracking_verification_failed"] = True
        write_json(output / "experiment_state.json", failure)
        raise
