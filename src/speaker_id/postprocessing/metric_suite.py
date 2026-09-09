"""S013 tracked CPU metric learning over unchanged public CAM++ embeddings."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.packaging.selected_sources import load_sources
from speaker_id.postprocessing.suite import (
    PACKAGE_PATH, PACKAGE_SHA, AUDIT_PATH, AUDIT_SHA, SOURCE_RUN, SOURCE_PARENT,
    SOURCE_CHILD, SOURCE_COMMIT, SOURCE_REPORT_SHA, BASELINE_MACRO_F1,
    BASELINE_THRESHOLD_ATOL, BASELINE_PROBABILITY_ATOL, _require, _path,
    _execution_environment, _seal_sources, _result_predictions, _verify_baseline_fold,
)
from speaker_id.postprocessing.decision_suite import (
    S011_AUDIT, S011_AUDIT_SHA, _checked_sources as _historical_sources,
    _array_fields, _zip_json, _mark_training_started,
)
from speaker_id.postprocessing.frozen_metric import METRIC_SPECS, _validate_payload
from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
from speaker_id.tracking.snapshot import git_provenance
from speaker_id.training.adaptation_comparison import paired_diagnostics, known_ranking_diagnostics
from speaker_id.training.fusion_suite import verify_predictions
from speaker_id.training.runner import write_csv, write_json

RECIPES = ["baseline", *[spec["id"] for spec in METRIC_SPECS], "overall"]
RESEARCH_NOTE = "reports/research/decision_postprocessing_20260908/frozen_metric_next_route.md"
RESEARCH_MANIFEST = "artifacts/infrastructure/S012_preparation/research/metric_next_route_manifest.json"
READINESS = "artifacts/infrastructure/S012_preparation/sklearn_install_verification.json"
PROTOCOL_DOC = "docs/local_metrics.fa.md"
SYNTHETIC_RECEIPT = "artifacts/infrastructure/S013_preparation/metric_validation.json"
PREREQUISITE_KEYS = {"run_path", "parent_run_id", "git_commit", "verification_path", "verification_sha256", "report_sha256"}
LIMITATIONS = [
    "Repeated development OOF is not an untouched test or hidden leaderboard estimate.",
    "Six fixed geometric recipes and their gates are selected from optimized nested meta scores; the selection score is not an independent test.",
    "The normalized identity geometry is a separate metric control, not the exact historical S008c baseline.",
    "Every meta-validation group is excluded from the fitted mean, covariance and entire reference gallery. Unknown recordings never enter the fitted mean or covariance.",
    "The mean weights known speakers equally; within-speaker covariance weights qualifying speakers equally and independent content groups equally.",
    "Nested gallery removal lowers speaker support; singleton groups remain reference-only, and support-domain mismatch remains a limitation.",
    "Each geometric recipe is exploratory. Only the prespecified overall selector or exact historical baseline fallback is the primary comparison.",
    "The fixed fusion weight is 0.5. Unlike decision-only heads, geometric transforms can change the known-identity ranking.",
    "Mean/covariance fitting, normalization and cosine matching execute on the local CPU. No CUDA scoring result or CPU/CUDA parity claim is made.",
    "No encoder updates, new audio extraction, test transduction, package replacement or leaderboard submission occurs in this suite.",
]


def _prerequisite_ready(config):
    return all(config["s012_prerequisite"].get(key) is not None for key in PREREQUISITE_KEYS)


def validate_config(config, *, allow_pending=True):
    from speaker_id.postprocessing.metric_scoring import (
        ADVANCED_WEIGHT, UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, THRESHOLD_QUANTILES,
        MINIMUM_META_GAIN, MAXIMUM_META_FOLD_LOSS)
    from speaker_id.postprocessing.nested_cases import META_FOLDS, META_SALT
    required = {"schema_version", "experiment_code", "run_name", "output_root", "source_package_config",
        "source_package_config_sha256", "source_verification", "source_verification_sha256",
        "s011_verification", "s011_verification_sha256", "s012_prerequisite", "device", "cpu_threads",
        "metric_fit_device", "matching_device", "metric_specs", "advanced_weight", "unknown_weights",
        "margin_weights", "threshold_quantiles", "probability_temperature", "meta_folds", "meta_assignment_salt",
        "promotion", "baseline_numerical_policy", "selection_policy", "research_notes", "local_readiness"}
    _require(isinstance(config, dict) and required <= set(config), "Incomplete S013 configuration")
    _require(config["schema_version"] == 1 and config["experiment_code"] == "S013"
        and config["run_name"] == "S013-campp-nested-frozen-metrics" and config["output_root"] == "artifacts/training",
        "Unexpected S013 identity")
    _require(config["source_package_config"] == PACKAGE_PATH and config["source_package_config_sha256"] == PACKAGE_SHA
        and config["source_verification"] == AUDIT_PATH and config["source_verification_sha256"] == AUDIT_SHA
        and config["s011_verification"] == S011_AUDIT and config["s011_verification_sha256"] == S011_AUDIT_SHA,
        "Historical source pins changed")
    _require(config["metric_specs"] == METRIC_SPECS and config["advanced_weight"] == ADVANCED_WEIGHT
        and config["unknown_weights"] == UNKNOWN_WEIGHTS and config["margin_weights"] == MARGIN_WEIGHTS
        and config["threshold_quantiles"] == THRESHOLD_QUANTILES and config["meta_folds"] == META_FOLDS
        and config["meta_assignment_salt"] == META_SALT, "The six geometric recipes or nested gate grid changed")
    _require(config["promotion"] == {"minimum_pooled_gain": MINIMUM_META_GAIN,
        "maximum_meta_fold_loss": MAXIMUM_META_FOLD_LOSS, "reference": "identity_metric_control", "otherwise": "exact_historical_baseline"},
        "Metric promotion or fallback rule changed")
    _require(config["device"] == "cuda" and config["metric_fit_device"] == config["matching_device"] == "cpu"
        and type(config["cpu_threads"]) is int and config["cpu_threads"] == 4 and config["probability_temperature"] == .05,
        "S013 requires this local GPU inventory and four-thread CPU metric computations")
    numeric = config["baseline_numerical_policy"]
    _require(isinstance(numeric, dict) and numeric.get("threshold_atol") == BASELINE_THRESHOLD_ATOL
        and numeric.get("probability_atol") == BASELINE_PROBABILITY_ATOL and numeric.get("relative_tolerance") == 0
        and bool(numeric.get("rationale")), "Historical baseline tolerances cannot be relaxed")
    _require(config["research_notes"] == RESEARCH_NOTE and config["local_readiness"] == READINESS
        and isinstance(config["selection_policy"], str) and bool(config["selection_policy"].strip()), "Required protocol/evidence identifiers differ")
    prior = config["s012_prerequisite"]
    _require(isinstance(prior, dict) and set(prior) == PREREQUISITE_KEYS, "S012 prerequisite fields differ")
    if not _prerequisite_ready(config):
        _require(all(value is None for value in prior.values()), "S012 prerequisite pins must be complete or wholly pending")
        _require(allow_pending, "S012 verification/report identities are not pinned; S013 execution is blocked")
    else:
        _require(isinstance(prior["run_path"], str)
            and re.fullmatch(r"artifacts/training/S012_\d{8}T\d{6}Z_[a-f0-9]{8}", prior["run_path"]) is not None
            and prior["verification_path"] == "artifacts/infrastructure/S012_verification/" + prior["run_path"].split("/")[-1] + "/verification.json",
            "S012 prerequisite paths must identify one completed original run")
        for name, length in (("parent_run_id", 32), ("git_commit", 40), ("verification_sha256", 64), ("report_sha256", 64)):
            _require(isinstance(prior[name], str) and re.fullmatch(r"[a-f0-9]{" + str(length) + "}", prior[name]) is not None,
                "Invalid completed S012 prerequisite identity")


def _checked_sources(root, config):
    checked = _historical_sources(root, config)
    prior = config["s012_prerequisite"]
    _require(_prerequisite_ready(config), "S012 independent verification is required before S013")
    audit_path = _path(root, prior["verification_path"])
    report_path = _path(root, prior["run_path"] + "/experiment_report.json")
    _require(file_sha256(audit_path) == prior["verification_sha256"]
        and file_sha256(report_path) == prior["report_sha256"], "Pinned S012 completion evidence changed")
    audit, report = [json.loads(path.read_text(encoding="utf-8")) for path in (audit_path, report_path)]
    _require(audit.get("status") == "verified" and report.get("status") == "complete"
        and audit.get("parent_run_id") == report.get("parent_run_id") == prior["parent_run_id"]
        and audit.get("git_commit") == prior["git_commit"] == report["source_binding"]["execution_git_commit"]
        and audit.get("run_path") == prior["run_path"] and audit.get("children") == report.get("children")
        and len(audit["children"]) == 6 and audit["results"]["baseline"]["macro_f1"] == BASELINE_MACRO_F1
        and report.get("encoder_updates") == 0 and report.get("source_arrays_unchanged") is True,
        "S012 has not completed and independently preserved the historical source")
    checked["paths"].update(s012_verification=audit_path, s012_report=report_path)
    return checked


def _supporting_inputs(root, config):
    paths = {"research_note": _path(root, config["research_notes"]), "research_manifest": _path(root, RESEARCH_MANIFEST),
        "local_readiness": _path(root, config["local_readiness"]), "protocol_implementation": _path(root, PROTOCOL_DOC),
        "metric_synthetic_validation": _path(root, SYNTHETIC_RECEIPT)}
    manifest = json.loads(paths["research_manifest"].read_text(encoding="utf-8"))
    matches = [entry for entry in manifest["outputs"] if entry["path"] == RESEARCH_NOTE]
    _require(len(matches) == 1 and matches[0]["sha256"] == file_sha256(paths["research_note"])
        and matches[0]["bytes"] == paths["research_note"].stat().st_size, "Metric research note differs from its manifest")
    ready = json.loads(paths["local_readiness"].read_text(encoding="utf-8"))
    _require(ready.get("status") == "verified" and ready.get("cuda_available") is True
        and ready.get("old_packages_changed_or_removed") == {}, "Local runtime readiness did not pass")
    tests = json.loads(paths["metric_synthetic_validation"].read_text(encoding="utf-8"))
    _require(tests.get("status") == "passed" and tests.get("real_project_data_used") is False,
        "Synthetic metric validation is required before a tracked run")
    files = tests.get("source_files", {})
    _require(files and all(file_sha256(_path(root, path)) == digest for path, digest in files.items()),
        "Metric validation receipt differs from tested source")
    return paths


def validate_inputs(root, config_path):
    root = Path(root).resolve()
    config = json.loads(_path(root, config_path).read_text(encoding="utf-8"))
    validate_config(config)
    _historical_sources(root, config)
    if not _prerequisite_ready(config):
        return {"status": "prepared_waiting_for_verified_S012_pins", "experiment": "S013", "execution_ready": False,
            "postprocessor_training_started": False, "encoder_updates": 0, "children": RECIPES}
    _checked_sources(root, config)
    supporting = _supporting_inputs(root, config)
    return {"status": "validated_no_experiment_started", "experiment": "S013", "execution_ready": True,
        "geometric_recipes": 6, "children": RECIPES, "metric_backend": "cpu", "encoder_updates": 0,
        "postprocessor_training_started": False, "supporting_evidence": {key: file_sha256(path) for key, path in supporting.items()}}


def _save_cases(directory, cases_by_metric, *, nested):
    """One NPZ and JSON archive per fold stage, with each metric payload indexed."""
    arrays, metadata = {}, {}
    for identifier, cases in cases_by_metric.items():
        for case in (cases if nested else [cases]):
            key = identifier + ("__meta" + str(case["meta_fold"]) if nested else "__full")
            payload = case["payload"]
            _validate_payload(payload)
            _require(identifier == payload["metadata"]["spec"]["id"], "Metric archive identifier differs from its fitted payload")
            arrays[key + "__mean"] = payload["mean"]
            dimension = payload["matrix"].shape[0]
            upper = payload["matrix"][np.triu_indices(dimension)]
            restored = _restore_symmetric_matrix(upper, dimension)
            _require(restored.tobytes() == payload["matrix"].tobytes(), "Metric matrix cannot be stored as an exact symmetric upper triangle")
            arrays[key + "__matrix_upper"] = upper
            # Transformed embeddings are reconstructable from sealed source +
            # mean/matrix; retain every actual score and row mapping instead.
            arrays.update({key + "__" + name: value for name, value in _array_fields(case).items()
                if name not in {"transformed_references", "transformed_queries"}})
            metadata[key + ".json"] = {"payload_metadata": payload["metadata"], "provenance": case["provenance"],
                "array_prefix": key + "__", "mean_array_key": key + "__mean", "matrix_upper_array_key": key + "__matrix_upper",
                "matrix_encoding": "symmetric_upper_triangle_including_diagonal", "matrix_dimension": dimension,
                "reconstructable_transformed_arrays": {name: {"shape": list(case[name].shape), "dtype": str(case[name].dtype),
                    "raw_array_sha256": hashlib.sha256(case[name].tobytes()).hexdigest()}
                    for name in ("transformed_references", "transformed_queries")}}
    path = directory / "metric_cases.npz"
    np.savez_compressed(path, **arrays)
    archive = _zip_json(directory / "metric_metadata.zip", metadata)
    return {"case_count": len(metadata), "arrays_file": path.name, "arrays_sha256": file_sha256(path),
        "metadata": archive, "format": "float64_mean_and_lossless_symmetric_matrix_upper_NPZ_finite_JSON_no_pickle",
        "transformed_embeddings": "reconstruct_from_sealed_fused_source_and_saved_metric_payload"}


def _restore_symmetric_matrix(upper, dimension):
    upper = np.asarray(upper)
    _require(type(dimension) is int and 0 < dimension <= 4096 and upper.dtype == np.float64
        and upper.shape == (dimension * (dimension + 1) // 2,) and np.isfinite(upper).all(),
        "Invalid lossless symmetric matrix encoding")
    indices = np.triu_indices(dimension)
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    matrix[indices] = upper
    matrix[(indices[1], indices[0])] = upper
    return matrix


def _save_search(output, outer, search, tracker, source_binding):
    directory = output / f"policy_search/fold_{outer}"
    directory.mkdir(parents=True, exist_ok=False)
    cases = _save_cases(directory, search["cases"], nested=True)
    _require(cases["case_count"] == 18, "Every geometric recipe needs three honest nested cases")
    arrays = {"source__" + name: value for name, value in search["source"].items()}
    rows = []
    for index, spec in enumerate(METRIC_SPECS):
        identifier = spec["id"]
        pooled = search["pooled_meta"][identifier]
        arrays.update({identifier + "__" + name: value for name, value in _array_fields(pooled).items()})
        for row in pooled["curves"]:
            rows.append([index, row["unknown_weight"], row["margin_weight"], row["threshold"], row["meta_macro_f1_447"], *row["meta_fold_macro_f1_447"]])
    np.savez_compressed(directory / "meta_score_arrays.npz", **arrays)
    np.savez_compressed(directory / "all_meta_curves.npz", values=np.asarray(rows, dtype=np.float64))
    write_json(directory / "selection.json", {"candidate_summary": search["candidate_summary"], "selection": search["selection"],
        "selection_sha256": search["selection_sha256"], "provenance": search["provenance"],
        "source_fingerprints": search["source_fingerprints"], "historical_baseline_binding": search["historical_baseline_binding"],
        "labels": search["labels"], "plan_provenance": search["plan"]["provenance"], "source_binding": source_binding,
        "cases": cases, "meta_score_arrays_sha256": file_sha256(directory / "meta_score_arrays.npz"),
        "curves_sha256": file_sha256(directory / "all_meta_curves.npz"), "curve_rows": len(rows),
        "curve_columns": ["metric_index", "unknown_weight", "margin_weight", "threshold", "meta_macro_f1_447",
            "fold0_meta_macro_f1_447", "fold1_meta_macro_f1_447", "fold2_meta_macro_f1_447"],
        "metric_ids": [spec["id"] for spec in METRIC_SPECS]})
    for path in directory.iterdir():
        tracker.add_artifact(path, f"policy_search/fold_{outer}/" + path.name)
    for candidate in search["candidate_summary"]:
        prefix = f"fold_{outer}/nested/{candidate['id']}/"
        tracker.log_metrics({prefix + "meta_macro_f1_447": candidate["calibration"]["meta_macro_f1_447"],
            prefix + "meta_gain_vs_identity": candidate["meta_gain_vs_identity"],
            prefix + "minimum_meta_fold_delta": min(candidate["meta_fold_delta_vs_identity"]),
            prefix + "promotion_eligible": int(candidate["promotion_eligible"]),
            prefix + "known_top1_accuracy": candidate["ranking"]["known_top1_accuracy"]}, sync=False)
    tracker.flush(strict=True)


def _save_outer(output, outer, result, tracker, source_binding):
    directory = output / f"outer_metrics/fold_{outer}"
    directory.mkdir(parents=True, exist_ok=False)
    cases = _save_cases(directory, result["full_reference_cases"], nested=False)
    _require(cases["case_count"] == 6, "All six full-reference geometry cases are required")
    report = {"cases": cases, "source_binding": source_binding, "selection_sha256": result["selection_sha256"],
        "selection": result["selection"], "original_outer_labels_read": result["original_outer_labels_read"],
        "outer_scores_computed_after_selection_freeze": result["outer_scores_computed_after_selection_freeze"],
        "historical_fallback_probability_bytes_preserved": result["historical_fallback_probability_bytes_preserved"],
        "metric_fit_backend": "numpy_float64_cpu", "matching_backend": "numpy_float32_cpu",
        "cuda_scoring_used": False, "CPU_CUDA_parity_claimed": False,
        "runtime_portability": "Saved NumPy mean/matrix need separate offline-package validation before promotion"}
    write_json(directory / "metric_manifest.json", report)
    for path in directory.iterdir():
        tracker.add_artifact(path, f"outer_metrics/fold_{outer}/" + path.name)
    tracker.flush(strict=True)
    return report


def _score_scope(recipe, result, baseline, search):
    retained = recipe == "baseline" or result.get("baseline_retained") is True
    identity = search["candidate_summary"][0]["calibration"]
    selected = None if retained else result["calibration"]
    return {"historical_baseline_retained": retained,
        "selected_geometry_meta_macro_f1_447": None if retained else selected["meta_macro_f1_447"],
        "selected_geometry_meta_fold_macro_f1_447": None if retained else selected["meta_fold_macro_f1_447"],
        "identity_control_meta_macro_f1_447": identity["meta_macro_f1_447"],
        "identity_control_meta_fold_macro_f1_447": identity["meta_fold_macro_f1_447"],
        "historical_full_pool_baseline_calibration": baseline["calibration"],
        "historical_full_pool_baseline_calibration_macro_f1_447": baseline["inner_macro_f1_447"],
        "score_scope": "A historical fallback has no selected geometric meta score; identity-control scores only explain the selection decision" if retained
            else "Nested geometry meta score optimized for gate/geometry selection; independent from reported outer OOF"}


def _fold_plot(directory, report, curves):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist([r["f1"] for r in report["outer"]["per_class"][1:]], bins=20, range=(0, 1))
    axes[0].set(xlabel="Known-speaker F1", ylabel="Speakers", title="Outer evaluation: 446 identities")
    if curves:
        axes[1].plot([r["threshold"] for r in curves], [r["meta_macro_f1_447"] for r in curves])
        axes[1].axvline(report["calibration"]["threshold"], color="orange", linestyle="--")
        axes[1].set(xlabel="Nested-selected gate threshold", ylabel="Nested meta Macro-F1 (447)", title="Geometric gate selection; no outer labels")
    else:
        axes[1].axis("off")
        axes[1].text(.05, .7, "Exact historical baseline retained\nIdentity geometry is a separate control")
    fig.savefig(directory / "evaluation.png", dpi=160)
    plt.close(fig)


def _persist_fold(record, contract, prepared, result, baseline, search, outer, manifest, check):
    directory = record["directory"] / f"fold_{outer}"
    directory.mkdir(exist_ok=False)
    predictions, metrics = _result_predictions(contract, prepared, result)
    scope = _score_scope(record["recipe"], result, baseline, search)
    reference, curves = None, []
    if not scope["historical_baseline_retained"]:
        identifier = result["policy"]["id"]
        prefix = identifier + "__full__"
        payload = result["payload"]
        reference = {"parent_arrays_artifact": f"outer_metrics/fold_{outer}/metric_cases.npz",
            "parent_metadata_artifact": f"outer_metrics/fold_{outer}/metric_metadata.zip", "metadata_member": identifier + "__full.json",
            "mean_array_key": prefix + "mean", "matrix_upper_array_key": prefix + "matrix_upper",
            "matrix_encoding": "symmetric_upper_triangle_including_diagonal", "matrix_dimension": payload["matrix"].shape[0],
            "mean_sha256": payload["metadata"]["mean_sha256"], "matrix_sha256": payload["metadata"]["matrix_sha256"],
            "arrays_sha256": manifest["cases"]["arrays_sha256"], "metadata_archive_sha256": manifest["cases"]["metadata"]["sha256"]}
        calibration = result["calibration"]
        curves = [r for r in search["pooled_meta"][identifier]["curves"] if all(r[k] == calibration[k] for k in ("unknown_weight", "margin_weight"))]
        write_json(directory / "selected_meta_curve.json", {"selected": calibration, "curve": curves})
    if record["recipe"] == "baseline":
        _zip_json(directory / "historical_baseline_calibration.zip", {"calibration.json": {"selected": baseline["calibration"],
            "curves": baseline["curves"], "alpha_candidates": baseline["baseline_alpha_candidates"]}})
    report = {"outer_fold": outer, "outer": metrics, "policy": result["policy"], "calibration": result["calibration"],
        **scope, "metric_reference": reference, "source_reproduction": check, "selection": search["selection"],
        "selection_sha256": search["selection_sha256"], "encoder_updates": 0,
        "metric_training_started": record["recipe"] not in {"baseline", "identity"},
        "metric_backend": "cpu", "cuda_scoring_used": False, "new_submission_built": False,
        "probability_semantics": "normalized decision scores, not calibrated posteriors"}
    indices = prepared["scores_by_alpha"][0.]["outer_indices"]
    report["outer_known_ranking"] = known_ranking_diagnostics([contract["manifest"][int(i)] for i in indices],
        result["scores"]["outer_known_scores"], result["probabilities"],
        prepared["scores_by_alpha"][0.]["outer_valid"], contract["labels"])
    write_json(directory / "evaluation.json", report)
    write_json(directory / "selected_policy.json", {"policy": result["policy"], "calibration": result["calibration"],
        "metric_reference": reference, "selection_sha256": search["selection_sha256"], **scope})
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "per_class.csv", metrics["per_class"])
    np.savez_compressed(directory / "outer_probabilities.npz", probabilities=result["probabilities"],
        audio_files=np.asarray([r["audio_file"] for r in predictions]), labels=np.asarray(contract["labels"]))
    _fold_plot(directory, report, curves)
    tracker = record["tracker"]
    for path in directory.iterdir():
        tracker.add_artifact(path, f"fold_{outer}/" + path.name)
    metric_values = {f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"],
        f"fold_{outer}/identity_control_meta_macro_f1_447": scope["identity_control_meta_macro_f1_447"],
        f"fold_{outer}/historical_baseline_calibration_macro_f1_447": scope["historical_full_pool_baseline_calibration_macro_f1_447"],
        f"fold_{outer}/historical_baseline_retained": int(scope["historical_baseline_retained"]),
        **{f"fold_{outer}/outer_{k}": v for k, v in metrics["errors"].items()}}
    metric_values.update({f"fold_{outer}/selected_{key}": result["calibration"][key]
        for key in ("threshold", "unknown_weight", "margin_weight")})
    metric_values[f"fold_{outer}/outer_known_rank1_accuracy"] = report["outer_known_ranking"]["rank1_accuracy"]
    if scope["selected_geometry_meta_macro_f1_447"] is not None:
        metric_values[f"fold_{outer}/selected_geometry_meta_macro_f1_447"] = scope["selected_geometry_meta_macro_f1_447"]
    tracker.log_metrics(metric_values, sync=True, strict=True)
    record["predictions"].extend(predictions)
    record["folds"].append(report)


def _finish_child(record, contract, baseline_predictions):
    recipe, tracker, directory = record["recipe"], record["tracker"], record["directory"]
    metrics = score_predictions(contract["manifest"], record["predictions"], contract["labels"])
    report = {"recipe": recipe, "oof": metrics, "folds": record["folds"],
        "paired_against_S008c": paired_diagnostics(contract["manifest"], baseline_predictions, record["predictions"], contract["labels"]),
        "oof_macro_f1_delta": metrics["macro_f1"] - BASELINE_MACRO_F1, "encoder_updates": 0,
        "metric_training_started": recipe not in {"baseline", "identity"}, "cuda_scoring_used": False,
        "comparison_role": "primary_prespecified_selector" if recipe == "overall" else "historical_baseline_control" if recipe == "baseline" else "exploratory_geometry",
        "limitations": LIMITATIONS, "new_submission_built": False}
    write_json(directory / "experiment_report.json", report)
    write_csv(directory / "oof_predictions.csv", record["predictions"])
    write_csv(directory / "oof_per_class.csv", metrics["per_class"])
    for name in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
        tracker.add_artifact(directory / name)
    tracker.log_metrics({"oof/macro_f1_447": metrics["macro_f1"], "oof/accuracy": metrics["accuracy"],
        "oof/macro_f1_delta_vs_S008c": report["oof_macro_f1_delta"], "encoder_updates": 0,
        **{"oof/" + k: v for k, v in metrics["errors"].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# S013 {recipe}\n\nOOF Macro-F1: {metrics['macro_f1']:.9f}; change from S008c: {report['oof_macro_f1_delta']:+.9f}.\n\nThree nested reference pools isolate every meta-validation group from mean/covariance fitting and scoring references. Unknown recordings do not contribute to fitted geometry. The identity geometry is a separate normalized control; exact historical fallback has no selected geometric meta score. Metric fitting and matching use CPU; encoder updates are zero.\n")
    tracker.finish("FINISHED", strict=True)
    return report


def execute_suite(root, config_path, binding_path):
    """Enforce four CPU BLAS threads, then execute only a completely pinned run."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        return _execute_suite(root, config_path, binding_path)


def _execute_suite(root, config_path, binding_path):
    from speaker_id.postprocessing.scoring import prepare_fold, select_baseline
    from speaker_id.postprocessing.metric_scoring import prepare_metric_search, evaluate_metric_outer
    root = Path(root).resolve()
    config_path, binding_path = _path(root, config_path), _path(root, binding_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, allow_pending=False)
    checked, supporting = _checked_sources(root, config), _supporting_inputs(root, config)
    source_state = git_provenance(root)
    _require(source_state["src_dirty"] is False and re.fullmatch(r"[a-f0-9]{40}", source_state["git_commit"] or ""),
        "Commit reviewed source before starting S013")
    inventory = _execution_environment(config)
    _require("1660 Ti" in inventory["device_name"], "S013 is limited to this local GTX 1660 Ti host")
    execution = {**inventory, "device": "cpu", "metric_fit_backend": "numpy_float64_cpu",
        "matching_backend": "numpy_float32_cpu", "cuda_scoring_used": False, "cpu_blas_threads": 4,
        "training_scope": "known_group_mean_and_within_speaker_covariance_only", "encoder_updates": 0}
    print(json.dumps({"stage": "source_verification", "status": "started"}), flush=True)
    sources = load_sources(root, checked["package"], verify_audio=False)
    contract = sources["contract"]
    _require(len(contract["manifest"]) == 4529 and len(contract["labels"]) == 447, "Original 4529-row/447-label source contract changed")
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    binding.validate()
    _require(binding.experiment_id == "1", "Use only the owned MLflow experiment 1")
    output = root / config["output_root"] / ("S013_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_binding = {"source_run": SOURCE_RUN, "parent_run_id": SOURCE_PARENT, "child_run_id": SOURCE_CHILD,
        "source_git_commit": SOURCE_COMMIT, "source_report_sha256": SOURCE_REPORT_SHA, "package_sha256": PACKAGE_SHA,
        "s008_verification_sha256": AUDIT_SHA, "s011_verification_sha256": S011_AUDIT_SHA,
        "completed_s012": config["s012_prerequisite"], "execution_git_commit": source_state["git_commit"],
        "suite_config_sha256": file_sha256(config_path)}
    write_json(output / "source_provenance.json", sources["proof"])
    array_receipt = _seal_sources(output, sources)
    source_binding["sealed_arrays_sha256"] = array_receipt["sha256"]
    resolved = {"suite": config, "source_binding": source_binding, "execution": execution,
        "data_input_hashes": contract["input_hashes"], "limitations": LIMITATIONS,
        "finite_search": {"geometries": 6, "outer_folds": 2, "nested_folds": 3, "nested_transform_fits_per_outer": 18,
            "full_reference_transform_fits_per_outer": 6, "unknown_weights": 5, "margin_weights": 2,
            "threshold_quantiles": 201, "children": 8, "total_runs": 9},
        "supporting_evidence": {name: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)} for name, path in supporting.items()}}
    write_json(output / "resolved_config.json", resolved)
    input_paths = {"suite_config": config_path, "launcher": root / "scripts/score_metrics.py", **checked["paths"], **supporting,
        **{key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")}}
    common = {"project_root": root, "binding": binding, "input_paths": input_paths,
        "run_kind": "local_nested_known_group_metric_learning", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=config["run_name"], config=resolved, **common)
    records, active, fit_started = {}, [], False
    try:
        for name in ("source_provenance.json", "verified_source_arrays.json", "verified_source_arrays.npz"):
            parent.add_artifact(output / name)
        for name, path in input_paths.items():
            parent.add_artifact(path, "input_evidence/" + name + path.suffix)
        parent.flush(strict=True)
        write_json(output / "initial_tracking_verification.json", {"artifacts": parent.verify_artifacts(), "metadata": parent.verify_remote_metadata()})
        parent.add_artifact(output / "initial_tracking_verification.json")
        parent.flush(strict=True)
        print(json.dumps({"stage": "source_verification", "status": "passed", "output": str(output), "parent_run_id": parent.run_id}), flush=True)
        prepared, baseline, checks, baseline_predictions = {}, {}, {}, []
        for outer in (0, 1):
            print(json.dumps({"stage": "baseline_reproduction", "fold": outer}), flush=True)
            prepared[outer] = prepare_fold(sources["vectors"]["public"], sources["vectors"]["advanced"], sources["valid"],
                contract["manifest"], contract["folds"], contract["labels"], outer)
            baseline[outer] = select_baseline(prepared[outer])
            checks[str(outer)] = _verify_baseline_fold(root, contract, prepared[outer], baseline[outer], outer)
            baseline_predictions.extend(_result_predictions(contract, prepared[outer], baseline[outer])[0])
            parent.log_metrics({"baseline/completed_verified_folds": outer + 1}, step=outer, sync=True, strict=True)
        pooled = score_predictions(contract["manifest"], baseline_predictions, contract["labels"])
        _require(pooled == checked["report"]["oof"] and pooled["macro_f1"] == BASELINE_MACRO_F1, "Both baseline folds must reproduce all pooled metrics")
        checks["pooled"] = {**verify_predictions(root / SOURCE_RUN / "S008c/oof_predictions.csv", baseline_predictions),
            "exact_pooled_metrics": True, "macro_f1": BASELINE_MACRO_F1}
        write_json(output / "baseline_control_checks.json", checks)
        parent.add_artifact(output / "baseline_control_checks.json")
        parent.flush(strict=True)
        for recipe in RECIPES:
            directory = output / recipe
            directory.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=directory / "tracking", parent_run_id=parent.run_id,
                run_name="S013-" + recipe, config={**resolved, "recipe": recipe}, **common)
            active.append(child)
            child.add_artifact(output / "baseline_control_checks.json")
            child.flush(strict=True)
            records[recipe] = {"recipe": recipe, "directory": directory, "tracker": child, "predictions": [], "folds": []}
        for outer in (0, 1):
            sequence, last_flush = [0], [time.monotonic()]
            def progress(info):
                sequence[0] += 1
                event = {"outer_fold": outer, **info}
                print(json.dumps(event), flush=True)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
                write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
                    "metric_training_started": fit_started, "encoder_updates": 0, "latest_progress": event})
                numeric = {f"fold_{outer}/progress/{k}": v for k, v in info.items() if type(v) in (int, float) and math.isfinite(v)}
                if numeric:
                    parent.log_metrics(numeric, step=sequence[0], sync=False)
                if sequence[0] % 4 == 0 or time.monotonic() - last_flush[0] >= 30:
                    parent.add_artifact(output / "progress.jsonl")
                    parent.flush(strict=True)
                    last_flush[0] = time.monotonic()
            if not fit_started:
                for tracker in [parent, *[records[r]["tracker"] for r in RECIPES if r not in {"baseline", "identity"}]]:
                    _mark_training_started(tracker)
                fit_started = True
            progress({"stage": "nested_metric_fitting", "status": "started"})
            search = prepare_metric_search(prepared[outer], baseline[outer], sources["vectors"]["public"], sources["vectors"]["advanced"],
                sources["valid"], contract["manifest"], contract["folds"], contract["labels"], outer, on_progress=progress)
            _save_search(output, outer, search, parent, source_binding)
            # The selection, all nested scores and every fit payload are durable
            # before any original outer geometry is fitted or evaluated.
            result = evaluate_metric_outer(search, baseline[outer], on_progress=progress)
            manifest = _save_outer(output, outer, result, parent, source_binding)
            for recipe, value in {"baseline": baseline[outer], **result["results"], "overall": result["overall"]}.items():
                _persist_fold(records[recipe], contract, prepared[outer], value, baseline[outer], search, outer, manifest, checks[str(outer)])
        reports = {}
        for recipe in RECIPES:
            reports[recipe] = _finish_child(records[recipe], contract, baseline_predictions)
            active.remove(records[recipe]["tracker"])
        for name, array in {**sources["vectors"], "valid": sources["valid"]}.items():
            _require(hashlib.sha256(array.tobytes()).hexdigest() == array_receipt["arrays"][name]["array_sha256"], "An original source cache was mutated")
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": reports,
            "children": {r: records[r]["tracker"].run_id for r in RECIPES}, "baseline_control_checks": checks,
            "source_binding": source_binding, "source_arrays_unchanged": True, "encoder_updates": 0,
            "metric_training_started": fit_started, "metric_backend": "cpu", "cuda_scoring_used": False,
            "finite_search": resolved["finite_search"], "elapsed_seconds": time.monotonic() - started,
            "selection_policy": config["selection_policy"], "limitations": LIMITATIONS, "new_submission_built": False}
        write_json(output / "experiment_report.json", report)
        for name in ("experiment_report.json", "resolved_config.json", "progress.jsonl"):
            parent.add_artifact(output / name)
        for recipe, item in reports.items():
            parent.log_metrics({recipe + "/oof_macro_f1_447": item["oof"]["macro_f1"], recipe + "/oof_delta": item["oof_macro_f1_delta"]}, sync=False)
        parent.write_report(report, markdown="# S013 frozen-encoder metric learning\n\nBoth original S008c folds reproduced all predictions and metrics before any mean/covariance fitting. Six preregistered geometries use fixed alpha 0.5 and three nested galleries. The primary selector requires pooled gain at least 0.001 over the normalized identity control and no meta-fold loss above 0.002; otherwise exact S008c is retained. All geometry/gate selection is frozen before outer evaluation. CPU fitting and cosine matching are explicit; there are no encoder updates or CUDA scoring claims.\n\n" + "\n".join(f"- {r}: OOF {v['oof']['macro_f1']:.9f}; delta {v['oof_macro_f1_delta']:+.9f}." for r, v in reports.items()) + "\n")
        parent.finish("FINISHED", strict=True)
        roundtrip = {}
        for recipe, tracker in [(r, records[r]["tracker"]) for r in RECIPES] + [("parent", parent)]:
            roundtrip[recipe] = {"artifacts": tracker.verify_artifacts(), "metadata": tracker.verify_remote_metadata()}
        write_json(output / "tracking_roundtrip_verification.json", {"status": "passed", "runs": roundtrip})
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id,
            "children": report["children"], "all_mlflow_finished_and_verified": True, "metric_training_started": True,
            "encoder_updates": 0, "git_commit": source_state["git_commit"]})
        return {"output": str(output), "parent_run_id": parent.run_id, "results": {r: v["oof"]["macro_f1"] for r, v in reports.items()},
            "all_mlflow_finished_and_verified": True, "new_submission_built": False}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id, "error_type": type(error).__name__,
            "error": parent.redactor.text(str(error)), "metric_training_started": fit_started,
            "encoder_updates": 0, "elapsed_seconds": time.monotonic() - started}
        write_json(output / "failure.json", failure)
        for tracker in active:
            tracker.write_report(failure)
            tracker.finish("FAILED", strict=False)
        if parent.state.get("remote_status") != "FINISHED":
            parent.write_report(failure)
            parent.finish("FAILED", strict=False)
        else:
            failure["computation_finished_tracking_verification_failed"] = True
        write_json(output / "experiment_state.json", failure)
        raise
