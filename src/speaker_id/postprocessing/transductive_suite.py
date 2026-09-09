"""Tracked, leakage-controlled S014 evaluation of one fixed batch decoder."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.packaging.selected_sources import load_sources
from speaker_id.postprocessing.suite import (
    PACKAGE_PATH, PACKAGE_SHA, AUDIT_PATH, AUDIT_SHA, SOURCE_RUN,
    SOURCE_PARENT, SOURCE_CHILD, SOURCE_COMMIT, SOURCE_REPORT_SHA,
    BASELINE_MACRO_F1, BASELINE_THRESHOLD_ATOL, BASELINE_PROBABILITY_ATOL,
    _execution_environment, _path, _require, _result_predictions,
    _seal_sources, _source_checks, _verify_baseline_fold,
)
from speaker_id.postprocessing.transductive_scoring import (
    BOOTSTRAP_LOWER_QUANTILE, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED,
    FIXED_ALIGNMENT_CONFIG, FIXED_POLICY, PROMOTION_RULE,
    align_probabilities, load_selection, promotion_decision, seal_fold, seal_selection,
    speaker_group_bootstrap,
)
from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
from speaker_id.tracking.snapshot import git_provenance
from speaker_id.training.adaptation_comparison import paired_diagnostics
from speaker_id.training.fusion_suite import verify_predictions
from speaker_id.training.runner import write_csv, write_json


RECIPES = ("baseline", "powered_prior")
PROTOCOL_DOC = "docs/s014_protocol.fa.md"
S013_RUN = "artifacts/training/S013_20260908T154916Z_0c36060b"
S013_PARENT = "45759a6510ee4783863511c1145d0a9b"
S013_COMMIT = "9d72ffb30393e4c3a32327d327bbdfef1eb01808"
S013_VERIFICATION = "artifacts/infrastructure/S013_verification/S013_20260908T154916Z_0c36060b/verification.json"
S013_VERIFICATION_SHA = "028c45e8486c317aba5c5ed0f8d65402c2a77d0dad1731384287e7a9f69cd077"
S013_REPORT_SHA = "e6e55e00ebdcafb39be040dc41f9a644666c03607e43a7a9786cff296c9898bd"
LIMITATIONS = [
    "The fixed S014 hyperparameters were chosen after exploratory inspection of these same development OOF folds.",
    "Repeated OOF is an engineering check, not independent confirmation or an unbiased hidden-leaderboard estimate.",
    "The 0.5 unknown and uniform-known design prior is not guaranteed by the organizer for the hidden batch.",
    "The method consumes only the unlabeled probability batch and validity mask; unknown recordings are never one speaker cluster.",
    "The bootstrap resamples content groups within true-class strata and does not preserve dependence of cross-label duplicate groups.",
    "No encoder update, raw-audio read, threshold search, query graph, ECAPA, WCCN, package build or leaderboard submission occurs.",
]


def _array_sha(value):
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _expected_prior_context():
    return {
        "run_path": S013_RUN,
        "parent_run_id": S013_PARENT,
        "execution_git_commit": S013_COMMIT,
        "verification_path": S013_VERIFICATION,
        "verification_sha256": S013_VERIFICATION_SHA,
        "report_sha256": S013_REPORT_SHA,
    }


def _require_clean_worktree(root):
    """Bind every tracked input and launcher to one clean execution commit."""
    environment = dict(os.environ)
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1",
             "--untracked-files=normal"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=environment, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Unable to verify the S014 Git worktree") from error
    _require(not result.stdout.strip(),
             "Commit every reviewed S014 input before execution")
    return True


def _without_outer_speaker_ids(manifest, folds, outer):
    """Remove every own-fold truth label from the pre-evaluation manifest."""
    by_name = {row["audio_file"]: int(row["fold"]) for row in folds}
    masked, removed = [], 0
    for row in manifest:
        item = dict(row)
        if by_name[item["audio_file"]] == outer:
            _require("speaker_id" in item,
                     "An outer row is missing its source speaker label")
            item.pop("speaker_id")
            removed += 1
        masked.append(item)
    _require(removed > 0 and all(
        ("speaker_id" not in row) == (by_name[row["audio_file"]] == outer)
        for row in masked), "Own-fold truth masking failed")
    return masked, {
        "outer_fold": outer,
        "own_fold_speaker_id_fields_removed": removed,
        "own_fold_speaker_ids_supplied_to_reconstruction": False,
    }


def _verify_reconstructed_alignment(root, config, contract, baseline, outer):
    """Prove Windows reconstruction preserves the pinned S014 decisions."""
    control = config["reconstruction_control"]
    entry = control["folds"][str(outer)]
    path = _path(root, entry["path"])
    _require(file_sha256(path) == entry["sha256"],
             "Pinned S008c probability control bytes changed")
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"probabilities", "audio_files", "labels"},
                 "Pinned S008c probability archive fields changed")
        historical = np.array(saved["probabilities"], copy=True)
        historical_files = tuple(saved["audio_files"].astype(str).tolist())
        historical_labels = tuple(saved["labels"].astype(str).tolist())
    scores = baseline["scores"]
    indices = np.asarray(scores["outer_indices"], dtype=np.int64)
    rebuilt_files = tuple(
        contract["manifest"][int(index)]["audio_file"] for index in indices)
    rebuilt = np.asarray(baseline["probabilities"])
    _require(historical.shape == rebuilt.shape
             and historical_files == rebuilt_files
             and historical_labels == tuple(contract["labels"]),
             "Reconstructed S014 row or label identity changed")
    maximum = float(np.max(np.abs(historical - rebuilt)))
    _require(np.allclose(
        historical, rebuilt, rtol=0,
        atol=control["maximum_probability_absolute_difference"]),
        "Windows probability drift exceeds the pinned S014 tolerance")
    historical_result = align_probabilities(
        historical, scores["outer_valid"])["adjusted_probabilities"]
    rebuilt_result = align_probabilities(
        rebuilt, scores["outer_valid"])["adjusted_probabilities"]
    historical_prediction = historical_result.argmax(axis=1)
    rebuilt_prediction = rebuilt_result.argmax(axis=1)
    digest = entry["powered_prediction_array_sha256"]
    _require(_array_sha(historical_prediction) == digest
             and _array_sha(rebuilt_prediction) == digest
             and np.array_equal(historical_prediction, rebuilt_prediction),
             "Windows drift changed a pinned powered-prior decision")
    return {
        "outer_fold": outer,
        "historical_probability_path": entry["path"],
        "historical_probability_sha256": entry["sha256"],
        "maximum_probability_absolute_difference": maximum,
        "probability_atol":
            control["maximum_probability_absolute_difference"],
        "powered_prediction_array_sha256": digest,
        "exact_powered_predictions": True,
        "rows": len(rebuilt_prediction),
    }


def _memory_snapshot():
    """Return host-memory evidence when the tracking environment exposes it."""
    try:
        import psutil
    except ImportError:
        return {"measurement_available": False}
    process = psutil.Process()
    info = process.memory_info()
    return {
        "measurement_available": True,
        "resident_memory_bytes": info.rss,
        "peak_resident_memory_bytes":
            int(getattr(info, "peak_wset", info.rss)),
        "available_host_memory_bytes": psutil.virtual_memory().available,
    }


def validate_config(config):
    required = {
        "schema_version", "experiment_code", "run_name", "output_root",
        "source_package_config", "source_package_config_sha256",
        "source_verification", "source_verification_sha256", "prior_context",
        "device", "alignment_device", "cpu_threads", "probability_temperature",
        "alignment", "reconstruction_control", "bootstrap", "promotion",
        "baseline_numerical_policy",
        "selection_policy", "research_notes",
    }
    _require(isinstance(config, dict) and set(config) == required,
             "S014 configuration fields changed")
    _require(config["schema_version"] == 1 and config["experiment_code"] == "S014"
             and config["run_name"] == "S014-campp-powered-design-prior-alignment"
             and config["output_root"] == "artifacts/training",
             "Unexpected S014 experiment identity")
    _require(config["source_package_config"] == PACKAGE_PATH
             and config["source_package_config_sha256"] == PACKAGE_SHA
             and config["source_verification"] == AUDIT_PATH
             and config["source_verification_sha256"] == AUDIT_SHA,
             "S014 requires the exact P002/S008c source")
    _require(config["prior_context"] == _expected_prior_context(),
             "S013 context binding changed")
    _require(config["device"] == "cuda" and config["alignment_device"] == "cpu"
             and type(config["cpu_threads"]) is int and config["cpu_threads"] == 4
             and config["probability_temperature"] == .05,
             "S014 is fixed to the local CUDA inventory and CPU alignment")
    _require(config["alignment"] == {
        "schema_version": FIXED_ALIGNMENT_CONFIG.schema_version,
        "design_unknown_prior": FIXED_ALIGNMENT_CONFIG.design_unknown_prior,
        "design_known_prior": FIXED_ALIGNMENT_CONFIG.design_known_prior,
        "unknown_strength": FIXED_ALIGNMENT_CONFIG.unknown_strength,
        "known_strength": FIXED_ALIGNMENT_CONFIG.known_strength,
        "epsilon": FIXED_ALIGNMENT_CONFIG.epsilon,
    }, "The fixed S014 powered-ratio formula changed")
    expected_reconstruction = {
        "folds": {
            "0": {
                "path": SOURCE_RUN + "/S008c/fold_0/outer_probabilities.npz",
                "sha256": "bdb53df3792e9305b9a9eac6dd1b0f335a597b26b96a61ccd90ae5760685600a",
                "powered_prediction_array_sha256":
                    "b188dcf480abac70508766d12ef95d5b3264e20dcfb0f19e3e4b595abff404c1",
            },
            "1": {
                "path": SOURCE_RUN + "/S008c/fold_1/outer_probabilities.npz",
                "sha256": "d08a211d42df193af367019859ccbd8792a8d41f079a576a0591006e069fcfdc",
                "powered_prediction_array_sha256":
                    "e63b5c8d5757691c9bbf717f1c895c605c0bcdea1f2bcc122545760476daa209",
            },
        },
        "maximum_probability_absolute_difference": BASELINE_PROBABILITY_ATOL,
        "require_exact_powered_predictions": True,
    }
    _require(config["reconstruction_control"] == expected_reconstruction,
             "The S014 Windows reconstruction control changed")
    _require(config["bootstrap"] == {
        "kind": "true_class_stratified_within_label_content_group_paired",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_SAMPLES,
        "lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
        "upper_quantile": 1.0 - BOOTSTRAP_LOWER_QUANTILE,
    }, "The fixed S014 bootstrap changed")
    expected_promotion = {
        "minimum_pooled_macro_f1_gain": PROMOTION_RULE["minimum_pooled_macro_f1_gain"],
        "minimum_each_fold_macro_f1_gain": PROMOTION_RULE["minimum_each_fold_macro_f1_gain"],
        "minimum_pooled_accuracy_gain": PROMOTION_RULE["minimum_accuracy_gain"],
        "maximum_unknown_to_known_increase": PROMOTION_RULE["maximum_unknown_to_known_error_increase"],
        "maximum_known_to_other_known_increase": PROMOTION_RULE["maximum_known_to_other_known_error_increase"],
        "minimum_bootstrap_lower_bound": PROMOTION_RULE["minimum_bootstrap_lower_gain"],
        "require_cpu_cuda_prediction_parity": True,
    }
    _require(config["promotion"] == expected_promotion,
             "S014 promotion gates changed")
    numeric = config["baseline_numerical_policy"]
    _require(isinstance(numeric, dict)
             and numeric.get("threshold_atol") == BASELINE_THRESHOLD_ATOL
             and numeric.get("probability_atol") == BASELINE_PROBABILITY_ATOL
             and numeric.get("relative_tolerance") == 0.0
             and bool(numeric.get("rationale")),
             "S008c numerical control cannot be relaxed")
    _require(config["research_notes"] == PROTOCOL_DOC
             and isinstance(config["selection_policy"], str)
             and bool(config["selection_policy"].strip()),
             "S014 protocol documentation is incomplete")


def _check_prior_context(root, config):
    verification = _path(root, config["prior_context"]["verification_path"])
    report_path = _path(root, config["prior_context"]["run_path"] + "/experiment_report.json")
    _require(file_sha256(verification) == S013_VERIFICATION_SHA
             and file_sha256(report_path) == S013_REPORT_SHA,
             "S013 verification or report bytes changed")
    audit = json.loads(verification.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(audit.get("status") == "passed"
             and audit.get("all_mlflow_finished_and_verified") is True
             and audit.get("parent_run_id") == report.get("parent_run_id") == S013_PARENT
             and audit.get("execution_git_commit") == S013_COMMIT
             and report.get("status") == "complete"
             and report.get("encoder_updates") == 0
             and report.get("source_arrays_unchanged") is True,
             "S013 context is not a completed, verified immutable-source run")
    return {"s013_verification": verification, "s013_report": report_path}


def validate_inputs(root, config_path):
    root = Path(root).resolve()
    config_path = _path(root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    checked = _source_checks(root, config)
    context = _check_prior_context(root, config)
    protocol = _path(root, config["research_notes"])
    reconstruction_hashes = {}
    for outer in (0, 1):
        entry = config["reconstruction_control"]["folds"][str(outer)]
        path = _path(root, entry["path"])
        digest = file_sha256(path)
        _require(digest == entry["sha256"],
                 "A pinned reconstruction-control cache changed")
        reconstruction_hashes[str(outer)] = digest
    return {
        "status": "validated_no_experiment_started",
        "experiment": "S014",
        "metadata_ready": True,
        "execution_preconditions_checked": False,
        "candidate_count": 1,
        "policy": FIXED_POLICY,
        "source_package_sha256": file_sha256(checked["paths"]["source_package_config"]),
        "source_verification_sha256": file_sha256(checked["paths"]["source_verification"]),
        "s013_verification_sha256": file_sha256(context["s013_verification"]),
        "protocol_sha256": file_sha256(protocol),
        "reconstruction_control_sha256": reconstruction_hashes,
        "encoder_updates": 0,
        "raw_audio_read": False,
    }


def _cuda_policy_parity(loaded):
    """Recompute the fixed arithmetic on CUDA from the sealed CPU inputs."""
    import torch
    base = np.array(loaded["base_probabilities"], dtype=np.float64, copy=True)
    factors = np.array(loaded["factors"], dtype=np.float64, copy=True)
    valid = np.array(loaded["valid"], dtype=np.bool_, copy=True)
    with torch.inference_mode():
        values = torch.as_tensor(base, dtype=torch.float64, device="cuda")
        scale = torch.as_tensor(factors, dtype=torch.float64, device="cuda")
        mask = torch.as_tensor(valid, dtype=torch.bool, device="cuda")
        adjusted = values.clone()
        weighted = values[mask] * scale[None, :]
        adjusted[mask] = weighted / weighted.sum(dim=1, keepdim=True)
        adjusted[~mask] = 0
        adjusted[~mask, 0] = 1
        cuda = adjusted.cpu().numpy()
    cpu = np.asarray(loaded["adjusted_probabilities"])
    return {
        "exact_predictions": bool(np.array_equal(cpu.argmax(axis=1), cuda.argmax(axis=1))),
        "maximum_probability_absolute_difference": float(np.max(np.abs(cpu - cuda))),
        "rows": len(cpu),
        "classes": cpu.shape[1],
        "cpu_dtype": str(cpu.dtype),
        "cuda_dtype": "torch.float64",
    }



def _persist_fold(directory, tracker, outer, metrics, predictions, report):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "evaluation.json", report)
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "per_class.csv", metrics["per_class"])
    for name in ("evaluation.json", "predictions.csv", "per_class.csv"):
        tracker.add_artifact(directory / name, f"fold_{outer}/{name}")


def _plots(output, baseline, adjusted, bootstrap_deltas):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ["P002", "S014"]
    axes[0].bar(names, [baseline["macro_f1"], adjusted["macro_f1"]],
                color=["#6b7280", "#166a77"])
    axes[0].set_ylim(min(baseline["macro_f1"], adjusted["macro_f1"]) - .004,
                     max(baseline["macro_f1"], adjusted["macro_f1"]) + .002)
    axes[0].set_ylabel("Macro-F1 (447 labels)")
    axes[0].set_title("Pooled OOF")
    errors = ("known_to_unknown", "unknown_to_known", "known_to_other_known")
    x = np.arange(len(errors))
    axes[1].bar(x - .18, [baseline["errors"][key] for key in errors], .36,
                label="P002", color="#6b7280")
    axes[1].bar(x + .18, [adjusted["errors"][key] for key in errors], .36,
                label="S014", color="#166a77")
    axes[1].set_xticks(x, ["K→U", "U→K", "K→K"])
    axes[1].set_ylabel("Errors")
    axes[1].set_title("Error directions")
    axes[1].legend()
    path = output / "s014_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(bootstrap_deltas, bins=50, color="#166a77", edgecolor="white")
    ax.axvline(0, color="#b45c37", linestyle="--", label="No change")
    ax.set(xlabel="Paired Macro-F1 gain", ylabel="Replicates",
           title="Speaker/content-group stratified bootstrap")
    ax.legend()
    path = output / "s014_bootstrap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _finish_child(record, contract, paired, promotion=None):
    tracker = record["tracker"]
    metrics = score_predictions(contract["manifest"], record["predictions"],
                                contract["labels"])
    report = {
        "recipe": record["recipe"],
        "oof": metrics,
        "folds": record["folds"],
        "paired_against_S008c": paired,
        "oof_macro_f1_delta": metrics["macro_f1"] - BASELINE_MACRO_F1,
        "promotion": promotion,
        "encoder_updates": 0,
        "raw_audio_read": False,
        "batch_alignment_executed": record["recipe"] == "powered_prior",
        "limitations": LIMITATIONS,
    }
    write_json(record["directory"] / "experiment_report.json", report)
    write_csv(record["directory"] / "oof_predictions.csv", record["predictions"])
    write_csv(record["directory"] / "oof_per_class.csv", metrics["per_class"])
    for name in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
        tracker.add_artifact(record["directory"] / name)
    tracker.log_metrics({
        "oof/macro_f1_447": metrics["macro_f1"],
        "oof/accuracy": metrics["accuracy"],
        "oof/macro_f1_delta_vs_S008c": report["oof_macro_f1_delta"],
        "oof/known_to_unknown": metrics["errors"]["known_to_unknown"],
        "oof/unknown_to_known": metrics["errors"]["unknown_to_known"],
        "oof/known_to_other_known": metrics["errors"]["known_to_other_known"],
        "encoder_updates": 0,
    }, sync=False)
    tracker.write_report(
        report,
        markdown=(
            f"# S014 {record['recipe']}\n\n"
            f"OOF Macro-F1: {metrics['macro_f1']:.9f}; "
            f"change from S008c: {report['oof_macro_f1_delta']:+.9f}.\n\n"
            "The fixed unlabeled batch output was sealed before outer "
            "evaluation. No encoder update or raw-audio read occurred.\n"
        ),
    )
    tracker.finish("FINISHED", strict=True)
    return report


def execute_suite(root, config_path, binding_path):
    from speaker_id.postprocessing.scoring import prepare_fold, select_baseline

    root = Path(root).resolve()
    config_path, binding_path = _path(root, config_path), _path(root, binding_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    checked = _source_checks(root, config)
    context = _check_prior_context(root, config)
    protocol = _path(root, config["research_notes"])

    source_state = git_provenance(root)
    _require(source_state["src_dirty"] is False
             and re.fullmatch(r"[a-f0-9]{40}", source_state["git_commit"] or ""),
             "Commit the reviewed S014 source before execution")
    source_state["git_worktree_clean"] = _require_clean_worktree(root)
    execution = {
        **_execution_environment(config),
        "alignment_device": "cpu",
        "alignment_dtype": "float64",
        "encoder_updates": 0,
        "raw_audio_read": False,
        "encoder_forward_calls": 0,
        "training_started": False,
        "transductive_parameters_fitted": "valid_batch_column_means_only",
    }
    _require("1660 Ti" in execution["device_name"],
             "S014 is authorized only on this local GTX 1660 Ti")

    print(json.dumps({"stage": "source_verification", "status": "started"}),
          flush=True)
    sources = load_sources(root, checked["package"], verify_audio=False)
    contract = sources["contract"]
    _require(len(contract["manifest"]) == 4529
             and len(contract["labels"]) == 447
             and set(sources["vectors"]) == {"public", "advanced"},
             "The immutable P002 source contract changed")
    binding = ExperimentBinding(
        **json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    binding.validate()
    _require(binding.experiment_id == "1",
             "Use only the owned project MLflow experiment 1")

    # Complete the largest deterministic reconstruction before creating any
    # remote run. This keeps an environment/RAM failure out of MLflow and lets
    # the successfully prepared objects flow unchanged into the sealed stage.
    preflight_started = time.monotonic()
    prepared, baseline, reconstruction = {}, {}, {}
    for outer in (0, 1):
        before = time.monotonic()
        masked_manifest, truth_isolation = _without_outer_speaker_ids(
            contract["manifest"], contract["folds"], outer)
        prepared[outer] = prepare_fold(
            sources["vectors"]["public"], sources["vectors"]["advanced"],
            sources["valid"], masked_manifest, contract["folds"],
            contract["labels"], outer)
        baseline[outer] = select_baseline(prepared[outer])
        reconstruction[str(outer)] = {
            "seconds": time.monotonic() - before,
            "rows": len(baseline[outer]["probabilities"]),
            "memory_after_fold": _memory_snapshot(),
            "selected_advanced_weight":
                baseline[outer]["policy"]["advanced_weight"],
            "truth_isolation": truth_isolation,
            "historical_decision_control":
                _verify_reconstructed_alignment(
                    root, config, contract, baseline[outer], outer),
        }
    local_preflight = {
        "status": "passed_before_mlflow_run_creation",
        "elapsed_seconds": time.monotonic() - preflight_started,
        "memory_at_completion": _memory_snapshot(),
        "fold_reconstruction": reconstruction,
        "remote_run_created": False,
    }
    print(json.dumps({"stage": "local_preflight", **local_preflight}),
          flush=True)

    output = root / config["output_root"] / (
        "S014_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_binding = {
        "source_run": SOURCE_RUN,
        "parent_run_id": SOURCE_PARENT,
        "child_run_id": SOURCE_CHILD,
        "source_git_commit": SOURCE_COMMIT,
        "source_report_sha256": SOURCE_REPORT_SHA,
        "package_sha256": PACKAGE_SHA,
        "s008_verification_sha256": AUDIT_SHA,
        "s013_parent_run_id": S013_PARENT,
        "s013_verification_sha256": S013_VERIFICATION_SHA,
        "s013_report_sha256": S013_REPORT_SHA,
        "execution_git_commit": source_state["git_commit"],
        "suite_config_sha256": file_sha256(config_path),
    }
    resolved = {
        "suite": config,
        "source_binding": source_binding,
        "execution": execution,
        "data_input_hashes": contract["input_hashes"],
        "fixed_policy": FIXED_POLICY,
        "finite_search": {
            "outer_folds": 2,
            "new_candidates": 1,
            "hyperparameter_search": 0,
            "query_graph_candidates": 0,
            "encoder_updates": 0,
        },
        "limitations": LIMITATIONS,
        "local_preflight": local_preflight,
        "supporting_evidence": {
            "protocol": {"path": PROTOCOL_DOC, "sha256": file_sha256(protocol)},
            "s013_verification": {"path": S013_VERIFICATION,
                                  "sha256": file_sha256(context["s013_verification"])},
            "s013_report": {"path": S013_RUN + "/experiment_report.json",
                            "sha256": file_sha256(context["s013_report"])},
        },
    }
    write_json(output / "source_provenance.json", sources["proof"])
    source_receipt = _seal_sources(output, sources)
    source_binding["sealed_arrays_sha256"] = source_receipt["sha256"]
    write_json(output / "resolved_config.json", resolved)

    input_paths = {
        "suite_config": config_path,
        "launcher": root / "scripts/score_transductive.py",
        "protocol": protocol,
        **checked["paths"],
        **context,
        **{key: root / contract["config"][key]
           for key in ("manifest", "folds", "roles", "label_map", "model_config")},
        **{
            f"historical_probabilities_fold_{outer}":
                _path(root, config["reconstruction_control"]["folds"][str(outer)]["path"])
            for outer in (0, 1)
        },
    }
    common = {
        "project_root": root,
        "binding": binding,
        "input_paths": input_paths,
        "run_kind": "local_label_free_batch_alignment",
        "training_started": False,
    }
    parent = DurableMLflowRun.prepare(
        spool_dir=output / "tracking", run_name=config["run_name"],
        config=resolved, **common)
    active = []
    try:
        for name in ("source_provenance.json", "verified_source_arrays.json",
                     "verified_source_arrays.npz", "resolved_config.json"):
            parent.add_artifact(output / name)
        for name, path in input_paths.items():
            parent.add_artifact(path, "input_evidence/" + name + path.suffix)
        parent.flush(strict=True)
        initial = {
            "artifacts": parent.verify_artifacts(),
            "metadata": parent.verify_remote_metadata(),
        }
        write_json(output / "initial_tracking_verification.json", initial)
        parent.add_artifact(output / "initial_tracking_verification.json")
        parent.flush(strict=True)
        print(json.dumps({
            "stage": "source_verification", "status": "passed",
            "output": str(output), "parent_run_id": parent.run_id,
        }), flush=True)

        build_times = {}
        seals_root = output / "sealed_alignment"
        for outer in (0, 1):
            scores = baseline[outer]["scores"]
            indices = np.asarray(scores["outer_indices"], dtype=np.int64)
            files = [contract["manifest"][int(i)]["audio_file"] for i in indices]
            before = time.monotonic()
            manifest = seal_fold(
                seals_root / f"fold_{outer}", outer,
                baseline[outer]["probabilities"], scores["outer_valid"],
                files, contract["labels"])
            build_times[str(outer)] = {
                "seconds": time.monotonic() - before,
                "rows": len(files),
                "valid_rows": manifest["valid_rows"],
                "invalid_rows": manifest["invalid_rows"],
            }
            print(json.dumps({
                "stage": "pretruth_alignment_sealed", "outer_fold": outer,
                "seal_sha256": manifest["seal_sha256"],
            }), flush=True)
        selection = seal_selection(seals_root)
        for path in sorted(seals_root.rglob("*")):
            if path.is_file():
                parent.add_artifact(
                    path, "sealed_alignment/"
                    + path.relative_to(seals_root).as_posix())
        parent.flush(strict=True)
        pretruth = {
            "status": "passed_before_outer_truth_evaluation",
            "selection_sha256": selection["selection_sha256"],
            "selection_receipt_sha256":
                file_sha256(seals_root / "selection_receipt.json"),
            "artifacts": parent.verify_artifacts(),
            "metadata": parent.verify_remote_metadata(),
            "build_times": build_times,
            "outer_truth_scored": False,
        }
        write_json(output / "pre_evaluation_tracking_verification.json", pretruth)
        parent.add_artifact(output / "pre_evaluation_tracking_verification.json")
        parent.flush(strict=True)
        write_json(output / "experiment_state.json", {
            "status": "sealed_waiting_for_evaluation",
            "parent_run_id": parent.run_id,
            "selection_sha256": selection["selection_sha256"],
            "outer_truth_scored": False,
            "encoder_updates": 0,
        })

        selection, sealed = load_selection(seals_root)
        records = {}
        for recipe in RECIPES:
            directory = output / recipe
            directory.mkdir()
            tracker = DurableMLflowRun.prepare(
                spool_dir=directory / "tracking",
                parent_run_id=parent.run_id,
                run_name="S014-" + recipe,
                config={**resolved, "recipe": recipe,
                        "selection_sha256": selection["selection_sha256"]},
                **common)
            active.append(tracker)
            tracker.add_artifact(
                seals_root / "selection_receipt.json",
                "pretruth/selection_receipt.json")
            tracker.flush(strict=True)
            records[recipe] = {
                "recipe": recipe, "directory": directory, "tracker": tracker,
                "predictions": [], "folds": [],
            }

        checks, parity, fold_metrics = {}, {}, {
            "baseline": [], "powered_prior": []}
        for outer in (0, 1):
            loaded = sealed[outer]
            scores = baseline[outer]["scores"]
            indices = np.asarray(scores["outer_indices"], dtype=np.int64)
            expected_files = tuple(
                contract["manifest"][int(i)]["audio_file"] for i in indices)
            _require(loaded["audio_files"] == expected_files
                     and loaded["class_labels"] == tuple(contract["labels"])
                     and np.array_equal(
                         loaded["base_probabilities"],
                         baseline[outer]["probabilities"]),
                     "The pre-truth seal differs from the reproduced P002 fold")
            checks[str(outer)] = _verify_baseline_fold(
                root, contract, prepared[outer], baseline[outer], outer)
            baseline_predictions, baseline_metric = _result_predictions(
                contract, prepared[outer], baseline[outer])
            adjusted_result = {
                "probabilities": loaded["adjusted_probabilities"]}
            adjusted_predictions, adjusted_metric = _result_predictions(
                contract, prepared[outer], adjusted_result)
            parity[str(outer)] = _cuda_policy_parity(loaded)
            _require(parity[str(outer)]["exact_predictions"],
                     "CPU/CUDA S014 decisions differ")
            fold_metrics["baseline"].append(baseline_metric)
            fold_metrics["powered_prior"].append(adjusted_metric)
            for recipe, predictions, metrics in (
                ("baseline", baseline_predictions, baseline_metric),
                ("powered_prior", adjusted_predictions, adjusted_metric),
            ):
                report = {
                    "outer_fold": outer,
                    "outer": metrics,
                    "policy": (
                        {"id": "S008c", "kind": "historical_baseline"}
                        if recipe == "baseline" else FIXED_POLICY),
                    "source_reproduction": checks[str(outer)],
                    "selection_sha256": selection["selection_sha256"],
                    "seal_sha256": loaded["manifest"]["seal_sha256"],
                    "sealed_before_outer_scoring": True,
                    "cpu_cuda_parity": parity[str(outer)],
                    "probability_semantics":
                        "normalized decision scores, not calibrated posteriors",
                }
                _persist_fold(
                    records[recipe]["directory"] / f"fold_{outer}",
                    records[recipe]["tracker"], outer, metrics, predictions,
                    report)
                records[recipe]["predictions"].extend(predictions)
                records[recipe]["folds"].append(report)
            parent.log_metrics({
                f"fold_{outer}/baseline_macro_f1": baseline_metric["macro_f1"],
                f"fold_{outer}/powered_prior_macro_f1":
                    adjusted_metric["macro_f1"],
                f"fold_{outer}/macro_f1_gain":
                    adjusted_metric["macro_f1"] - baseline_metric["macro_f1"],
            }, step=outer, sync=True, strict=True)

        pooled = {
            recipe: score_predictions(
                contract["manifest"], records[recipe]["predictions"],
                contract["labels"])
            for recipe in RECIPES
        }
        _require(pooled["baseline"] == checked["report"]["oof"]
                 and pooled["baseline"]["macro_f1"] == BASELINE_MACRO_F1,
                 "Pooled P002 control differs from the completed S008c result")
        checks["pooled"] = {
            **verify_predictions(
                root / SOURCE_RUN / "S008c/oof_predictions.csv",
                records["baseline"]["predictions"]),
            "exact_pooled_metrics": True,
            "macro_f1": BASELINE_MACRO_F1,
        }
        write_json(output / "baseline_control_checks.json", checks)

        labels = {label: index for index, label in enumerate(contract["labels"])}
        baseline_by_file = {
            row["audio_file"]: labels[row["speaker_id"]]
            for row in records["baseline"]["predictions"]}
        adjusted_by_file = {
            row["audio_file"]: labels[row["speaker_id"]]
            for row in records["powered_prior"]["predictions"]}
        fold_by_file = {row["audio_file"]: row for row in contract["folds"]}
        truth = np.asarray(
            [labels[row["speaker_id"]] for row in contract["manifest"]],
            dtype=np.int64)
        before = np.asarray(
            [baseline_by_file[row["audio_file"]]
             for row in contract["manifest"]], dtype=np.int64)
        after = np.asarray(
            [adjusted_by_file[row["audio_file"]]
             for row in contract["manifest"]], dtype=np.int64)
        groups = np.asarray(
            [fold_by_file[row["audio_file"]]["group_id"]
             for row in contract["manifest"]])
        bootstrap = speaker_group_bootstrap(
            truth, before, after, groups,
            samples=config["bootstrap"]["replicates"],
            seed=config["bootstrap"]["seed"],
            lower_quantile=config["bootstrap"]["lower_quantile"])
        deltas = bootstrap.pop("deltas")
        np.savez_compressed(output / "bootstrap_deltas.npz", deltas=deltas)
        bootstrap["archive_sha256"] = file_sha256(
            output / "bootstrap_deltas.npz")
        bootstrap["deltas_array_sha256"] = _array_sha(deltas)
        write_json(output / "bootstrap_summary.json", bootstrap)

        parity_passed = all(
            row["exact_predictions"] for row in parity.values())
        decision = promotion_decision(
            pooled["baseline"], pooled["powered_prior"],
            fold_metrics["baseline"], fold_metrics["powered_prior"],
            bootstrap, parity_passed)
        decision["evidence_role"] = (
            "repeated_development_OOF_engineering_gate_not_independent_confirmation")
        paired = paired_diagnostics(
            contract["manifest"], records["baseline"]["predictions"],
            records["powered_prior"]["predictions"], contract["labels"])
        write_json(output / "promotion_decision.json", decision)
        write_json(output / "paired_diagnostics.json", paired)
        plot_paths = _plots(
            output, pooled["baseline"], pooled["powered_prior"], deltas)

        reports = {}
        baseline_pair = paired_diagnostics(
            contract["manifest"], records["baseline"]["predictions"],
            records["baseline"]["predictions"], contract["labels"])
        reports["baseline"] = _finish_child(
            records["baseline"], contract, baseline_pair)
        active.remove(records["baseline"]["tracker"])
        reports["powered_prior"] = _finish_child(
            records["powered_prior"], contract, paired, decision)
        active.remove(records["powered_prior"]["tracker"])

        for name, array in {**sources["vectors"],
                            "valid": sources["valid"]}.items():
            receipt = source_receipt["arrays"][name]
            _require(list(array.shape) == receipt["shape"]
                     and str(array.dtype) == receipt["dtype"]
                     and hashlib.sha256(array.tobytes()).hexdigest()
                     == receipt["array_sha256"],
                     "An immutable source cache was mutated")
        import torch
        memory = _memory_snapshot()
        resource_usage = {
            "host_memory": memory,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "fold_alignment_build_times": build_times,
            "local_reconstruction_preflight": local_preflight,
        }
        report = {
            "status": "complete",
            "parent_run_id": parent.run_id,
            "children": {
                recipe: records[recipe]["tracker"].run_id
                for recipe in RECIPES},
            "results": reports,
            "baseline_control_checks": checks,
            "selection_receipt": selection,
            "promotion": decision,
            "paired_against_S008c": paired,
            "bootstrap": bootstrap,
            "cpu_cuda_parity": parity,
            "source_binding": source_binding,
            "source_arrays_unchanged": True,
            "encoder_updates": 0,
            "raw_audio_read": False,
            "new_submission_built": False,
            "elapsed_seconds": time.monotonic() - started,
            "resource_usage": resource_usage,
            "limitations": LIMITATIONS,
        }
        write_json(output / "experiment_report.json", report)
        for name in (
            "experiment_report.json", "resolved_config.json",
            "baseline_control_checks.json", "bootstrap_summary.json",
            "bootstrap_deltas.npz", "promotion_decision.json",
            "paired_diagnostics.json",
        ):
            parent.add_artifact(output / name)
        for path in plot_paths:
            parent.add_artifact(path)
        parent.log_metrics({
            "baseline/oof_macro_f1_447": pooled["baseline"]["macro_f1"],
            "powered_prior/oof_macro_f1_447":
                pooled["powered_prior"]["macro_f1"],
            "powered_prior/oof_macro_f1_gain":
                decision["observed"]["pooled_macro_f1_gain"],
            "powered_prior/oof_accuracy":
                pooled["powered_prior"]["accuracy"],
            "bootstrap/lower_gain": bootstrap["lower"],
            "bootstrap/upper_gain": bootstrap["upper"],
            "promotion/passed": float(decision["promoted"]),
            "encoder_updates": 0,
        }, sync=False)
        parent.write_report(
            report,
            markdown=(
                "# S014 powered design-prior batch alignment\n\n"
                "Both fixed unlabeled outer-batch outputs were sealed and "
                "round-trip verified before outer scoring. Own-fold speaker "
                "IDs were removed from each reconstruction manifest. "
                f"P002 OOF Macro-F1: {pooled['baseline']['macro_f1']:.9f}. "
                f"S014 OOF Macro-F1: {pooled['powered_prior']['macro_f1']:.9f}; "
                f"gain {decision['observed']['pooled_macro_f1_gain']:+.9f}. "
                f"Promotion gate: {decision['promoted']}. No encoder update, "
                "audio read, package build or leaderboard submission occurred.\n"
            ),
        )
        parent.flush(strict=True)
        pre_finish_roundtrip = {
            "status": "verified_while_parent_running_before_finish",
            "children": {
                recipe: {
                    "artifacts": records[recipe]["tracker"].verify_artifacts(),
                    "metadata": records[recipe]["tracker"].verify_remote_metadata(),
                }
                for recipe in RECIPES
            },
            "parent_before_receipt": {
                "artifacts": parent.verify_artifacts(),
                "metadata": parent.verify_remote_metadata(),
            },
        }
        write_json(
            output / "tracking_verification_before_finish.json",
            pre_finish_roundtrip)
        parent.add_artifact(output / "tracking_verification_before_finish.json")
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.finish("FINISHED", strict=True)
        roundtrip = {}
        for recipe in RECIPES:
            tracker = records[recipe]["tracker"]
            roundtrip[recipe] = {
                "artifacts": tracker.verify_artifacts(),
                "metadata": tracker.verify_remote_metadata(),
            }
        roundtrip["parent"] = {
            "artifacts": parent.verify_artifacts(),
            "metadata": parent.verify_remote_metadata(),
        }
        write_json(output / "tracking_roundtrip_verification.json", {
            "status": "passed", "runs": roundtrip})
        write_json(output / "experiment_state.json", {
            "status": "complete",
            "parent_run_id": parent.run_id,
            "children": report["children"],
            "selection_sha256": selection["selection_sha256"],
            "all_mlflow_finished_and_verified": True,
            "encoder_updates": 0,
            "git_commit": source_state["git_commit"],
        })
        return {
            "output": str(output),
            "parent_run_id": parent.run_id,
            "results": {
                recipe: pooled[recipe]["macro_f1"] for recipe in RECIPES},
            "promotion": decision,
            "all_mlflow_finished_and_verified": True,
            "new_submission_built": False,
        }
    except BaseException as error:
        failure = {
            "status": "failed",
            "parent_run_id": parent.run_id,
            "error_type": type(error).__name__,
            "error": parent.redactor.text(str(error)),
            "encoder_updates": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
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
