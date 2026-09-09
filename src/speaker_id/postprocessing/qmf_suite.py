"""S017: nested quality-measure fusion over the immutable C002b cache.

The only learned component is a small linear logistic gate for the binary
``known`` versus ``unknown`` decision.  Every meta-validation content group is
removed from the complete score-building path.  The already selected known
speaker is never reordered.  Both outer policies are sealed before this module
reads outer truth for reporting.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import time
import uuid

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.postprocessing.decision_scoring import FEATURE_NAMES, decision_features
from speaker_id.postprocessing.nested_cases import META_FOLDS, META_SALT, make_nested_cases
from speaker_id.postprocessing.qmf_scoring import (
    DEFAULT_L2_PENALTY,
    DEFAULT_SCALE_FLOOR,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD_QUANTILES,
    META_MAXIMUM_FOLD_LOSS,
    META_MINIMUM_POOLED_GAIN,
    fit_qmf_logistic,
    is_known_targets,
    predict_qmf_logit,
    qmf_feature_view,
    qmf_probabilities,
    select_qmf_policy,
)
from speaker_id.postprocessing.scoring import prepare_fold, select_baseline
from speaker_id.tracking.snapshot import git_provenance
from speaker_id.training.adaptation_comparison import paired_diagnostics
from speaker_id.training.contracts import load_contract
from speaker_id.training.fusion_suite import verify_predictions
from speaker_id.training.gain_suite import verify_gain_cache
from speaker_id.training.runner import write_csv, write_json


BASELINE_MACRO_F1 = 0.9565282892229405
BASELINE_ACCURACY = 0.9589313314197394
BASELINE_ERRORS = {
    "known_to_unknown": 114,
    "unknown_to_known": 60,
    "known_to_other_known": 12,
}
SOURCE_PARENT = "37589a011a8c4f9aa0c5de0936ac7e46"
SOURCE_CHILD = "14a0f144b27a46e7a36e3fc3b2ce24c4"
SOURCE_COMMIT = "adf820c253c985364ccc36b24d2943508a661fc4"
READINESS_CONFIG = "configs/train/campp_coverage_c002.json"
BINDING_PATH = "artifacts/infrastructure/C002_preparation/mlflow_state.json"
RUN_DIR = "/workspace/Speaker-identification-2-c002/artifacts/training/cuda_gain_c002/C002_20260908T230219Z_479a211696d64a9495854f29006f72f7"
FEATURE_SETS = {
    "scores_only": [
        "fused_known_top",
        "fused_known_gap",
        "fused_unknown_top",
        "fused_unknown_top3_mean",
        "fused_unknown_top50_mean",
        "fused_unknown_top50_std",
        "public_known_minus_unknown",
        "advanced_known_minus_unknown",
        "encoder_winner_agreement",
    ],
    "scores_quality": [
        "fused_known_top",
        "fused_known_gap",
        "fused_unknown_top",
        "fused_unknown_top3_mean",
        "fused_unknown_top50_mean",
        "fused_unknown_top50_std",
        "public_known_minus_unknown",
        "advanced_known_minus_unknown",
        "encoder_winner_agreement",
        "log1p_duration_capped180",
        "rms_dbfs_clipped",
    ],
}
RECIPES = ("baseline", "qmf_scores", "qmf_scores_quality", "selector")
SOURCE_ARTIFACTS = {
    "experiment_report": ("experiment_report.json", "9fa9e223a099e946bd41f68a4ea8f115b13189abe49c1a5ca11491653304c096"),
    "paired_execution_report": ("paired_execution_report.json", "052578058fe6f2d6c21f2a06006072455fda97adff304fbc3ae7ad10b5267af2"),
    "identity_cache_manifest": ("identity_cache_manifest.json", "7e9a9906f4708ccdea20851e31c5db6ef92f7b454e0d21539c8016ac3a0ca31a"),
    "identity_cache_identity": ("identity_cache_identity.json", "18b5df7f2e222f11762eed343e96ddbf1b92ea88c2b0981d2f56923f747018b1"),
    "frozen_inner_choices": ("frozen_inner_choices.json", "af8815572a739d84f4e58999b947cfeeaf3c85801d80682a0ce5d3a7ae35a38c"),
    "c002b_experiment_report": ("C002b/experiment_report.json", "01e403a65cf01d3bd8d267706174383c77b53a92317773f0d24f49f7a5628d39"),
    "c002b_oof_predictions": ("C002b/oof_predictions.csv", "80eb46d160d042674b450ac7070e4396889f3644ee0de1ab4e8bf5f591ecc3a6"),
    "c002b_fold0_probabilities": ("C002b/fold_0/outer_probabilities.npz", "bdb53df3792e9305b9a9eac6dd1b0f335a597b26b96a61ccd90ae5760685600a"),
    "c002b_fold1_probabilities": ("C002b/fold_1/outer_probabilities.npz", "d08a211d42df193af367019859ccbd8792a8d41f079a576a0591006e069fcfdc"),
}
LIMITATIONS = [
    "Repeated development OOF is not an untouched test or hidden-leaderboard estimate.",
    "QMF only accepts or rejects the fixed best known identity; it cannot repair known-to-known ranking.",
    "Every meta-validation content group is absent from gallery, background cohort, baseline fitting, scaling and logistic fitting.",
    "Only two preregistered feature sets and one fixed L2 penalty are evaluated.",
    "No raw audio, embeddings, model weights or credentials are uploaded to MLflow.",
    "No server artifact is transferred to the local workstation unless the final promotion gate passes.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_bytes(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_sha(value) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _project_file(root: Path, value: str | Path) -> Path:
    original = root / value
    _require(not original.is_symlink(), "Input cannot be a symlink")
    resolved = original.resolve()
    _require(resolved.is_relative_to(root) and resolved.is_file(), "Expected project file is missing")
    return resolved


def _source_directory(root: Path, value: str) -> Path:
    path = Path(value)
    _require(path.is_absolute() and path.as_posix() == RUN_DIR, "S017 source run path changed")
    resolved = path.resolve()
    _require(resolved.is_relative_to(root) and resolved.is_dir() and not path.is_symlink(),
             "C002 source must be the pinned directory inside the active server workspace")
    return resolved


def _feature_columns(names: list[str], feature_set: str) -> np.ndarray:
    _require(names == list(FEATURE_NAMES), "Decision feature schema changed")
    required = FEATURE_SETS[feature_set]
    mapping = {name: index for index, name in enumerate(names)}
    _require(set(required).issubset(mapping), "A preregistered QMF feature is unavailable")
    return np.asarray([mapping[name] for name in required], dtype=np.int64)


def validate_config(config: dict) -> None:
    top = {
        "schema_version", "experiment_code", "run_name", "output_root", "source", "candidates",
        "feature_sets", "logistic", "nested_validation", "probability_temperature", "execution",
        "bootstrap", "promotion", "selection_policy", "mlflow", "retention", "limitations",
    }
    _require(isinstance(config, dict) and set(config) == top, "Incomplete or extra S017 configuration")
    _require(config["schema_version"] == 1 and config["experiment_code"] == "S017"
             and config["run_name"] == "S017-c002b-nested-open-set-qmf"
             and config["output_root"] == "artifacts/training/qmf_s017",
             "S017 experiment identity changed")
    source = config["source"]
    _require(source.get("run_dir") == RUN_DIR and source.get("parent_run_id") == SOURCE_PARENT
             and source.get("child_run_id") == SOURCE_CHILD and source.get("recipe") == "C002b"
             and source.get("recipe_identity") == "C002b_fresh_cuda_identity"
             and source.get("git_commit") == SOURCE_COMMIT and source.get("embedding_cache") == "identity_embedding_cache"
             and source.get("read_only") is True and source.get("encoder_updates") == 0
             and source.get("fresh_audio_extraction") is False,
             "C002b source binding changed")
    artifacts = source.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == set(SOURCE_ARTIFACTS),
             "C002b source-artifact coverage changed")
    for name, (relative, digest) in SOURCE_ARTIFACTS.items():
        _require(artifacts[name] == {"path": relative, "sha256": digest},
                 "Pinned C002b source-artifact hash changed")
    expected_candidates = [
        {"id": "baseline", "kind": "baseline", "feature_set": "baseline"},
        {"id": "qmf_scores", "kind": "qmf_logistic", "feature_set": "scores_only"},
        {"id": "qmf_scores_quality", "kind": "qmf_logistic", "feature_set": "scores_quality"},
    ]
    _require(config["candidates"] == expected_candidates and config["feature_sets"] == FEATURE_SETS,
             "S017 candidate or feature grid changed")
    _require(config["logistic"] == {
        "target": "is_known", "l2_penalty": DEFAULT_L2_PENALTY,
        "scale_floor": DEFAULT_SCALE_FLOOR, "weighting": "group_equal_then_binary_balanced",
        "solver": "scipy_L-BFGS-B", "model_format": "finite_json_coefficients_numpy_inference",
    }, "S017 logistic contract changed")
    nested = config["nested_validation"]
    _require(nested.get("meta_folds") == META_FOLDS and nested.get("meta_assignment_salt") == META_SALT
             and nested.get("assignment_unit") == "whole_content_group"
             and nested.get("threshold_quantiles") == DEFAULT_THRESHOLD_QUANTILES
             and nested.get("selection_metric") == "macro_f1_447"
             and nested.get("candidate_tie_order") == ["baseline", "qmf_scores", "qmf_scores_quality"]
             and nested.get("minimum_pooled_meta_gain") == META_MINIMUM_POOLED_GAIN
             and nested.get("maximum_meta_fold_loss") == META_MAXIMUM_FOLD_LOSS
             and nested.get("otherwise") == "baseline"
             and nested.get("outer_labels_forbidden_until_policy_sealed") is True,
             "S017 nested-selection contract changed")
    _require(set(nested.get("full_chain_group_exclusion", [])) == {
        "logistic_fit", "feature_standardization", "known_reference_gallery", "unknown_reference_cohort",
        "baseline_calibration", "threshold_selection",
    }, "S017 full nested exclusion scope changed")
    execution = config["execution"]
    _require(execution == {"qmf_device": "cpu", "cpu_threads": 4, "source_cache_read_only": True,
        "no_encoder_forward": True, "no_encoder_updates": True, "no_audio_loading": True,
        "no_test_transduction": True}, "S017 execution policy changed")
    _require(config["probability_temperature"] == DEFAULT_TEMPERATURE,
             "S017 probability temperature changed")
    bootstrap = config["bootstrap"]
    _require(bootstrap == {"kind": "paired_true_class_stratified_content_group", "seed": 20260909,
        "replicates": 5000, "lower_quantile": .025, "upper_quantile": .975},
        "S017 bootstrap contract changed")
    promotion = config["promotion"]
    _require(promotion == {"minimum_pooled_macro_f1_delta": .0015,
        "minimum_each_fold_macro_f1_delta": 0.0, "minimum_pooled_accuracy_delta": 0.0,
        "maximum_unknown_to_known_increase": 2, "maximum_known_to_other_known_increase": 0,
        "minimum_bootstrap_lower_bound": -.001, "require_exact_cpu_cuda_prediction_parity": True,
        "all_conditions_required": True, "otherwise": "baseline"},
        "S017 promotion gate changed")
    _require(config["mlflow"].get("experiment_id") == "1"
             and set(config["mlflow"].get("forbidden", [])) == {"raw_audio", "embeddings", "model_weights", "credentials"},
             "S017 MLflow scope changed")
    _require(config["retention"].get("local_transfer") == "promotion_only",
             "S017 local retention policy changed")


def validate(root: Path, config_path: Path) -> dict:
    """Validate the committed experiment contract without starting a run."""
    root = Path(root).resolve()
    path = _project_file(root, config_path)
    config = _json(path)
    validate_config(config)
    source_present = Path(config["source"]["run_dir"]).is_dir()
    return {"status": "validated_no_experiment_started", "experiment": "S017",
            "config_sha256": file_sha256(path), "source_present_on_this_host": source_present,
            "source_cache_transfer_required": False, "candidate_count": len(config["candidates"]),
            "local_transfer_policy": "promotion_only"}


def _verify_source(root: Path, config: dict, contract: dict) -> dict:
    source = _source_directory(root, config["source"]["run_dir"])
    paths = {}
    for name, (relative, digest) in SOURCE_ARTIFACTS.items():
        path = (source / relative).resolve()
        _require(path.is_relative_to(source) and path.is_file() and not path.is_symlink()
                 and file_sha256(path) == digest, "Pinned C002b artifact changed: " + name)
        paths[name] = path
    report = _json(paths["experiment_report"])
    paired = _json(paths["paired_execution_report"])
    state = _json(source / "experiment_state.json")
    result = next((row for row in report.get("results", []) if row.get("recipe") == "C002b"), None)
    _require(report.get("status") == "complete" and report.get("parent_run_id") == SOURCE_PARENT
             and report.get("encoder_updates") == 0 and report.get("embedding_artifacts_uploaded") is False
             and state == {"status": "complete", "parent_run_id": SOURCE_PARENT},
             "C002 parent is not the exact completed source")
    _require(result is not None and result.get("oof", {}).get("macro_f1") == BASELINE_MACRO_F1
             and result["oof"].get("accuracy") == BASELINE_ACCURACY
             and result["oof"].get("errors") == BASELINE_ERRORS,
             "C002b baseline report changed")
    _require(paired.get("status") == "complete" and paired.get("completed_pairs") == 4529
             and paired.get("encoder_updates") == 0 and paired.get("embedding_artifacts_uploaded") is False
             and paired.get("no_op_bitwise_parity_verified") is True
             and paired.get("raw_audio_sha_before_and_after_each_frontend") is True
             and paired.get("recognition_scoring_or_calibration") is False,
             "C002 paired execution integrity is incomplete")
    identity = _json(paths["identity_cache_identity"])
    receipt = _json(paths["identity_cache_manifest"])
    _require(identity.get("frontend") == "identity" and identity.get("embedding_dims") == {"public": 512, "advanced": 192}
             and identity.get("encoder_updates") == 0 and identity.get("labels") == contract["labels"]
             and identity.get("data_input_hashes") == contract["input_hashes"],
             "C002 identity does not match the current data contract")
    vectors, valid = verify_gain_cache(source / config["source"]["embedding_cache"], identity,
                                       contract["manifest"], receipt)
    _require(vectors["public"].shape == (4529, 512) and vectors["advanced"].shape == (4529, 192)
             and valid.shape == (4529,) and valid.dtype == np.bool_, "C002 cache dimensions changed")
    before = {name: _array_sha(value) for name, value in {**vectors, "valid": valid}.items()}
    return {"directory": source, "paths": paths, "report": report, "paired": paired,
            "identity": identity, "receipt": receipt, "vectors": vectors, "valid": valid,
            "array_sha256": before}


def _verify_remote_source(client, binding, source: dict, destination: Path) -> dict:
    parent = client.get_run(SOURCE_PARENT)
    child = client.get_run(SOURCE_CHILD)
    _require(str(parent.info.experiment_id) == str(binding.experiment_id) and parent.info.status == "FINISHED"
             and parent.data.tags.get("speaker_id.project") == binding.project
             and parent.data.tags.get("speaker_id.scope_id") == binding.scope_id
             and parent.data.tags.get("mlflow.source.git.commit") == SOURCE_COMMIT,
             "Remote C002 parent run identity/status changed")
    _require(str(child.info.experiment_id) == str(binding.experiment_id) and child.info.status == "FINISHED"
             and child.data.tags.get("speaker_id.project") == binding.project
             and child.data.tags.get("speaker_id.scope_id") == binding.scope_id
             and child.data.tags.get("mlflow.parentRunId") == SOURCE_PARENT
             and child.data.tags.get("mlflow.source.git.commit") == SOURCE_COMMIT,
             "Remote C002b child run identity/status changed")
    destination.mkdir(parents=True, exist_ok=False)
    downloaded = Path(client.download_artifacts(SOURCE_CHILD, "experiment_report.json", str(destination)))
    _require(downloaded.is_file() and file_sha256(downloaded) == SOURCE_ARTIFACTS["c002b_experiment_report"][1],
             "Remote C002b report bytes differ from the pinned server source")
    return {"status": "passed", "parent_run_id": SOURCE_PARENT, "child_run_id": SOURCE_CHILD,
            "experiment_id": str(binding.experiment_id), "child_report_sha256": file_sha256(downloaded)}


def _execution_environment(config: dict) -> dict:
    _require(all(os.environ.get(name) == "4" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")),
             "Start S017 with the exact four-thread BLAS environment")
    import scipy
    import torch
    _require(torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(0),
             "S017 execution requires the active RTX 3090 server")
    torch.set_num_threads(config["execution"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0),
            "cpu_threads": torch.get_num_threads(), "blas_threads": {name: os.environ[name] for name in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "qmf_fit_device": "cpu", "feature_primary_device": "cpu", "feature_parity_device": "cuda",
            "encoder_forward_calls": 0, "encoder_updates": 0, "raw_audio_reads": 0, "tf32_enabled": False}


def _masked_manifest(contract: dict, outer: int) -> list[dict]:
    folds = {row["audio_file"]: int(row["fold"]) for row in contract["folds"]}
    rows = []
    for row in contract["manifest"]:
        value = dict(row)
        if folds[row["audio_file"]] == outer:
            value["speaker_id"] = "__outer_truth_withheld__"
        rows.append(value)
    return rows


def _prediction_indices(result: dict) -> np.ndarray:
    values = np.asarray(result["probabilities"])
    _require(values.ndim == 2 and values.shape[1] == 447 and np.isfinite(values).all()
             and np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10),
             "Invalid 447-class decision probabilities")
    return values.argmax(axis=1).astype(np.int64)


def _prediction_rows(contract: dict, outer_indices: np.ndarray, predictions: np.ndarray) -> list[dict]:
    labels = contract["labels"]
    _require(predictions.shape == (len(outer_indices),) and np.all((predictions >= 0) & (predictions < len(labels))),
             "Prediction index shape/range changed")
    return [{"audio_file": contract["manifest"][int(index)]["audio_file"], "speaker_id": labels[int(label)]}
            for index, label in zip(outer_indices, predictions)]


def _verify_baseline_without_truth(source: dict, contract: dict, prepared: dict, baseline: dict, outer: int) -> dict:
    anchor = prepared["scores_by_alpha"][0.0]
    predictions = _prediction_rows(contract, anchor["outer_indices"], _prediction_indices(baseline))
    reference = source["directory"] / "C002b"
    exact = verify_predictions(reference / f"fold_{outer}/predictions.csv", predictions)
    frozen = _json(source["paths"]["frozen_inner_choices"])
    expected = frozen["inner_fits"][str(outer)]["identity"]["selected"]
    _require(baseline["policy"]["advanced_weight"] == expected["advanced_weight"]
             and all(baseline["calibration"][key] == expected["calibration"][key]
                     for key in ("unknown_weight", "margin_weight", "inner_macro_f1_447"))
             and abs(baseline["calibration"]["threshold"] - expected["calibration"]["threshold"]) <= 1e-6,
             "Fresh C002b alpha/gate no longer reproduces")
    with np.load(reference / f"fold_{outer}/outer_probabilities.npz", allow_pickle=False) as saved:
        observed = np.asarray(baseline["probabilities"])
        _require(np.array_equal(saved["audio_files"], np.asarray([row["audio_file"] for row in predictions]))
                 and np.array_equal(saved["labels"], np.asarray(contract["labels"]))
                 and np.allclose(saved["probabilities"], observed, rtol=0, atol=1e-5),
                 "Fresh C002b probability control changed")
        difference = float(np.max(np.abs(saved["probabilities"] - observed), initial=0))
    return {**exact, "selected_alpha_and_gate_match": True, "probability_atol": 1e-5,
            "probability_max_abs_difference": difference, "outer_truth_accessed": False}


def _case_feature_matrix(case: dict, scope: str, feature_set: str) -> tuple[np.ndarray, list[str]]:
    values = case[scope]
    return qmf_feature_view(values["features"], values["feature_names"], feature_set)


def _fit_search(prepared: dict, baseline: dict, nested: dict, *, device: str) -> dict:
    query = np.asarray(nested["query_global_indices"], dtype=np.int64)
    _require(np.array_equal(query, prepared["scores_by_alpha"][0.0]["calibration_indices"]),
             "Nested query order changed")
    lookup = {int(index): position for position, index in enumerate(query)}
    count = len(query)
    truth = np.asarray(prepared["inner_truth"], dtype=np.int64)
    guess = np.empty(count, dtype=np.int64)
    valid = np.ones(count, dtype=bool)
    assignments = np.full(count, -1, dtype=np.int64)
    baseline_predictions = np.empty(count, dtype=np.int64)
    logits = {name: np.full(count, np.nan) for name in FEATURE_SETS}
    meta_models = {name: [] for name in FEATURE_SETS}
    for case in nested["cases"]:
        fit, validation = case["fit"], case["validation"]
        rows = np.asarray([lookup[int(index)] for index in validation["global_indices"]], dtype=np.int64)
        _require(np.all(assignments[rows] == -1) and np.array_equal(truth[rows], validation["truth"]),
                 "Nested validation truth/order changed")
        assignments[rows] = int(case["meta_fold"])
        guess[rows] = validation["guess"]
        baseline_predictions[rows] = np.where(validation["margin"] > 0, validation["guess"], 0)
        fit_target = is_known_targets(fit["truth"], classes=447)
        _require(np.array_equal(fit_target, np.asarray(fit["truth"]) != 0),
                 "QMF target must mark every known row known, including misranked known rows")
        for feature_set in FEATURE_SETS:
            fit_x, names = _case_feature_matrix(case, "fit", feature_set)
            validation_x, validation_names = _case_feature_matrix(case, "validation", feature_set)
            _require(names == validation_names, "QMF fit/validation feature order changed")
            model = fit_qmf_logistic(fit_x, fit_target, fit["groups"], names,
                                     l2_penalty=DEFAULT_L2_PENALTY, scale_floor=DEFAULT_SCALE_FLOOR)
            logits[feature_set][rows] = predict_qmf_logit(validation_x, model, feature_order=names)
            meta_models[feature_set].append({"meta_fold": int(case["meta_fold"]), "model": model})
    _require(np.all(assignments >= 0) and np.array_equal(assignments, nested["assignments"])
             and all(np.isfinite(value).all() for value in logits.values()),
             "Nested QMF coverage is incomplete")
    selection = select_qmf_policy(truth, guess, valid, baseline_predictions, assignments, [
        {"id": "qmf_scores", "feature_set": "scores_only", "known_logits": logits["scores_only"]},
        {"id": "qmf_scores_quality", "feature_set": "scores_quality", "known_logits": logits["scores_quality"]},
    ], classes=447)
    fit_features = decision_features(prepared, baseline, "inner", device=device)
    outer_features = decision_features(prepared, baseline, "outer", device=device)
    final_models, results = {}, {"baseline": {**baseline, "model": None,
        "meta_selection": selection["candidates"][0]}}
    for feature_set, recipe in (("scores_only", "qmf_scores"), ("scores_quality", "qmf_scores_quality")):
        fit_x, names = qmf_feature_view(fit_features["features"], fit_features["feature_names"], feature_set)
        outer_x, outer_names = qmf_feature_view(outer_features["features"], outer_features["feature_names"], feature_set)
        _require(names == outer_names, "Full-fit and outer QMF feature order changed")
        model = fit_qmf_logistic(fit_x,
            is_known_targets(prepared["inner_truth"], classes=447),
            prepared["groups"][fit_features["indices"]], names,
            l2_penalty=DEFAULT_L2_PENALTY, scale_floor=DEFAULT_SCALE_FLOOR)
        candidate = next(row for row in selection["candidates"] if row["id"] == recipe)
        outer_logits = predict_qmf_logit(outer_x, model, feature_order=names)
        probabilities = qmf_probabilities(outer_features["known_scores"], outer_logits,
            candidate["threshold"], outer_features["valid"], temperature=DEFAULT_TEMPERATURE)
        known_winner = outer_features["known_scores"].argmax(axis=1) + 1
        predicted = probabilities.argmax(axis=1)
        _require(np.all((predicted == 0) | (predicted == known_winner)), "QMF reordered a known identity")
        policy = {"id": recipe, "kind": "qmf_open_set_logistic", "feature_set": feature_set,
            "feature_names": names, "threshold": candidate["threshold"], "model_sha256": _sha_bytes(model),
            "target": "is_known", "identity_ranking": "unchanged_C002b_max_reference",
            "base_policy": baseline["policy"], "base_calibration": baseline["calibration"]}
        results[recipe] = {"policy": policy, "calibration": {"threshold": candidate["threshold"],
            "inner_macro_f1_447": candidate["meta_macro_f1_447"]}, "probabilities": probabilities,
            "scores": {**baseline["scores"], "outer_qmf_logit": outer_logits}, "model": model,
            "meta_selection": candidate}
        final_models[feature_set] = model
    selected = selection["selected"]["id"]
    results["selector"] = deepcopy(results[selected])
    results["selector"]["policy"] = {**results["selector"]["policy"], "id": "selector",
        "selected_recipe": selected, "baseline_fallback": selection["baseline_fallback"]}
    return {"results": results, "selection": selection, "meta_models": meta_models,
            "final_models": final_models, "meta_logits": logits, "meta_truth": truth,
            "meta_guess": guess, "meta_baseline_predictions": baseline_predictions,
            "assignments": assignments, "full_fit_features": fit_features, "outer_features": outer_features}


def _parity(prepared: dict, baseline: dict, search: dict) -> dict:
    cuda = decision_features(prepared, baseline, "outer", device="cuda")
    cpu = search["outer_features"]
    _require(cuda["feature_names"] == cpu["feature_names"] and np.array_equal(cuda["indices"], cpu["indices"]),
             "CPU/CUDA feature row/schema mismatch")
    families = {"baseline": {"exact_predictions": True, "decision_disagreements": 0}}
    for feature_set, recipe in (("scores_only", "qmf_scores"), ("scores_quality", "qmf_scores_quality")):
        cuda_x, names = qmf_feature_view(cuda["features"], cuda["feature_names"], feature_set)
        cpu_x, cpu_names = qmf_feature_view(cpu["features"], cpu["feature_names"], feature_set)
        _require(names == cpu_names, "CPU/CUDA QMF feature order changed")
        model = search["final_models"][feature_set]
        logits = predict_qmf_logit(cuda_x, model, feature_order=names)
        threshold = search["results"][recipe]["policy"]["threshold"]
        probabilities = qmf_probabilities(cuda["known_scores"], logits, threshold, cuda["valid"])
        expected = _prediction_indices(search["results"][recipe])
        observed = probabilities.argmax(axis=1)
        differences = observed != expected
        families[recipe] = {"exact_predictions": not bool(differences.any()),
            "decision_disagreements": int(differences.sum()),
            "disagreement_indices": cuda["indices"][differences].tolist(),
            "feature_max_abs_difference": float(np.max(np.abs(cuda_x - cpu_x), initial=0)),
            "logit_max_abs_difference": float(np.max(np.abs(logits - search["results"][recipe]["scores"]["outer_qmf_logit"]), initial=0))}
    selected = search["selection"]["selected"]["id"]
    families["selector"] = dict(families[selected])
    return {"scope": "independent_cuda_feature_recompute_same_json_model_no_refit",
            "families": families, "all_exact_predictions": all(row["exact_predictions"] for row in families.values()),
            "encoder_updates": 0, "additional_model_fits": 0}


def _seal_fold(output: Path, outer: int, prepared: dict, baseline_check: dict,
               search: dict, parity: dict, source_binding: dict) -> dict:
    directory = output / "pretruth" / f"fold_{outer}"
    directory.mkdir(parents=True, exist_ok=False)
    predictions = {recipe: _prediction_indices(search["results"][recipe]) for recipe in RECIPES}
    np.savez_compressed(directory / "prediction_indices.npz", **predictions,
                        outer_indices=search["outer_features"]["indices"])
    models = {feature_set: model for feature_set, model in search["final_models"].items()}
    write_json(directory / "qmf_models.json", models)
    selection = deepcopy(search["selection"])
    for row in selection["candidates"]:
        row.pop("predictions", None)
        row.pop("curve", None)
    selection["selected"].pop("predictions", None)
    selection["selected"].pop("curve", None)
    write_json(directory / "selection.json", selection)
    write_json(directory / "parity.json", parity)
    seal = {"schema_version": 1, "outer_fold": outer, "outer_truth_accessed": False,
        "source_binding": source_binding, "baseline_control": baseline_check,
        "prediction_indices_sha256": file_sha256(directory / "prediction_indices.npz"),
        "qmf_models_sha256": file_sha256(directory / "qmf_models.json"),
        "selection_sha256": file_sha256(directory / "selection.json"),
        "parity_sha256": file_sha256(directory / "parity.json"),
        "selected_recipe": search["selection"]["selected"]["id"],
        "baseline_fallback": search["selection"]["baseline_fallback"],
        "outer_rows": len(search["outer_features"]["indices"]),
        "known_ranking_unchanged": True, "invalid_forced_unknown": True}
    write_json(directory / "seal.json", seal)
    return {"directory": directory, "seal": seal, "seal_sha256": file_sha256(directory / "seal.json"),
            "search": search, "prepared": prepared, "parity": parity}


def _selection_metadata(row: dict) -> dict:
    """Remove row-level arrays/curves before JSON reports and MLflow upload."""
    result = deepcopy(row)
    result.pop("predictions", None)
    result.pop("curve", None)
    return result


def _read_sealed_predictions(fold: dict, recipe: str) -> tuple[np.ndarray, np.ndarray]:
    directory, seal = fold["directory"], fold["seal"]
    saved_seal = _json(directory / "seal.json")
    _require(saved_seal == seal
             and file_sha256(directory / "seal.json") == fold["seal_sha256"]
             and file_sha256(directory / "prediction_indices.npz") == seal["prediction_indices_sha256"]
             and file_sha256(directory / "qmf_models.json") == seal["qmf_models_sha256"]
             and file_sha256(directory / "selection.json") == seal["selection_sha256"]
             and file_sha256(directory / "parity.json") == seal["parity_sha256"],
             "Pretruth seal changed before evaluation")
    with np.load(directory / "prediction_indices.npz", allow_pickle=False) as arrays:
        return arrays["outer_indices"].copy(), arrays[recipe].copy()


def _bootstrap(contract: dict, before: list[dict], after: list[dict], config: dict) -> dict:
    label_index = {label: index for index, label in enumerate(contract["labels"])}
    old = {row["audio_file"]: label_index[row["speaker_id"]] for row in before}
    new = {row["audio_file"]: label_index[row["speaker_id"]] for row in after}
    truth = np.asarray([label_index[row["speaker_id"]] for row in contract["manifest"]], dtype=np.int64)
    old_values = np.asarray([old[row["audio_file"]] for row in contract["manifest"]], dtype=np.int64)
    new_values = np.asarray([new[row["audio_file"]] for row in contract["manifest"]], dtype=np.int64)
    fold_rows = {row["audio_file"]: row for row in contract["folds"]}
    groups = np.asarray([fold_rows[row["audio_file"]]["group_id"] for row in contract["manifest"]])
    strata = {}
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        values = np.unique(truth[indices])
        _require(len(values) == 1, "Bootstrap content group mixes true classes")
        strata.setdefault(int(values[0]), []).append(indices)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    deltas = np.empty(config["bootstrap"]["replicates"], dtype=np.float64)
    from speaker_id.training.scoring import macro_f1_indices
    for repeat in range(len(deltas)):
        sampled = []
        for label in sorted(strata):
            values = strata[label]
            sampled.extend(values[int(index)] for index in rng.integers(0, len(values), size=len(values)))
        indices = np.concatenate(sampled)
        deltas[repeat] = (macro_f1_indices(truth[indices], new_values[indices], 447)
                          - macro_f1_indices(truth[indices], old_values[indices], 447))
    lower, upper = np.quantile(deltas, [config["bootstrap"]["lower_quantile"], config["bootstrap"]["upper_quantile"]])
    return {"kind": config["bootstrap"]["kind"], "seed": config["bootstrap"]["seed"],
        "replicates": len(deltas), "lower": float(lower), "median": float(np.median(deltas)),
        "upper": float(upper), "positive_fraction": float(np.mean(deltas > 0)),
        "delta_array_sha256": _array_sha(deltas), "true_class_strata": len(strata),
        "content_groups": int(sum(len(values) for values in strata.values()))}


def _promotion(config: dict, baseline: dict, selector: dict, bootstrap: dict,
               fold_metrics: dict, parity: bool) -> dict:
    rule = config["promotion"]
    deltas = [fold_metrics["selector"][str(f)]["macro_f1"] - fold_metrics["baseline"][str(f)]["macro_f1"]
              for f in (0, 1)]
    conditions = {
        "pooled_macro_f1_delta": selector["macro_f1"] - baseline["macro_f1"] >= rule["minimum_pooled_macro_f1_delta"],
        "each_fold_macro_f1_delta": min(deltas) >= rule["minimum_each_fold_macro_f1_delta"],
        "pooled_accuracy_delta": selector["accuracy"] - baseline["accuracy"] >= rule["minimum_pooled_accuracy_delta"],
        "unknown_to_known_increase": selector["errors"]["unknown_to_known"] - baseline["errors"]["unknown_to_known"] <= rule["maximum_unknown_to_known_increase"],
        "known_to_other_known_increase": selector["errors"]["known_to_other_known"] - baseline["errors"]["known_to_other_known"] <= rule["maximum_known_to_other_known_increase"],
        "bootstrap_lower_bound": bootstrap["lower"] >= rule["minimum_bootstrap_lower_bound"],
        "exact_cpu_cuda_prediction_parity": bool(parity),
    }
    return {"passed": all(conditions.values()), "conditions": conditions,
        "pooled_macro_f1_delta": selector["macro_f1"] - baseline["macro_f1"],
        "pooled_accuracy_delta": selector["accuracy"] - baseline["accuracy"],
        "fold_macro_f1_deltas": deltas, "bootstrap": bootstrap, "rule": rule,
        "retained_recipe": "selector" if all(conditions.values()) else "baseline",
        "local_transfer_allowed": bool(all(conditions.values()))}


def _mark_fit_started(trackers) -> None:
    for tracker in trackers:
        for key, value in {"speaker_id.training_started": "true",
                           "speaker_id.training_scope": "cpu_qmf_postprocessor_only",
                           "speaker_id.encoder_updates": "0"}.items():
            tracker.client.set_tag(tracker.run_id, key, value)
            tracker.state["tags"][key] = value
        tracker._save()
        tracker.log_metrics({"postprocessor_training_started": 1, "encoder_updates": 0}, sync=True, strict=True)


def _mark_postfinish_failure(tracker, error: BaseException) -> None:
    """Prevent a failed verification from remaining a silent FINISHED run."""
    message = tracker.redactor.text(f"{type(error).__name__}: {error}")
    try:
        tracker.client.set_tag(tracker.run_id, "speaker_id.postfinish_verification_failed", "true")
        tracker.client.set_tag(tracker.run_id, "speaker_id.postfinish_verification_error", message[:5000])
        tracker.state["tags"]["speaker_id.postfinish_verification_failed"] = "true"
        tracker.state["tags"]["speaker_id.postfinish_verification_error"] = message[:5000]
        tracker._save()
    except Exception:
        pass
    try:
        tracker.finish("FAILED", strict=False)
    except Exception:
        pass


def _finish_verified(tracker) -> dict:
    """Verify while RUNNING, finish, then attest the terminal status."""
    before = {"artifacts": tracker.verify_artifacts(), "metadata": tracker.verify_remote_metadata()}
    try:
        tracker.finish("FINISHED", strict=True)
        after = tracker.verify_remote_metadata()
        _require(after["remote_run_status"] == "FINISHED", "MLflow run did not reach FINISHED")
        return {"pretermination": before, "posttermination": after}
    except BaseException as error:
        _mark_postfinish_failure(tracker, error)
        raise


def execute(root: Path, config_path: Path, binding_path: Path) -> dict:
    """Execute S017; no source arrays or model weights are transferred or uploaded."""
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    root = Path(root).resolve()
    config_path = _project_file(root, config_path)
    config = _json(config_path)
    validate_config(config)
    _require(_project_file(root, READINESS_CONFIG).is_file(), "C002 data contract is missing")
    binding_path = _project_file(root, binding_path)
    _require(binding_path == (root / BINDING_PATH).resolve(), "S017 must use the isolated C002 MLflow binding")
    provenance = git_provenance(root)
    _require(provenance["src_dirty"] is False and re.fullmatch(r"[a-f0-9]{40}", provenance.get("git_commit") or ""),
             "Commit clean S017 source before execution")
    contract = load_contract(root / READINESS_CONFIG, root, verify_audio=False)
    _require(len(contract["manifest"]) == 4529 and len(contract["labels"]) == 447,
             "S017 data/label population changed")
    source = _verify_source(root, config, contract)
    execution = _execution_environment(config)
    binding = ExperimentBinding(**_json(binding_path)["binding"])
    binding.validate()
    _require(binding.experiment_id == "1", "S017 must use owned MLflow experiment 1")
    output = root / config["output_root"] / ("S017_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    source_binding = {"run_dir": RUN_DIR, "parent_run_id": SOURCE_PARENT, "child_run_id": SOURCE_CHILD,
        "source_git_commit": SOURCE_COMMIT, "execution_git_commit": provenance["git_commit"],
        "suite_config_sha256": file_sha256(config_path),
        "artifacts": {name: {"path": relative, "sha256": digest}
                      for name, (relative, digest) in SOURCE_ARTIFACTS.items()},
        "cache_array_sha256": source["array_sha256"]}
    resolved = {"suite": config, "source_binding": source_binding, "execution": execution,
        "data_input_hashes": contract["input_hashes"], "limitations": LIMITATIONS,
        "outer_truth_sealing": "both folds sealed before any outer metric",
        "retention": "server-only unless final promotion passes"}
    write_json(output / "resolved_config.json", resolved)
    input_paths = {"suite_config": config_path, "launcher": root / "scripts/score_qmf.py",
        "readiness_config": root / READINESS_CONFIG, **source["paths"],
        **{key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")}}
    common = {"project_root": root, "binding": binding, "input_paths": input_paths,
              "run_kind": "nested_qmf_open_set", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=config["run_name"], config=resolved, **common)
    children, active, started = {}, [], time.monotonic()
    try:
        parent.flush(strict=True)
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
            "postprocessor_training_started": False, "encoder_updates": 0})
        remote = _verify_remote_source(parent.client, binding, source, output / "remote_source_verification")
        write_json(output / "source_verification.json", {"status": "passed", "remote": remote,
            "cache_files_verified": len(contract["manifest"]), "cache_array_sha256": source["array_sha256"],
            "embedding_artifacts_uploaded": False, "raw_audio_read": False, "encoder_updates": 0})
        for name in ("resolved_config.json", "source_verification.json"):
            parent.add_artifact(output / name)
        for name, path in (("suite_config", config_path), ("launcher", root / "scripts/score_qmf.py"),
                           ("readiness_config", root / READINESS_CONFIG)):
            parent.add_artifact(path, "input_configs/" + name + path.suffix)
        for name, path in source["paths"].items():
            if name not in {"identity_cache_manifest", "c002b_fold0_probabilities", "c002b_fold1_probabilities"}:
                parent.add_artifact(path, "source_evidence/" + name + path.suffix)
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        for recipe in RECIPES:
            directory = output / recipe
            directory.mkdir()
            tracker = DurableMLflowRun.prepare(spool_dir=directory / "tracking", parent_run_id=parent.run_id,
                run_name="S017-" + recipe, config={**resolved, "recipe": recipe}, **common)
            children[recipe] = {"tracker": tracker, "directory": directory, "predictions": [], "folds": []}
            active.append(tracker)
            tracker.flush(strict=True)
        _mark_fit_started([parent, *active])
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
            "postprocessor_training_started": True, "encoder_updates": 0})

        sealed = {}
        for outer in (0, 1):
            print(json.dumps({"stage": "nested_qmf", "outer_fold": outer, "status": "started"}), flush=True)
            manifest = _masked_manifest(contract, outer)
            prepared = prepare_fold(source["vectors"]["public"], source["vectors"]["advanced"], source["valid"],
                manifest, contract["folds"], contract["labels"], outer)
            baseline = select_baseline(prepared)
            check = _verify_baseline_without_truth(source, contract, prepared, baseline, outer)
            nested = make_nested_cases(source["vectors"]["public"], source["vectors"]["advanced"], source["valid"],
                manifest, contract["folds"], contract["labels"], outer,
                lambda p, b, scope: decision_features(p, b, scope, device="cpu"), prepared=prepared)
            search = _fit_search(prepared, baseline, nested, device="cpu")
            parity = _parity(prepared, baseline, search)
            sealed[str(outer)] = _seal_fold(output, outer, prepared, check, search, parity, source_binding)
            parent.log_metrics({f"fold_{outer}/meta_baseline_macro_f1_447": search["selection"]["candidates"][0]["meta_macro_f1_447"],
                f"fold_{outer}/meta_selected_macro_f1_447": search["selection"]["selected"]["meta_macro_f1_447"],
                f"fold_{outer}/meta_baseline_fallback": int(search["selection"]["baseline_fallback"]),
                f"fold_{outer}/cpu_cuda_exact_predictions": int(parity["all_exact_predictions"])}, sync=True, strict=True)
            print(json.dumps({"stage": "nested_qmf", "outer_fold": outer, "status": "sealed",
                "selected": search["selection"]["selected"]["id"]}), flush=True)
        root_seal = {"schema_version": 1, "both_outer_folds_sealed": True, "outer_truth_accessed": False,
            "folds": {outer: {"seal_sha256": value["seal_sha256"], "selected_recipe": value["seal"]["selected_recipe"]}
                      for outer, value in sealed.items()}, "source_binding": source_binding}
        write_json(output / "pretruth_seal.json", root_seal)
        parent.add_artifact(output / "pretruth_seal.json")
        for outer, value in sealed.items():
            for name in ("seal.json", "selection.json", "parity.json", "qmf_models.json"):
                parent.add_artifact(value["directory"] / name, f"pretruth/fold_{outer}/" + name)
        parent.flush(strict=True)
        # All sealed model/prediction identities must round-trip before the first
        # outer truth value is used for metrics.
        parent.verify_artifacts()
        parent.verify_remote_metadata()

        reports, child_verification = {}, {}
        fold_metrics = {recipe: {} for recipe in RECIPES}
        baseline_predictions = None
        for recipe in RECIPES:
            record = children[recipe]
            for outer in (0, 1):
                indices, values = _read_sealed_predictions(sealed[str(outer)], recipe)
                rows = _prediction_rows(contract, indices, values)
                references = [contract["manifest"][int(index)] for index in indices]
                metrics = score_predictions(references, rows, contract["labels"])
                fold_metrics[recipe][str(outer)] = metrics
                directory = record["directory"] / f"fold_{outer}"
                directory.mkdir()
                fold_report = {"outer_fold": outer, "outer": metrics,
                    "policy": sealed[str(outer)]["search"]["results"][recipe]["policy"],
                    "meta_selection": _selection_metadata(
                        sealed[str(outer)]["search"]["results"][recipe]["meta_selection"]),
                    "pretruth_seal_sha256": sealed[str(outer)]["seal_sha256"],
                    "deployment_parity": sealed[str(outer)]["parity"]["families"][recipe],
                    "outer_truth_first_access": "after both fold seals", "encoder_updates": 0}
                write_json(directory / "evaluation.json", fold_report)
                write_csv(directory / "predictions.csv", rows)
                write_csv(directory / "per_class.csv", metrics["per_class"])
                for name in ("evaluation.json", "predictions.csv", "per_class.csv"):
                    record["tracker"].add_artifact(directory / name, f"fold_{outer}/" + name)
                record["tracker"].log_metrics({f"fold_{outer}/macro_f1_447": metrics["macro_f1"],
                    f"fold_{outer}/accuracy": metrics["accuracy"],
                    **{f"fold_{outer}/{key}": value for key, value in metrics["errors"].items()}}, sync=True, strict=True)
                record["predictions"].extend(rows)
                record["folds"].append(fold_report)
            pooled = score_predictions(contract["manifest"], record["predictions"], contract["labels"])
            report = {"recipe": recipe, "oof": pooled, "folds": record["folds"],
                "oof_macro_f1_delta_vs_C002b": pooled["macro_f1"] - BASELINE_MACRO_F1,
                "paired_against_C002b": None, "encoder_updates": 0, "limitations": LIMITATIONS}
            if recipe == "baseline":
                _require(pooled["macro_f1"] == BASELINE_MACRO_F1 and pooled["accuracy"] == BASELINE_ACCURACY
                         and pooled["errors"] == BASELINE_ERRORS, "Sealed C002b pooled baseline changed")
                exact = verify_predictions(source["paths"]["c002b_oof_predictions"], record["predictions"])
                report["source_control"] = {**exact, "exact_pooled_metrics": True}
                baseline_predictions = list(record["predictions"])
            else:
                _require(baseline_predictions is not None, "Baseline evaluation must finish first")
                report["paired_against_C002b"] = paired_diagnostics(contract["manifest"], baseline_predictions,
                    record["predictions"], contract["labels"])
            write_json(record["directory"] / "experiment_report.json", report)
            write_csv(record["directory"] / "oof_predictions.csv", record["predictions"])
            write_csv(record["directory"] / "oof_per_class.csv", pooled["per_class"])
            for name in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
                record["tracker"].add_artifact(record["directory"] / name)
            record["tracker"].log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                "oof/macro_f1_delta_vs_C002b": report["oof_macro_f1_delta_vs_C002b"],
                **{"oof/" + key: value for key, value in pooled["errors"].items()}}, sync=False)
            record["tracker"].write_report(report, markdown=f"# S017 {recipe}\n\nOOF Macro-F1: {pooled['macro_f1']:.9f}; change from C002b: {report['oof_macro_f1_delta_vs_C002b']:+.9f}. The open-set gate was selected with fully nested group-held-out meta predictions. Known identity ranking and both CAM++ encoders remained fixed.\n")
            child_verification[recipe] = _finish_verified(record["tracker"])
            active.remove(record["tracker"])
            reports[recipe] = report

        bootstrap = _bootstrap(contract, children["baseline"]["predictions"], children["selector"]["predictions"], config)
        all_parity = all(sealed[str(outer)]["parity"]["families"]["selector"]["exact_predictions"] for outer in (0, 1))
        promotion = _promotion(config, reports["baseline"]["oof"], reports["selector"]["oof"],
                               bootstrap, fold_metrics, all_parity)
        for name, array in {**source["vectors"], "valid": source["valid"]}.items():
            _require(_array_sha(array) == source["array_sha256"][name], "Read-only C002 cache mutated in memory")
        final = {"status": "complete", "parent_run_id": parent.run_id,
            "children": {recipe: children[recipe]["tracker"].run_id for recipe in RECIPES},
            "results": {recipe: report["oof"] for recipe, report in reports.items()},
            "promotion": promotion, "source_binding": source_binding, "source_arrays_unchanged": True,
            "all_mlflow_finished_and_verified": False, "encoder_updates": 0,
            "embedding_artifacts_uploaded": False, "raw_audio_read": False,
            "elapsed_seconds": time.monotonic() - started,
            "local_transfer_performed": False,
            "local_transfer_policy": "promotion_only_and_separate_packaging_step"}
        write_json(output / "bootstrap_report.json", bootstrap)
        write_json(output / "promotion_decision.json", promotion)
        write_json(output / "experiment_report.json", final)
        for name in ("bootstrap_report.json", "promotion_decision.json", "experiment_report.json", "resolved_config.json"):
            parent.add_artifact(output / name)
        for recipe, report in reports.items():
            parent.log_metrics({f"{recipe}/oof_macro_f1_447": report["oof"]["macro_f1"],
                f"{recipe}/oof_accuracy": report["oof"]["accuracy"]}, sync=False)
        parent.log_metrics({"promotion/passed": int(promotion["passed"]),
            "promotion/macro_f1_delta": promotion["pooled_macro_f1_delta"],
            "promotion/bootstrap_lower": bootstrap["lower"], "encoder_updates": 0}, sync=False)
        parent.write_report(final, markdown="# S017 nested open-set QMF\n\n" + "\n".join(
            f"- {recipe}: OOF Macro-F1 {report['oof']['macro_f1']:.9f}; delta {report['oof_macro_f1_delta_vs_C002b']:+.9f}."
            for recipe, report in reports.items()) + f"\n\nPromotion gate passed: {promotion['passed']}. No local transfer was performed.\n")
        write_json(output / "tracking_pretermination_verification.json",
                   {"status": "passed", "children": child_verification,
                    "parent": "verified again immediately before FINISHED"})
        parent.add_artifact(output / "tracking_pretermination_verification.json")
        parent_verification = _finish_verified(parent)
        roundtrip = {"parent": parent_verification, "children": child_verification}
        write_json(output / "tracking_roundtrip_verification.json", {"status": "passed", "runs": roundtrip})
        final["all_mlflow_finished_and_verified"] = True
        write_json(output / "experiment_report.json", final)
        parent.write_report(final, markdown="# S017 nested open-set QMF\n\n" + "\n".join(
            f"- {recipe}: OOF Macro-F1 {report['oof']['macro_f1']:.9f}; delta {report['oof_macro_f1_delta_vs_C002b']:+.9f}."
            for recipe, report in reports.items()) + f"\n\nPromotion gate passed: {promotion['passed']}. All five MLflow runs and tracked artifacts were read back successfully. No local transfer was performed.\n")
        parent.add_artifact(output / "experiment_report.json")
        parent.add_artifact(output / "tracking_roundtrip_verification.json")
        try:
            parent.flush(strict=True)
            parent.verify_artifacts()
            parent.verify_remote_metadata()
        except BaseException as error:
            _mark_postfinish_failure(parent, error)
            raise
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id,
            "children": final["children"], "all_mlflow_finished_and_verified": True,
            "postprocessor_training_started": True, "encoder_updates": 0,
            "git_commit": provenance["git_commit"], "promotion_passed": promotion["passed"],
            "local_transfer_performed": False})
        return {"output": str(output), "parent_run_id": parent.run_id,
            "results": {recipe: report["oof"]["macro_f1"] for recipe, report in reports.items()},
            "promotion": promotion, "all_mlflow_finished_and_verified": True,
            "local_transfer_performed": False}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id,
            "error_type": type(error).__name__, "error": parent.redactor.text(str(error)),
            "encoder_updates": 0, "elapsed_seconds": time.monotonic() - started,
            "local_transfer_performed": False}
        write_json(output / "failure.json", failure)
        for tracker in list(active):
            try:
                tracker.write_report(failure)
                tracker.finish("FAILED", strict=False)
            except Exception:
                continue
        try:
            parent.write_report(failure)
            parent.finish("FAILED", strict=False)
        except Exception:
            pass
        write_json(output / "experiment_state.json", failure)
        raise
