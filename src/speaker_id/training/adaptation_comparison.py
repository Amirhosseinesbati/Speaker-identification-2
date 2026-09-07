"""S006: preregistered comparison of two independently attested adapted encoders.

This module cannot fit a model or extract audio. Historical S004/S005 schemas
stay unchanged. A real config must supply completed identities; there is no
default or partially filled source config.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import uuid
import zipfile

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.adapted_scoring import (
    assert_fixed_role_groups, load_adapted_fold_cache, validate_adapted_identity,
    validate_adapted_suite, verified_export_inventory,
)
from speaker_id.training.contracts import load_contract, read_csv
from speaker_id.training.frozen_suite import fixed_role_scores
from speaker_id.training.fusion_suite import project_path, verify_predictions
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities
from speaker_id.training.runner import write_csv, write_json
from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_total_steps
from speaker_id.training.scoring import fit_threshold, score_probabilities


SOURCE_CONFIGS = {"F003": "configs/train/campp_finetune_fp32.json",
                  "F004": "configs/train/campp_finetune_head600.json"}
HEAD_STEPS = {"F003": 100, "F004": 600}
FIXED = "fixed_inner_holdout"
EXPANDED = "original_heldout_queries_expanded_gallery"
SELECTION_POLICY = "inner_queries_only; fixed final checkpoints; no outer recipe selection"
DECISION_RULE = {"minimum_pooled_improvement": .003, "maximum_fold_decline": .005}
LIMITATIONS = [
    "Repeated development-fold comparisons are not an unbiased final test or a hidden leaderboard score.",
    "Longer head training also extends the head LR ramp and shifts global-step-indexed tail batches; this compares two training recipes.",
    "Fixed inner queries exclude encoder-fit groups; this is not the broader frozen S002f crossfit protocol.",
    "Expanded inner queries exclude their own whole content group; restored outer galleries have greater true-class support.",
    "All 4529 files and 447 classes remain in pooled OOF, including 89 zero-signal unknown fallbacks.",
]


def comparison_recipes(expanded_arm: bool) -> list[dict]:
    """Frozen before F004 results; the optional arm has its own mandatory control."""
    rows = []
    specifications = [
        ("a", "F003", "prototype", FIXED, "F003_prototype"),
        ("b", "F004", "prototype", FIXED, "F004_prototype"),
        ("c", "F003", "max_reference", FIXED, "S004d"),
        ("d", "F004", "max_reference", FIXED, None),
    ]
    if expanded_arm:
        specifications += [("e", "F003", "max_reference", EXPANDED, "S005d"),
                           ("f", "F004", "max_reference", EXPANDED, None)]
    for letter, source, method, protocol, control in specifications:
        rows.append({"id": "S006" + letter, "source": source, "method": method,
            "protocol": protocol, "control": control,
            "unknown_weights": [0.0] if method == "prototype" else [0.0, .25, .5, .75, 1.0],
            "margin_weights": [0.0] if method == "prototype" else [0.0, .5]})
    return rows


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{" + str(length) + r"}", value) is not None


def validate_comparison_suite(suite: dict) -> None:
    required = {"schema_version", "experiment_code", "run_name", "readiness_config", "output_root",
                "calibration_protocol", "threshold_candidates", "probability_temperature", "expanded_arm",
                "sources", "controls", "recipes", "selection_policy", "decision_rule"}
    if (not isinstance(suite, dict) or set(suite) != required or suite["schema_version"] != 1
            or suite["experiment_code"] != "S006" or not isinstance(suite["run_name"], str)
            or not re.fullmatch(r"S006-[A-Za-z0-9_-]+", suite["run_name"])
            or suite["readiness_config"] != "configs/train/campp_coverage.json"
            or suite["output_root"] != "artifacts/training" or suite["calibration_protocol"] != FIXED
            or type(suite["expanded_arm"]) is not bool or type(suite["threshold_candidates"]) is not int
            or suite["threshold_candidates"] != 201 or type(suite["probability_temperature"]) is not float
            or suite["probability_temperature"] != .05 or suite["selection_policy"] != SELECTION_POLICY
            or suite["decision_rule"] != DECISION_RULE
            or suite["recipes"] != comparison_recipes(suite["expanded_arm"])
            or not isinstance(suite["sources"], dict) or set(suite["sources"]) != set(SOURCE_CONFIGS)):
        raise ValueError("S006 requires its exact preregistered recipes, decision rule and two adapted sources")
    source_keys = {"config", "run", "parent_run_id", "signature", "git_commit", "export_manifest_paths",
                   "export_manifest_sha256", "completed_steps", "folds"}
    identities, checkpoints = set(), set()
    for name, source in suite["sources"].items():
        if (not isinstance(source, dict) or set(source) != source_keys or source["config"] != SOURCE_CONFIGS[name]
                or not isinstance(source["run"], str) or not source["run"].startswith("artifacts/training/" + name + "_")
                or not _hex(source["parent_run_id"], 32) or not _hex(source["signature"], 64)
                or not _hex(source["git_commit"], 40) or not _hex(source["export_manifest_sha256"], 64)
                or type(source["completed_steps"]) is not int or source["completed_steps"] != HEAD_STEPS[name] + 500
                or not isinstance(source["export_manifest_paths"], list) or not source["export_manifest_paths"]
                or any(not isinstance(p, str) or not p.startswith("artifacts/") for p in source["export_manifest_paths"])
                or not isinstance(source["folds"], dict) or set(source["folds"]) != {"0", "1"}):
            raise ValueError("Each source must bind its own completed curriculum, export and original configuration")
        for fold in source["folds"].values():
            if (not isinstance(fold, dict) or set(fold) != {"child_run_id", "checkpoint_sha256"}
                    or not _hex(fold["child_run_id"], 32) or not _hex(fold["checkpoint_sha256"], 64)):
                raise ValueError("Each source fold requires its completed child and final checkpoint SHA")
            identities.add(fold["child_run_id"])
            checkpoints.add(fold["checkpoint_sha256"])
        identities.add(source["parent_run_id"])
    if len(identities) != 6 or len(checkpoints) != 4:
        raise ValueError("Both adapted parents, children and all four checkpoints must be distinct")
    controls = suite["controls"]
    if not isinstance(controls, dict) or set(controls) != ({"fixed", "expanded"} if suite["expanded_arm"] else {"fixed"}):
        raise ValueError("Enabled comparison arms require their own completed historical controls")
    for arm, control in controls.items():
        if (not isinstance(control, dict) or set(control) != {"run", "parent_run_id", "child_run_id", "recipe_id",
                "report_sha256", "recipe_report_sha256", "source_provenance_sha256", "resolved_config_sha256"}
                or control["recipe_id"] != ("S004d" if arm == "fixed" else "S005d")
                or not isinstance(control["run"], str) or not control["run"].startswith("artifacts/training/" + control["recipe_id"][:4] + "_")
                or not _hex(control["parent_run_id"], 32) or not _hex(control["child_run_id"], 32)
                or control["parent_run_id"] == control["child_run_id"]
                or any(not _hex(control[key], 64) for key in ("report_sha256", "recipe_report_sha256",
                                                            "source_provenance_sha256", "resolved_config_sha256"))):
            raise ValueError("Historical controls need explicit completed run IDs and captured artifact hashes")


def validate_comparison_contracts(suite: dict, contracts: dict) -> None:
    """The only training-policy difference permitted is the preregistered head duration."""
    validate_comparison_suite(suite)
    first, second = contracts["F003"], contracts["F004"]
    for key in ("model", "input_hashes", "manifest", "folds", "roles", "labels"):
        if first[key] != second[key] or first[key] != contracts["readiness"][key]:
            raise ValueError("Both adapted encoders and readiness must share original data/model/roles/labels")
    comparable = []
    for name in SOURCE_CONFIGS:
        config = contracts[name]["config"]
        fit = config["fit"]
        if (config["experiment_code"] != name or config["mode"] != "fine_tune" or config["fold_ids"] != [0, 1]
                or config["inference"] != {"seconds": 180.0, "maximum_windows": 1}
                or config["expected_source_files"] != 4529 or config["known_classes"] != 446
                or config["evaluation_classes"] != 447 or fit["mixed_precision"] is not False
                or fit["epoch_selection"] != "fixed_steps_no_outer_selection"
                or fit["adaptation_schedule"]["head_only_steps"] != HEAD_STEPS[name]
                or fit["epochs"] * fit["steps_per_epoch"] != 500
                or adaptation_total_steps(fit) != suite["sources"][name]["completed_steps"]):
            raise ValueError("Source-specific FP32 final curriculum must be F003 100+500 or F004 600+500")
        normalized = deepcopy(config)
        for key in ("experiment_code", "run_name", "hypothesis"):
            normalized.pop(key, None)
        normalized["fit"]["adaptation_schedule"]["head_only_steps"] = 0
        comparable.append(normalized)
        for outer in (0, 1):
            assert_fixed_role_groups(contracts[name], outer)
    if comparable[0] != comparable[1]:
        raise ValueError("Unregistered training-policy difference beyond the head-only duration")


def load_comparison_contracts(root: Path, suite: dict, *, verify_audio=False) -> dict:
    validate_comparison_suite(suite)
    contracts = {"readiness": load_contract(project_path(root, suite["readiness_config"], "configs/train"),
                                           root, verify_audio=verify_audio)}
    for name, source in suite["sources"].items():
        contracts[name] = load_contract(project_path(root, source["config"], "configs/train"), root)
        project_path(root, source["run"], "artifacts/training", exists=False)
        for value in source["export_manifest_paths"]:
            project_path(root, value, "artifacts", exists=False)
    for control in suite["controls"].values():
        project_path(root, control["run"], "artifacts/training", exists=False)
    validate_comparison_contracts(suite, contracts)
    return contracts


def _read_json(directory: Path, relative: str) -> dict:
    path = directory / relative
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
            or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(directory))):
        raise ValueError("Evidence must be a confined regular file")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source_snapshot(directory: Path, source: dict, original: dict, inventory: dict) -> dict:
    """Read the immutable captured ZIP; never replace it with today's source."""
    attempt = _read_json(directory, "experiment_state.json")["attempt"]
    prefix = f"tracking/{attempt}/artifacts/"
    if any(prefix + name not in inventory for name in ("source_manifest.json", "source_snapshot.zip", "resolved_config.json")):
        raise ValueError("Audited export must contain the complete original source snapshot")
    if _read_json(directory, prefix + "resolved_config.json") != original:
        raise ValueError("Captured parent configuration differs from the original source contract")
    manifest = _read_json(directory, prefix + "source_manifest.json")
    archive_path = directory / (prefix + "source_snapshot.zip")
    if (manifest.get("schema_version") != 2 or manifest.get("archive_format") != "zip"
            or manifest.get("src_dirty") is not False or manifest.get("git_commit") != source["git_commit"]
            or manifest.get("archive_sha256") != file_sha256(archive_path)):
        raise ValueError("Captured source snapshot hash/revision/clean state differs")
    files = {row["path"]: row for row in manifest["files"]}
    if len(files) != len(manifest["files"]) or len(files) != manifest["file_count"]:
        raise ValueError("Captured source inventory has duplicate or missing files")
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) != len(files) or set(archive.namelist()) != set(files):
            raise ValueError("Captured source ZIP differs from its complete inventory")
        for entry in archive.infolist():
            name, row = entry.filename, files[entry.filename]
            part = PurePosixPath(name)
            if (not name.startswith("src/") or "\\" in name or ":" in name or part.as_posix() != name
                    or ".." in part.parts or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.file_size != row["bytes"]
                    or hashlib.sha256(archive.read(entry)).hexdigest() != row["sha256"]):
                raise ValueError("Captured source ZIP entry has unsafe path or changed bytes")
    for name, digest in original["code_hashes"].items():
        if name.startswith("src/") and files.get(name, {}).get("sha256") != digest:
            raise ValueError("Original contract code hashes disagree with its actual source snapshot")
    return {"git_commit": source["git_commit"], "archive_sha256": manifest["archive_sha256"],
            "verified_source_files": len(files), "captured_snapshot_preserved": True}


def verify_final_schedule(directory: Path, source: dict, contract: dict, inventory: dict, *, checkpoint_loader=None) -> dict:
    """Inspect final checkpoint metadata with weights_only=True; no model is instantiated."""
    if checkpoint_loader is None:
        import torch
        checkpoint_loader = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    fit = contract["config"]["fit"]
    total = adaptation_total_steps(fit)
    expected = adaptation_checkpoint_state(fit, total)
    report = _read_json(directory, "experiment_report.json")
    proof = {}
    for outer in (0, 1):
        relative = f"fold_{outer}/last.pt"
        path = directory / relative
        if (inventory.get(relative, {}).get("sha256") != source["folds"][str(outer)]["checkpoint_sha256"]
                or file_sha256(path) != source["folds"][str(outer)]["checkpoint_sha256"]):
            raise ValueError("Final checkpoint bytes no longer match the audited fold")
        saved = checkpoint_loader(path)
        matches = [row for row in report["folds"] if row["outer_fold"] == outer]
        recorded = matches[0].get("fit", {}) if len(matches) == 1 else {}
        if (saved.get("format_version") != 2 or saved.get("signature") != source["signature"]
                or saved.get("outer_fold") != outer or saved.get("completed_steps") != total
                or source["completed_steps"] != total or saved.get("schedule_state") != expected
                or saved.get("adaptation_schedule") != fit["adaptation_schedule"]
                or recorded.get("schedule_state") != expected or recorded.get("completed_steps") != total
                or recorded.get("adaptation_schedule") != fit["adaptation_schedule"]
                or recorded.get("initial_step") != 0
                or recorded.get("epoch_selection") != "fixed_steps_no_outer_selection"):
            raise ValueError("Final checkpoint/report does not represent the complete source-specific curriculum")
        proof[str(outer)] = expected
        del saved
    return proof


def verified_historical_control(root: Path, suite: dict, arm: str, f003_proof: dict) -> tuple[Path, dict]:
    """Pin completed S004d/S005d and prove they used these exact F003 fold caches."""
    specification = suite["controls"][arm]
    directory = project_path(root, specification["run"], "artifacts/training")
    recipe_id = specification["recipe_id"]
    pinned = {"experiment_report.json": "report_sha256", "source_provenance.json": "source_provenance_sha256",
              "tracking/artifacts/resolved_config.json": "resolved_config_sha256",
              recipe_id + "/experiment_report.json": "recipe_report_sha256"}
    for relative, key in pinned.items():
        _read_json(directory, relative)
        if file_sha256(directory / relative) != specification[key]:
            raise ValueError("Historical control artifact differs from its pinned hash")
    state, report = _read_json(directory, "experiment_state.json"), _read_json(directory, "experiment_report.json")
    parent = _read_json(directory, "tracking/run_state.json")
    old_suite = _read_json(directory, "tracking/artifacts/resolved_config.json")["suite"]
    if arm == "fixed":
        validate_adapted_suite(old_suite)
    else:
        from speaker_id.training.expanded_gallery import validate_expanded_suite
        validate_expanded_suite(old_suite)
        fixed = suite["controls"]["fixed"]
        if (old_suite["control"]["run"] != fixed["run"]
                or old_suite["control"]["parent_run_id"] != fixed["parent_run_id"]
                or old_suite["control"]["recipes"]["adapted"] != {"id": "S004d", "child_run_id": fixed["child_run_id"]}):
            raise ValueError("Expanded control must descend from the same completed S004d control")
    expected = next(row for row in old_suite["recipes"] if row["id"] == recipe_id)
    child = _read_json(directory, recipe_id + "/tracking/run_state.json")
    captured = _read_json(directory, recipe_id + "/tracking/artifacts/resolved_config.json")
    recipe_report = _read_json(directory, recipe_id + "/experiment_report.json")
    if (state.get("status") != "complete" or report.get("status") != "complete"
            or state.get("parent_run_id") != specification["parent_run_id"]
            or report.get("parent_run_id") != specification["parent_run_id"]
            or parent.get("run_id") != specification["parent_run_id"] or parent.get("remote_status") != "FINISHED"
            or old_suite["sources"]["adapted"] != suite["sources"]["F003"]
            or _read_json(directory, "source_provenance.json").get("adapted") != f003_proof
            or set(report.get("source_control_checks", {})) != {"public", "adapted"}
            or not all(row.get("exact_prediction_reproduction") is True for row in report["source_control_checks"].values())
            or child.get("run_id") != specification["child_run_id"] or child.get("remote_status") != "FINISHED"
            or child.get("tags", {}).get("mlflow.parentRunId") != specification["parent_run_id"]
            or captured.get("suite") != old_suite or captured.get("recipe") != expected
            or recipe_report.get("recipe") != expected
            or sorted(row["outer_fold"] for row in recipe_report["folds"]) != [0, 1]):
        raise ValueError("Historical control is incomplete or has changed source/cache/recipe identity")
    return directory / recipe_id, {**specification, "recipe": expected, "source_identity_exact": True}


def verify_control_fold(directory: Path, outer: int, predictions: list[dict], metrics: dict, calibration: dict) -> dict:
    evidence = verify_predictions(directory / f"fold_{outer}/predictions.csv", predictions)
    recorded = _read_json(directory, f"fold_{outer}/evaluation.json")
    old_calibration = _read_json(directory, f"fold_{outer}/calibration.json")
    if "selected" in old_calibration:
        expected = old_calibration["selected"]
    else:
        best = max(old_calibration["curve"], key=lambda row: (row["inner_macro_f1_447"], row["threshold"]))
        expected = {"unknown_weight": 0.0, "margin_weight": 0.0, **best}
        if old_calibration["threshold"] != expected["threshold"]:
            raise ValueError("Source prototype threshold no longer matches its recorded inner selection")
    if recorded["outer"] != metrics or expected != calibration:
        raise ValueError("Control predictions match but exact metrics/calibration do not")
    return {**evidence, "exact_metrics_reproduction": True, "exact_calibration_reproduction": True}


def require_controls_before(recipe: dict, completed: dict) -> None:
    required = {"F003_prototype", "F004_prototype", "S004d"}
    if recipe["id"] == "S006f":
        required.add("S005d")
    if recipe["control"] is None and any(completed.get(key, {}).get("exact_prediction_reproduction") is not True for key in required):
        raise ValueError("All preregistered exact source controls must pass before the new F004 recipe")


def paired_diagnostics(manifest: list[dict], before: list[dict], after: list[dict], labels: list[str]) -> dict:
    old, new = ({row["audio_file"]: row["speaker_id"] for row in rows} for rows in (before, after))
    expected = {row["audio_file"] for row in manifest}
    if len(old) != len(before) or len(new) != len(after) or set(old) != expected or set(new) != expected:
        raise ValueError("Paired diagnostics require exact original filename coverage")
    subsets = {
        "all": lambda row: True,
        "zero_signal": lambda row: not truth(row["has_nonzero_signal"]),
        "nonzero_under_5s": lambda row: truth(row["has_nonzero_signal"]) and float(row["duration_seconds"]) < 5,
        "nonzero_5s_to_30s": lambda row: truth(row["has_nonzero_signal"]) and 5 <= float(row["duration_seconds"]) < 30,
        "nonzero_at_least_30s": lambda row: truth(row["has_nonzero_signal"]) and float(row["duration_seconds"]) >= 30,
        "nonzero_below_minus_35_dbfs": lambda row: truth(row["has_nonzero_signal"]) and float(row["mono_rms_dbfs"]) < -35,
        "nonzero_below_minus_50_dbfs": lambda row: truth(row["has_nonzero_signal"]) and float(row["mono_rms_dbfs"]) < -50,
    }
    result = {}
    for name, condition in subsets.items():
        rows = [row for row in manifest if condition(row)]
        if not rows:
            continue
        result[name] = {"files": len(rows),
            "corrected": sum(old[row["audio_file"]] != row["speaker_id"] == new[row["audio_file"]] for row in rows),
            "regressed": sum(old[row["audio_file"]] == row["speaker_id"] != new[row["audio_file"]] for row in rows),
            "changed_predictions": sum(old[row["audio_file"]] != new[row["audio_file"]] for row in rows)}
        for side, indexed in (("before", old), ("after", new)):
            metrics = score_predictions(rows, [{"audio_file": row["audio_file"], "speaker_id": indexed[row["audio_file"]]} for row in rows], labels)
            result[name][side] = {key: metrics[key] for key in ("macro_f1", "accuracy", "errors")}
    return result


def known_ranking_diagnostics(references: list[dict], scores: np.ndarray, probabilities: np.ndarray,
                              valid: np.ndarray, labels: list[str]) -> dict:
    """Observational outer diagnostic, called only after inner policy is fixed."""
    if (scores.shape != (len(references), len(labels) - 1)
            or probabilities.shape != (len(references), len(labels)) or valid.shape != (len(references),)
            or not np.isfinite(scores).all() or not np.isfinite(probabilities).all()):
        raise ValueError("Known-rank diagnostic requires aligned finite outer arrays")
    indexed = {label: index for index, label in enumerate(labels)}
    actual = np.asarray([indexed[row["speaker_id"]] for row in references])
    eligible = valid & (actual > 0)
    rank1, predicted = scores.argmax(axis=1) + 1, probabilities.argmax(axis=1)
    correct_rank, rejected = rank1 == actual, predicted == 0
    if np.any(eligible & ~rejected & (predicted != rank1)):
        raise ValueError("Reference scorer changed known ranking after calibration")
    count = int(eligible.sum())
    correct = int((eligible & correct_rank).sum())
    return {"nonzero_known_files": count, "rank1_correct": correct,
            "rank1_accuracy": correct / count if count else 0.0,
            "rejected_correct_rank": int((eligible & rejected & correct_rank).sum()),
            "rejected_wrong_rank": int((eligible & rejected & ~correct_rank).sum()),
            "accepted_wrong": int((eligible & ~rejected & ~correct_rank).sum()),
            "accepted_correct": int((eligible & ~rejected & correct_rank).sum()),
            "role": "observational diagnostic after inner selection; not a selection metric"}


def compare_results(before: dict, after: dict) -> dict:
    old = {row["outer_fold"]: row["outer"]["macro_f1"] for row in before["folds"]}
    new = {row["outer_fold"]: row["outer"]["macro_f1"] for row in after["folds"]}
    if set(old) != {0, 1} or set(new) != {0, 1}:
        raise ValueError("The paired decision requires both original folds")
    delta = after["oof"]["macro_f1"] - before["oof"]["macro_f1"]
    folds = {str(key): new[key] - old[key] for key in old}
    return {"pooled_macro_f1_delta": delta, "fold_macro_f1_deltas": folds,
            "engineering_gate_passed": delta >= .003 and min(folds.values()) >= -.005,
            "decision_rule": dict(DECISION_RULE), "statistical_significance_claimed": False}


def _verify_remote_evidence(client, binding, requests: list[tuple], destination: Path) -> list[dict]:
    """Read-only source run/status and byte round-trip verification in the owned experiment."""
    records = []
    for run_id, parent_id, artifacts in requests:
        remote = client.get_run(run_id)
        if (remote.info.status != "FINISHED" or str(remote.info.experiment_id) != str(binding.experiment_id)
                or remote.data.tags.get("speaker_id.project") != binding.project
                or remote.data.tags.get("speaker_id.scope_id") != binding.scope_id
                or (parent_id is not None and remote.data.tags.get("mlflow.parentRunId") != parent_id)):
            raise ValueError("A required completed source/control MLflow run has wrong status or ownership")
        for index, (remote_path, local_path) in enumerate(artifacts):
            folder = destination / run_id / str(index)
            folder.mkdir(parents=True, exist_ok=False)
            downloaded = Path(client.download_artifacts(run_id, remote_path, str(folder)))
            if not downloaded.is_file() or file_sha256(downloaded) != file_sha256(local_path):
                raise ValueError("Completed MLflow evidence bytes differ from the attested local source/control")
            records.append({"run_id": run_id, "artifact": remote_path, "sha256": file_sha256(local_path), "status": "FINISHED"})
    return records


def _remote_requests(suite: dict, source_directories: dict, control_directories: dict) -> list[tuple]:
    requests = []
    for name, source in suite["sources"].items():
        directory = source_directories[name]
        attempt = _read_json(directory, "experiment_state.json")["attempt"]
        requests.append((source["parent_run_id"], None, [
            ("resolved_config.json", directory / f"tracking/{attempt}/artifacts/resolved_config.json"),
            ("source_manifest.json", directory / f"tracking/{attempt}/artifacts/source_manifest.json"),
            ("source_snapshot.zip", directory / f"tracking/{attempt}/artifacts/source_snapshot.zip"),
            ("experiment_report.json", directory / "experiment_report.json"),
            ("oof_predictions.csv", directory / "oof_predictions.csv")]))
        for outer in (0, 1):
            # Training children put calibration and predictions under evaluation/.
            child_artifacts = [("resolved_config.json", directory / f"fold_{outer}/tracking/{attempt}/artifacts/resolved_config.json")]
            child_artifacts += [("evaluation/" + filename, directory / f"fold_{outer}/{filename}")
                               for filename in ("evaluation.json", "calibration.json", "predictions.csv")]
            requests.append((source["folds"][str(outer)]["child_run_id"], source["parent_run_id"], child_artifacts))
    for arm, control in suite["controls"].items():
        recipe_dir, directory = control_directories[arm], control_directories[arm].parent
        requests.append((control["parent_run_id"], None, [(name, directory / path) for name, path in (
            ("experiment_report.json", "experiment_report.json"), ("resolved_config.json", "tracking/artifacts/resolved_config.json"),
            ("source_provenance.json", "source_provenance.json"))]))
        files = ["experiment_report.json", "oof_predictions.csv"] + [f"fold_{outer}/{name}" for outer in (0, 1)
            for name in ("evaluation.json", "calibration.json", "predictions.csv")]
        requests.append((control["child_run_id"], control["parent_run_id"], [(name, recipe_dir / name) for name in files]))
    return requests


def execute_comparison(root: Path, config_path: Path, suite: dict, contracts: dict, binding_path: Path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.training.plots import evaluation_plots

    validate_comparison_contracts(suite, contracts)
    readiness = contracts["readiness"]
    validate_readiness_for_execution(root, readiness)
    if (os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available()
            or "3090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("S006 may execute only on the authorized RTX 3090 instance")
    torch.set_num_threads(4)
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    output = root / "artifacts/training" / ("S006_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"suite_config": config_path, "launcher": root / "scripts/score_adaptation_comparison.py"}
    inputs.update({key: root / readiness["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")})
    for name, source in suite["sources"].items():
        inputs[name + "_config"] = root / source["config"]
        inputs[name + "_captured_config"] = project_path(root, source["run"], "artifacts/training") / "resolved_config.json"
    resolved = {"suite": suite, "contracts": {name: {key: value[key] for key in ("config", "model", "input_hashes", "code_hashes", "signature")}
                                             for name, value in contracts.items()}, "limitations": LIMITATIONS}
    write_json(output / "resolved_config.json", resolved)
    common = {"project_root": root, "binding": binding, "input_paths": inputs,
              "run_kind": "adaptation_curriculum_comparison", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["run_name"], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        arrays, provenance, source_directories, schedules, snapshots = {}, {}, {}, {}, {}
        for name, source in suite["sources"].items():
            directory, exported, inventory = verified_export_inventory(root, source)
            source_directories[name] = directory
            original, fold_proofs = validate_adapted_identity(directory, source, contracts[name], inventory)
            snapshots[name] = verify_source_snapshot(directory, source, original, inventory)
            schedules[name] = verify_final_schedule(directory, source, contracts[name], inventory)
            arrays[name] = {}
            for outer in (0, 1):
                vectors, valid, evidence = load_adapted_fold_cache(directory, contracts[name], source["signature"], inventory, outer)
                arrays[name][outer] = vectors, valid
                fold_proofs[str(outer)]["files"] = evidence
            provenance[name] = {"parent_run_id": source["parent_run_id"], "source_signature": source["signature"],
                "git_commit": source["git_commit"], "export_manifest_sha256": source["export_manifest_sha256"],
                "verified_export_files": len(inventory), "folds": fold_proofs}
            write_json(output / (name + "_audited_export_manifest.json"), exported)
            parent.add_artifact(output / (name + "_audited_export_manifest.json"))
            print(json.dumps({"stage": "adapted_source_verified", "source": name, "files": len(inventory)}), flush=True)
        for outer in (0, 1):
            if not np.array_equal(arrays["F003"][outer][1], arrays["F004"][outer][1]):
                raise ValueError("The two source encoders changed zero-signal eligibility")
        control_directories, control_proofs = {}, {}
        for arm in suite["controls"]:
            control_directories[arm], control_proofs[arm] = verified_historical_control(root, suite, arm, provenance["F003"])
        remote = _verify_remote_evidence(parent.client, binding,
            _remote_requests(suite, source_directories, control_directories), output / "verified_remote_evidence")
        proof = {"sources": provenance, "snapshots": snapshots, "final_schedules": schedules,
                 "historical_controls": control_proofs, "remote_evidence": remote,
                 "no_encoder_fitting_or_audio_extraction": True}
        write_json(output / "source_provenance.json", proof)
        parent.add_artifact(output / "source_provenance.json")
        for name, value in inputs.items():
            if name.endswith("_config") or name == "launcher":
                parent.add_artifact(value, "input_configs/" + name + value.suffix)
        targets = {"F003_prototype": source_directories["F003"], "F004_prototype": source_directories["F004"],
                   "S004d": control_directories["fixed"]}
        if suite["expanded_arm"]:
            targets["S005d"] = control_directories["expanded"]
        contract = contracts["F003"]
        labels, manifest = contract["labels"], contract["manifest"]
        label_index = {label: index for index, label in enumerate(labels)}
        checks, results, prediction_sets = {}, [], {}
        for recipe in suite["recipes"]:
            require_controls_before(recipe, checks)
            name = recipe["source"]
            recipe_path = output / recipe["id"]
            recipe_path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=recipe_path / "tracking", parent_run_id=parent.run_id,
                run_name=recipe["id"] + "-" + name + "-" + recipe["method"], config={**resolved, "recipe": recipe}, **common)
            child.flush(strict=True)
            child.add_artifact(output / "source_provenance.json")
            all_predictions, fold_reports = [], []
            for outer in (0, 1):
                fold_path = recipe_path / f"fold_{outer}"
                fold_path.mkdir()
                vectors, valid = arrays[name][outer]
                if recipe["protocol"] == EXPANDED:
                    scores = heldout_reference_scores(contracts[name], vectors, valid, outer)
                else:
                    scores = fixed_role_scores(contracts[name], vectors, valid, outer, recipe["method"])
                query, evaluation = scores["calibration_indices"], scores["outer_indices"]
                inner_truth = np.asarray([label_index[manifest[int(i)]["speaker_id"]] for i in query])
                if recipe["method"] == "prototype":
                    threshold, original_curve = fit_threshold(scores["inner_known_scores"], inner_truth, 201)
                    curve = [{"unknown_weight": 0.0, "margin_weight": 0.0, **row} for row in original_curve]
                    calibration = next(row for row in curve if row["threshold"] == threshold)
                    probabilities = score_probabilities(scores["outer_known_scores"], threshold, .05, valid[evaluation])
                else:
                    calibration, curve = calibrate_gate(scores["inner_known_scores"], inner_truth,
                        scores["inner_unknown_similarity"], recipe["unknown_weights"], recipe["margin_weights"], 201)
                    probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"], calibration, valid[evaluation], .05)
                references = [manifest[int(i)] for i in evaluation]
                predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(guess)]}
                               for row, guess in zip(references, probabilities.argmax(axis=1))]
                metrics = score_predictions(references, predictions, labels)
                ranking = known_ranking_diagnostics(references, scores["outer_known_scores"], probabilities, valid[evaluation], labels)
                reproduction = (verify_control_fold(targets[recipe["control"]], outer, predictions, metrics, calibration)
                                if recipe["control"] else None)
                counts = scores["reference_counts"]
                support = {key: value for key, value in counts.items() if isinstance(value, np.ndarray)}
                report = {"recipe": recipe, "outer_fold": outer, "outer": metrics, "calibration": calibration,
                    "threshold": calibration["threshold"], "threshold_axis_label": "Original held-out query gate score",
                    "reference_counts": {key: value for key, value in counts.items() if key not in support},
                    "nonzero_known_ranking": ranking,
                    "inner_query_files": len(query), "role_proof": assert_fixed_role_groups(contracts[name], outer),
                    "source_reproduction": reproduction, "checkpoint_binding": {key: value for key, value in provenance[name]["folds"][str(outer)].items() if key != "files"},
                    "probability_semantics": "normalized scores, not calibrated posteriors"}
                if support:
                    np.savez_compressed(fold_path / "reference_support.npz", **support, calibration_indices=query, known_labels=np.asarray(labels[1:]))
                    report["reference_counts"]["support_arrays_artifact"] = "reference_support.npz"
                    report["scoring_provenance"] = scores["provenance"]
                    child.add_artifact(fold_path / "reference_support.npz", f"fold_{outer}/reference_support.npz")
                write_json(fold_path / "evaluation.json", report)
                write_json(fold_path / "calibration.json", {"selected": calibration, "curve": curve})
                write_csv(fold_path / "predictions.csv", predictions)
                write_csv(fold_path / "per_class.csv", metrics["per_class"])
                write_csv(fold_path / "file_diagnostics.csv", [{"audio_file": row["audio_file"], "true_speaker_id": row["speaker_id"],
                    "predicted_speaker_id": prediction["speaker_id"], "correct": row["speaker_id"] == prediction["speaker_id"],
                    "duration_seconds": row["duration_seconds"], "mono_rms_dbfs": row["mono_rms_dbfs"]} for row, prediction in zip(references, predictions)])
                np.savez_compressed(fold_path / "outer_probabilities.npz", probabilities=probabilities,
                                    audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(labels))
                for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "file_diagnostics.csv", "outer_probabilities.npz"):
                    child.add_artifact(fold_path / filename, f"fold_{outer}/" + filename)
                selected_curve = [row for row in curve if all(row[key] == calibration[key] for key in ("unknown_weight", "margin_weight"))]
                for figure in evaluation_plots(fold_path, report, selected_curve):
                    child.add_artifact(figure, f"fold_{outer}/figures/" + figure.name)
                child.log_metrics({f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"], f"fold_{outer}/outer_accuracy": metrics["accuracy"],
                    **{f"fold_{outer}/inner_{key}": value for key, value in calibration.items()},
                    **{f"fold_{outer}/known_ranking/{key}": value for key, value in ranking.items() if key != "role"},
                    **{f"fold_{outer}/{key}": value for key, value in metrics["errors"].items()}}, sync=True)
                fold_reports.append(report)
                all_predictions.extend(predictions)
                print(json.dumps({"stage": "adaptation_comparison", "recipe": recipe["id"], "fold": outer, "control_exact": reproduction is not None}), flush=True)
            pooled = score_predictions(manifest, all_predictions, labels)
            if recipe["control"]:
                check = verify_predictions(targets[recipe["control"]] / "oof_predictions.csv", all_predictions)
                old_report = _read_json(targets[recipe["control"]], "experiment_report.json")
                if old_report["oof"] != pooled:
                    raise ValueError("Pooled source/control metrics differ despite exact predictions")
                checks[recipe["control"]] = {**check, "exact_fold_metrics_and_calibration": True, "exact_pooled_metrics": True}
                write_json(output / "source_control_checks.json", checks)
            report = {"recipe": recipe, "oof": pooled, "folds": fold_reports, "source_control_checks": deepcopy(checks), "limitations": LIMITATIONS}
            write_json(recipe_path / "experiment_report.json", report)
            write_csv(recipe_path / "oof_predictions.csv", all_predictions)
            write_csv(recipe_path / "oof_per_class.csv", pooled["per_class"])
            for filename in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
                child.add_artifact(recipe_path / filename, filename)
            child.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                               **{"oof/" + key: value for key, value in pooled["errors"].items()}}, sync=False)
            child.write_report(report, markdown=f"# {recipe['id']} {name}\n\n{recipe['protocol']}; original held-out queries only. OOF Macro-F1: {pooled['macro_f1']:.6f}. No encoder fitting or outer-label selection.\n")
            child.finish("FINISHED", strict=True)
            child = None
            results.append(report)
            prediction_sets[recipe["id"]] = all_predictions
        by_recipe = {row["recipe"]["id"]: row for row in results}
        comparisons = {}
        for arm, first, second in [("fixed", "S006c", "S006d")] + ([("expanded", "S006e", "S006f")] if suite["expanded_arm"] else []):
            comparisons[arm] = {**compare_results(by_recipe[first], by_recipe[second]),
                "paired_quality_slices": paired_diagnostics(manifest, prediction_sets[first], prediction_sets[second], labels),
                "primary_decision": arm == "fixed"}
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": results,
                  "source_control_checks": checks, "comparisons": comparisons, "selection_policy": SELECTION_POLICY,
                  "elapsed_seconds": time.monotonic() - started, "limitations": LIMITATIONS}
        write_json(output / "experiment_report.json", report)
        for filename in ("experiment_report.json", "source_control_checks.json", "resolved_config.json"):
            parent.add_artifact(output / filename)
        for result in results:
            parent.log_metrics({result["recipe"]["id"] + "/oof_macro_f1_447": result["oof"]["macro_f1"]}, sync=False)
        parent.log_metrics({"primary/pooled_macro_f1_delta": comparisons["fixed"]["pooled_macro_f1_delta"],
                            "primary/engineering_gate_passed": int(comparisons["fixed"]["engineering_gate_passed"])}, sync=False)
        parent.write_report(report, markdown="# S006 F003/F004 curriculum comparison\n\nExact prototype and historical controls passed. Coefficients use only original calibration queries. The engineering gate is a development criterion, not statistical significance. No automatic package promotion.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), "parent_run_id": parent.run_id,
                "results": {row["recipe"]["id"]: row["oof"]["macro_f1"] for row in results}, "comparisons": comparisons}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id, "error_type": type(error).__name__, "error": str(error)}
        write_json(output / "failure.json", failure)
        if child is not None:
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        parent.write_report(failure)
        parent.finish("FAILED", strict=False)
        write_json(output / "experiment_state.json", failure)
        raise
