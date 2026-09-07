"""S008: five preregistered same-reference policies for two public encoders.

Historical candidate cache signatures are verified from their captured source
identity, never regenerated from today's all-source fingerprint. No model is
loaded, optimized or used to extract audio by this module.
"""
from __future__ import annotations

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

from speaker_id.candidates.campp_advanced import validate_advanced_config
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.adapted_scoring import verified_export_inventory
from speaker_id.training.adaptation_comparison import _verify_remote_evidence, known_ranking_diagnostics, paired_diagnostics, verify_control_fold
from speaker_id.training.candidate_comparison import (
    SOURCE, UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, _public_remote_requests,
    require_public_control, validate_candidate_suite, verify_candidate_cache,
)
from speaker_id.training.contracts import load_contract
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.frozen_suite import validated_cache
from speaker_id.training.fusion import require_aligned_scores
from speaker_id.training.fusion_suite import project_path, verify_control_run, verify_predictions
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities
from speaker_id.training.runner import write_csv, write_json


ALPHAS = (0.0, .25, .5, .75, 1.0)
TIE_ORDER = (0.0, 1.0, .25, .5, .75)
SELECTION_POLICY = "Select alpha and gate on group-excluded inner queries only; exact S002f/S007b endpoint controls precede mixed outer evaluation."
EXECUTION_POLICY = "Manual dispatch after completed S006 and S007 only if S006d, S006f and S007b all remain below pooled OOF Macro-F1 0.965; no automatic chaining."
LIMITATIONS = [
    "Repeated development selection is not an untouched test or a hidden leaderboard estimate.",
    "Both models must remain public frozen encoders; adapted-query crossfit is forbidden.",
    "Crossfit removes the whole query group while outer scoring restores full training-reference support.",
    "Selected alpha may differ by fold; pooled OOF evaluates the inner-selection procedure.",
    "Mixed policies require two encoders at inference; an endpoint policy requires only one.",
]


def validate_fusion_suite(suite: dict) -> None:
    required = {"schema_version", "experiment_code", "run_name", "readiness_config", "output_root", "source_public",
                "source_advanced", "alphas", "alpha_tie_order", "unknown_weights", "margin_weights",
                "threshold_candidates", "probability_temperature", "selection_policy", "execution_policy"}
    if (not isinstance(suite, dict) or set(suite) != required or suite["schema_version"] != 1
            or suite["experiment_code"] != "S008" or suite["run_name"] != "S008-campp-public512-advanced192-paired-reference"
            or suite["readiness_config"] != "configs/train/campp_coverage.json" or suite["output_root"] != "artifacts/training"
            or suite["source_public"] != SOURCE or suite["alphas"] != list(ALPHAS) or suite["alpha_tie_order"] != list(TIE_ORDER)
            or suite["unknown_weights"] != UNKNOWN_WEIGHTS or suite["margin_weights"] != MARGIN_WEIGHTS
            or type(suite["threshold_candidates"]) is not int or suite["threshold_candidates"] != 201
            or type(suite["probability_temperature"]) is not float or suite["probability_temperature"] != .05
            or suite["selection_policy"] != SELECTION_POLICY or suite["execution_policy"] != EXECUTION_POLICY):
        raise ValueError("S008 requires exactly the preregistered same-reference policies and conditional manual dispatch")
    source = suite["source_advanced"]
    if (not isinstance(source, dict) or set(source) != {"run", "parent_run_id", "source_signature", "git_commit", "export_manifest_paths", "export_manifest_sha256", "children"}
            or not isinstance(source["run"], str) or not source["run"].startswith("artifacts/training/S007_")
            or not isinstance(source["export_manifest_paths"], list) or not source["export_manifest_paths"]
            or any(not isinstance(path, str) or not path.startswith("artifacts/") for path in source["export_manifest_paths"])
            or not isinstance(source["children"], dict) or set(source["children"]) != {"S007a", "S007b"}):
        raise ValueError("S008 needs an actual completed S007 source and both children; no placeholders")
    for key, length in (("parent_run_id", 32), ("source_signature", 64), ("git_commit", 40), ("export_manifest_sha256", 64)):
        if not isinstance(source[key], str) or not re.fullmatch(r"[a-f0-9]{" + str(length) + "}", source[key]):
            raise ValueError("Invalid completed S007 identity")
    if (any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value) for value in source["children"].values())
            or len({source["parent_run_id"], *source["children"].values()}) != 3):
        raise ValueError("S007 parent and both completed child IDs must be distinct")


def weighted_encoder_pair(public, advanced, valid, alpha: float) -> np.ndarray:
    """A 704d cosine combines both encoders for the same reference recording."""
    public, advanced, mask = np.asarray(public), np.asarray(advanced), np.asarray(valid)
    if (alpha not in ALPHAS or public.ndim != 2 or public.shape[1] != 512
            or advanced.shape != (len(public), 192) or mask.shape != (len(public),) or mask.dtype != np.bool_):
        raise ValueError("Fusion requires aligned public512/advanced192 rows, validity and a preregistered alpha")
    views = []
    for values in (public, advanced):
        if values.dtype != np.float32 or not np.isfinite(values).all() or np.any(values[~mask]):
            raise ValueError("Fusion requires original finite float32 embeddings and exact invalid zeros")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms[mask], 1, atol=1e-5):
            raise ValueError("Fusion source vectors must be unit normalized")
        normalized = np.zeros_like(values)
        normalized[mask] = values[mask] / norms[mask, None]
        views.append(normalized)
    public_weight = np.float32(np.sqrt(np.float32(1.0 - alpha)))
    advanced_weight = np.float32(np.sqrt(np.float32(alpha)))
    return np.concatenate((public_weight * views[0], advanced_weight * views[1]), axis=1).astype(np.float32)


def scores_for_alpha(public, advanced, valid, manifest, folds, outer, alpha, endpoints, *, classes=446):
    """Endpoints preserve their exact original arithmetic, with no concatenation."""
    if alpha == 0:
        return endpoints["public"]
    if alpha == 1:
        return endpoints["advanced"]
    values = weighted_encoder_pair(public, advanced, valid, alpha)
    result = crossfit_scores(values, valid, manifest, folds, outer, "max_reference", classes)
    require_aligned_scores(endpoints["public"], result)
    result["provenance"] = {**result["provenance"], "fusion": "same_reference_sqrt_weighted_encoder_concatenation", "advanced_weight": float(alpha)}
    return result


def select_inner_alpha(candidates: dict, inner_truth, *, classes=447) -> tuple[dict, dict]:
    if set(candidates) != set(ALPHAS) or any(set(entry) != {"known", "unknown"} for entry in candidates.values()):
        raise ValueError("Alpha selection accepts exactly five INNER known/unknown score candidates, never outer arrays")
    curves = {}
    for alpha in ALPHAS:
        entry = candidates[alpha]
        selected, curve = calibrate_gate(entry["known"], inner_truth, entry["unknown"], UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201, classes)
        curves[str(alpha)] = {"advanced_weight": alpha, "selected": selected, "curve": curve}
    chosen = max(TIE_ORDER, key=lambda alpha: curves[str(alpha)]["selected"]["inner_macro_f1_447"])
    return {"advanced_weight": chosen, "calibration": curves[str(chosen)]["selected"],
            "alpha_tie_order": list(TIE_ORDER), "selection_scope": "group-excluded inner queries only"}, curves


def require_endpoint_controls(checks: dict) -> None:
    if set(checks) != {"public", "advanced"}:
        raise ValueError("Both exact endpoint controls must finish before mixed outer evaluation")
    for control in checks.values():
        require_public_control(control)


def _read(directory: Path, relative: str, inventory: dict | None = None):
    path = directory / relative
    if (inventory is not None and relative not in inventory) or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Historical candidate evidence must be present in the confined audited export")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_historical_candidate(directory: Path, source: dict, contract: dict, inventory: dict, root: Path) -> tuple[dict, dict]:
    """Authenticate old all-src identity without requiring equality to today's identity."""
    original = _read(directory, "resolved_config.json", inventory)
    captured = _read(directory, "tracking/artifacts/resolved_config.json", inventory)
    identity = _read(directory, "candidate_identity.json", inventory)
    state, report = _read(directory, "experiment_state.json", inventory), _read(directory, "experiment_report.json", inventory)
    parent = _read(directory, "tracking/run_state.json", inventory)
    validate_candidate_suite(original["suite"])
    computed = hashlib.sha256(json.dumps({key: value for key, value in identity.items() if key != "signature"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    validate_advanced_config(identity["model"])
    if (original != captured or original.get("candidate_identity") != identity or computed != identity.get("signature")
            or computed != source["source_signature"] or identity.get("embedding_dim") != 192
            or identity.get("encoder_kind") != "public_frozen_campp_advanced_192" or identity.get("encoder_updates") != 0
            or identity.get("weights_sha256") != identity["model"]["weights_sha256"]
            or identity.get("inference") != {"seconds": 180.0, "maximum_windows": 1}
            or identity.get("data_input_hashes") != {key: value for key, value in contract["input_hashes"].items() if key != "model_config"}
            or identity.get("labels") != contract["labels"] or state.get("status") != "complete" or report.get("status") != "complete"
            or state.get("parent_run_id") != source["parent_run_id"] or report.get("parent_run_id") != source["parent_run_id"]
            or parent.get("run_id") != source["parent_run_id"] or parent.get("remote_status") != "FINISHED"
            or parent.get("tags", {}).get("mlflow.source.git.commit") != source["git_commit"] or report.get("encoder_updates") != 0):
        raise ValueError("Historical S007 is not the complete pinned frozen candidate identity")
    require_public_control(report["source_control_checks"])
    snapshot = _read(directory, "tracking/artifacts/source_manifest.json", inventory)
    archive_path = directory / "tracking/artifacts/source_snapshot.zip"
    if ("tracking/artifacts/source_snapshot.zip" not in inventory or snapshot.get("schema_version") != 2
            or snapshot.get("archive_format") != "zip" or snapshot.get("src_dirty") is not False
            or snapshot.get("git_commit") != source["git_commit"] or file_sha256(archive_path) != snapshot.get("archive_sha256")):
        raise ValueError("Historical candidate source ZIP identity or hash differs")
    files = {row["path"]: row for row in snapshot["files"]}
    if len(files) != len(snapshot["files"]) or len(files) != snapshot["file_count"]:
        raise ValueError("Historical source snapshot has duplicate or missing files")
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) != len(files) or set(archive.namelist()) != set(files):
            raise ValueError("Historical ZIP differs from its complete source inventory")
        for entry in archive.infolist():
            name, row = entry.filename, files[entry.filename]
            part = PurePosixPath(name)
            if (not name.startswith("src/") or "\\" in name or ":" in name or ".." in part.parts or part.as_posix() != name
                    or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size != row["bytes"]
                    or hashlib.sha256(archive.read(entry)).hexdigest() != row["sha256"]):
                raise ValueError("Historical source ZIP entry has unsafe path or changed bytes")
    code = identity["source_code_hashes"]
    for name, digest in code.items():
        if name.startswith("src/") and files.get(name, {}).get("sha256") != digest:
            raise ValueError("Historical candidate source hash differs from actual captured ZIP bytes")
        if not name.startswith("src/") and name != "scripts/score_candidate.py":
            raise ValueError("Unexpected historical candidate source dependency")
    cli_path = "tracking/artifacts/input_configs/launcher.py"
    if cli_path not in inventory or file_sha256(directory / cli_path) != code.get("scripts/score_candidate.py"):
        raise ValueError("Historical CLI bytes are not the separately archived original candidate launcher")
    critical = {"src/speaker_id/candidates/campp_advanced.py", "src/speaker_id/models/campp.py",
                "src/speaker_id/training/candidate_comparison.py", "scripts/score_candidate.py"}
    critical.update(name for name in code if name.startswith("src/speaker_id/models/vendor/campplus/") and name.endswith(".py"))
    for name in critical:
        if not (root / name).is_file() or file_sha256(root / name) != code.get(name):
            raise ValueError("Feature-critical candidate source changed; historical cache reuse requires a new reviewed extraction")
    model_path = "tracking/artifacts/candidate_model/model_config.json"
    weights_path = "tracking/artifacts/candidate_model/" + Path(identity["model"]["weights_path"]).name
    if (model_path not in inventory or weights_path not in inventory
            or file_sha256(directory / model_path) != identity["model_config_sha256"]
            or _read(directory, model_path, inventory) != identity["model"]
            or (directory / weights_path).stat().st_size != identity["model"]["weights_bytes"]
            or file_sha256(directory / weights_path) != identity["weights_sha256"]):
        raise ValueError("Historical candidate model assets differ from their original signature")
    receipt = _read(directory, "candidate_cache_manifest.json", inventory)
    if (receipt.get("identity") != identity or receipt.get("weights_sha256_before") != identity["weights_sha256"]
            or receipt.get("weights_sha256_after") != identity["weights_sha256"]
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("encoder_state_sha256_before", ""))
            or receipt.get("encoder_state_sha256_before") != receipt.get("encoder_state_sha256_after")):
        raise ValueError("Historical extraction does not attest frozen weights before and after")
    for recipe in original["suite"]["recipes"]:
        name = recipe["id"]
        child = _read(directory, name + "/tracking/run_state.json", inventory)
        child_config = _read(directory, name + "/tracking/artifacts/resolved_config.json", inventory)
        child_report = _read(directory, name + "/experiment_report.json", inventory)
        if (child.get("run_id") != source["children"][name] or child.get("remote_status") != "FINISHED"
                or child.get("tags", {}).get("mlflow.parentRunId") != source["parent_run_id"]
                or child.get("tags", {}).get("mlflow.source.git.commit") != source["git_commit"]
                or child_config != {**original, "recipe": recipe} or child_report.get("recipe") != recipe
                or sorted(row["outer_fold"] for row in child_report["folds"]) != [0, 1]):
            raise ValueError("Historical S007 child identity, complete folds or captured source differs")
    return identity, {"source_parent_run_id": source["parent_run_id"], "source_signature": computed,
        "source_git_commit": source["git_commit"], "captured_snapshot_sha256": snapshot["archive_sha256"],
        "source_files_verified": len(files), "captured_cli_sha256": file_sha256(directory / cli_path),
        "critical_source_unchanged": True, "historical_all_src_identity_preserved": True}


def load_fusion_inputs(root: Path, suite: dict, *, verify_audio=False) -> dict:
    validate_fusion_suite(suite)
    contract = load_contract(project_path(root, suite["readiness_config"], "configs/train"), root, verify_audio=verify_audio)
    if (contract["config"]["mode"] != "frozen_baseline" or contract["config"]["experiment_code"] != "B002"
            or contract["config"]["fold_ids"] != [0, 1] or contract["config"]["expected_source_files"] != 4529
            or contract["config"]["evaluation_classes"] != 447
            or contract["config"]["inference"] != {"seconds": 180.0, "maximum_windows": 1}):
        raise ValueError("S008 needs original B002 full-utterance frozen data and readiness")
    for value in (suite["source_public"]["source_run"], suite["source_public"]["control_run"], suite["source_advanced"]["run"]):
        project_path(root, value, "artifacts/training", exists=False)
    for value in suite["source_advanced"]["export_manifest_paths"]:
        project_path(root, value, "artifacts", exists=False)
    return contract


def _candidate_remote_requests(directory: Path, source: dict) -> list[tuple]:
    parent_files = [(name, directory / name) for name in ("experiment_report.json", "candidate_identity.json", "candidate_cache_manifest.json", "source_provenance.json", "candidate_embedding_cache.zip")]
    parent_files += [(name, directory / "tracking/artifacts" / name) for name in ("resolved_config.json", "source_manifest.json", "source_snapshot.zip", "input_configs/launcher.py", "candidate_model/model_config.json", "candidate_model/campplus_cn_en_common.pt")]
    requests = [(source["parent_run_id"], None, parent_files)]
    for name, identifier in source["children"].items():
        files = [(path, directory / name / path) for path in ["experiment_report.json", "oof_predictions.csv"]
            + [f"fold_{outer}/{filename}" for outer in (0, 1) for filename in ("evaluation.json", "calibration.json", "predictions.csv")]]
        files.append(("resolved_config.json", directory / name / "tracking/artifacts/resolved_config.json"))
        # Deterministic child snapshots must equal the attested parent bytes.
        # Immutable exports may omit these identical child artifact copies.
        files += [(path, directory / "tracking/artifacts" / path) for path in ("source_manifest.json", "source_snapshot.zip")]
        requests.append((identifier, source["parent_run_id"], files))
    return requests


def _record_fold(directory: Path, tracker, contract, outer, scores, valid, calibration, curve, policy, control_directory=None):
    """Outer labels are read only here, after a policy has been selected internally."""
    from speaker_id.training.plots import evaluation_plots
    directory.mkdir()
    indices, labels = scores["outer_indices"], contract["labels"]
    probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"], calibration, valid[indices], .05)
    references = [contract["manifest"][int(i)] for i in indices]
    predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(guess)]} for row, guess in zip(references, probabilities.argmax(axis=1))]
    metrics = score_predictions(references, predictions, labels)
    check = verify_control_fold(control_directory, outer, predictions, metrics, calibration) if control_directory else None
    ranking = known_ranking_diagnostics(references, scores["outer_known_scores"], probabilities, valid[indices], labels)
    support = {key: value for key, value in scores["reference_counts"].items() if isinstance(value, np.ndarray)}
    report = {"outer_fold": outer, "outer": metrics, "calibration": calibration, "policy": policy,
        "threshold": calibration["threshold"], "threshold_axis_label": "Inner-selected paired-reference gate score",
        "reference_counts": {key: value for key, value in scores["reference_counts"].items() if key not in support},
        "scoring_provenance": scores["provenance"], "source_reproduction": check, "nonzero_known_ranking": ranking,
        "probability_semantics": "normalized scores, not calibrated posteriors"}
    np.savez_compressed(directory / "reference_support.npz", **support, calibration_indices=scores["calibration_indices"], known_labels=np.asarray(labels[1:]))
    report["reference_counts"]["support_arrays_artifact"] = "reference_support.npz"
    write_json(directory / "evaluation.json", report)
    write_json(directory / "calibration.json", {"selected": calibration, "curve": curve})
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "per_class.csv", metrics["per_class"])
    write_csv(directory / "file_diagnostics.csv", [{"audio_file": row["audio_file"], "true_speaker_id": row["speaker_id"],
        "predicted_speaker_id": pred["speaker_id"], "correct": row["speaker_id"] == pred["speaker_id"],
        "duration_seconds": row["duration_seconds"], "mono_rms_dbfs": row["mono_rms_dbfs"]} for row, pred in zip(references, predictions)])
    np.savez_compressed(directory / "outer_probabilities.npz", probabilities=probabilities,
                        audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(labels))
    for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "file_diagnostics.csv", "outer_probabilities.npz", "reference_support.npz"):
        tracker.add_artifact(directory / filename, f"fold_{outer}/" + filename)
    selected_curve = [row for row in curve if all(row[key] == calibration[key] for key in ("unknown_weight", "margin_weight"))]
    for figure in evaluation_plots(directory, report, selected_curve):
        tracker.add_artifact(figure, f"fold_{outer}/figures/" + figure.name)
    tracker.log_metrics({f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"], f"fold_{outer}/outer_accuracy": metrics["accuracy"],
        **{f"fold_{outer}/inner_{key}": value for key, value in calibration.items()},
        **{f"fold_{outer}/known_ranking/{key}": value for key, value in ranking.items() if key != "role"},
        **{f"fold_{outer}/{key}": value for key, value in metrics["errors"].items()}}, sync=True)
    return predictions, report, check


def _finish_recipe(directory: Path, tracker, contract, predictions, folds, recipe, checks):
    pooled = score_predictions(contract["manifest"], predictions, contract["labels"])
    report = {"recipe": recipe, "oof": pooled, "folds": folds, "source_control_checks": checks, "limitations": LIMITATIONS}
    write_json(directory / "experiment_report.json", report)
    write_csv(directory / "oof_predictions.csv", predictions)
    write_csv(directory / "oof_per_class.csv", pooled["per_class"])
    for filename in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
        tracker.add_artifact(directory / filename)
    tracker.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                        **{"oof/" + key: value for key, value in pooled["errors"].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# {recipe}\n\nOOF Macro-F1: {pooled['macro_f1']:.6f}. Policies selected on group-excluded inner queries only. No encoder fitting or audio extraction.\n")
    tracker.finish("FINISHED", strict=True)
    return report


def execute_candidate_fusion(root: Path, config_path: Path, suite: dict, contract: dict, binding_path: Path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    validate_fusion_suite(suite)
    validate_readiness_for_execution(root, contract)
    if os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("S008 execution requires the authorized RTX 3090 instance")
    torch.set_num_threads(4)
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    output = root / "artifacts/training" / ("S008_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"suite_config": config_path, "launcher": root / "scripts/score_candidate_fusion.py"}
    inputs.update({key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")})
    resolved = {"suite": suite, "data_readiness_contract": {key: contract[key] for key in ("config", "model", "input_hashes", "code_hashes", "signature")}, "limitations": LIMITATIONS}
    write_json(output / "resolved_config.json", resolved)
    common = {"project_root": root, "binding": binding, "input_paths": inputs, "run_kind": "frozen_encoder_same_reference_fusion", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["run_name"], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        public_source, advanced_source = suite["source_public"], suite["source_advanced"]
        public, valid, public_proof = validated_cache(root, project_path(root, public_source["source_run"], "artifacts/training"), contract, public_source["source_parent_run_id"])
        endpoint_proof = verify_control_run(root, public_source, public_proof)
        directory, exported, inventory = verified_export_inventory(root, advanced_source)
        identity, candidate_proof = validate_historical_candidate(directory, advanced_source, contract, inventory, root)
        child_resolved = {**resolved, "captured_candidate_identity": identity}
        if _read(directory, "source_provenance.json", inventory)["public_cache"] != public_proof:
            raise ValueError("Historical S007 used a different public control cache")
        receipt = _read(directory, "candidate_cache_manifest.json", inventory)
        advanced, advanced_valid = verify_candidate_cache(directory / "candidate_embedding_cache", identity, contract["manifest"], receipt)
        if not np.array_equal(valid, advanced_valid):
            raise ValueError("Historical encoders disagree about original zero-signal rows")
        requests = _public_remote_requests(root, public_source) + _candidate_remote_requests(directory, advanced_source)
        remote = _verify_remote_evidence(parent.client, binding, requests, output / "verified_remote_evidence")
        proof = {"public_cache": public_proof, "public_control": endpoint_proof, "historical_candidate": candidate_proof,
                 "remote_evidence": remote, "same_audio_manifest_order": True, "no_encoder_fitting_or_extraction": True}
        for filename, value in (("source_provenance.json", proof), ("audited_S007_export.json", exported), ("captured_candidate_identity.json", identity), ("captured_candidate_cache_manifest.json", receipt)):
            write_json(output / filename, value)
            parent.add_artifact(output / filename)
        for name in ("suite_config", "launcher"):
            parent.add_artifact(inputs[name], "input_configs/" + name + inputs[name].suffix)
        labels, manifest = contract["labels"], contract["manifest"]
        label_index = {label: index for index, label in enumerate(labels)}
        historical = {"public": project_path(root, public_source["control_run"], "artifacts/training") / public_source["control_recipe_id"], "advanced": directory / "S007b"}
        arrays = {"public": public, "advanced": advanced}
        endpoints, checks, results, all_sets = {0: {}, 1: {}}, {}, [], {}
        for source_name, recipe in (("public", "S008a"), ("advanced", "S008b")):
            recipe_path = output / recipe
            recipe_path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=recipe_path / "tracking", parent_run_id=parent.run_id,
                run_name=recipe + "-" + source_name + "-exact-control", config={**child_resolved, "recipe": recipe}, **common)
            child.flush(strict=True)
            child.add_artifact(output / "source_provenance.json")
            predictions, folds, fold_checks = [], [], {}
            for outer in (0, 1):
                scores = crossfit_scores(arrays[source_name], valid, manifest, contract["folds"], outer, "max_reference")
                if scores["known_labels"] != labels[1:]:
                    raise ValueError("Endpoint score columns differ from original labels")
                endpoints[outer][source_name] = scores
                query = scores["calibration_indices"]
                inner_truth = np.asarray([label_index[manifest[int(i)]["speaker_id"]] for i in query])
                calibration, curve = calibrate_gate(scores["inner_known_scores"], inner_truth, scores["inner_unknown_similarity"], UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201)
                rows, report, check = _record_fold(recipe_path / f"fold_{outer}", child, contract, outer, scores, valid, calibration, curve,
                                                  {"advanced_weight": 0.0 if source_name == "public" else 1.0}, historical[source_name])
                predictions.extend(rows)
                folds.append(report)
                fold_checks[str(outer)] = check
            pooled = score_predictions(manifest, predictions, labels)
            if _read(historical[source_name], "experiment_report.json")["oof"] != pooled:
                raise ValueError("Endpoint pooled metrics differ despite exact fold controls")
            checks[source_name] = {**verify_predictions(historical[source_name] / "oof_predictions.csv", predictions), "exact_pooled_metrics": True, "folds": fold_checks}
            write_json(output / "source_control_checks.json", checks)
            results.append(_finish_recipe(recipe_path, child, contract, predictions, folds, recipe, dict(checks)))
            all_sets[source_name] = predictions
            child = None
        require_endpoint_controls(checks)
        recipe_path = output / "S008c"
        recipe_path.mkdir()
        child = DurableMLflowRun.prepare(spool_dir=recipe_path / "tracking", parent_run_id=parent.run_id,
            run_name="S008c-inner-selected-same-reference-fusion", config={**child_resolved, "recipe": "S008c"}, **common)
        child.flush(strict=True)
        child.add_artifact(output / "source_provenance.json")
        child.add_artifact(output / "source_control_checks.json")
        predictions, folds = [], []
        for outer in (0, 1):
            require_endpoint_controls(checks)
            require_aligned_scores(endpoints[outer]["public"], endpoints[outer]["advanced"])
            scored = {alpha: scores_for_alpha(public, advanced, valid, manifest, contract["folds"], outer, alpha, endpoints[outer]) for alpha in ALPHAS}
            query = endpoints[outer]["public"]["calibration_indices"]
            inner_truth = np.asarray([label_index[manifest[int(i)]["speaker_id"]] for i in query])
            selected, curves = select_inner_alpha({alpha: {"known": value["inner_known_scores"], "unknown": value["inner_unknown_similarity"]} for alpha, value in scored.items()}, inner_truth)
            selected_alpha = selected["advanced_weight"]
            rows, report, _ = _record_fold(recipe_path / f"fold_{outer}", child, contract, outer, scored[selected_alpha], valid,
                selected["calibration"], curves[str(selected_alpha)]["curve"], selected)
            path = recipe_path / f"fold_{outer}/inner_alpha_calibration.json"
            write_json(path, {"selected": selected, "candidates": curves})
            child.add_artifact(path, f"fold_{outer}/inner_alpha_calibration.json")
            child.log_metrics({f"fold_{outer}/selected_advanced_weight": selected_alpha}, sync=True)
            predictions.extend(rows)
            folds.append(report)
            print(json.dumps({"stage": "candidate_fusion", "fold": outer, "selected_advanced_weight": selected_alpha}), flush=True)
        results.append(_finish_recipe(recipe_path, child, contract, predictions, folds, "S008c", checks))
        child = None
        comparisons = {name: {"pooled_macro_f1_delta": results[2]["oof"]["macro_f1"] - results[index]["oof"]["macro_f1"],
            "paired_quality_slices": paired_diagnostics(manifest, all_sets[name], predictions, labels)} for index, name in enumerate(("public", "advanced"))}
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": results, "comparisons": comparisons,
            "source_control_checks": checks, "selection_policy": SELECTION_POLICY, "execution_policy": EXECUTION_POLICY,
            "encoder_updates": 0, "elapsed_seconds": time.monotonic() - started, "limitations": LIMITATIONS}
        write_json(output / "experiment_report.json", report)
        for filename in ("experiment_report.json", "source_control_checks.json", "resolved_config.json"):
            parent.add_artifact(output / filename)
        for row in results:
            parent.log_metrics({row["recipe"] + "/oof_macro_f1_447": row["oof"]["macro_f1"]}, sync=False)
        parent.write_report(report, markdown="# S008 frozen same-reference fusion\n\nExact public512 and advanced192 controls passed before mixed outer evaluation. Alpha/gate chosen on inner queries only; all source caches preserve their captured identities. No extraction or training.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), "parent_run_id": parent.run_id, "results": {row["recipe"]: row["oof"]["macro_f1"] for row in results}}
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
