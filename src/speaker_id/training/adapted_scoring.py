"""Paired public/adapted scoring with immutable fold caches and held-out queries.

This module never fits an encoder or extracts audio. Adapted caches cannot use
group crossfit: their original inner calibration roles must remain untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.contracts import load_contract
from speaker_id.training.frozen_suite import fixed_role_scores, validated_cache
from speaker_id.training.fusion_suite import project_path, verify_predictions
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities
from speaker_id.training.runner import write_csv, write_json


RECIPES = (
    {"id": "S004a", "source": "public", "method": "prototype", "unknown_weights": [0.0], "margin_weights": [0.0]},
    {"id": "S004b", "source": "adapted", "method": "prototype", "unknown_weights": [0.0], "margin_weights": [0.0]},
    {"id": "S004c", "source": "public", "method": "max_reference", "unknown_weights": [0.0, .25, .5, .75, 1.0], "margin_weights": [0.0, .5]},
    {"id": "S004d", "source": "adapted", "method": "max_reference", "unknown_weights": [0.0, .25, .5, .75, 1.0], "margin_weights": [0.0, .5]},
)


def validate_adapted_suite(suite: dict) -> None:
    if (not isinstance(suite, dict) or suite.get("schema_version") != 1 or suite.get("experiment_code") != "S004"
            or suite.get("readiness_config") != "configs/train/campp_coverage.json"
            or suite.get("output_root") != "artifacts/training" or suite.get("recipes") != list(RECIPES)
            or suite.get("calibration_protocol") != "fixed_inner_holdout"
            or type(suite.get("threshold_candidates")) is not int or suite["threshold_candidates"] != 201
            or suite.get("probability_temperature") != .05 or set(suite.get("sources", {})) != {"public", "adapted"}):
        raise ValueError("S004 requires the four paired fixed-role recipes and unchanged gate grids")
    for name, source in suite["sources"].items():
        expected = "configs/train/campp_coverage.json" if name == "public" else "configs/train/campp_finetune_fp32.json"
        if (source.get("config") != expected or not re.fullmatch(r"[a-f0-9]{32}", source.get("parent_run_id", ""))):
            raise ValueError("Each source needs its explicit completed baseline contract and parent")
    source = suite["sources"]["adapted"]
    if (not re.fullmatch(r"[a-f0-9]{64}", source.get("export_manifest_sha256", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", source.get("signature", ""))
            or not re.fullmatch(r"[a-f0-9]{40}", source.get("git_commit", ""))
            or source.get("completed_steps") != 600 or set(source.get("folds", {})) != {"0", "1"}
            or not isinstance(source.get("export_manifest_paths"), list) or not source["export_manifest_paths"]):
        raise ValueError("Adapted source requires its audited export, source revision and both completed folds")
    for fold in source["folds"].values():
        if (not re.fullmatch(r"[a-f0-9]{32}", fold.get("child_run_id", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", fold.get("checkpoint_sha256", ""))):
            raise ValueError("Each adapted cache must bind to its distinct child and checkpoint")
    if len({fold["checkpoint_sha256"] for fold in source["folds"].values()}) != 2:
        raise ValueError("Adapted fold checkpoints must be distinct")


def assert_fixed_role_groups(contract: dict, outer: int) -> dict:
    rows = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    groups = {
        "fit": {row["group_id"] for row in rows if truth(row["encoder_fit_allowed"])},
        "reference": {row["group_id"] for row in rows if truth(row["enrollment_allowed"]) or
                      (row["role"] == "unknown_development" and truth(row["encoder_fit_allowed"]))},
        "query": {row["group_id"] for row in rows if truth(row["calibration_query"])},
        "outer": {row["group_id"] for row in rows if truth(row["outer_evaluation_included"])},
    }
    if not rows or any(not value for value in groups.values()):
        raise ValueError("Missing fixed-role reference, fit, calibration or outer content groups")
    for first, second in (("fit", "query"), ("fit", "outer"), ("reference", "query"),
                          ("reference", "outer"), ("query", "outer")):
        if groups[first] & groups[second]:
            raise ValueError(f"Adapted fixed-role content-group leakage: {first}/{second}")
    return {"protocol": "fixed_inner_holdout", "fit_query_outer_content_groups_disjoint": True,
            "group_counts": {key: len(value) for key, value in groups.items()}}


def load_adapted_contracts(root: Path, suite: dict, *, verify_audio: bool = False) -> dict:
    validate_adapted_suite(suite)
    contracts = {}
    for name, source in suite["sources"].items():
        path = project_path(root, source["config"], "configs/train")
        contracts[name] = load_contract(path, root, verify_audio=verify_audio and name == "public")
        project_path(root, source["run"], "artifacts/training", exists=False)
    for path in suite["sources"]["adapted"]["export_manifest_paths"]:
        project_path(root, path, "artifacts", exists=False)
    public, adapted = contracts["public"], contracts["adapted"]
    if public["config"]["mode"] != "frozen_baseline" or adapted["config"]["mode"] != "fine_tune":
        raise ValueError("Expected the completed public B002 and adapted F003 modes")
    for key in ("model", "input_hashes", "manifest", "folds", "roles", "labels"):
        if public[key] != adapted[key]:
            raise ValueError("Paired sources must use identical data, roles, labels and public initialization")
    if (public["config"]["inference"] != adapted["config"]["inference"]
            or public["config"]["inference"] != {"seconds": 180.0, "maximum_windows": 1}
            or public["config"]["fold_ids"] != [0, 1] or adapted["config"]["fold_ids"] != [0, 1]):
        raise ValueError("S004 requires the same full-utterance views and both original folds")
    for outer in (0, 1):
        assert_fixed_role_groups(adapted, outer)
    return contracts


def verified_export_inventory(root: Path, source: dict) -> tuple[Path, dict, dict]:
    """Verify the pinned receipt before trusting ANY cache-to-fold association."""
    candidates = [project_path(root, value, "artifacts", exists=False) for value in source["export_manifest_paths"]]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError("The audited adapted export manifest is unavailable")
    for path in existing:
        if file_sha256(path) != source["export_manifest_sha256"]:
            raise ValueError("Adapted export manifest bytes differ from the preregistered audit")
    manifest = json.loads(existing[0].read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("parent_run_id") != source["parent_run_id"]
            or manifest.get("run_name") != Path(source["run"]).name
            or manifest.get("selected_file_count") != len(manifest.get("files", []))):
        raise ValueError("Adapted export identity or file coverage differs")
    indexed = {}
    directory = project_path(root, source["run"], "artifacts/training")
    for row in manifest["files"]:
        relative = row["path"]
        part = PurePosixPath(relative)
        if (not relative or "\\" in relative or ":" in relative or part.is_absolute()
                or any(value in (".", "..") for value in part.parts) or part.as_posix() != relative
                or relative in indexed or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))):
            raise ValueError("Unsafe, duplicate or malformed adapted export entry")
        path = directory / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory)
                or any(parent.is_symlink() for parent in path.parents if parent != directory and parent.is_relative_to(directory))
                or path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]):
            raise ValueError(f"Adapted exported file hash or path mismatch: {relative}")
        indexed[relative] = row
    return directory, manifest, indexed


def validate_adapted_identity(directory: Path, source: dict, contract: dict, inventory: dict) -> tuple[dict, dict]:
    def captured(relative):
        if relative not in inventory:
            raise ValueError("Required adapted source identity file is absent from audited export")
        return json.loads((directory / relative).read_text(encoding="utf-8"))
    state, original = captured("experiment_state.json"), captured("resolved_config.json")
    report = captured("experiment_report.json")
    computed = hashlib.sha256(json.dumps({"config": original["experiment"], "model": original["model"],
        "input_hashes": original["input_hashes"], "code_hashes": original["code_hashes"]}, sort_keys=True).encode()).hexdigest()
    if (state.get("status") != "complete" or state.get("parent_run_id") != source["parent_run_id"]
            or report.get("status") != "complete" or original.get("resume") is not False
            or computed != source["signature"] or original.get("signature") != computed
            or state.get("signature") != computed or report.get("signature") != computed
            or original["experiment"] != contract["config"] or original["model"] != contract["model"]
            or original["input_hashes"] != contract["input_hashes"]):
        raise ValueError("Adapted source is not the completed audited contract")
    critical = {"src/speaker_id/training/runner.py", "src/speaker_id/training/fit.py",
                "src/speaker_id/training/schedules.py", "src/speaker_id/training/contracts.py",
                "src/speaker_id/training/scoring.py", "scripts/train.py"}
    for name in set(contract["code_hashes"]) | set(original["code_hashes"]):
        if ((name.startswith("src/speaker_id/models/") or name in critical)
                and original["code_hashes"].get(name) != contract["code_hashes"].get(name)):
            raise ValueError("Adapted training/feature/role implementation changed since the audited run")
    attempt = state["attempt"]
    parent = captured(f"tracking/{attempt}/run_state.json")
    source_manifest = captured(f"tracking/{attempt}/artifacts/source_manifest.json")
    if (parent.get("run_id") != source["parent_run_id"] or parent.get("remote_status") != "FINISHED"
            or source_manifest.get("git_commit") != source["git_commit"]):
        raise ValueError("Adapted completed parent or source revision differs")
    folds = {}
    for outer in contract["config"]["fold_ids"]:
        specification = source["folds"][str(outer)]
        child = captured(f"fold_{outer}/tracking/{attempt}/run_state.json")
        child_config = captured(f"fold_{outer}/tracking/{attempt}/artifacts/resolved_config.json")
        checkpoint = inventory.get(f"fold_{outer}/last.pt", {})
        matching = [row for row in report["folds"] if row["outer_fold"] == outer]
        if (child.get("run_id") != specification["child_run_id"] or child.get("remote_status") != "FINISHED"
                or child.get("tags", {}).get("mlflow.parentRunId") != source["parent_run_id"]
                or child.get("tags", {}).get("mlflow.source.git.commit") != source["git_commit"]
                or child_config.get("outer_fold") != outer or child_config.get("signature") != computed
                or child_config.get("experiment") != original["experiment"]
                or child_config.get("model") != original["model"]
                or child_config.get("input_hashes") != original["input_hashes"]
                or child_config.get("code_hashes") != original["code_hashes"]
                or checkpoint.get("sha256") != specification["checkpoint_sha256"]
                or len(matching) != 1 or matching[0].get("fit", {}).get("completed_steps") != source["completed_steps"]):
            raise ValueError("Adapted fold/child/checkpoint/completed-step binding differs")
        folds[str(outer)] = {**specification, "completed_steps": source["completed_steps"],
                             "role_proof": assert_fixed_role_groups(contract, outer)}
    return original, folds


def load_adapted_fold_cache(directory: Path, contract: dict, signature: str, inventory: dict, outer: int):
    """Per-file hashes provide fold identity even when NPZ signatures are equal."""
    cache = directory / f"fold_{outer}/embedding_cache"
    expected = {Path(row["audio_file"]).stem + ".npz" for row in contract["manifest"]}
    if cache.is_symlink() or not cache.is_dir() or {path.name for path in cache.iterdir()} != expected:
        raise ValueError("Adapted fold cache must cover every original audio file exactly")
    vectors, validity, evidence = [], [], []
    for row in contract["manifest"]:
        path = cache / (Path(row["audio_file"]).stem + ".npz")
        relative = path.relative_to(directory).as_posix()
        digest = file_sha256(path)
        if path.is_symlink() or relative not in inventory or inventory[relative]["sha256"] != digest:
            raise ValueError("Adapted cache fold binding/hash mismatch; shared signatures are insufficient")
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["signature"]) != signature or str(saved["audio_sha256"]) != row["input_sha256"]
                    or saved["valid"].shape != ()):
                raise ValueError("Adapted cache audio or source identity differs")
            vector, valid = saved["embedding"].copy(), bool(saved["valid"])
        if (vector.shape != (512,) or not np.isfinite(vector).all() or valid != truth(row["has_nonzero_signal"])
                or (valid and not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)) or (not valid and np.any(vector))):
            raise ValueError("Invalid adapted embedding shape, norm or zero-signal policy")
        vectors.append(vector)
        validity.append(valid)
        evidence.append({"audio_file": row["audio_file"], "cache_path": relative, "cache_sha256": digest,
                         "audio_sha256": row["input_sha256"]})
    return np.asarray(vectors), np.asarray(validity, dtype=bool), evidence


def execute_adapted_suite(root: Path, config_path: Path, suite: dict, contracts: dict, binding_path: Path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.training.plots import evaluation_plots
    validate_adapted_suite(suite)
    contract = contracts["public"]
    validate_readiness_for_execution(root, contract)
    if (os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available()
            or "3090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("S004 execution requires the authorized RTX 3090 instance")
    torch.set_num_threads(contract["config"]["cpu_threads"])
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    output = root / "artifacts/training" / ("S004_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"suite_config": config_path}
    for name, source in suite["sources"].items():
        inputs[name + "_config"] = root / source["config"]
        inputs[name + "_source_config"] = project_path(root, source["run"], "artifacts/training") / "resolved_config.json"
    inputs.update({key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")})
    common = {"project_root": root, "binding": binding, "input_paths": inputs,
              "run_kind": "paired_adapted_fixed_role_scoring", "training_started": False}
    resolved = {"suite": suite, "contract_signatures": {key: value["signature"] for key, value in contracts.items()}}
    write_json(output / "resolved_config.json", resolved)
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["run_name"], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        public = suite["sources"]["public"]
        public_dir = project_path(root, public["run"], "artifacts/training")
        public_vectors, public_valid, public_proof = validated_cache(root, public_dir, contract, public["parent_run_id"])
        adapted = suite["sources"]["adapted"]
        adapted_dir, exported, inventory = verified_export_inventory(root, adapted)
        original, fold_proofs = validate_adapted_identity(adapted_dir, adapted, contracts["adapted"], inventory)
        arrays = {"public": {fold: (public_vectors, public_valid) for fold in (0, 1)}, "adapted": {}}
        for outer in (0, 1):
            vectors, valid, file_proof = load_adapted_fold_cache(adapted_dir, contracts["adapted"], original["signature"], inventory, outer)
            if not np.array_equal(valid, public_valid):
                raise ValueError("Paired sources changed valid or zero-signal rows")
            arrays["adapted"][outer] = vectors, valid
            fold_proofs[str(outer)]["files"] = file_proof
        proof = {"public": public_proof, "adapted": {"parent_run_id": adapted["parent_run_id"],
                 "source_signature": original["signature"], "git_commit": adapted["git_commit"],
                 "export_manifest_sha256": adapted["export_manifest_sha256"], "verified_export_files": len(inventory),
                 "folds": fold_proofs}, "calibration_protocol": "fixed_inner_holdout", "no_encoder_fitting_or_audio_extraction": True}
        write_json(output / "source_provenance.json", proof)
        write_json(output / "audited_export_manifest.json", exported)
        parent.add_artifact(output / "source_provenance.json")
        parent.add_artifact(output / "audited_export_manifest.json")
        label_index = {label: index for index, label in enumerate(contract["labels"])}
        control_checks, results = {}, []
        for recipe in RECIPES:
            is_control = recipe["method"] == "prototype"
            if not is_control and set(control_checks) != {"public", "adapted"}:
                raise ValueError("Both exact source OOF controls must pass before improved recipes")
            name = recipe["source"]
            source_dir = public_dir if name == "public" else adapted_dir
            recipe_path = output / recipe["id"]
            recipe_path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=recipe_path / "tracking", parent_run_id=parent.run_id,
                run_name=f"{recipe['id']}-campp-{name}-{recipe['method']}",
                config={**resolved, "recipe": recipe, "source_parent_run_id": suite["sources"][name]["parent_run_id"],
                        "source_signature": proof[name]["source_signature"]}, **common)
            child.flush(strict=True)
            child.add_artifact(output / "source_provenance.json")
            all_predictions, fold_reports = [], []
            for outer in (0, 1):
                fold_path = recipe_path / f"fold_{outer}"
                fold_path.mkdir()
                vectors, valid = arrays[name][outer]
                role_proof = assert_fixed_role_groups(contracts[name], outer)
                scores = fixed_role_scores(contracts[name], vectors, valid, outer, recipe["method"])
                query, evaluation = scores["calibration_indices"], scores["outer_indices"]
                inner_truth = np.asarray([label_index[contract["manifest"][int(i)]["speaker_id"]] for i in query])
                calibration, curve = calibrate_gate(scores["inner_known_scores"], inner_truth, scores["inner_unknown_similarity"],
                    recipe["unknown_weights"], recipe["margin_weights"], 201)
                probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"], calibration, valid[evaluation], .05)
                references = [contract["manifest"][int(i)] for i in evaluation]
                predictions = [{"audio_file": row["audio_file"], "speaker_id": contract["labels"][int(guess)]}
                               for row, guess in zip(references, probabilities.argmax(axis=1))]
                reproduction = verify_predictions(source_dir / f"fold_{outer}/predictions.csv", predictions) if is_control else None
                metrics = score_predictions(references, predictions, contract["labels"])
                report = {"recipe": dict(recipe), "outer_fold": outer, "outer": metrics, "calibration": calibration,
                    "threshold": calibration["threshold"], "threshold_axis_label": "Fixed-role reference gate score",
                    "reference_counts": scores["reference_counts"], "inner_query_files": len(query),
                    "role_proof": role_proof, "source_reproduction": reproduction,
                    "checkpoint_binding": fold_proofs[str(outer)] if name == "adapted" else {"encoder": "public_frozen"},
                    "probability_semantics": "normalized scores, not calibrated posteriors"}
                # Full per-file attestation is kept once in source_provenance.json.
                report["checkpoint_binding"] = {key: value for key, value in report["checkpoint_binding"].items() if key != "files"}
                write_json(fold_path / "evaluation.json", report)
                write_json(fold_path / "calibration.json", {"selected": calibration, "curve": curve})
                write_csv(fold_path / "predictions.csv", predictions)
                write_csv(fold_path / "per_class.csv", metrics["per_class"])
                write_csv(fold_path / "file_diagnostics.csv", [{"audio_file": row["audio_file"], "true_speaker_id": row["speaker_id"],
                    "predicted_speaker_id": prediction["speaker_id"], "correct": row["speaker_id"] == prediction["speaker_id"]}
                    for row, prediction in zip(references, predictions)])
                np.savez_compressed(fold_path / "outer_probabilities.npz", probabilities=probabilities,
                    audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(contract["labels"]))
                for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "file_diagnostics.csv", "outer_probabilities.npz"):
                    child.add_artifact(fold_path / filename, f"fold_{outer}/" + filename)
                selected_curve = [row for row in curve if all(row[key] == calibration[key] for key in ("unknown_weight", "margin_weight"))]
                for figure in evaluation_plots(fold_path, report, selected_curve):
                    child.add_artifact(figure, f"fold_{outer}/figures/" + figure.name)
                child.log_metrics({f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"],
                    f"fold_{outer}/outer_accuracy": metrics["accuracy"],
                    **{f"fold_{outer}/inner_{key}": value for key, value in calibration.items()},
                    **{f"fold_{outer}/{key}": value for key, value in metrics["errors"].items()}}, sync=True)
                all_predictions.extend(predictions)
                fold_reports.append(report)
                print(json.dumps({"stage": "adapted_scoring", "recipe": recipe["id"], "fold": outer,
                                  "source_control_exact": reproduction is not None}), flush=True)
            if is_control:
                control_checks[name] = verify_predictions(source_dir / "oof_predictions.csv", all_predictions)
                write_json(output / "source_control_checks.json", control_checks)
            pooled = score_predictions(contract["manifest"], all_predictions, contract["labels"])
            report = {"recipe": dict(recipe), "oof": pooled, "folds": fold_reports,
                      "source_control_checks": dict(control_checks), "limitations": suite["limitations"]}
            write_json(recipe_path / "experiment_report.json", report)
            write_csv(recipe_path / "oof_predictions.csv", all_predictions)
            write_csv(recipe_path / "oof_per_class.csv", pooled["per_class"])
            for filename in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
                child.add_artifact(recipe_path / filename, filename)
            child.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                               **{"oof/" + key: value for key, value in pooled["errors"].items()}}, sync=False)
            child.write_report(report, markdown=f"# {recipe['id']} {name}\n\nFixed disjoint inner roles; no encoder fitting. Both outer folds evaluated. OOF Macro-F1: {pooled['macro_f1']:.6f}. This is a paired scoring ablation, not a comparison under the S002f crossfit protocol.\n")
            child.finish("FINISHED", strict=True)
            child = None
            results.append({"recipe": dict(recipe), "oof": pooled, "folds": fold_reports})
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": results,
                  "source_control_checks": control_checks, "selection_policy": suite["selection_policy"],
                  "elapsed_seconds": time.monotonic() - started, "limitations": suite["limitations"]}
        write_json(output / "experiment_report.json", report)
        for filename in ("experiment_report.json", "source_control_checks.json", "resolved_config.json"):
            parent.add_artifact(output / filename, filename)
        for result in results:
            parent.log_metrics({result["recipe"]["id"] + "/oof_macro_f1_447": result["oof"]["macro_f1"]}, sync=False)
        parent.write_report(report, markdown="# S004 paired public/adapted scoring\n\nBoth source baselines reproduced exactly. Four preregistered recipes share the original fixed inner holdout. No outer labels selected coefficients, checkpoints or recipes; no training or audio extraction occurred.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), "parent_run_id": parent.run_id,
                "results": {row["recipe"]["id"]: row["oof"]["macro_f1"] for row in results}}
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
