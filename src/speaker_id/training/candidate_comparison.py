"""S007: one frozen 192d candidate after an exact public S002f control.

The original B002 contract supplies data and operational readiness only. The
candidate owns a separate model, source fingerprint, signature and cache schema.
No optimization, checkpoint selection, resume or automatic dispatch is provided.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np

from speaker_id.candidates.campp_advanced import validate_advanced_config, load_advanced, extract_advanced_embedding
from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.adaptation_comparison import (
    _verify_remote_evidence, known_ranking_diagnostics, paired_diagnostics, verify_control_fold,
)
from speaker_id.training.contracts import load_contract
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.frozen_suite import validated_cache
from speaker_id.training.fusion_suite import project_path, verify_control_run, verify_predictions
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities
from speaker_id.training.runner import write_csv, write_json


RECIPES = [
    {"id": "S007a", "source": "public", "method": "max_reference", "calibration_protocol": "leave_content_group_out"},
    {"id": "S007b", "source": "advanced", "method": "max_reference", "calibration_protocol": "leave_content_group_out"},
]
UNKNOWN_WEIGHTS, MARGIN_WEIGHTS = [0.0, .25, .5, .75, 1.0], [0.0, .5]
SELECTION_POLICY = "Both-fold and pooled exact S002f controls finish before candidate extraction. Each coefficient and threshold is fit only on group-excluded outer-training queries. One candidate, no outer-based parameter or checkpoint selection."
EXECUTION_POLICY = "Prepare during F004; start only after completed S006 if neither F004 recipe S006d nor S006f reaches pooled OOF Macro-F1 0.965. Manual dispatch; no automatic chaining."
SOURCE = {
    "baseline_config": "configs/train/campp_coverage.json",
    "source_run": "artifacts/training/B002_20260907T150724Z_9b11fe4b",
    "source_parent_run_id": "9b5ef17e61d24b9ba438128bf7f24a7a",
    "control_run": "artifacts/training/S002_20260907T153007Z_ec71bae8",
    "control_parent_run_id": "7f72cd37eee249589d405d8e4bc63e61",
    "control_recipe_id": "S002f",
}


def validate_candidate_suite(suite: dict) -> None:
    fields = {"schema_version", "experiment_code", "run_name", "readiness_config", "output_root", "source",
              "candidate_config", "recipes", "unknown_weights", "margin_weights", "threshold_candidates",
              "probability_temperature", "hypothesis", "selection_policy", "execution_policy", "limitations"}
    if (not isinstance(suite, dict) or set(suite) != fields or suite["schema_version"] != 1
            or suite["experiment_code"] != "S007" or suite["run_name"] != "S007-campp-advanced192-frozen-full"
            or suite["readiness_config"] != "configs/train/campp_coverage.json"
            or suite["output_root"] != "artifacts/training" or suite["source"] != SOURCE
            or suite["candidate_config"] != "configs/model/campp_advanced.json" or suite["recipes"] != RECIPES
            or suite["unknown_weights"] != UNKNOWN_WEIGHTS or suite["margin_weights"] != MARGIN_WEIGHTS
            or type(suite["threshold_candidates"]) is not int or suite["threshold_candidates"] != 201
            or type(suite["probability_temperature"]) is not float or suite["probability_temperature"] != .05
            or suite["selection_policy"] != SELECTION_POLICY or suite["execution_policy"] != EXECUTION_POLICY
            or not isinstance(suite["hypothesis"], str) or not suite["hypothesis"].strip()
            or not isinstance(suite["limitations"], list) or not suite["limitations"]
            or any(not isinstance(item, str) or not item.strip() for item in suite["limitations"])):
        raise ValueError("S007 requires its exact two frozen recipes, completed S002f source and preregistered policies")


def candidate_identity(root: Path, config_path: Path, contract: dict) -> dict:
    model = json.loads(config_path.read_text(encoding="utf-8"))
    validate_advanced_config(model)
    # All src bytes (including isolated loader and shared frontend/vendor) plus
    # this CLI enter the signature. The legacy 512d model identity does not.
    paths = sorted((root / "src").rglob("*.py")) + [root / "scripts/score_candidate.py"]
    code = {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
    payload = {"schema_version": 1, "encoder_kind": "public_frozen_campp_advanced_192",
        "model": model, "model_config_sha256": file_sha256(config_path),
        "embedding_dim": 192, "weights_sha256": model["weights_sha256"], "inference": model["inference"],
        "data_input_hashes": {key: value for key, value in contract["input_hashes"].items() if key != "model_config"},
        "labels": contract["labels"], "source_code_hashes": code, "encoder_updates": 0}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**payload, "signature": signature}


def load_candidate_contract(root: Path, suite: dict, *, verify_audio=False) -> tuple[dict, dict]:
    validate_candidate_suite(suite)
    contract = load_contract(project_path(root, suite["readiness_config"], "configs/train"), root, verify_audio=verify_audio)
    config = contract["config"]
    if (config["mode"] != "frozen_baseline" or config["experiment_code"] != "B002"
            or config["fold_ids"] != [0, 1] or config["inference"] != {"seconds": 180.0, "maximum_windows": 1}
            or config["expected_source_files"] != 4529 or config["evaluation_classes"] != 447):
        raise ValueError("S007 requires original B002 full-utterance data and readiness; no adapted query crossfit")
    for key in ("source_run", "control_run"):
        project_path(root, suite["source"][key], "artifacts/training", exists=False)
    model_path = project_path(root, suite["candidate_config"], "configs/model")
    return contract, candidate_identity(root, model_path, contract)


def require_public_control(control: dict) -> None:
    if (not isinstance(control, dict) or control.get("exact_prediction_reproduction") is not True
            or control.get("exact_pooled_metrics") is not True or set(control.get("folds", {})) != {"0", "1"}
            or any(any(row.get(key) is not True for key in ("exact_prediction_reproduction", "exact_metrics_reproduction", "exact_calibration_reproduction"))
                   for row in control["folds"].values())):
        raise ValueError("Exact S002f predictions, calibration and metrics in both folds and pooled OOF must precede candidate loading")


def verify_candidate_weights(root: Path, identity: dict) -> Path:
    model = identity["model"]
    validate_advanced_config(model)
    path = project_path(root, model["weights_path"], "artifacts/models")
    if path.stat().st_size != model["weights_bytes"] or file_sha256(path) != model["weights_sha256"]:
        raise ValueError("Candidate weights differ from the pinned official bytes")
    return path


def encoder_state_sha256(encoder) -> str:
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Candidate encoder must stay fully frozen in eval mode")
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_candidate_vector(vector, valid, row: dict) -> None:
    if (not isinstance(vector, np.ndarray) or vector.dtype != np.float32 or vector.shape != (192,)
            or type(valid) not in (bool, np.bool_) or valid != truth(row["has_nonzero_signal"])
            or not np.isfinite(vector).all() or (valid and not np.isclose(np.linalg.norm(vector), 1, atol=1e-5))
            or (not valid and np.any(vector))):
        raise ValueError("Candidate cache requires finite float32 192d unit vectors and unchanged zero-signal fallback")


def verify_candidate_cache(cache: Path, identity: dict, manifest: list[dict], receipt: dict):
    """Strict, standalone 192d cache verification; old 512d NPZs are ineligible."""
    names = [row["audio_file"] for row in manifest]
    expected = {Path(name).stem + ".npz" for name in names}
    if (cache.is_symlink() or not cache.is_dir() or len(expected) != len(names)
            or {path.name for path in cache.iterdir()} != expected
            or receipt.get("schema_version") != 1 or receipt.get("identity") != identity
            or receipt.get("embedding_dim") != 192 or receipt.get("encoder_updates") != 0
            or receipt.get("file_count") != len(manifest) or len(receipt.get("files", [])) != len(manifest)):
        raise ValueError("Candidate cache inventory or identity differs from its complete 192d receipt")
    records = {row["audio_file"]: row for row in receipt["files"]}
    if set(records) != set(names) or len(records) != len(receipt["files"]):
        raise ValueError("Candidate cache receipt has duplicate or missing audio files")
    vectors, validity = [], []
    for row in manifest:
        filename = Path(row["audio_file"]).stem + ".npz"
        path, entry = cache / filename, records[row["audio_file"]]
        if (path.is_symlink() or not path.is_file() or entry.get("cache_file") != filename
                or entry.get("audio_sha256") != row["input_sha256"]
                or entry.get("bytes") != path.stat().st_size or entry.get("cache_sha256") != file_sha256(path)):
            raise ValueError("Candidate cache bytes/audio mapping differ from its manifest")
        with np.load(path, allow_pickle=False) as saved:
            fields = {"embedding", "valid", "audio_file", "audio_sha256", "signature", "model_sha256", "embedding_dim"}
            if (set(saved.files) != fields or saved["valid"].shape != () or saved["valid"].dtype != np.bool_
                    or saved["embedding_dim"].shape != () or int(saved["embedding_dim"]) != 192
                    or str(saved["audio_file"]) != row["audio_file"] or str(saved["audio_sha256"]) != row["input_sha256"]
                    or str(saved["signature"]) != identity["signature"] or str(saved["model_sha256"]) != identity["weights_sha256"]):
                raise ValueError("Candidate cache is not bound to its own audio/model/signature/dimension")
            vector, valid = saved["embedding"].copy(), bool(saved["valid"])
        validate_candidate_vector(vector, valid, row)
        vectors.append(vector)
        validity.append(valid)
    return np.asarray(vectors, dtype=np.float32), np.asarray(validity, dtype=bool)


def extract_candidate_cache(root: Path, contract: dict, identity: dict, output: Path, tracker,
                            control: dict, *, loader=None, extractor=None):
    """One extraction shared by both folds; partial caches are preserved, never resumed."""
    require_public_control(control)
    weights = verify_candidate_weights(root, identity)
    loader, extractor = loader or load_advanced, extractor or extract_advanced_embedding
    cache = output / "candidate_embedding_cache"
    cache.mkdir(parents=True, exist_ok=False)
    encoder = loader(identity["model"], root, device="cuda")
    before = encoder_state_sha256(encoder)
    records, started = [], time.monotonic()
    for i, row in enumerate(contract["manifest"]):
        filename = row["audio_file"]
        if Path(filename).name != filename:
            raise ValueError("Candidate input filename must remain flat")
        vector, info = extractor(encoder, root / contract["config"]["data_dir"] / filename,
                                 device="cuda", **identity["inference"])
        valid = info["nonzero_signal"]
        validate_candidate_vector(vector, valid, row)
        path = cache / (Path(filename).stem + ".npz")
        if path.exists():
            raise ValueError("Candidate audio filenames collided in the fresh cache")
        temporary = path.with_suffix(".partial")
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, embedding=vector, valid=bool(valid), audio_file=filename,
                audio_sha256=row["input_sha256"], signature=identity["signature"],
                model_sha256=identity["weights_sha256"], embedding_dim=np.int64(192))
        temporary.replace(path)
        records.append({"audio_file": filename, "audio_sha256": row["input_sha256"], "cache_file": path.name,
                        "bytes": path.stat().st_size, "cache_sha256": file_sha256(path), "valid": bool(valid)})
        if (i + 1) % 50 == 0 or i + 1 == len(contract["manifest"]):
            tracker.log_metrics({"extraction/completed_files": i + 1, "extraction/elapsed_seconds": time.monotonic() - started}, step=i + 1, sync=True)
            print(json.dumps({"stage": "advanced192_extraction", "completed_files": i + 1, "total_files": len(contract["manifest"])}), flush=True)
    after = encoder_state_sha256(encoder)
    if before != after or file_sha256(weights) != identity["weights_sha256"]:
        raise ValueError("Frozen candidate weights or in-memory parameters changed during extraction")
    del encoder
    receipt = {"schema_version": 1, "identity": identity, "embedding_dim": 192, "encoder_updates": 0,
        "file_count": len(records), "files": records, "weights_sha256_before": identity["weights_sha256"],
        "weights_sha256_after": file_sha256(weights), "encoder_state_sha256_before": before,
        "encoder_state_sha256_after": after, "elapsed_seconds": time.monotonic() - started,
        "cache_policy": "fresh one-time extraction; no implicit resume; shared across both outer folds"}
    write_json(output / "candidate_cache_manifest.json", receipt)
    vectors, valid = verify_candidate_cache(cache, identity, contract["manifest"], receipt)
    with zipfile.ZipFile(output / "candidate_embedding_cache.zip", "x", compression=zipfile.ZIP_STORED) as archive:
        archive.write(output / "candidate_cache_manifest.json", "candidate_cache_manifest.json")
        for entry in records:
            archive.write(cache / entry["cache_file"], "candidate_embedding_cache/" + entry["cache_file"])
    return vectors, valid, receipt


def _public_remote_requests(root: Path, source: dict) -> list[tuple]:
    baseline = project_path(root, source["source_run"], "artifacts/training")
    control = project_path(root, source["control_run"], "artifacts/training")
    recipe = control / source["control_recipe_id"]
    requests = [(source["source_parent_run_id"], None, [("experiment_report.json", baseline / "experiment_report.json")]),
        (source["control_parent_run_id"], None, [("experiment_report.json", control / "experiment_report.json"),
            ("cache_provenance.json", control / "cache_provenance.json"),
            (source["control_recipe_id"] + "/experiment_report.json", recipe / "experiment_report.json"),
            (source["control_recipe_id"] + "/oof_predictions.csv", recipe / "oof_predictions.csv")])]
    for outer in (0, 1):
        directory = recipe / f"fold_{outer}"
        state = json.loads((directory / "tracking/run_state.json").read_text(encoding="utf-8"))
        if state.get("remote_status") != "FINISHED" or state.get("tags", {}).get("mlflow.parentRunId") != source["control_parent_run_id"]:
            raise ValueError("Historical S002f fold child is incomplete or belongs to another parent")
        requests.append((state["run_id"], source["control_parent_run_id"],
                         [("evaluation/" + name, directory / name) for name in ("evaluation.json", "calibration.json", "predictions.csv")]))
    return requests


def execute_candidate_comparison(root: Path, config_path: Path, suite: dict, contract: dict, identity: dict, binding_path: Path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.training.plots import evaluation_plots

    validate_candidate_suite(suite)
    if identity != candidate_identity(root, project_path(root, suite["candidate_config"], "configs/model"), contract):
        raise ValueError("Candidate source/config identity changed after validation")
    validate_readiness_for_execution(root, contract)
    if (os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available()
            or "3090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("S007 execution requires the authorized RTX 3090 instance")
    torch.set_num_threads(4)
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    output = root / "artifacts/training" / ("S007_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"suite_config": config_path, "candidate_model_config": root / suite["candidate_config"],
              "launcher": root / "scripts/score_candidate.py"}
    inputs.update({key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")})
    resolved = {"suite": suite, "candidate_identity": identity,
        "data_readiness_contract": {key: contract[key] for key in ("config", "input_hashes", "code_hashes", "signature")}}
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "candidate_identity.json", identity)
    common = {"project_root": root, "binding": binding, "input_paths": inputs, "run_kind": "frozen_public_candidate_comparison", "training_started": False}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["run_name"], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        source = suite["source"]
        vectors, public_valid, cache_proof = validated_cache(root, project_path(root, source["source_run"], "artifacts/training"), contract, source["source_parent_run_id"])
        control_proof = verify_control_run(root, source, cache_proof)
        remote = _verify_remote_evidence(parent.client, binding, _public_remote_requests(root, source), output / "verified_remote_evidence")
        proof = {"public_cache": cache_proof, "historical_control": control_proof, "remote_evidence": remote,
                 "candidate_identity": identity, "candidate_extraction_requires_exact_controls": True}
        write_json(output / "source_provenance.json", proof)
        parent.add_artifact(output / "source_provenance.json")
        parent.add_artifact(output / "candidate_identity.json")
        for name, path in inputs.items():
            if name.endswith("config") or name == "launcher":
                parent.add_artifact(path, "input_configs/" + name + path.suffix)
        labels, manifest = contract["labels"], contract["manifest"]
        label_index = {label: index for index, label in enumerate(labels)}
        historical = project_path(root, source["control_run"], "artifacts/training") / source["control_recipe_id"]
        control, results, prediction_sets = {}, [], {}
        for recipe in suite["recipes"]:
            candidate = recipe["source"] == "advanced"
            if candidate:
                require_public_control(control)
            recipe_path = output / recipe["id"]
            recipe_path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=recipe_path / "tracking", parent_run_id=parent.run_id,
                run_name=recipe["id"] + "-" + recipe["source"], config={**resolved, "recipe": recipe}, **common)
            child.flush(strict=True)
            child.add_artifact(output / "source_provenance.json")
            if candidate:
                vectors, valid, receipt = extract_candidate_cache(root, contract, identity, output, child, control)
                if not np.array_equal(valid, public_valid):
                    raise ValueError("Candidate validity changed original signal eligibility")
                for filename in ("candidate_cache_manifest.json", "candidate_embedding_cache.zip"):
                    parent.add_artifact(output / filename, filename)
                parent.add_artifact(verify_candidate_weights(root, identity), "candidate_model/" + Path(identity["model"]["weights_path"]).name)
                parent.add_artifact(root / suite["candidate_config"], "candidate_model/model_config.json")
                child.add_artifact(output / "candidate_cache_manifest.json")
                parent.flush(strict=True)
            else:
                valid = public_valid
            all_predictions, fold_reports, fold_checks = [], [], {}
            for outer in (0, 1):
                fold_path = recipe_path / f"fold_{outer}"
                fold_path.mkdir()
                scores = crossfit_scores(vectors, valid, manifest, contract["folds"], outer, method="max_reference")
                if scores["known_labels"] != labels[1:]:
                    raise ValueError("Crossfit known columns differ from the fixed 447-label map")
                query, evaluation = scores["calibration_indices"], scores["outer_indices"]
                inner_truth = np.asarray([label_index[manifest[int(i)]["speaker_id"]] for i in query])
                calibration, curve = calibrate_gate(scores["inner_known_scores"], inner_truth, scores["inner_unknown_similarity"], UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201)
                probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"], calibration, valid[evaluation], .05)
                references = [manifest[int(i)] for i in evaluation]
                predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(guess)]}
                               for row, guess in zip(references, probabilities.argmax(axis=1))]
                metrics = score_predictions(references, predictions, labels)
                reproduction = None if candidate else verify_control_fold(historical, outer, predictions, metrics, calibration)
                if reproduction is not None:
                    fold_checks[str(outer)] = reproduction
                ranking = known_ranking_diagnostics(references, scores["outer_known_scores"], probabilities, valid[evaluation], labels)
                support = {key: value for key, value in scores["reference_counts"].items() if isinstance(value, np.ndarray)}
                report = {"recipe": recipe, "outer_fold": outer, "outer": metrics, "calibration": calibration,
                    "threshold": calibration["threshold"], "threshold_axis_label": "Group-excluded reference gate score",
                    "reference_counts": {key: value for key, value in scores["reference_counts"].items() if key not in support},
                    "provenance": scores["provenance"], "inner_query_files": len(query), "source_reproduction": reproduction,
                    "nonzero_known_ranking": ranking, "embedding_dim": 192 if candidate else 512,
                    "probability_semantics": "normalized scores, not calibrated posteriors"}
                np.savez_compressed(fold_path / "reference_support.npz", **support, calibration_indices=query, known_labels=np.asarray(labels[1:]))
                report["reference_counts"]["support_arrays_artifact"] = "reference_support.npz"
                write_json(fold_path / "evaluation.json", report)
                write_json(fold_path / "calibration.json", {"selected": calibration, "curve": curve})
                write_csv(fold_path / "predictions.csv", predictions)
                write_csv(fold_path / "per_class.csv", metrics["per_class"])
                write_csv(fold_path / "file_diagnostics.csv", [{"audio_file": row["audio_file"], "true_speaker_id": row["speaker_id"],
                    "predicted_speaker_id": pred["speaker_id"], "correct": row["speaker_id"] == pred["speaker_id"],
                    "duration_seconds": row["duration_seconds"], "mono_rms_dbfs": row["mono_rms_dbfs"]} for row, pred in zip(references, predictions)])
                np.savez_compressed(fold_path / "outer_probabilities.npz", probabilities=probabilities,
                                    audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(labels))
                for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "file_diagnostics.csv", "outer_probabilities.npz", "reference_support.npz"):
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
                print(json.dumps({"stage": "candidate_scoring", "recipe": recipe["id"], "fold": outer, "macro_f1": metrics["macro_f1"]}), flush=True)
            pooled = score_predictions(manifest, all_predictions, labels)
            if not candidate:
                control = {**verify_predictions(historical / "oof_predictions.csv", all_predictions), "folds": fold_checks}
                if json.loads((historical / "experiment_report.json").read_text(encoding="utf-8"))["oof"] != pooled:
                    raise ValueError("Historical S002f pooled metric differs despite exact predictions")
                control["exact_pooled_metrics"] = True
                require_public_control(control)
                write_json(output / "source_control_checks.json", control)
                parent.add_artifact(output / "source_control_checks.json")
            report = {"recipe": recipe, "oof": pooled, "folds": fold_reports, "source_control_checks": control,
                      "candidate_identity": identity if candidate else None, "limitations": suite["limitations"]}
            write_json(recipe_path / "experiment_report.json", report)
            write_csv(recipe_path / "oof_predictions.csv", all_predictions)
            write_csv(recipe_path / "oof_per_class.csv", pooled["per_class"])
            for filename in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
                child.add_artifact(recipe_path / filename)
            child.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                               **{"oof/" + key: value for key, value in pooled["errors"].items()}}, sync=False)
            child.write_report(report, markdown=f"# {recipe['id']}\n\nFrozen {192 if candidate else 512}d public encoder; group-excluded inner calibration only. OOF Macro-F1: {pooled['macro_f1']:.6f}. No learned encoder updates.\n")
            child.finish("FINISHED", strict=True)
            child = None
            results.append(report)
            prediction_sets[recipe["id"]] = all_predictions
        comparison = {"pooled_macro_f1_delta": results[1]["oof"]["macro_f1"] - results[0]["oof"]["macro_f1"],
            "fold_macro_f1_deltas": {str(outer): results[1]["folds"][outer]["outer"]["macro_f1"] - results[0]["folds"][outer]["outer"]["macro_f1"] for outer in (0, 1)},
            "paired_quality_slices": paired_diagnostics(manifest, prediction_sets["S007a"], prediction_sets["S007b"], labels)}
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": results, "comparison": comparison,
                  "source_control_checks": control, "encoder_updates": 0, "elapsed_seconds": time.monotonic() - started,
                  "selection_policy": SELECTION_POLICY, "execution_policy": EXECUTION_POLICY, "limitations": suite["limitations"]}
        write_json(output / "experiment_report.json", report)
        for filename in ("experiment_report.json", "resolved_config.json"):
            parent.add_artifact(output / filename)
        for result in results:
            parent.log_metrics({result["recipe"]["id"] + "/oof_macro_f1_447": result["oof"]["macro_f1"]}, sync=False)
        parent.log_metrics({"comparison/pooled_macro_f1_delta": comparison["pooled_macro_f1_delta"]}, sync=False)
        parent.write_report(report, markdown="# S007 frozen advanced192 comparison\n\nBoth folds and pooled S002f predictions, metrics and inner calibration reproduced before candidate loading. One frozen candidate, one preregistered group-crossfit protocol, no outer parameter selection.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), "parent_run_id": parent.run_id,
                "results": {row["recipe"]["id"]: row["oof"]["macro_f1"] for row in results}, "comparison": comparison}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id, "error_type": type(error).__name__, "error": str(error), "resume_supported": False}
        write_json(output / "failure.json", failure)
        if child is not None:
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        parent.write_report(failure)
        parent.finish("FAILED", strict=False)
        write_json(output / "experiment_state.json", failure)
        raise
