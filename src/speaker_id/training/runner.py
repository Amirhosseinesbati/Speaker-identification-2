"""Execute a CAM++ development experiment after explicit user authorization."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import extract_embedding, load_campp
from speaker_id.training.scoring import build_prototypes, fit_threshold, score_probabilities


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tracking_fold_report(report: dict, config: dict) -> dict:
    """Remove per-class label rows from externally tracked research reports."""
    if config.get("retention", {}).get("mlflow_upload_private_reports", True):
        return report
    safe = dict(report)
    def compact_metric(value):
        return ({key: item for key, item in value.items() if key != "per_class"}
                if isinstance(value, dict) else value)

    safe["outer"] = compact_metric(safe.get("outer"))
    safe["oof"] = compact_metric(safe.get("oof"))
    folds = safe.get("folds")
    if isinstance(folds, list):
        safe["folds"] = [
            {**row, "outer": compact_metric(row.get("outer"))}
            if isinstance(row, dict) else row
            for row in folds
        ]
    safe["private_label_rows_uploaded"] = False
    safe["per_file_predictions_uploaded"] = False
    return safe


def extract_all(encoder, contract: dict, root: Path, cache: Path, tracker) -> tuple[np.ndarray, np.ndarray]:
    config = contract["config"]
    cache.mkdir(parents=True, exist_ok=True)
    embeddings, valid = [], []
    started = time.monotonic()
    for index, row in enumerate(contract["manifest"]):
        filename = row["audio_file"]
        target = cache / (Path(filename).stem + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if str(saved["signature"]) != contract["signature"] or str(saved["audio_sha256"]) != row["input_sha256"]:
                    raise ValueError("Embedding cache signature does not match this experiment")
                vector, is_valid = saved["embedding"].copy(), bool(saved["valid"])
        else:
            vector, info = extract_embedding(encoder, root / config["data_dir"] / filename,
                                             device=config["device"], **config["inference"])
            is_valid = info["nonzero_signal"]
            temporary = target.with_suffix(".partial")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, embedding=vector, valid=is_valid,
                                    signature=contract["signature"], audio_sha256=row["input_sha256"])
            temporary.replace(target)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("Invalid cached embedding")
        embeddings.append(vector)
        valid.append(is_valid)
        if (index + 1) % 50 == 0 or index + 1 == len(contract["manifest"]):
            tracker.log_metrics({"extraction/completed_files": index + 1,
                                 "extraction/elapsed_seconds": time.monotonic() - started}, step=index + 1, sync=True, strict=False)
            print(json.dumps({"stage": "embedding", "files": index + 1, "total": len(contract["manifest"])}), flush=True)
    return np.asarray(embeddings), np.asarray(valid, dtype=bool)


def evaluate_fold(contract: dict, roles: list[dict], embeddings: np.ndarray,
                  valid: np.ndarray, output: Path, tracker) -> tuple[list[dict], dict]:
    labels, manifest, config = contract["labels"], contract["manifest"], contract["config"]
    by_name = {row["audio_file"]: index for index, row in enumerate(manifest)}
    label_index = {label: index for index, label in enumerate(labels)}
    enrollment = [row for row in roles if truth(row["enrollment_allowed"])]
    queries = [row for row in roles if truth(row["calibration_query"])]
    outer = [row for row in roles if truth(row["outer_evaluation_included"])]
    def indices(rows):
        return np.asarray([by_name[row["audio_file"]] for row in rows], dtype=int)
    def targets(rows):
        return np.asarray([label_index[row["speaker_id"]] for row in rows], dtype=int)
    enrollment_idx, query_idx, outer_idx = indices(enrollment), indices(queries), indices(outer)
    if not valid[enrollment_idx].all() or not valid[query_idx].all():
        raise ValueError("Invalid signal in frozen fit or calibration roles")
    prototypes = build_prototypes(embeddings[enrollment_idx], targets(enrollment))
    inner_scores = embeddings[query_idx] @ prototypes.T
    threshold, curve = fit_threshold(inner_scores, targets(queries), config["scoring"]["threshold_candidates"])
    outer_scores = embeddings[outer_idx] @ prototypes.T
    probabilities = score_probabilities(outer_scores, threshold, config["scoring"]["probability_temperature"], valid[outer_idx])
    predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(guess)]}
                   for row, guess in zip(outer, probabilities.argmax(axis=1))]
    metrics = score_predictions(outer, predictions, labels)
    error_rows = []
    for row, prediction in zip(outer, predictions):
        source = manifest[by_name[row["audio_file"]]]
        error_rows.append({"audio_file": row["audio_file"], "true_speaker_id": row["speaker_id"],
                           "predicted_speaker_id": prediction["speaker_id"],
                           "correct": row["speaker_id"] == prediction["speaker_id"],
                           "duration_seconds": source["duration_seconds"],
                           "nonzero_signal": bool(valid[by_name[row["audio_file"]]]),
                           "mono_rms_dbfs": source["mono_rms_dbfs"]})
    slices = {}
    for name, condition in {
        "zero_signal": lambda row: not truth(row["has_nonzero_signal"]),
        "nonzero_under_5s": lambda row: truth(row["has_nonzero_signal"]) and float(row["duration_seconds"]) < 5,
        "nonzero_5s_to_30s": lambda row: truth(row["has_nonzero_signal"]) and 5 <= float(row["duration_seconds"]) < 30,
        "nonzero_at_least_30s": lambda row: truth(row["has_nonzero_signal"]) and float(row["duration_seconds"]) >= 30,
    }.items():
        selected = [i for i, row in enumerate(outer) if condition(manifest[by_name[row["audio_file"]]])]
        if selected:
            detail = score_predictions([outer[i] for i in selected], [predictions[i] for i in selected], labels)
            slices[name] = {key: detail[key] for key in ("row_count", "macro_f1", "accuracy", "errors")}
    report = {"outer_fold": int(roles[0]["outer_fold"]), "threshold": threshold,
              "calibration_query_files": len(queries), "enrollment_files": len(enrollment),
              "threshold_selection": "inner_queries_only_fixed_447_macro_f1",
              "probability_semantics": "normalized cosine scores, not calibrated posterior estimates",
              "outer": metrics,
              "diagnostic_slices_fixed_447_labels": slices,
              "limitations": ["Content-group disjointness does not prove session/identity disjointness.",
                              "Unknown identities are not labeled.", "Zero-signal files remain in outer evaluation with unknown fallback."]}
    write_json(output / "evaluation.json", report)
    write_json(output / "calibration.json", {"threshold": threshold, "temperature": config["scoring"]["probability_temperature"], "curve": curve})
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "per_class.csv", metrics["per_class"])
    write_csv(output / "file_diagnostics.csv", error_rows)
    np.savez_compressed(output / "gallery.npz", prototypes=prototypes, labels=np.asarray(labels[1:]))
    # Preserve the same precision used for decisions, including threshold ties.
    np.savez_compressed(output / "outer_probabilities.npz", probabilities=probabilities, audio_files=np.asarray([row["audio_file"] for row in outer]), labels=np.asarray(labels))
    tracker.log_metrics({"outer/macro_f1_447": metrics["macro_f1"], "outer/accuracy": metrics["accuracy"],
                         "inner/best_macro_f1_447": max(row["inner_macro_f1_447"] for row in curve),
                         "inner/threshold": threshold,
                         **{"outer/" + key: value for key, value in metrics["errors"].items()}}, step=0, sync=True)
    retention = config.get("retention", {})
    if retention.get("mlflow_upload_evaluation_artifacts", True):
        for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "file_diagnostics.csv", "gallery.npz", "outer_probabilities.npz"):
            tracker.add_artifact(output / filename, "evaluation/" + filename)
        from speaker_id.training.plots import evaluation_plots
        for plot in evaluation_plots(output, report, curve):
            tracker.add_artifact(plot, "figures/" + plot.name)
    return predictions, report


def execute(contract: dict, root: Path, binding_path: Path, *, resume_dir: Path | None = None) -> dict:
    import torch
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    config = contract["config"]
    # A recipe may bind execution to a versioned, experiment-specific readiness
    # report.  Historical recipes keep the repository default path; new
    # screens must never accidentally validate against stale evidence from a
    # different config or server checkout.
    readiness_path = root / config.get("readiness_report", "artifacts/infrastructure/readiness.json")
    validate_readiness_for_execution(root, contract, readiness_path)
    if os.environ.get("VAST_INSTANCE_ID") != str(config["expected_vast_instance_id"]):
        raise RuntimeError("Execution requires the expected Vast instance marker; local training is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote experiment execution")
    if "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The initial experiment is bound to the user-authorized RTX 3090 server")
    torch.set_num_threads(config["cpu_threads"])
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    execution = config.get("execution", {})
    run_fold_ids = execution.get("run_fold_ids", config["fold_ids"])
    if (not isinstance(run_fold_ids, list) or not run_fold_ids
            or len(set(run_fold_ids)) != len(run_fold_ids)
            or any(type(outer) is not int or outer not in config["fold_ids"] for outer in run_fold_ids)):
        raise ValueError("Execution fold subset must be a nonempty subset of configured folds")
    complete_oof = set(run_fold_ids) == set(config["fold_ids"])
    input_paths = {key: root / config[key] for key in ("manifest", "folds", "roles", "label_map", "model_config")}
    input_paths["public_weights"] = root / contract["model"]["weights_path"]
    if resume_dir:
        output = resume_dir.resolve()
        previous = json.loads((output / "experiment_state.json").read_text(encoding="utf-8"))
        if previous["signature"] != contract["signature"]:
            raise ValueError("Resume configuration, model or data signature differs")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / config["output_root"] / (config["experiment_code"] + "_" + stamp + "_" + uuid.uuid4().hex[:8])
        output.mkdir(parents=True, exist_ok=False)
    # Each process attempt is a separately documented parent/child run; a resumed
    # optimizer state is linked through the shared experiment_state and checkpoints.
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    common = {"project_root": root, "binding": binding, "input_paths": input_paths,
              "run_kind": config["mode"], "training_started": config["mode"] == "fine_tune"}
    resolved = {"experiment": config, "model": contract["model"], "input_hashes": contract["input_hashes"], "code_hashes": contract["code_hashes"],
                "signature": contract["signature"], "resume": bool(resume_dir), "output_directory": str(output)}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking" / attempt,
                                     run_name=config["run_name"] + ("-resume" if resume_dir else ""),
                                     config=resolved, **common)
    try:
        parent.flush(strict=True)  # No extraction, calibration or optimizer step before live confirmation.
        parent.verify_artifacts()
        parent.verify_remote_metadata()
    except BaseException:
        parent.finish("FAILED", strict=False)
        raise
    state = {"signature": contract["signature"], "status": "running", "parent_run_id": parent.run_id,
             "attempt": attempt, "training_started": config["mode"] == "fine_tune",
             "configured_fold_ids": list(config["fold_ids"]),
             "run_fold_ids": list(run_fold_ids), "complete_oof": complete_oof}
    write_json(output / "experiment_state.json", state)
    write_json(output / "resolved_config.json", resolved)
    all_predictions, fold_reports = [], []
    current_child = None
    try:
        shared_embeddings = shared_valid = None
        if config["mode"] == "frozen_baseline":
            encoder = load_campp(contract["model"], root, config["device"])
            shared_embeddings, shared_valid = extract_all(encoder, contract, root, output / "frozen_embedding_cache", parent)
            del encoder
            torch.cuda.empty_cache()
        for outer in run_fold_ids:
            roles = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
            fold_output = output / f"fold_{outer}"
            fold_output.mkdir(parents=True, exist_ok=True)
            current_child = DurableMLflowRun.prepare(spool_dir=fold_output / "tracking" / attempt,
                                                    run_name=f"{config['run_name']}-fold{outer}",
                                                    config={**resolved, "outer_fold": outer}, parent_run_id=parent.run_id, **common)
            current_child.flush(strict=True)
            fit_report = None
            if config["mode"] == "fine_tune":
                from speaker_id.training.fit import fit_encoder
                encoder = load_campp(contract["model"], root, config["device"])
                fit_report = fit_encoder(encoder, roles, contract["labels"], root, config,
                                         contract["signature"], fold_output, current_child, resume=bool(resume_dir))
                embeddings, valid = extract_all(encoder, contract, root, fold_output / "embedding_cache", current_child)
                del encoder
                torch.cuda.empty_cache()
            else:
                embeddings, valid = shared_embeddings, shared_valid
            predictions, report = evaluate_fold(contract, roles, embeddings, valid, fold_output, current_child)
            report["fit"] = fit_report
            all_predictions.extend(predictions)
            fold_reports.append(report)
            tracked_fold_report = _tracking_fold_report(report, config)
            current_child.write_report(tracked_fold_report, markdown=f"# CAM++ fold {outer}\n\nMode: {config['mode']}. Threshold fitted from independent inner queries. Outer Macro-F1 (447 labels): {report['outer']['macro_f1']:.6f}.\n")
            current_child.finish("FINISHED", strict=True)
            current_child = None
        pooled = score_predictions(contract["manifest"], all_predictions, contract["labels"]) if complete_oof else None
        if complete_oof:
            write_csv(output / "oof_predictions.csv", all_predictions)
            write_csv(output / "oof_per_class.csv", pooled["per_class"])
        report = {"status": "complete" if complete_oof else "screen_complete",
                  "mode": config["mode"], "signature": contract["signature"],
                  "oof": pooled, "folds": fold_reports, "output": str(output),
                  "configured_fold_ids": list(config["fold_ids"]),
                  "evaluated_fold_ids": list(run_fold_ids),
                  "complete_oof": complete_oof,
                  "selection_policy": "Fixed recipe; outer labels never select checkpoint or rejection threshold."}
        write_json(output / "experiment_report.json", report)
        if complete_oof:
            parent.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                                **{"oof/" + key: value for key, value in pooled["errors"].items()}}, step=0)
        else:
            parent.log_metrics({"screen/evaluated_folds": float(len(run_fold_ids)),
                                "screen/configured_folds": float(len(config["fold_ids"]))}, step=0)
        retention = config.get("retention", {})
        if retention.get("mlflow_upload_predictions", True) and complete_oof:
            for filename in ("oof_predictions.csv", "oof_per_class.csv"):
                parent.add_artifact(output / filename, filename)
        tracked_report = _tracking_fold_report(report, config)
        if retention.get("mlflow_upload_private_reports", True):
            parent.add_artifact(output / "experiment_report.json", "experiment_report.json")
        else:
            safe_path = output / "experiment_report_mlflow_safe.json"
            write_json(safe_path, tracked_report)
            parent.add_artifact(safe_path, "experiment_report.json")
        summary_line = (
            f"Pooled OOF Macro-F1 on all 447 labels and {pooled['row_count']} files: {pooled['macro_f1']:.6f}."
            if complete_oof else
            f"Screened folds: {run_fold_ids}; pooled OOF is intentionally unavailable until all configured folds run."
        )
        parent.write_report(tracked_report, markdown=f"# {config['run_name']}\n\n{config['hypothesis']}\n\n{summary_line}\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {**state, "status": "complete"})
        return {"output": str(output), "parent_run_id": parent.run_id,
                "macro_f1": None if pooled is None else pooled["macro_f1"],
                "complete_oof": complete_oof, "evaluated_fold_ids": list(run_fold_ids)}
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "signature": contract["signature"], "resume_directory": str(output)}
        write_json(output / "failure.json", failure)
        if current_child is not None:
            current_child.write_report(failure)
            current_child.finish("FAILED", strict=False)
        parent.write_report(failure)
        parent.finish("FAILED", strict=False)
        write_json(output / "experiment_state.json", {**state, "status": "failed"})
        raise
