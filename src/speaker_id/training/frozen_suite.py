"""Controlled scoring comparisons using an attested, public frozen CAM++ cache."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.contracts import read_csv
from speaker_id.training.reference_scoring import known_scores, calibrate_gate, reference_probabilities
from speaker_id.training.runner import write_csv, write_json


def validate_suite(suite: dict):
    if (not isinstance(suite, dict) or suite.get("schema_version") != 1
            or not isinstance(suite.get("experiment_code"), str)
            or not re.fullmatch(r"S[0-9]{3}", suite["experiment_code"])
            or not isinstance(suite.get("recipes"), list)
            or not suite["recipes"] or suite.get("output_root") != "artifacts/training"
            or type(suite.get("threshold_candidates")) is not int or suite["threshold_candidates"] < 2
            or not isinstance(suite.get("probability_temperature"), (int, float))
            or not np.isfinite(suite["probability_temperature"]) or suite["probability_temperature"] <= 0):
        raise ValueError("Invalid scoring suite configuration")
    identifiers = set()
    for recipe in suite["recipes"]:
        if not isinstance(recipe, dict):
            raise ValueError("Each scoring recipe must be an object")
        identifier = recipe.get("id", "")
        if (not isinstance(identifier, str)
                or not re.fullmatch(suite["experiment_code"] + r"[a-z]", identifier) or identifier in identifiers
                or recipe.get("method") not in {"prototype", "max_reference"}
                or recipe.get("calibration_protocol", "fixed_inner_holdout") not in {"fixed_inner_holdout", "leave_content_group_out"}):
            raise ValueError("Invalid or duplicate recipe ID, method, or calibration protocol")
        identifiers.add(identifier)
        for key in ("unknown_weights", "margin_weights"):
            values = recipe.get(key)
            if (not isinstance(values, list) or not values
                    or any(type(v) not in {float, int} or not np.isfinite(v) or v < 0 for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError("Coefficient grids must contain distinct finite nonnegative numbers")
    # Run the complete source-reproduction control before any comparison recipe.
    control = suite["recipes"][0]
    if (control["id"] != suite["experiment_code"] + "a" or control["method"] != "prototype"
            or control.get("calibration_protocol", "fixed_inner_holdout") != "fixed_inner_holdout"
            or control["unknown_weights"] != [0.0] or control["margin_weights"] != [0.0]):
        raise ValueError("The first recipe must be the suite's fixed-role zero-weight prototype reproduction control")


def verify_source_predictions(source: Path, outer: int, predictions: list[dict]) -> dict:
    """A named control cannot silently pass with changed or duplicated predictions."""
    path = source / f"fold_{outer}" / "predictions.csv"
    original_rows = read_csv(path)
    original = {row["audio_file"]: row["speaker_id"] for row in original_rows}
    observed = {row["audio_file"]: row["speaker_id"] for row in predictions}
    if (not original_rows or len(original) != len(original_rows) or len(observed) != len(predictions)
            or observed != original):
        raise ValueError("Frozen source control does not reproduce exactly; comparisons are invalid")
    return {"exact_prediction_reproduction": True, "files": len(original),
            "source_prediction_sha256": file_sha256(path)}


def validated_cache(root: Path, source: Path, contract: dict, expected_run_id: str):
    """Allow code evolution outside feature extraction, but never silent cache reuse."""
    if contract["config"].get("mode") != "frozen_baseline":
        raise ValueError("Scoring suites require a public frozen baseline contract")
    state = json.loads((source / "experiment_state.json").read_text())
    original = json.loads((source / "resolved_config.json").read_text())
    if state["status"] != "complete" or state["parent_run_id"] != expected_run_id:
        raise ValueError("Source baseline is incomplete or has an unexpected run ID")
    if state["signature"] != original["signature"] or original["experiment"]["mode"] != "frozen_baseline":
        raise ValueError("Only a completed public frozen encoder cache is eligible")
    computed_signature = hashlib.sha256(json.dumps({"config": original["experiment"], "model": original["model"],
        "input_hashes": original["input_hashes"], "code_hashes": original["code_hashes"]}, sort_keys=True).encode()).hexdigest()
    if computed_signature != original["signature"]:
        raise ValueError("Source resolved configuration no longer matches its cache signature")
    if original["model"] != contract["model"] or original["input_hashes"] != contract["input_hashes"]:
        raise ValueError("Source model or data/roles differ from the current contract")
    if original["experiment"]["inference"] != contract["config"]["inference"]:
        raise ValueError("Feature extraction settings differ")
    for name, digest in contract["code_hashes"].items():
        if (name.startswith("src/speaker_id/models/") or name == "src/speaker_id/training/runner.py") and original["code_hashes"].get(name) != digest:
            raise ValueError("Feature extraction implementation changed; old embeddings cannot be reused")
    vectors, validity, manifest = [], [], []
    for row in contract["manifest"]:
        path = source / "frozen_embedding_cache" / (Path(row["audio_file"]).stem + ".npz")
        if path.is_symlink() or not path.is_file():
            raise ValueError("Cache entry must be a regular file")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) != original["signature"] or str(saved["audio_sha256"]) != row["input_sha256"]:
                raise ValueError("Cached embedding has another source signature or audio hash")
            vector, valid = saved["embedding"].copy(), bool(saved["valid"])
        if (vector.shape != (512,) or not np.isfinite(vector).all()
                or valid != truth(row["has_nonzero_signal"])
                or (valid and not np.isclose(np.linalg.norm(vector), 1, atol=1e-5))
                or (not valid and np.any(vector))):
            raise ValueError("Invalid embedding shape, norm or signal flag")
        vectors.append(vector)
        validity.append(valid)
        manifest.append({"audio_file": row["audio_file"], "audio_sha256": row["input_sha256"],
                         "cache_sha256": file_sha256(path)})
    return np.asarray(vectors), np.asarray(validity, dtype=bool), {
        "source_parent_run_id": expected_run_id, "source_signature": original["signature"],
        "source_git_commit": json.loads(next((source / "tracking").glob("*/artifacts/source_manifest.json")).read_text())["git_commit"],
        "feature_implementation_unchanged": True, "files": manifest}


def fixed_role_scores(contract, embeddings, valid, outer, method):
    rows = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    positions = {row["audio_file"]: index for index, row in enumerate(contract["manifest"])}
    targets = {label: index for index, label in enumerate(contract["labels"])}
    def select(predicate):
        return np.asarray([positions[row["audio_file"]] for row in rows if predicate(row)], dtype=int)
    enrollment = select(lambda row: truth(row["enrollment_allowed"]))
    unknown = select(lambda row: row["role"] == "unknown_development" and truth(row["encoder_fit_allowed"]))
    query = select(lambda row: truth(row["calibration_query"]))
    evaluation = select(lambda row: truth(row["outer_evaluation_included"]))
    if not len(unknown) or not valid[np.r_[enrollment, unknown, query]].all():
        raise ValueError("Fixed-role scoring needs valid independent references and queries")
    if any(set(first) & set(second) for first, second in ((enrollment, query), (unknown, query),
                                                        (enrollment, evaluation), (unknown, evaluation), (query, evaluation))):
        raise ValueError("Reference/query/outer roles overlap")
    enrollment_targets = np.asarray([targets[contract["manifest"][i]["speaker_id"]] for i in enrollment])
    def score(indices):
        return known_scores(embeddings[indices], embeddings[enrollment], enrollment_targets, method), \
               (embeddings[indices] @ embeddings[unknown].T).max(axis=1)
    inner_known, inner_unknown = score(query)
    outer_known, outer_unknown = score(evaluation)
    return {"calibration_indices": query, "inner_known_scores": inner_known,
            "inner_unknown_similarity": inner_unknown, "outer_indices": evaluation,
            "outer_known_scores": outer_known, "outer_unknown_similarity": outer_unknown,
            "reference_counts": {"known_files": len(enrollment), "unknown_files": len(unknown)},
            "provenance": {"protocol": "fixed_inner_holdout", "calibration_queries_excluded_from_all_references": True}}


def execute_suite(root: Path, suite_path: Path, contract: dict, binding_path: Path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    validate_readiness_for_execution(root, contract)
    if os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available():
        raise RuntimeError("Experiments are restricted to the authorized remote CUDA instance")
    torch.set_num_threads(contract["config"]["cpu_threads"])
    suite = json.loads(suite_path.read_text())
    validate_suite(suite)
    source = (root / suite["source_run"]).resolve()
    if not source.is_relative_to(root / "artifacts/training"):
        raise ValueError("Source run must stay inside training artifacts")
    binding = ExperimentBinding(**json.loads(binding_path.read_text())["binding"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / suite["output_root"] / (suite["experiment_code"] + "_" + stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    common = {"project_root": root, "binding": binding, "run_kind": "frozen_scoring_comparison", "training_started": False,
              "input_paths": {"suite_config": suite_path, "source_config": source / "resolved_config.json",
                              "source_report": source / "experiment_report.json",
                              **{name: root / contract["config"][name] for name in ("manifest", "folds", "roles", "label_map", "model_config")}}}
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["experiment_code"] + "-campp-reference-comparisons",
                                     config={"suite": suite, "contract_signature": contract["signature"]}, **common)
    child = None
    started = time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        embeddings, valid, provenance = validated_cache(root, source, contract, suite["source_parent_run_id"])
        write_json(output / "cache_provenance.json", provenance)
        parent.add_artifact(output / "cache_provenance.json", "cache_provenance.json")
        # Numerical diagnostics on saved predictions do not fit or select models.
        from speaker_id.evaluation.error_analysis import analyze_saved_probabilities, write_analysis
        diagnosis, file_rows, class_rows = analyze_saved_probabilities(source,
            root / contract["config"]["manifest"], root / contract["config"]["roles"], root / contract["config"]["label_map"])
        write_analysis(output / "baseline_diagnostics", diagnosis, file_rows, class_rows)
        for artifact in sorted((output / "baseline_diagnostics").iterdir()):
            parent.add_artifact(artifact, "baseline_diagnostics/" + artifact.name)
        labels = contract["labels"]
        label_to_index = {label: index for index, label in enumerate(labels)}
        truth_all = np.asarray([label_to_index[row["speaker_id"]] for row in contract["manifest"]])
        results = []
        for recipe in suite["recipes"]:
            recipe_path = output / recipe["id"]
            recipe_path.mkdir()
            all_predictions, fold_reports = [], []
            for outer in contract["config"]["fold_ids"]:
                fold_path = recipe_path / f"fold_{outer}"
                fold_path.mkdir()
                child = DurableMLflowRun.prepare(spool_dir=fold_path / "tracking",
                    run_name=f"{recipe['id']}-campp-{recipe['method']}-fold{outer}", parent_run_id=parent.run_id,
                    config={"suite": suite, "recipe": recipe, "outer_fold": outer,
                            "source_signature": provenance["source_signature"]}, **common)
                child.flush(strict=True)
                if recipe.get("calibration_protocol", "fixed_inner_holdout") == "leave_content_group_out":
                    from speaker_id.training.crossfit_references import crossfit_scores
                    scores = crossfit_scores(embeddings, valid, contract["manifest"], contract["folds"], outer, method=recipe["method"])
                else:
                    scores = fixed_role_scores(contract, embeddings, valid, outer, recipe["method"])
                if scores.get("known_labels", labels[1:]) != labels[1:]:
                    raise ValueError("Reference score columns differ from the fixed label map")
                query, evaluation = scores["calibration_indices"], scores["outer_indices"]
                calibration, curve = calibrate_gate(scores["inner_known_scores"], truth_all[query], scores["inner_unknown_similarity"],
                    recipe["unknown_weights"], recipe["margin_weights"], suite["threshold_candidates"])
                probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"],
                    calibration, valid[evaluation], suite["probability_temperature"])
                references = [contract["manifest"][i] for i in evaluation]
                predictions = [{"audio_file": row["audio_file"], "speaker_id": labels[int(pred)]}
                               for row, pred in zip(references, probabilities.argmax(axis=1))]
                reproduction = (verify_source_predictions(source, outer, predictions)
                                if recipe["id"] == suite["experiment_code"] + "a" else None)
                metrics = score_predictions(references, predictions, labels)
                support_arrays = {key: value for key, value in scores["reference_counts"].items() if isinstance(value, np.ndarray)}
                support_metadata = {key: value for key, value in scores["reference_counts"].items() if key not in support_arrays}
                if support_arrays:
                    np.savez_compressed(fold_path / "reference_support.npz", **support_arrays,
                                        calibration_indices=query, known_labels=np.asarray(labels[1:]))
                    support_metadata["array_shapes"] = {key: list(value.shape) for key, value in support_arrays.items()}
                    support_metadata["support_arrays_artifact"] = "reference_support.npz"
                report = {"recipe": recipe, "outer_fold": outer, "outer": metrics, "calibration": calibration,
                          "source_reproduction": reproduction,
                          "threshold": calibration["threshold"], "threshold_axis_label": "Calibrated reference gate score",
                          "reference_counts": support_metadata, "provenance": scores["provenance"],
                          "inner_query_files": len(query), "probability_semantics": "normalized scores, not calibrated posteriors"}
                write_json(fold_path / "evaluation.json", report)
                write_json(fold_path / "calibration.json", {"selected": calibration, "curve": curve})
                write_csv(fold_path / "predictions.csv", predictions)
                write_csv(fold_path / "per_class.csv", metrics["per_class"])
                np.savez_compressed(fold_path / "outer_probabilities.npz", probabilities=probabilities,
                                    audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(labels))
                selected_curve = [row for row in curve if all(row[k] == calibration[k] for k in ("unknown_weight", "margin_weight"))]
                from speaker_id.training.plots import evaluation_plots
                for figure in evaluation_plots(fold_path, report, selected_curve):
                    child.add_artifact(figure, "figures/" + figure.name)
                for filename in ("evaluation.json", "calibration.json", "predictions.csv", "per_class.csv", "outer_probabilities.npz"):
                    child.add_artifact(fold_path / filename, "evaluation/" + filename)
                if support_arrays:
                    child.add_artifact(fold_path / "reference_support.npz", "evaluation/reference_support.npz")
                child.log_metrics({"outer/macro_f1_447": metrics["macro_f1"], "outer/accuracy": metrics["accuracy"],
                                   **{"outer/" + k: v for k, v in metrics["errors"].items()},
                                   **{"inner/" + k: v for k, v in calibration.items()}}, sync=False)
                child.write_report(report, markdown=f"# {recipe['id']} fold {outer}\n\n{recipe['purpose']}\n\nOuter Macro-F1: {metrics['macro_f1']:.6f}. Coefficients and threshold selected from inner queries only.\n")
                child.finish("FINISHED", strict=True)
                child = None
                all_predictions.extend(predictions)
                fold_reports.append(report)
                print(json.dumps({"stage": "scoring", "recipe": recipe["id"], "fold": outer, "macro_f1": metrics["macro_f1"]}), flush=True)
            pooled = score_predictions(contract["manifest"], all_predictions, labels)
            write_csv(recipe_path / "oof_predictions.csv", all_predictions)
            write_csv(recipe_path / "oof_per_class.csv", pooled["per_class"])
            report = {"recipe": recipe, "oof": pooled, "folds": fold_reports}
            write_json(recipe_path / "experiment_report.json", report)
            for filename in ("oof_predictions.csv", "oof_per_class.csv", "experiment_report.json"):
                parent.add_artifact(recipe_path / filename, recipe["id"] + "/" + filename)
            summary = {"recipe": recipe["id"], "macro_f1": pooled["macro_f1"], "accuracy": pooled["accuracy"],
                       "errors": pooled["errors"], "fold_macro_f1": [row["outer"]["macro_f1"] for row in fold_reports]}
            results.append(summary)
            parent.log_metrics({recipe["id"] + "/oof_macro_f1_447": pooled["macro_f1"],
                                recipe["id"] + "/oof_accuracy": pooled["accuracy"]})
        report = {"status": "complete", "results": results, "elapsed_seconds": time.monotonic() - started,
                  "parent_run_id": parent.run_id, "selection_policy": suite["selection_policy"],
                  "limitations": ["Repeated outer comparisons are development estimates, not hidden leaderboard scores.",
                                  "Unknown-person and session separation cannot be established from shared unknown labels.",
                                  "Crossfit calibration uses one fewer content group than the final outer-training gallery."]}
        write_json(output / "experiment_report.json", report)
        parent.add_artifact(output / "experiment_report.json", "experiment_report.json")
        parent.write_report(report, markdown="# CAM++ controlled scoring comparisons\n\n" + "\n".join(
            f"- {row['recipe']}: OOF Macro-F1 {row['macro_f1']:.6f}; fold scores {row['fold_macro_f1']}" for row in results)
            + "\n\nAll coefficients selected on inner queries. Crossfit variants change enrollment support and are reported separately.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), **report}
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "parent_run_id": parent.run_id}
        write_json(output / "failure.json", failure)
        if child is not None:
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        parent.write_report(failure)
        parent.finish("FAILED", strict=False)
        write_json(output / "experiment_state.json", failure)
        raise
