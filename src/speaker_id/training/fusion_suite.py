"""One attested dual-view experiment, with exact controls before outer results."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.training.contracts import load_contract, read_csv
from speaker_id.training.frozen_suite import validated_cache
from speaker_id.training.fusion import CANDIDATES, MARGIN_WEIGHTS, UNKNOWN_WEIGHTS, dual_view_scores, select_inner_policy
from speaker_id.training.reference_scoring import reference_probabilities
from speaker_id.training.runner import write_csv, write_json


def project_path(root: Path, value: str | Path, area: str, *, exists: bool = True) -> Path:
    supplied = root / value
    resolved = supplied.resolve(strict=exists)
    if supplied.is_symlink() or not resolved.is_relative_to((root / area).resolve()):
        raise ValueError("Fusion input/output path escaped its committed project scope")
    return resolved


def validate_fusion_config(suite: dict) -> None:
    if (suite.get("schema_version") != 1 or suite.get("experiment_code") != "S003"
            or suite.get("readiness_config") != "configs/train/campp_coverage.json"
            or suite.get("output_root") != "artifacts/training"
            or suite.get("candidates") != list(CANDIDATES)
            or suite.get("unknown_weights") != UNKNOWN_WEIGHTS or suite.get("margin_weights") != MARGIN_WEIGHTS
            or type(suite.get("threshold_candidates")) is not int or suite["threshold_candidates"] != 201
            or suite.get("probability_temperature") != .05
            or set(suite.get("sources", {})) != {"full", "short"}):
        raise ValueError("S003 requires exactly the eight preregistered candidates and unchanged gate grids")
    expected = {"full": ("configs/train/campp_coverage.json", "S002f"),
                "short": ("configs/train/campp_baseline.json", "S001f")}
    for view, source in suite["sources"].items():
        if (source.get("baseline_config") != expected[view][0] or source.get("control_recipe_id") != expected[view][1]
                or any(not isinstance(source.get(key), str) or not re.fullmatch(r"[a-f0-9]{32}", source[key])
                       for key in ("source_parent_run_id", "control_parent_run_id"))):
            raise ValueError("Fusion source must identify its baseline and completed scoring control")


def load_fusion_contracts(root: Path, suite: dict, *, verify_audio: bool = False) -> dict:
    validate_fusion_config(suite)
    contracts = {}
    for view, source in suite["sources"].items():
        config = project_path(root, source["baseline_config"], "configs/train")
        contracts[view] = load_contract(config, root, verify_audio=verify_audio and view == "full")
        if contracts[view]["config"]["mode"] != "frozen_baseline":
            raise ValueError("Fine-tuned encoders cannot enter frozen dual-view crossfit")
        for key in ("source_run", "control_run"):
            project_path(root, source[key], "artifacts/training", exists=False)
    full, short = contracts["full"], contracts["short"]
    for key in ("model", "input_hashes", "manifest", "folds", "roles", "labels"):
        if full[key] != short[key]:
            raise ValueError(f"Dual-view source contracts disagree: {key}")
    if (full["config"]["inference"] != {"seconds": 180.0, "maximum_windows": 1}
            or short["config"]["inference"] != {"seconds": 6.0, "maximum_windows": 3}
            or full["config"]["fold_ids"] != short["config"]["fold_ids"]):
        raise ValueError("Expected the original B002 full and B001 short extraction policies")
    return contracts


def verify_predictions(path: Path, predictions: list[dict]) -> dict:
    original = read_csv(path)
    expected = {row["audio_file"]: row["speaker_id"] for row in original}
    actual = {row["audio_file"]: row["speaker_id"] for row in predictions}
    if not original or len(expected) != len(original) or len(actual) != len(predictions) or actual != expected:
        raise ValueError("A frozen source control failed exact filename/label reproduction")
    return {"exact_prediction_reproduction": True, "files": len(original), "source_sha256": file_sha256(path)}


def verify_control_run(root: Path, source: dict, cache_provenance: dict) -> dict:
    directory = project_path(root, source["control_run"], "artifacts/training")
    state = json.loads((directory / "experiment_state.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "experiment_report.json").read_text(encoding="utf-8"))
    resolved = json.loads((directory / "tracking/artifacts/resolved_config.json").read_text(encoding="utf-8"))
    original_suite = resolved["suite"]
    if (state.get("status") != "complete" or report.get("status") != "complete"
            or state.get("parent_run_id") != source["control_parent_run_id"]
            or report.get("parent_run_id") != source["control_parent_run_id"]
            or original_suite.get("source_parent_run_id") != source["source_parent_run_id"]
            or original_suite.get("source_run") != source["source_run"]
            or original_suite.get("baseline_config") != source["baseline_config"]
            or original_suite.get("unknown_weights", UNKNOWN_WEIGHTS) != UNKNOWN_WEIGHTS
            or original_suite.get("threshold_candidates") != 201 or original_suite.get("probability_temperature") != .05):
        raise ValueError("Fusion control must be the completed, pinned original scoring run")
    recipes = [row for row in original_suite["recipes"] if row["id"] == source["control_recipe_id"]]
    if (len(recipes) != 1 or recipes[0].get("method") != "max_reference"
            or recipes[0].get("calibration_protocol") != "leave_content_group_out"
            or recipes[0].get("unknown_weights") != UNKNOWN_WEIGHTS or recipes[0].get("margin_weights") != MARGIN_WEIGHTS):
        raise ValueError("Source scoring control changed its reference protocol or gate grid")
    recipe_report = json.loads((directory / source["control_recipe_id"] / "experiment_report.json").read_text(encoding="utf-8"))
    if recipe_report.get("recipe") != recipes[0]:
        raise ValueError("Completed source recipe differs from its captured configuration")
    recorded_cache = json.loads((directory / "cache_provenance.json").read_text(encoding="utf-8"))
    for key in ("source_parent_run_id", "source_signature", "files"):
        if recorded_cache.get(key) != cache_provenance[key]:
            raise ValueError("Current frozen cache bytes differ from the completed scoring control")
    return {"parent_run_id": source["control_parent_run_id"], "recipe": recipes[0],
            "source_signature": cache_provenance["source_signature"], "status": "complete",
            "source_report_sha256": file_sha256(directory / "experiment_report.json"),
            "resolved_config_sha256": file_sha256(directory / "tracking/artifacts/resolved_config.json")}


def _prediction_rows(contract, indices, scores, calibration, valid, temperature):
    probabilities = reference_probabilities(scores["outer_known_scores"], scores["outer_unknown_similarity"],
                                            calibration, valid[indices], temperature)
    rows = [{"audio_file": contract["manifest"][int(index)]["audio_file"], "speaker_id": contract["labels"][int(guess)]}
            for index, guess in zip(indices, probabilities.argmax(axis=1))]
    return rows, probabilities


def execute_fusion(root: Path, config_path: Path, suite: dict, contracts: dict, binding_path: Path) -> dict:
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.training.plots import evaluation_plots
    contract = contracts["full"]
    validate_fusion_config(suite)
    validate_readiness_for_execution(root, contract)
    if (os.environ.get("VAST_INSTANCE_ID") != "50079023" or not torch.cuda.is_available()
            or "3090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("S003 execution requires the authorized RTX 3090 instance")
    torch.set_num_threads(contract["config"]["cpu_threads"])
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    output = root / "artifacts/training" / ("S003_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"fusion_config": config_path}
    for view, source in suite["sources"].items():
        for key, suffix in (("source_run", "resolved_config.json"), ("control_run", "experiment_report.json")):
            inputs[view + "_" + key] = project_path(root, source[key], "artifacts/training") / suffix
        inputs[view + "_baseline_config"] = root / source["baseline_config"]
    inputs.update({key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")})
    common = {"project_root": root, "binding": binding, "input_paths": inputs,
              "run_kind": "frozen_dual_view_fusion", "training_started": False}
    resolved = {"suite": suite, "contract_signatures": {name: value["signature"] for name, value in contracts.items()}}
    write_json(output / "resolved_config.json", resolved)
    parent = DurableMLflowRun.prepare(spool_dir=output / "tracking", run_name=suite["run_name"], config=resolved, **common)
    children, pending, controls_oof = {}, {}, {"full": [], "short": []}
    started = time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id})
        embeddings, cache_provenance, control_provenance, valid = {}, {}, {}, None
        for view, source in suite["sources"].items():
            directory = project_path(root, source["source_run"], "artifacts/training")
            vectors, view_valid, proof = validated_cache(root, directory, contracts[view], source["source_parent_run_id"])
            if valid is not None and not np.array_equal(valid, view_valid):
                raise ValueError("Frozen views disagree about valid/zero signal rows")
            embeddings[view], cache_provenance[view], valid = vectors, proof, view_valid
            control_provenance[view] = verify_control_run(root, source, proof)
        provenance = {"caches": cache_provenance, "completed_controls": control_provenance,
                      "row_alignment": "identical input manifests, fold assignments, roles and label maps"}
        write_json(output / "cache_reference_provenance.json", provenance)
        parent.add_artifact(output / "cache_reference_provenance.json")
        label_index = {label: index for index, label in enumerate(contract["labels"])}
        # Phase1: select from INNER labels, and reproduce both source controls.
        # Do not compute primary outer predictions/metrics until every fold and
        # pooled OOF control has passed exact comparison.
        for outer in contract["config"]["fold_ids"]:
            fold_path = output / f"fold_{outer}"
            fold_path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=fold_path / "tracking",
                run_name=f"S003-campp-inner-selected-fusion-fold{outer}", parent_run_id=parent.run_id,
                config={**resolved, "outer_fold": outer}, **common)
            children[outer] = child
            child.flush(strict=True)
            child.add_artifact(output / "cache_reference_provenance.json")
            candidates = dual_view_scores(embeddings["full"], embeddings["short"], valid,
                                          contract["manifest"], contract["folds"], outer)
            inner_indices = candidates["full"]["calibration_indices"]
            if candidates["full"]["known_labels"] != contract["labels"][1:]:
                raise ValueError("Fusion class columns differ from the fixed label map")
            inner_truth = np.asarray([label_index[contract["manifest"][int(i)]["speaker_id"]] for i in inner_indices])
            selected, curves = select_inner_policy({name: {"known": scores["inner_known_scores"],
                "unknown": scores["inner_unknown_similarity"]} for name, scores in candidates.items()}, inner_truth)
            write_json(fold_path / "inner_candidate_calibration.json", {"selected_policy": selected, "candidates": curves})
            checks = {}
            for view in ("full", "short"):
                scores = candidates[view]
                rows, _ = _prediction_rows(contract, scores["outer_indices"], scores, curves[view]["selected"], valid, .05)
                source = suite["sources"][view]
                control_dir = root / source["control_run"] / source["control_recipe_id"]
                checks[view] = verify_predictions(control_dir / f"fold_{outer}" / "predictions.csv", rows)
                checks[view]["calibration"] = curves[view]["selected"]
                write_csv(fold_path / f"control_{view}_predictions.csv", rows)
                controls_oof[view].extend(rows)
            write_json(fold_path / "source_control_checks.json", checks)
            chosen = candidates[selected["id"]]
            pending[outer] = {"scores": chosen, "selected": selected,
                              "curve": curves[selected["id"]]["curve"], "checks": checks}
            child.add_artifact(fold_path / "inner_candidate_calibration.json", "calibration/inner_candidate_calibration.json")
            child.add_artifact(fold_path / "source_control_checks.json", "controls/source_control_checks.json")
            child.log_metrics({"inner/selected_macro_f1_447": selected["calibration"]["inner_macro_f1_447"],
                               "inner/short_weight": selected["short_weight"]}, sync=True)
            print(json.dumps({"stage": "fusion_controls", "fold": outer, "selected_on_inner": selected["id"],
                              "control_predictions_exact": True}), flush=True)
            del candidates
        oof_control_checks = {}
        for view, source in suite["sources"].items():
            path = root / source["control_run"] / source["control_recipe_id"] / "oof_predictions.csv"
            oof_control_checks[view] = verify_predictions(path, controls_oof[view])
            write_csv(output / f"control_{view}_oof_predictions.csv", controls_oof[view])
        write_json(output / "all_source_controls_verified.json", {"all_controls_verified_before_primary_outer": True,
                   "completed_runs": control_provenance, "oof": oof_control_checks})
        parent.add_artifact(output / "all_source_controls_verified.json")
        all_predictions, fold_reports = [], []
        # Phase2: coefficients are frozen; outer labels are used only for reports.
        for outer, item in pending.items():
            scores, selected = item["scores"], item["selected"]
            indices = scores["outer_indices"]
            predictions, probabilities = _prediction_rows(contract, indices, scores, selected["calibration"], valid, .05)
            references = [contract["manifest"][int(i)] for i in indices]
            metrics = score_predictions(references, predictions, contract["labels"])
            arrays = {key: value for key, value in scores["reference_counts"].items() if isinstance(value, np.ndarray)}
            support = {key: value for key, value in scores["reference_counts"].items() if key not in arrays}
            support["array_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
            fold_path, child = output / f"fold_{outer}", children[outer]
            report = {"outer_fold": outer, "outer": metrics, "selected_policy": selected,
                      "threshold": selected["calibration"]["threshold"], "threshold_axis_label": "Selected fused reference gate score",
                      "calibration_query_files": len(scores["calibration_indices"]), "reference_counts": support,
                      "provenance": scores["provenance"], "source_control_checks": item["checks"],
                      "all_fold_and_oof_controls_verified_before_primary_outer": True,
                      "probability_semantics": "normalized scores, not calibrated posteriors", "limitations": suite["limitations"]}
            write_json(fold_path / "evaluation.json", report)
            write_json(fold_path / "selected_policy.json", selected)
            write_csv(fold_path / "predictions.csv", predictions)
            write_csv(fold_path / "per_class.csv", metrics["per_class"])
            np.savez_compressed(fold_path / "outer_probabilities.npz", probabilities=probabilities,
                audio_files=np.asarray([row["audio_file"] for row in references]), labels=np.asarray(contract["labels"]))
            np.savez_compressed(fold_path / "reference_support.npz", **arrays,
                calibration_indices=scores["calibration_indices"], known_labels=np.asarray(scores["known_labels"]))
            for name in ("evaluation.json", "selected_policy.json", "predictions.csv", "per_class.csv", "outer_probabilities.npz", "reference_support.npz"):
                child.add_artifact(fold_path / name, "evaluation/" + name)
            for view in ("full", "short"):
                child.add_artifact(fold_path / f"control_{view}_predictions.csv", f"controls/{view}_predictions.csv")
            child.add_artifact(output / "all_source_controls_verified.json", "controls/all_source_controls_verified.json")
            curve = [row for row in item["curve"] if all(row[key] == selected["calibration"][key]
                                                         for key in ("unknown_weight", "margin_weight"))]
            for path in evaluation_plots(fold_path, report, curve):
                child.add_artifact(path, "figures/" + path.name)
            child.log_metrics({"outer/macro_f1_447": metrics["macro_f1"], "outer/accuracy": metrics["accuracy"],
                               **{"outer/" + key: value for key, value in metrics["errors"].items()},
                               **{"inner/" + key: value for key, value in selected["calibration"].items()}}, sync=False)
            child.write_report(report, markdown=f"# S003 fold {outer}\n\nSelected only on inner queries: {selected['id']}. Both source controls passed all fold and OOF checks before primary outer evaluation.\n\nOuter Macro-F1: {metrics['macro_f1']:.6f}.\n")
            child.finish("FINISHED", strict=True)
            del children[outer]
            all_predictions.extend(predictions)
            fold_reports.append(report)
        pooled = score_predictions(contract["manifest"], all_predictions, contract["labels"])
        control_metrics = {view: score_predictions(contract["manifest"], rows, contract["labels"]) for view, rows in controls_oof.items()}
        report = {"status": "complete", "oof": pooled, "controls": control_metrics, "folds": fold_reports,
                  "parent_run_id": parent.run_id, "selection_policy": suite["selection_policy"],
                  "source_controls_verified": oof_control_checks, "elapsed_seconds": time.monotonic() - started,
                  "limitations": suite["limitations"]}
        write_csv(output / "oof_predictions.csv", all_predictions)
        write_csv(output / "oof_per_class.csv", pooled["per_class"])
        write_json(output / "experiment_report.json", report)
        for name in ("resolved_config.json", "oof_predictions.csv", "oof_per_class.csv", "experiment_report.json",
                     "control_full_oof_predictions.csv", "control_short_oof_predictions.csv"):
            parent.add_artifact(output / name, name)
        parent.log_metrics({"oof/macro_f1_447": pooled["macro_f1"], "oof/accuracy": pooled["accuracy"],
                            **{"oof/" + key: value for key, value in pooled["errors"].items()},
                            **{"control/" + view + "_oof_macro_f1_447": result["macro_f1"] for view, result in control_metrics.items()}})
        parent.write_report(report, markdown=f"# S003 frozen short/full fusion\n\nBoth completed source controls were reproduced exactly. All eight candidate policies were selected using inner queries only.\n\nPrimary OOF Macro-F1: {pooled['macro_f1']:.6f}. Fold choices: {', '.join(row['selected_policy']['id'] for row in fold_reports)}.\n\nThese are development estimates, not hidden leaderboard results.\n")
        parent.finish("FINISHED", strict=True)
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id})
        return {"output": str(output), "parent_run_id": parent.run_id, "macro_f1": pooled["macro_f1"]}
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "parent_run_id": parent.run_id}
        write_json(output / "failure.json", failure)
        for child in children.values():
            child.write_report(failure)
            child.finish("FAILED", strict=False)
        parent.write_report(failure)
        parent.finish("FAILED", strict=False)
        write_json(output / "experiment_state.json", failure)
        raise
