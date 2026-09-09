"""S012 tracked local CPU decision heads over sealed CAM++ score caches."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid
import zipfile

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.models.campp import file_sha256
from speaker_id.packaging.selected_sources import load_sources
from speaker_id.postprocessing.suite import (
    PACKAGE_PATH, PACKAGE_SHA, AUDIT_PATH, AUDIT_SHA, SOURCE_RUN, SOURCE_PARENT,
    SOURCE_CHILD, SOURCE_COMMIT, SOURCE_REPORT_SHA, BASELINE_MACRO_F1,
    BASELINE_THRESHOLD_ATOL, BASELINE_PROBABILITY_ATOL, _require, _path,
    _source_checks, _execution_environment, _seal_sources, _result_predictions,
    _verify_baseline_fold,
)
from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
from speaker_id.tracking.snapshot import git_provenance
from speaker_id.training.adaptation_comparison import paired_diagnostics
from speaker_id.training.fusion_suite import verify_predictions
from speaker_id.training.runner import write_csv, write_json

S011_AUDIT = "artifacts/infrastructure/S011_verification/S011_20260908T114324Z_4d842981/verification.json"
S011_AUDIT_SHA = "926689b0ce8b4f3cdc0618a2df8ba815e3acac27b7f85f66a001f05ea90abc5e"
S011_PARENT = "a56df5c8e3c84d86b09475bedf2afb6f"
FAMILIES = ["decision_tree", "random_forest", "extra_trees", "gradient_boosting"]
RECIPES = ["baseline", *FAMILIES, "overall"]
PREPARATION = "artifacts/infrastructure/S012_preparation/"
LIMITATIONS = [
    "Repeated development OOF is not an untouched test or hidden leaderboard estimate.",
    "The 84 candidates and threshold grid are selected on nested meta validation; this selection score is itself optimized and not an independent test.",
    "Each meta-validation group is absent from learner-fit queries, reference gallery, cohort statistics and baseline calibration fitting.",
    "Reduced meta galleries lower known-speaker support and leave some singleton classes reference-only; honest nesting does not remove this support-domain shift.",
    "Four family OOF comparisons are exploratory; the prespecified overall selector with baseline fallback is the primary comparison.",
    "Historical full-pool baseline calibration F1, nested selection F1, and outer OOF F1 have different scopes and must never be substituted for one another.",
    "Known identity ranking is fixed; learned heads only accept or reject the existing best known identity.",
    "The quality feature variant requires exactly matching original-rate channel-mean EDA duration/RMS in a future offline package.",
    "No audio loading, encoder updates, test transduction or package replacement occurs in S012.",
]


def validate_config(config):
    from speaker_id.postprocessing.decision_scoring import (
        FEATURE_NAMES, FEATURE_SETS, MODES, META_MIN_GAIN, META_MAX_FOLD_LOSS, THRESHOLD_QUANTILES)
    from speaker_id.postprocessing.nested_cases import META_FOLDS, META_SALT
    from speaker_id.postprocessing.tree_models import MODEL_SPECS, SEED, SKLEARN_VERSION
    required = {"schema_version", "experiment_code", "run_name", "output_root", "source_package_config",
        "source_package_config_sha256", "source_verification", "source_verification_sha256",
        "s011_verification", "s011_verification_sha256", "device", "cpu_threads", "probability_temperature",
        "model_specs", "feature_names", "feature_sets", "modes", "meta_folds", "meta_assignment_salt",
        "seed", "sklearn_version", "loss_weighting", "threshold_quantiles", "promotion", "selection_policy",
        "baseline_numerical_policy", "local_readiness", "research_notes"}
    _require(isinstance(config, dict) and required <= set(config), "Incomplete S012 configuration")
    _require(config["schema_version"] == 1 and config["experiment_code"] == "S012"
        and config["run_name"] == "S012-campp-nested-decision-heads"
        and config["output_root"] == "artifacts/training", "Unexpected S012 experiment identity")
    _require(config["source_package_config"] == PACKAGE_PATH and config["source_package_config_sha256"] == PACKAGE_SHA
        and config["source_verification"] == AUDIT_PATH and config["source_verification_sha256"] == AUDIT_SHA
        and config["s011_verification"] == S011_AUDIT and config["s011_verification_sha256"] == S011_AUDIT_SHA,
        "S012 must use the exact verified P002/S008 and completed S011 source bindings")
    _require(config["model_specs"] == MODEL_SPECS and config["feature_names"] == FEATURE_NAMES
        and config["feature_sets"] == FEATURE_SETS and config["modes"] == MODES
        and config["meta_folds"] == META_FOLDS and config["meta_assignment_salt"] == META_SALT
        and config["seed"] == SEED and config["sklearn_version"] == SKLEARN_VERSION
        and config["loss_weighting"] == "binary_balanced" and config["threshold_quantiles"] == THRESHOLD_QUANTILES,
        "Decision models, feature order, weighting or nested selection grid changed")
    _require(config["promotion"] == {"minimum_pooled_gain": META_MIN_GAIN,
        "maximum_meta_fold_loss": META_MAX_FOLD_LOSS, "otherwise": "baseline"}, "Promotion/fallback policy changed")
    _require(config["device"] == "cuda" and type(config["cpu_threads"]) is int
        and config["cpu_threads"] == 4 and config["probability_temperature"] == .05,
        "Use the fixed local CUDA similarities and four-thread CPU trees")
    numeric = config["baseline_numerical_policy"]
    _require(isinstance(numeric, dict) and numeric.get("threshold_atol") == BASELINE_THRESHOLD_ATOL
        and numeric.get("probability_atol") == BASELINE_PROBABILITY_ATOL and numeric.get("relative_tolerance") == 0
        and bool(numeric.get("rationale")), "Historical baseline tolerances cannot be relaxed")
    _require(isinstance(config["selection_policy"], str) and bool(config["selection_policy"].strip()), "Record the selection rule")
    _require(config["local_readiness"] == PREPARATION + "sklearn_install_verification.json"
        and config["research_notes"] == "reports/research/decision_postprocessing_20260908/research_notes.md",
        "Only the curated S012 preparation inputs are permitted")


def _checked_sources(root, config):
    checked = _source_checks(root, config)
    path = _path(root, config["s011_verification"])
    _require(file_sha256(path) == S011_AUDIT_SHA, "Completed S011 verification receipt changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require(receipt.get("status") == "verified" and receipt.get("parent_run_id") == S011_PARENT
        and receipt.get("results", {}).get("baseline", {}).get("macro_f1") == BASELINE_MACRO_F1,
        "S011 does not identify the verified historical baseline")
    checked["paths"]["s011_verification"] = path
    return checked


def _supporting_inputs(root, config):
    paths = {"research_notes": _path(root, config["research_notes"]),
        "research_manifest": _path(root, PREPARATION + "research/manifest.json"),
        "export_addendum": _path(root, "reports/research/decision_postprocessing_20260908/sklearn_1_8_export_addendum.md"),
        "addendum_manifest": _path(root, PREPARATION + "research/addendum_manifest.json"),
        "local_readiness": _path(root, config["local_readiness"]),
        "protocol_implementation": _path(root, "docs/local_decisions.fa.md"),
        "nested_validation": _path(root, PREPARATION + "nested_cases_validation.json"),
        "tree_validation": _path(root, PREPARATION + "tree_validation.json"),
        "scoring_validation": _path(root, PREPARATION + "scoring_validation.json"),
        "dependency_audit": _path(root, PREPARATION + "dependency_audit.json")}
    for note, manifest in (("research_notes", "research_manifest"), ("export_addendum", "addendum_manifest")):
        data = json.loads(paths[manifest].read_text(encoding="utf-8"))
        matches = [r for r in data["outputs"] if r["path"] == paths[note].relative_to(root).as_posix()]
        _require(len(matches) == 1 and matches[0]["sha256"] == file_sha256(paths[note])
            and matches[0]["bytes"] == paths[note].stat().st_size, "Curated research note changed after its manifest")
    ready = json.loads(paths["local_readiness"].read_text(encoding="utf-8"))
    _require(ready.get("status") == "verified" and ready.get("old_packages_changed_or_removed") == {}
        and ready.get("old_lock_entries_changed") == [] and ready.get("cuda_available") is True,
        "The isolated local decision-head dependency readiness did not pass")
    for name in ("nested_validation", "tree_validation", "scoring_validation"):
        receipt = json.loads(paths[name].read_text(encoding="utf-8"))
        _require(receipt.get("status") == "passed", "A required synthetic protocol/export validation did not pass")
        entries = dict(receipt.get("source_sha256", receipt.get("source_files", {})))
        entries.update({item["path"]: item["sha256"] for item in receipt.get("files", [])})
        _require(entries and all(file_sha256(_path(root, path)) == digest for path, digest in entries.items()),
            "A protocol/export validation receipt differs from the tested source")
    return paths


def validate_inputs(root, config_path):
    root = Path(root).resolve()
    config = json.loads(_path(root, config_path).read_text(encoding="utf-8"))
    validate_config(config)
    _checked_sources(root, config)
    supporting = _supporting_inputs(root, config)
    return {"status": "validated_no_experiment_started", "experiment": "S012", "candidates_per_fold": 84,
        "meta_folds": 3, "children": RECIPES, "encoder_updates": 0, "postprocessor_training_started": False,
        "supporting_evidence": {name: file_sha256(path) for name, path in supporting.items()}}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array_fields(value):
    arrays = {k: v for k, v in value.items() if isinstance(v, np.ndarray)}
    _require(all(v.dtype.kind != "O" for v in arrays.values()), "Object arrays and pickle-dependent artifacts are forbidden")
    _require(all(v.dtype.kind not in "biufc" or np.isfinite(v).all() for v in arrays.values()), "Artifact arrays must be finite")
    return arrays


def _zip_json(path, members):
    """One compressed, nonexecutable JSON archive, with byte receipts per member."""
    index = {}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(members.items()):
            _require(re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.json", name)
                and ".." not in name.split("/"), "Unsafe archive member")
            data = _json_bytes(payload)
            index[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return {"archive": path.name, "sha256": file_sha256(path), "members": index}


def _save_nested(directory, nested, parent, outer):
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {"query_global_indices": nested["query_global_indices"], "assignments": nested["assignments"]}
    plans, baselines = [], {}
    for case in nested["cases"]:
        meta = case["meta_fold"]
        plan = {"meta_fold": meta, "provenance": case["provenance"], "feature_names": case["fit"]["feature_names"]}
        for key in ("fit_global_indices", "validation_global_indices", "reference_global_indices", "local_to_global_indices"):
            arrays[f"case{meta}_{key}"] = case[key]
        for scope in ("fit", "validation"):
            for key, value in _array_fields(case[scope]).items():
                arrays[f"case{meta}_{scope}_{key}"] = value
        base = case["baseline"]
        baselines[f"case{meta}.json"] = {"policy": base["policy"], "calibration": base["calibration"],
            "historical_calibration_scope": "meta_training_group_excluded_only",
            "curves": base["curves"], "alpha_candidates": base["baseline_alpha_candidates"]}
        plans.append(plan)
    np.savez_compressed(directory / "cases.npz", **arrays)
    baseline_receipt = _zip_json(directory / "baseline_calibrations.zip", baselines)
    write_json(directory / "case_plan.json", {"provenance": nested["provenance"], "cases": plans,
        "arrays_sha256": file_sha256(directory / "cases.npz"), "baseline_calibrations": baseline_receipt})
    for path in directory.iterdir():
        parent.add_artifact(path, f"nested_cases/fold_{outer}/" + path.name)
    parent.flush(strict=True)


def _save_search(directory, search, parent, outer, source_binding):
    from speaker_id.inference.decision_trees import validate_export
    directory.mkdir(parents=True, exist_ok=False)
    candidates = search["candidate_summary"]
    _require(len(candidates) == 84 and len({r["id"] for r in candidates}) == 84, "Incomplete finite decision search")
    curves = search["curves"]
    values = np.asarray(curves["values"])
    _require(values.ndim == 2 and values.shape[1] == 7 and np.isfinite(values).all(), "Invalid nested selection curves")
    np.savez_compressed(directory / "all_meta_curves.npz", values=values)
    arrays = _array_fields(search["meta_predictions"])
    for scope in ("full_fit_features", "outer_features"):
        arrays.update({scope + "_" + k: v for k, v in _array_fields(search[scope]).items()})
    np.savez_compressed(directory / "decision_arrays.npz", **arrays)
    members = {}
    for key, models in search["meta_models"].items():
        _require(len(models) == 3, "Every model requires three nested fitted payloads")
        for item in models:
            payload = item["payload"]
            validate_export(payload)
            members[f"meta/{key}/fold{item['meta_fold']}.json"] = payload
    for key, payload in search["final_models"].items():
        validate_export(payload)
        members[f"final/{key}.json"] = payload
    _require(len(search["meta_models"]) == 12, "Missing model/feature combinations")
    archive = _zip_json(directory / "models.zip", members)
    write_json(directory / "model_manifest.json", {**archive, "source_binding": source_binding,
        "format": "finite_json_node_arrays_no_pickle", "feature_names": search["full_fit_features"]["feature_names"],
        "final_model_keys": sorted(search["final_models"]), "meta_model_count": 36})
    write_json(directory / "candidate_summary.json", {"candidates": candidates, "selection": search["selection"],
        "columns": curves["columns"], "candidate_ids": [r["id"] for r in candidates], "curve_rows": len(values),
        "curves_sha256": file_sha256(directory / "all_meta_curves.npz"),
        "decision_arrays_sha256": file_sha256(directory / "decision_arrays.npz"),
        "score_scope": "nested_meta_validation_optimized_for_policy_and_threshold_selection"})
    for path in directory.iterdir():
        parent.add_artifact(path, f"policy_search/fold_{outer}/" + path.name)
    parent.flush(strict=True)
    return archive


def _selection_fields(result, baseline, selection):
    retained = result.get("model") is None
    meta = result.get("meta_selection", {})
    score = selection["baseline_meta_macro_f1_447"] if retained else meta["meta_macro_f1_447"]
    folds = selection["baseline_meta_fold_macro_f1_447"] if retained else meta["meta_fold_macro_f1_447"]
    return {"baseline_retained": retained, "selection_meta_macro_f1_447": score,
        "selection_meta_fold_macro_f1_447": folds,
        "baseline_meta_macro_f1_447": selection["baseline_meta_macro_f1_447"],
        "historical_full_pool_baseline_calibration": baseline["calibration"],
        "historical_full_pool_baseline_calibration_macro_f1_447": baseline["inner_macro_f1_447"],
        "selection_meta_scope": "honest_nested_gallery_out_of_fit_scores_with_optimized_policy_and_threshold",
        "meta_selection": meta}


def _deployment_parity(directory, prepared, baseline, search, parent, outer):
    """Evaluate identical exported models on independently recomputed CPU features."""
    from speaker_id.postprocessing.decision_scoring import (
        decision_features, feature_columns, apply_cascade, decision_probabilities)
    from speaker_id.inference.decision_trees import predict_export
    cpu = decision_features(prepared, baseline, "outer", device="cpu")
    gpu = search["outer_features"]
    _require(cpu["feature_names"] == gpu["feature_names"] and np.array_equal(cpu["indices"], gpu["indices"]),
        "CPU/CUDA feature parity has different row or column identities")
    arrays = {"cpu_" + k: v for k, v in _array_fields(cpu).items()}
    comparisons = {}
    for family, result in search["results"].items():
        if result.get("model") is None:
            comparisons[family] = {"baseline_retained": True, "exact_decisions": True,
                "promotion_blocked": False, "probability_max_abs_difference": 0.0}
            continue
        policy = result["policy"]
        confidence = predict_export(result["model"], cpu["features"][:, feature_columns(policy["feature_set"])])
        margin = apply_cascade(cpu["margin"], confidence, policy["mode"], policy["band"], policy["threshold"])
        probabilities = decision_probabilities(cpu["known_scores"], margin, cpu["valid"])
        differences = probabilities.argmax(axis=1) != result["probabilities"].argmax(axis=1)
        arrays.update({family + "_cpu_confidence": confidence, family + "_cpu_margin": margin,
            family + "_cpu_probabilities": probabilities})
        comparisons[family] = {"baseline_retained": False,
            "exact_decisions": not bool(differences.any()), "decision_disagreements": int(differences.sum()),
            "disagreement_indices": cpu["indices"][differences].tolist(),
            "promotion_blocked": bool(differences.any()),
            "probability_max_abs_difference": float(np.max(np.abs(probabilities - result["probabilities"]), initial=0)),
            "confidence_max_abs_difference": float(np.max(np.abs(confidence - result["scores"]["outer_tree_confidence"]), initial=0))}
    np.savez_compressed(directory / "deployment_cpu_arrays.npz", **arrays)
    report = {"scope": "independent_CPU_outer_feature_recomputation_and_same_exported_model_no_refit",
        "feature_names": cpu["feature_names"], "row_count": len(cpu["indices"]),
        "feature_max_abs_difference": float(np.max(np.abs(cpu["features"] - gpu["features"]), initial=0)),
        "feature_max_abs_difference_by_column": np.max(np.abs(cpu["features"] - gpu["features"]), axis=0).tolist(),
        "exact_feature_values": bool(np.array_equal(cpu["features"], gpu["features"])),
        "families": comparisons, "cpu_arrays_sha256": file_sha256(directory / "deployment_cpu_arrays.npz"),
        "promotion_rule": "Every selected policy requires exact CPU/CUDA class decisions before packaging; mismatches block promotion without rewriting OOF",
        "probability_difference_policy": "Report full absolute drift; exact decisions are mandatory, numeric probabilities remain diagnostic",
        "encoder_updates": 0, "additional_learner_fits": 0}
    write_json(directory / "deployment_parity.json", report)
    for name in ("deployment_cpu_arrays.npz", "deployment_parity.json"):
        parent.add_artifact(directory / name, f"policy_search/fold_{outer}/" + name)
    parent.flush(strict=True)
    return report


def _fold_plots(directory, report, curve):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist([r["f1"] for r in report["outer"]["per_class"][1:]], bins=20, range=(0, 1))
    axes[0].set(xlabel="Known-speaker F1", ylabel="Speakers", title="Outer validation: 446 known identities")
    if curve is not None:
        axes[1].plot(curve[:, 1], curve[:, 2])
        axes[1].axvline(report["policy"]["threshold"], color="orange", linestyle="--")
        axes[1].set(xlabel="Nested-selected correctness threshold", ylabel="Meta Macro-F1 (447)", title="Nested selection curve; outer labels withheld")
    else:
        axes[1].axis("off")
        axes[1].text(.05, .7, "Historical baseline retained\nNested meta F1: " + f"{report['selection_meta_macro_f1_447']:.6f}")
    fig.savefig(directory / "evaluation.png", dpi=160)
    plt.close(fig)


def _persist_fold(record, contract, prepared, result, baseline, search, outer, archive, check, parity):
    directory = record["directory"] / f"fold_{outer}"
    directory.mkdir(exist_ok=False)
    predictions, metrics = _result_predictions(contract, prepared, result)
    fields = _selection_fields(result, baseline, search["selection"])
    model = result.get("model")
    model_reference = None
    curve = None
    if model is not None:
        key = result["meta_selection"]["model_key"]
        member = f"final/{key}.json"
        digest = hashlib.sha256(_json_bytes(model)).hexdigest()
        _require(archive["members"][member]["sha256"] == digest == result["policy"]["model_sha256"],
            "Selected decision policy differs from its portable fitted model")
        model_reference = {"parent_artifact": f"policy_search/fold_{outer}/models.zip", "member": member,
            "model_sha256": digest, "archive_sha256": archive["sha256"]}
        idx = [r["id"] for r in search["candidate_summary"]].index(result["policy"]["id"])
        curve = search["curves"]["values"][search["curves"]["values"][:, 0] == idx]
        np.savez_compressed(directory / "selected_meta_curve.npz", values=curve)
    report = {"outer_fold": outer, "outer": metrics, "policy": result["policy"], **fields,
        "source_reproduction": check, "model_reference": model_reference, "encoder_updates": 0,
        "probability_semantics": "normalized decision scores, not calibrated posteriors",
        "postprocessor_training_started": record["family"] != "baseline",
        "identity_ranking": "unchanged_S008c_max_reference", "deployment_parity": parity,
        "promotion_blocked": parity.get("promotion_blocked", False)}
    if record["family"] == "baseline":
        _zip_json(directory / "historical_baseline_calibration.zip", {"calibration.json": {
            "selected": baseline["calibration"], "curves": baseline["curves"],
            "alpha_candidates": baseline["baseline_alpha_candidates"],
            "scope": "historical_full_pool_group_excluded_calibration_not_nested_meta_OOF"}})
    write_json(directory / "evaluation.json", report)
    write_json(directory / "selected_policy.json", {"policy": result["policy"], "model_reference": model_reference, **fields})
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "per_class.csv", metrics["per_class"])
    np.savez_compressed(directory / "outer_probabilities.npz", probabilities=result["probabilities"],
        audio_files=np.asarray([r["audio_file"] for r in predictions]), labels=np.asarray(contract["labels"]))
    _fold_plots(directory, report, curve)
    tracker = record["tracker"]
    for path in directory.iterdir():
        tracker.add_artifact(path, f"fold_{outer}/" + path.name)
    tracker.log_metrics({f"fold_{outer}/outer_macro_f1_447": metrics["macro_f1"],
        f"fold_{outer}/selection_meta_macro_f1_447": fields["selection_meta_macro_f1_447"],
        f"fold_{outer}/historical_baseline_calibration_macro_f1_447": fields["historical_full_pool_baseline_calibration_macro_f1_447"],
        f"fold_{outer}/baseline_retained": int(fields["baseline_retained"]),
        **{f"fold_{outer}/outer_{k}": v for k, v in metrics["errors"].items()}}, sync=True, strict=True)
    record["predictions"].extend(predictions)
    record["folds"].append(report)


def _finish_child(record, contract, baseline_predictions):
    family, tracker, directory = record["family"], record["tracker"], record["directory"]
    metrics = score_predictions(contract["manifest"], record["predictions"], contract["labels"])
    report = {"recipe": family, "oof": metrics, "folds": record["folds"],
        "paired_against_S008c": paired_diagnostics(contract["manifest"], baseline_predictions, record["predictions"], contract["labels"]),
        "oof_macro_f1_delta": metrics["macro_f1"] - BASELINE_MACRO_F1, "encoder_updates": 0,
        "postprocessor_training_started": family != "baseline", "limitations": LIMITATIONS,
        "comparison_role": "primary_prespecified_selector" if family == "overall" else "baseline_control" if family == "baseline" else "exploratory_family",
        "promotion_blocked": any(f["promotion_blocked"] for f in record["folds"])}
    write_json(directory / "experiment_report.json", report)
    write_csv(directory / "oof_predictions.csv", record["predictions"])
    write_csv(directory / "oof_per_class.csv", metrics["per_class"])
    for name in ("experiment_report.json", "oof_predictions.csv", "oof_per_class.csv"):
        tracker.add_artifact(directory / name)
    tracker.log_metrics({"oof/macro_f1_447": metrics["macro_f1"], "oof/accuracy": metrics["accuracy"],
        "oof/macro_f1_delta_vs_S008c": report["oof_macro_f1_delta"], "encoder_updates": 0,
        **{"oof/" + k: v for k, v in metrics["errors"].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# S012 {family}\n\nOOF Macro-F1: {metrics['macro_f1']:.9f}; change from S008c: {report['oof_macro_f1_delta']:+.9f}.\n\nThree nested galleries refit the baseline and CPU decision head without meta-validation groups. Policy/threshold selection uses nested meta scores, separately recorded from historical calibration and outer OOF. Known identity ranking and encoder weights are fixed. Family results are exploratory; overall is the prespecified selector. No new submission is built.\n")
    tracker.finish("FINISHED", strict=True)
    return report


def _mark_training_started(tracker):
    """Record entry into CPU postprocessor fitting, never claim encoder training."""
    tracker.flush(strict=True)
    for key, value in {"speaker_id.training_started": "true", "speaker_id.training_scope": "cpu_postprocessor_only",
        "speaker_id.encoder_updates": "0"}.items():
        tracker.client.set_tag(tracker.run_id, key, value)
        tracker.state["tags"][key] = value
    tracker._save()
    tracker.log_metrics({"postprocessor_training_started": 1, "encoder_updates": 0}, sync=True, strict=True)


def execute_suite(root, config_path, binding_path):
    from speaker_id.postprocessing.scoring import prepare_fold, select_baseline
    from speaker_id.postprocessing.nested_cases import make_nested_cases
    from speaker_id.postprocessing.decision_scoring import decision_features, evaluate_decisions
    from speaker_id.postprocessing.tree_models import SKLEARN_VERSION
    import sklearn
    root = Path(root).resolve()
    config_path, binding_path = _path(root, config_path), _path(root, binding_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    checked, supporting = _checked_sources(root, config), _supporting_inputs(root, config)
    source_state = git_provenance(root)
    _require(source_state["src_dirty"] is False and re.fullmatch(r"[a-f0-9]{40}", source_state["git_commit"] or ""),
        "Commit reviewed source before starting S012")
    _require(sklearn.__version__ == SKLEARN_VERSION, "Use the pinned, verified local scikit-learn")
    execution = {**_execution_environment(config), "sklearn": sklearn.__version__, "decision_fit_device": "cpu",
        "encoder_updates": 0, "training_scope": "classical_postprocessor_only"}
    _require("1660 Ti" in execution["device_name"], "S012 is authorized only on this local GTX 1660 Ti; do not use a remote GPU")
    print(json.dumps({"stage": "source_verification", "status": "started"}), flush=True)
    sources = load_sources(root, checked["package"], verify_audio=False)
    contract = sources["contract"]
    _require(len(contract["manifest"]) == 4529 and len(contract["labels"]) == 447, "Original source contract changed")
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    binding.validate()
    _require(binding.experiment_id == "1", "Use only the owned project MLflow experiment 1")
    output = root / config["output_root"] / ("S012_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_binding = {"source_run": SOURCE_RUN, "parent_run_id": SOURCE_PARENT, "child_run_id": SOURCE_CHILD,
        "source_git_commit": SOURCE_COMMIT, "source_report_sha256": SOURCE_REPORT_SHA,
        "package_sha256": PACKAGE_SHA, "s008_verification_sha256": AUDIT_SHA, "s011_verification_sha256": S011_AUDIT_SHA,
        "execution_git_commit": source_state["git_commit"], "suite_config_sha256": file_sha256(config_path)}
    resolved = {"suite": config, "source_binding": source_binding, "execution": execution,
        "data_input_hashes": contract["input_hashes"], "limitations": LIMITATIONS,
        "finite_search": {"outer_folds": 2, "nested_galleries_per_outer": 3, "model_specs": 6,
            "feature_sets": 2, "cascade_modes": 7, "candidates_per_outer": 84, "baseline_entries_per_outer": 1,
            "meta_model_fits_per_outer": 36, "maximum_final_refits_per_outer": 5, "children": 6},
        "supporting_evidence": {name: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)} for name, path in supporting.items()}}
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "source_provenance.json", sources["proof"])
    receipt = _seal_sources(output, sources)
    source_binding["sealed_arrays_sha256"] = receipt["sha256"]
    write_json(output / "resolved_config.json", resolved)
    input_paths = {"suite_config": config_path, "launcher": root / "scripts/score_decisions.py", **checked["paths"], **supporting,
        **{key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")}}
    common = {"project_root": root, "binding": binding, "input_paths": input_paths,
        "run_kind": "local_nested_cpu_decision_heads", "training_started": False}
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
        _require(pooled == checked["report"]["oof"] and pooled["macro_f1"] == BASELINE_MACRO_F1, "Both baseline folds must reproduce every pooled metric")
        checks["pooled"] = {**verify_predictions(root / SOURCE_RUN / "S008c/oof_predictions.csv", baseline_predictions),
            "exact_pooled_metrics": True, "macro_f1": BASELINE_MACRO_F1}
        write_json(output / "baseline_control_checks.json", checks)
        parent.add_artifact(output / "baseline_control_checks.json")
        parent.flush(strict=True)
        for family in RECIPES:
            directory = output / family
            directory.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=directory / "tracking", parent_run_id=parent.run_id,
                run_name="S012-" + family, config={**resolved, "recipe": family}, **common)
            active.append(child)
            child.add_artifact(output / "baseline_control_checks.json")
            child.flush(strict=True)
            records[family] = {"family": family, "directory": directory, "tracker": child, "predictions": [], "folds": []}
        for outer in (0, 1):
            sequence, last_flush = [0], [time.monotonic()]
            def progress(info):
                sequence[0] += 1
                event = {"outer_fold": outer, **info}
                print(json.dumps(event), flush=True)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
                write_json(output / "experiment_state.json", {"status": "running", "parent_run_id": parent.run_id,
                    "postprocessor_training_started": fit_started, "encoder_updates": 0, "latest_progress": event})
                values = {f"fold_{outer}/progress/{k}": v for k, v in info.items() if type(v) in (int, float) and math.isfinite(v)}
                if values:
                    parent.log_metrics(values, step=sequence[0], sync=False)
                if sequence[0] % 4 == 0 or time.monotonic() - last_flush[0] >= 30:
                    parent.add_artifact(output / "progress.jsonl")
                    parent.flush(strict=True)
                    last_flush[0] = time.monotonic()
            nested = make_nested_cases(sources["vectors"]["public"], sources["vectors"]["advanced"], sources["valid"],
                contract["manifest"], contract["folds"], contract["labels"], outer,
                lambda p, b, scope: decision_features(p, b, scope, config["device"]), on_progress=progress, prepared=prepared[outer])
            _save_nested(output / f"nested_cases/fold_{outer}", nested, parent, outer)
            if not fit_started:
                for tracker in [parent, *[records[f]["tracker"] for f in RECIPES if f != "baseline"]]:
                    _mark_training_started(tracker)
                fit_started = True
            progress({"stage": "cpu_postprocessor_fit", "status": "started"})
            search = evaluate_decisions(prepared[outer], baseline[outer], nested, device=config["device"], on_progress=progress)
            _require(set(search["results"]) == set(FAMILIES + ["overall"]), "Unexpected decision family coverage")
            archive = _save_search(output / f"policy_search/fold_{outer}", search, parent, outer, source_binding)
            parity = _deployment_parity(output / f"policy_search/fold_{outer}", prepared[outer], baseline[outer], search, parent, outer)
            for family, result in {"baseline": baseline[outer], **search["results"]}.items():
                parity_family = parity["families"].get(family, {"baseline_retained": True, "exact_decisions": True, "promotion_blocked": False})
                _persist_fold(records[family], contract, prepared[outer], result, baseline[outer], search, outer, archive, checks[str(outer)], parity_family)
        reports = {}
        for family in RECIPES:
            reports[family] = _finish_child(records[family], contract, baseline_predictions)
            active.remove(records[family]["tracker"])
        for name, array in {**sources["vectors"], "valid": sources["valid"]}.items():
            _require(hashlib.sha256(array.tobytes()).hexdigest() == receipt["arrays"][name]["array_sha256"], "An original source cache was mutated")
        import torch
        report = {"status": "complete", "parent_run_id": parent.run_id, "results": reports,
            "children": {f: records[f]["tracker"].run_id for f in RECIPES}, "baseline_control_checks": checks,
            "source_binding": source_binding, "source_arrays_unchanged": True, "encoder_updates": 0,
            "postprocessor_training_started": fit_started, "finite_search": resolved["finite_search"],
            "elapsed_seconds": time.monotonic() - started, "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "selection_policy": config["selection_policy"], "limitations": LIMITATIONS, "new_submission_built": False}
        write_json(output / "experiment_report.json", report)
        for name in ("experiment_report.json", "resolved_config.json", "progress.jsonl"):
            parent.add_artifact(output / name)
        for family, item in reports.items():
            parent.log_metrics({family + "/oof_macro_f1_447": item["oof"]["macro_f1"], family + "/oof_delta": item["oof_macro_f1_delta"]}, sync=False)
        parent.write_report(report, markdown="# S012 nested decision-head evaluation\n\nBoth original S008c folds reproduced all predictions and 447-class metrics before any CPU learner fitting. Three nested galleries refit baseline calibration without each meta-validation group. The primary selector requires pooled meta gain at least 0.001 and no meta-fold loss above 0.002; otherwise it retains S008c. Historical calibration, nested selection and outer OOF scores are separately labeled. Only classical CPU postprocessors were trained; encoder updates are zero.\n\n" + "\n".join(f"- {f}: OOF {r['oof']['macro_f1']:.9f}; delta {r['oof_macro_f1_delta']:+.9f}." for f, r in reports.items()) + "\n")
        parent.finish("FINISHED", strict=True)
        roundtrip = {}
        for family, tracker in [(f, records[f]["tracker"]) for f in RECIPES] + [("parent", parent)]:
            roundtrip[family] = {"artifacts": tracker.verify_artifacts(), "metadata": tracker.verify_remote_metadata()}
        write_json(output / "tracking_roundtrip_verification.json", {"status": "passed", "runs": roundtrip})
        write_json(output / "experiment_state.json", {"status": "complete", "parent_run_id": parent.run_id,
            "children": report["children"], "all_mlflow_finished_and_verified": True, "postprocessor_training_started": True,
            "encoder_updates": 0, "git_commit": source_state["git_commit"]})
        return {"output": str(output), "parent_run_id": parent.run_id, "results": {f: r["oof"]["macro_f1"] for f, r in reports.items()},
            "all_mlflow_finished_and_verified": True, "new_submission_built": False}
    except BaseException as error:
        failure = {"status": "failed", "parent_run_id": parent.run_id, "error_type": type(error).__name__,
            "error": parent.redactor.text(str(error)), "postprocessor_training_started": fit_started,
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
