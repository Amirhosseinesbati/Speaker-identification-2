"""Tracked, nested screen for the S033 frozen-winner stability veto.

S033 is deliberately narrow.  It replays C002b exactly, then asks whether a
*frozen* known prediction survives balanced re-sampling of a role-disjoint
known enrollment gallery.  The postprocessor may only veto a known prediction
to ``unknown``.  It cannot rerank known speakers, refit an encoder, or touch
audio/model files.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import score_predictions, validate_labels
from speaker_id.evaluation.scoring_bridge import (
    PROBABILITY_TEMPERATURE,
    _load_c002_identity_cache,
    _load_historical_policy,
    _load_historical_probabilities,
    _score_with_fixed_policy,
)
from speaker_id.research.s033_reference_resampling import (
    apply_stability_veto,
    group_disjoint_frozen_winner_stability,
    make_balanced_resampled_galleries,
)
from speaker_id.training.candidate_fusion import weighted_encoder_pair
from speaker_id.training.f005_contract import load_f005_contract
from speaker_id.training.reference_scoring import reference_probabilities
from speaker_id.training.scoring import macro_f1_indices
from speaker_id.tracking.snapshot import git_provenance, sha256_file, write_json


SCHEMA_VERSION = 1
BASELINE_MACRO_F1 = 0.9565282892229405
BASELINE_ACCURACY = 0.9589313314197394
BASELINE_ERRORS = {
    "known_to_unknown": 114,
    "unknown_to_known": 60,
    "known_to_other_known": 12,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_sha(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _project_path(root: Path, value: Path | str) -> Path:
    root = Path(root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _require(path.is_relative_to(root), "S033 project input escapes the project root")
    return path


def _compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _valid_thresholds(values) -> list[float]:
    _require(isinstance(values, list) and values, "S033 threshold grid must be a nonempty list")
    result = [float(value) for value in values]
    _require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result),
             "S033 thresholds must be finite values in [0, 1]")
    _require(result == sorted(set(result)) and result[0] == 0.0,
             "S033 thresholds must be unique, ascending, and include baseline threshold zero")
    return result


def validate_config(config: dict) -> dict:
    """Validate the small fixed S033 screen, without opening any data/cache."""
    _require(isinstance(config, dict), "S033 config must be an object")
    required = {
        "schema_version", "experiment_code", "run_name", "hypothesis", "source",
        "output_root", "fold_ids", "sampling", "selection", "screen_gate",
        "tracking", "retention",
    }
    _require(set(config) == required, "S033 config fields changed")
    _require(config["schema_version"] == SCHEMA_VERSION and config["experiment_code"] == "S033",
             "Unexpected S033 experiment identity")
    _require(isinstance(config["run_name"], str) and config["run_name"].startswith("S033-"),
             "S033 requires an explicit run name")
    _require(isinstance(config["hypothesis"], str) and config["hypothesis"],
             "S033 hypothesis is required")
    source = config["source"]
    _require(set(source) == {"f005_source_config", "recipe", "cache_kind"}
             and source["f005_source_config"] == "configs/train/campp_f005_consistency.json"
             and source["recipe"] == "C002b" and source["cache_kind"] == "identity",
             "S033 source binding changed")
    _require(config["output_root"] == "artifacts/research/s033_reference_resampling", 
             "S033 output retention location changed")
    _require(config["fold_ids"] == [0, 1], "S033 requires both original outer folds")

    sampling = config["sampling"]
    _require(set(sampling) == {
        "n_resamples", "per_class", "seed", "replace", "backend", "query_batch_size",
        "reference_scope", "query_scope", "group_exclusion",
    }, "S033 sampling fields changed")
    _require(type(sampling["n_resamples"]) is int and sampling["n_resamples"] >= 16
             and type(sampling["per_class"]) is int and sampling["per_class"] == 1
             and type(sampling["seed"]) is int and sampling["seed"] >= 0
             and sampling["replace"] is False and sampling["backend"] == "cuda"
             and type(sampling["query_batch_size"]) is int and sampling["query_batch_size"] >= 1
             and sampling["reference_scope"] == "known_valid_enrollment_allowed_only"
             and sampling["query_scope"] == "original_calibration_query_then_outer_evaluation"
             and sampling["group_exclusion"] == "role_groups_must_be_disjoint",
             "S033 sampling protocol changed")
    selection = config["selection"]
    _require(set(selection) == {"metric", "threshold_grid", "tie_break", "outer_labels_forbidden_until_sealed"}
             and selection["metric"] == "inner_macro_f1_447"
             and selection["tie_break"] == "lowest_threshold_least_intervention"
             and selection["outer_labels_forbidden_until_sealed"] is True,
             "S033 inner-selection protocol changed")
    _valid_thresholds(selection["threshold_grid"])
    gate = config["screen_gate"]
    _require(set(gate) == {
        "minimum_pooled_macro_f1", "require_positive_each_fold", "minimum_unknown_to_known_reduction",
        "maximum_known_to_unknown_increase", "require_no_known_rerank",
    } and type(gate["minimum_pooled_macro_f1"]) in {float, int}
             and float(gate["minimum_pooled_macro_f1"]) >= BASELINE_MACRO_F1
             and gate["require_positive_each_fold"] is True
             and type(gate["minimum_unknown_to_known_reduction"]) is int
             and type(gate["maximum_known_to_unknown_increase"]) is int
             and gate["require_no_known_rerank"] is True,
             "S033 screen gates changed")
    tracking = config["tracking"]
    _require(tracking == {
        "binding_state": "artifacts/infrastructure/C002_preparation/mlflow_state.json",
        "experiment_id": "1", "upload": ["resolved_config", "source_provenance", "src_snapshot", "pretruth_seals", "reports"],
        "forbidden": ["raw_audio", "embeddings", "model_weights", "optimizer_state", "credentials", "per_file_predictions"],
    }, "S033 MLflow policy changed")
    _require(config["retention"] == {
        "local_transfer": "none_for_screen_only", "server_only": ["cache_reuse_receipts", "run_reports"],
    }, "S033 retention policy changed")
    return config


def _verify_c002_source(source: dict) -> tuple[Path, Path, dict]:
    c002_run = Path(source["run_dir"])
    _require(c002_run.is_dir() and not c002_run.is_symlink(), "Pinned C002 source run is unavailable")
    c002b_dir = c002_run / "C002b"
    _require(c002b_dir.is_dir() and not c002b_dir.is_symlink(), "Pinned C002b source directory is unavailable")
    _require(source.get("recipe") == "C002b" and source.get("recipe_identity") == "C002b_fresh_cuda_identity",
             "S033 source is not the immutable C002b recipe")
    expected = source.get("artifacts")
    _require(isinstance(expected, dict) and expected, "Pinned C002 artifact receipt is missing")
    verified = {}
    for relative, digest in expected.items():
        path = c002_run / relative
        _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(c002_run.resolve()),
                 "Pinned C002 artifact is unavailable or unsafe")
        _require(isinstance(digest, str) and len(digest) == 64 and sha256_file(path) == digest,
                 "Pinned C002 artifact bytes changed: " + relative)
        verified[relative] = digest
    return c002_run, c002b_dir, {
        "run_dir": str(c002_run), "parent_run_id": source["parent_run_id"],
        "child_run_id": source["child_run_id"], "git_commit": source["git_commit"],
        "recipe": source["recipe"], "recipe_identity": source["recipe_identity"],
        "artifact_sha256": verified,
    }


def _role_topology(contract: dict, outer: int) -> tuple[dict[str, int], dict[str, dict], dict[str, dict], np.ndarray]:
    manifest = contract["manifest"]
    positions = {row["audio_file"]: index for index, row in enumerate(manifest)}
    _require(len(positions) == len(manifest), "S033 manifest audio identities are not unique")
    folds = {row["audio_file"]: row for row in contract["folds"]}
    roles_list = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    roles = {row["audio_file"]: row for row in roles_list}
    _require(set(positions) == set(folds) == set(roles) and len(roles) == len(roles_list),
             "S033 metadata coverage differs between manifest, folds, and roles")
    groups = np.asarray([folds[row["audio_file"]]["group_id"] for row in manifest], dtype=object)
    _require(all(isinstance(value, str) and value for value in groups), "S033 requires nonempty content groups")
    return positions, folds, roles, groups


def _endpoint_for_alpha(vectors: dict[str, np.ndarray], valid: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return vectors["public"]
    if alpha == 1.0:
        return vectors["advanced"]
    return weighted_encoder_pair(vectors["public"], vectors["advanced"], valid, alpha)


def _select_threshold(truth_values: np.ndarray, baseline: np.ndarray, stability: np.ndarray,
                      valid: np.ndarray, thresholds: list[float]) -> dict:
    _require(truth_values.ndim == baseline.ndim == stability.ndim == valid.ndim == 1
             and len(truth_values) == len(baseline) == len(stability) == len(valid) and len(truth_values),
             "S033 inner selection arrays must align and be nonempty")
    _require(np.issubdtype(truth_values.dtype, np.integer) and np.issubdtype(baseline.dtype, np.integer)
             and valid.dtype == np.bool_, "S033 inner labels and validity changed")
    candidates = []
    for threshold in thresholds:
        predicted = apply_stability_veto(baseline, stability, valid, minimum_stability=threshold)
        candidates.append({
            "minimum_stability": float(threshold),
            "inner_macro_f1_447": macro_f1_indices(truth_values, predicted, 447),
            "vetoed_known_rows": int(((baseline != 0) & (predicted == 0)).sum()),
            "unknown_to_known": int(((truth_values == 0) & (predicted != 0)).sum()),
            "known_to_unknown": int(((truth_values != 0) & (predicted == 0)).sum()),
        })
    selected = max(candidates, key=lambda row: (row["inner_macro_f1_447"], -row["minimum_stability"]))
    return {
        "baseline_inner_macro_f1_447": macro_f1_indices(truth_values, baseline, 447),
        "selected": dict(selected), "candidates": candidates,
        "selection_scope": "original_calibration_query_roles_only",
        "tie_break": "lowest_threshold_least_intervention",
    }


def _histogram(values: np.ndarray) -> dict[str, int]:
    return {str(value): int(count) for value, count in sorted(Counter(np.asarray(values).tolist()).items())}


def _build_fold_pretruth(
        contract: dict, vectors: dict[str, np.ndarray], valid: np.ndarray,
        c002_run: Path, c002b_dir: Path, c002_cache: dict, source: dict,
        outer: int, config: dict,
) -> dict:
    """Build one fold's frozen baseline, stability feature, and sealed choice.

    This function intentionally does not call ``score_predictions`` or inspect
    the speaker ID of an outer row.  It returns only arrays needed after both
    folds have been sealed.
    """
    labels = validate_labels(contract["labels"])
    label_index = {label: index for index, label in enumerate(labels)}
    positions, folds, roles, groups = _role_topology(contract, outer)
    policy = _load_historical_policy(c002_run, c002b_dir, outer)
    historical_probabilities, historical_names, historical_receipt = _load_historical_probabilities(
        c002b_dir, outer, labels,
    )
    replayed_probabilities, replayed_names, score = _score_with_fixed_policy(
        vectors["public"], vectors["advanced"], valid, contract["manifest"], contract["folds"], labels,
        outer, policy,
    )
    _require(replayed_names == historical_names and np.array_equal(replayed_probabilities, historical_probabilities),
             f"C002b exact probability replay failed for fold {outer}")
    outer_indices = np.asarray(score["outer_indices"], dtype=np.int64)
    _require([contract["manifest"][int(index)]["audio_file"] for index in outer_indices] == historical_names,
             "S033 C002b outer row order differs from its probability artifact")
    outer_role_indices = {
        positions[name] for name, role in roles.items() if truth(role["outer_evaluation_included"])
    }
    _require(set(outer_indices.tolist()) == outer_role_indices,
             "S033 outer scoring rows differ from immutable outer evaluation roles")

    calibration_indices = np.asarray(score["calibration_indices"], dtype=np.int64)
    calibration_lookup = {int(index): position for position, index in enumerate(calibration_indices)}
    selected_calibration_indices = np.asarray([
        positions[name] for name, role in roles.items() if truth(role["calibration_query"])
    ], dtype=np.int64)
    _require(len(selected_calibration_indices) and set(selected_calibration_indices).issubset(calibration_lookup),
             "S033 original calibration queries are unavailable from frozen C002b scoring")
    calibration_locations = np.asarray(
        sorted(calibration_lookup[int(index)] for index in selected_calibration_indices), dtype=np.int64,
    )
    selected_calibration_indices = calibration_indices[calibration_locations]

    inner_probabilities = reference_probabilities(
        score["inner_known_scores"], score["inner_unknown_similarity"], policy["calibration"],
        valid[calibration_indices], PROBABILITY_TEMPERATURE,
    )
    inner_baseline = inner_probabilities[calibration_locations].argmax(axis=1).astype(np.int64)
    outer_baseline = historical_probabilities.argmax(axis=1).astype(np.int64)
    _require(np.array_equal(outer_baseline, replayed_probabilities.argmax(axis=1)),
             "S033 historical and replayed C002b labels differ")

    enrollment_indices = np.asarray([
        positions[name] for name, role in roles.items()
        if truth(role["enrollment_allowed"]) and contract["manifest"][positions[name]]["speaker_id"] != "unknown"
    ], dtype=np.int64)
    _require(len(enrollment_indices), "S033 has no eligible known enrollment references")
    enrollment_labels = np.asarray([
        label_index[contract["manifest"][int(index)]["speaker_id"]] for index in enrollment_indices
    ], dtype=np.int64)
    _require(valid[enrollment_indices].all() and np.array_equal(np.unique(enrollment_labels), np.arange(1, len(labels))),
             "S033 enrollment gallery lacks valid coverage of the fixed known labels")
    query_indices = np.concatenate((selected_calibration_indices, outer_indices))
    query_groups = groups[query_indices]
    reference_groups = groups[enrollment_indices]
    _require(not (set(query_groups.tolist()) & set(reference_groups.tolist())),
             "S033 role assignment does not keep enrollment and query groups disjoint")

    sampling = config["sampling"]
    galleries = make_balanced_resampled_galleries(
        np.arange(len(enrollment_indices), dtype=np.int64), enrollment_labels,
        n_resamples=sampling["n_resamples"], per_class=sampling["per_class"],
        seed=sampling["seed"] + outer, replace=sampling["replace"],
    )
    endpoint = _endpoint_for_alpha(vectors, valid, policy["advanced_weight"])
    query_baseline = np.concatenate((inner_baseline, outer_baseline))
    stability = group_disjoint_frozen_winner_stability(
        endpoint[query_indices], valid[query_indices], query_groups,
        endpoint[enrollment_indices], valid[enrollment_indices], reference_groups,
        galleries, query_baseline, device=sampling["backend"],
        query_batch_size=sampling["query_batch_size"],
    )
    inner_count = len(selected_calibration_indices)
    inner_stability, outer_stability = stability[:inner_count], stability[inner_count:]
    inner_truth = np.asarray([
        label_index[contract["manifest"][int(index)]["speaker_id"]] for index in selected_calibration_indices
    ], dtype=np.int64)
    selection = _select_threshold(
        inner_truth, inner_baseline, inner_stability, valid[selected_calibration_indices],
        _valid_thresholds(config["selection"]["threshold_grid"]),
    )
    threshold = float(selection["selected"]["minimum_stability"])
    outer_veto = apply_stability_veto(
        outer_baseline, outer_stability, valid[outer_indices], minimum_stability=threshold,
    )
    _require(not np.any((outer_baseline != outer_veto) & ~((outer_baseline != 0) & (outer_veto == 0))),
             "S033 veto attempted a forbidden known identity change")
    seal = {
        "schema_version": SCHEMA_VERSION, "outer_fold": outer,
        "outer_truth_accessed": False, "source": source, "c002_cache": c002_cache,
        "policy": policy, "historical_probability_artifact": historical_receipt,
        "exact_c002b_probability_replay": True,
        "sampling": {
            "n_resamples": sampling["n_resamples"], "per_class": sampling["per_class"],
            "seed": sampling["seed"] + outer, "replace": sampling["replace"],
            "backend": sampling["backend"], "query_batch_size": sampling["query_batch_size"],
            "reference_scope": sampling["reference_scope"], "group_disjoint": True,
            "enrollment_reference_rows": int(len(enrollment_indices)),
            "known_reference_count_histogram": _histogram(np.bincount(enrollment_labels, minlength=len(labels))[1:]),
        },
        "calibration": {
            "rows": int(inner_count), "indices_sha256": _array_sha(selected_calibration_indices),
            "baseline_prediction_sha256": _array_sha(inner_baseline),
            "stability_sha256": _array_sha(inner_stability), "selection": selection,
        },
        "outer_prediction_commitment": {
            "rows": int(len(outer_indices)), "indices_sha256": _array_sha(outer_indices),
            "audio_files_sha256": _sha(historical_names),
            "baseline_prediction_sha256": _array_sha(outer_baseline),
            "stability_sha256": _array_sha(outer_stability),
            "veto_prediction_sha256": _array_sha(outer_veto),
            "vetoed_known_rows": int(((outer_baseline != 0) & (outer_veto == 0)).sum()),
            "known_rerank_count": 0,
        },
    }
    seal["seal_sha256"] = _sha(seal)
    return {
        "outer": outer, "seal": seal, "outer_indices": outer_indices,
        "outer_names": historical_names, "baseline": outer_baseline, "veto": outer_veto,
        "outer_stability": outer_stability,
    }


def _prediction_rows(names: list[str], values: np.ndarray, labels: list[str]) -> list[dict]:
    _require(len(names) == len(values), "S033 prediction names and labels differ")
    return [
        {"audio_file": name, "speaker_id": labels[int(value)]}
        for name, value in zip(names, values, strict=True)
    ]


def _transition_summary(reference: list[dict], baseline: list[dict], veto: list[dict]) -> dict:
    actual = {row["audio_file"]: row["speaker_id"] for row in reference}
    before = {row["audio_file"]: row["speaker_id"] for row in baseline}
    after = {row["audio_file"]: row["speaker_id"] for row in veto}
    _require(set(actual) == set(before) == set(after), "S033 transition rows are incomplete")
    modes: Counter[str] = Counter()
    corrected = regressed = vetoes = reranks = 0
    for name, expected in actual.items():
        old, new = before[name], after[name]
        if old != new:
            vetoes += 1
        if old != "unknown" and new != "unknown" and old != new:
            reranks += 1
        if old != expected and new == expected:
            corrected += 1
        if old == expected and new != expected:
            regressed += 1
        def mode(prediction: str) -> str:
            if prediction == expected:
                return "correct"
            if expected == "unknown":
                return "unknown_to_known"
            if prediction == "unknown":
                return "known_to_unknown"
            return "known_to_other_known"
        modes[mode(old) + "->" + mode(new)] += 1
    return {
        "vetoed_rows": vetoes, "known_rerank_count": reranks,
        "corrected": corrected, "regressed": regressed,
        "error_mode_transitions": dict(sorted(modes.items())),
    }


def _screen_gate(config: dict, baseline: dict, candidate: dict, folds: list[dict], transitions: dict) -> dict:
    gate = config["screen_gate"]
    fold_delta = {str(item["outer_fold"]): item["candidate"]["macro_f1"] - item["baseline"]["macro_f1"] for item in folds}
    unknown_reduction = baseline["errors"]["unknown_to_known"] - candidate["errors"]["unknown_to_known"]
    known_increase = candidate["errors"]["known_to_unknown"] - baseline["errors"]["known_to_unknown"]
    checks = {
        "minimum_pooled_macro_f1": candidate["macro_f1"] >= float(gate["minimum_pooled_macro_f1"]),
        "positive_each_fold": all(value > 0.0 for value in fold_delta.values()),
        "minimum_unknown_to_known_reduction": unknown_reduction >= gate["minimum_unknown_to_known_reduction"],
        "maximum_known_to_unknown_increase": known_increase <= gate["maximum_known_to_unknown_increase"],
        "no_known_rerank": transitions["known_rerank_count"] == 0,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "pooled_macro_f1_delta_vs_c002b": candidate["macro_f1"] - baseline["macro_f1"],
        "fold_macro_f1_delta_vs_c002b": fold_delta,
        "unknown_to_known_reduction": unknown_reduction,
        "known_to_unknown_increase": known_increase,
        "purpose": "research_screen_only_not_a_promotion_or_submission_decision",
    }


def _finish_tracker(tracker) -> dict:
    tracker.flush(strict=True)
    tracker.finish("FINISHED", strict=True)
    return {"run_id": tracker.run_id, "remote_status": tracker.state.get("remote_status")}


def validate(root: Path, config_path: Path) -> dict:
    """Validate S033 configuration only; it does not inspect server cache/data."""
    root = Path(root).resolve()
    config_path = _project_path(root, config_path)
    config = validate_config(_read_json(config_path))
    return {
        "status": "validated", "experiment_code": config["experiment_code"],
        "config_sha256": sha256_file(config_path), "execution_required": "server_cuda_cached_embeddings_only",
    }


def execute(root: Path, config_path: Path, binding_path: Path) -> dict:
    """Run one tracked S033 screen.  No local transfer is performed."""
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding

    root = Path(root).resolve()
    config_path = _project_path(root, config_path)
    config = validate_config(_read_json(config_path))
    binding_path = _project_path(root, binding_path)
    provenance = git_provenance(root)
    _require(provenance["src_dirty"] is False and isinstance(provenance["git_commit"], str),
             "Commit clean S033 source before execution")
    binding = ExperimentBinding(**_read_json(binding_path)["binding"])
    binding.validate()
    _require(binding.experiment_id == config["tracking"]["experiment_id"], "S033 MLflow experiment changed")
    f005_config = _project_path(root, config["source"]["f005_source_config"])
    contract = load_f005_contract(f005_config, root, verify_audio=False, verify_sources=False)
    _require(len(contract["manifest"]) == 4529 and len(contract["labels"]) == 447,
             "S033 source data population changed")
    c002_run, c002b_dir, source = _verify_c002_source(contract["config"]["source_c002b"])
    vectors, valid, c002_cache = _load_c002_identity_cache(c002_run, contract["manifest"])

    output = root / config["output_root"] / (
        "S033_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    )
    output.mkdir(parents=True, exist_ok=False)
    source_provenance = {
        "schema_version": SCHEMA_VERSION, "source": source, "c002_cache": c002_cache,
        "raw_audio_read": False, "model_loaded": False, "embedding_extraction": False,
        "encoder_updates": 0, "cache_files_verified": len(contract["manifest"]),
    }
    write_json(output / "source_provenance.json", source_provenance)
    resolved = {
        "experiment": config, "execution_git_commit": provenance["git_commit"],
        "data_input_hashes": contract["readiness"]["input_hashes"], "source": source,
        "c002_cache": c002_cache, "raw_audio_read": False, "model_loaded": False,
        "embedding_extraction": False, "encoder_updates": 0,
        "outer_truth_sealing": "both folds sealed before the first outer metric",
        "local_transfer": "none_for_screen_only",
    }
    write_json(output / "resolved_config.json", resolved)
    inputs = {
        "s033_config": config_path, "source_f005_config": f005_config,
        "mlflow_binding": binding_path, "frozen_choices": c002_run / "frozen_inner_choices.json",
        "identity_cache_receipt": c002_run / "identity_cache_manifest.json",
    }
    tracker = DurableMLflowRun.prepare(
        project_root=root, spool_dir=output / "tracking", binding=binding, run_name=config["run_name"],
        config=resolved, input_paths=inputs, run_kind="nested_open_set_stability_veto_screen", training_started=False,
    )
    started = time.monotonic()
    try:
        tracker.flush(strict=True)
        for name in ("resolved_config.json", "source_provenance.json"):
            tracker.add_artifact(output / name)
        tracker.add_artifact(config_path, "input_configs/s033_config.json")
        tracker.flush(strict=True)

        sealed: dict[str, dict] = {}
        for outer in config["fold_ids"]:
            prepared = _build_fold_pretruth(
                contract, vectors, valid, c002_run, c002b_dir, c002_cache, source, outer, config,
            )
            fold_dir = output / f"fold_{outer}"
            fold_dir.mkdir()
            seal_path = fold_dir / "pretruth_seal.json"
            write_json(seal_path, prepared["seal"])
            reloaded = _read_json(seal_path)
            _require(reloaded.get("seal_sha256") == _sha({key: value for key, value in reloaded.items() if key != "seal_sha256"}),
                     "S033 pretruth seal was not durably reproduced")
            tracker.add_artifact(seal_path, f"pretruth/fold_{outer}/pretruth_seal.json")
            selected = prepared["seal"]["calibration"]["selection"]["selected"]
            tracker.log_metrics({
                f"fold_{outer}/inner_baseline_macro_f1_447": prepared["seal"]["calibration"]["selection"]["baseline_inner_macro_f1_447"],
                f"fold_{outer}/inner_selected_macro_f1_447": selected["inner_macro_f1_447"],
                f"fold_{outer}/selected_minimum_stability": selected["minimum_stability"],
                f"fold_{outer}/inner_vetoed_known_rows": selected["vetoed_known_rows"],
            }, sync=True, strict=True)
            sealed[str(outer)] = prepared
        root_seal = {
            "schema_version": SCHEMA_VERSION, "both_outer_folds_sealed": True,
            "outer_truth_accessed": False,
            "folds": {key: value["seal"]["seal_sha256"] for key, value in sealed.items()},
        }
        write_json(output / "pretruth_seal.json", root_seal)
        tracker.add_artifact(output / "pretruth_seal.json")
        tracker.flush(strict=True)

        # This is the first point at which outer labels are used: only for the
        # sealed baseline/candidate evaluation below.
        labels = validate_labels(contract["labels"])
        baseline_rows, veto_rows, folds = [], [], []
        for outer in config["fold_ids"]:
            item = sealed[str(outer)]
            references = [contract["manifest"][int(index)] for index in item["outer_indices"]]
            before = _prediction_rows(item["outer_names"], item["baseline"], labels)
            after = _prediction_rows(item["outer_names"], item["veto"], labels)
            baseline_metrics = _compact_metrics(score_predictions(references, before, labels))
            veto_metrics = _compact_metrics(score_predictions(references, after, labels))
            transitions = _transition_summary(references, before, after)
            _require(transitions["known_rerank_count"] == 0, "S033 produced a forbidden known rerank")
            fold_report = {
                "outer_fold": outer, "pretruth_seal_sha256": item["seal"]["seal_sha256"],
                "baseline": baseline_metrics, "candidate": veto_metrics, "transitions": transitions,
                "selected_minimum_stability": item["seal"]["calibration"]["selection"]["selected"]["minimum_stability"],
                "outer_truth_first_access": "after_both_folds_pretruth_sealed",
            }
            fold_dir = output / f"fold_{outer}"
            write_json(fold_dir / "evaluation.json", fold_report)
            tracker.add_artifact(fold_dir / "evaluation.json", f"reports/fold_{outer}/evaluation.json")
            tracker.log_metrics({
                f"fold_{outer}/baseline_macro_f1_447": baseline_metrics["macro_f1"],
                f"fold_{outer}/candidate_macro_f1_447": veto_metrics["macro_f1"],
                f"fold_{outer}/macro_f1_delta_vs_c002b": veto_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                f"fold_{outer}/unknown_to_known_reduction": baseline_metrics["errors"]["unknown_to_known"] - veto_metrics["errors"]["unknown_to_known"],
                f"fold_{outer}/known_to_unknown_increase": veto_metrics["errors"]["known_to_unknown"] - baseline_metrics["errors"]["known_to_unknown"],
            }, sync=True, strict=True)
            baseline_rows.extend(before)
            veto_rows.extend(after)
            folds.append(fold_report)
        baseline_oof = _compact_metrics(score_predictions(contract["manifest"], baseline_rows, labels))
        veto_oof = _compact_metrics(score_predictions(contract["manifest"], veto_rows, labels))
        _require(math.isclose(baseline_oof["macro_f1"], BASELINE_MACRO_F1, rel_tol=0.0, abs_tol=1e-15)
                 and math.isclose(baseline_oof["accuracy"], BASELINE_ACCURACY, rel_tol=0.0, abs_tol=1e-15)
                 and baseline_oof["errors"] == BASELINE_ERRORS,
                 "S033 frozen C002b baseline metrics changed")
        pooled_transitions = _transition_summary(contract["manifest"], baseline_rows, veto_rows)
        screen = _screen_gate(config, baseline_oof, veto_oof, folds, pooled_transitions)
        report = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "experiment_code": "S033",
            "run_id": tracker.run_id, "source": source, "baseline_oof": baseline_oof,
            "candidate_oof": veto_oof, "folds": folds, "pooled_transitions": pooled_transitions,
            "screen_gate": screen, "raw_audio_read": False, "model_loaded": False,
            "embedding_extraction": False, "encoder_updates": 0, "local_transfer_performed": False,
            "retention": config["retention"], "elapsed_seconds": time.monotonic() - started,
        }
        markdown = (
            "# S033 frozen-winner stability veto\n\n"
            f"C002b OOF Macro-F1: **{baseline_oof['macro_f1']:.9f}**  \n"
            f"S033 OOF Macro-F1: **{veto_oof['macro_f1']:.9f}**  \n"
            f"Delta: **{screen['pooled_macro_f1_delta_vs_c002b']:+.9f}**  \n"
            f"unknown→known reduction: **{screen['unknown_to_known_reduction']}**; "
            f"known→unknown increase: **{screen['known_to_unknown_increase']}**.\n\n"
            "The candidate only vetoes frozen known predictions to unknown; it never reranks a known speaker. "
            f"Research screen passed: **{screen['passed']}**. This is not a model-promotion or submission decision.\n"
        )
        write_json(output / "experiment_report.json", report)
        (output / "experiment_report.md").write_text(markdown, encoding="utf-8")
        for name in ("experiment_report.json", "experiment_report.md"):
            tracker.add_artifact(output / name, "reports/" + name)
        tracker.log_metrics({
            "baseline/oof_macro_f1_447": baseline_oof["macro_f1"],
            "candidate/oof_macro_f1_447": veto_oof["macro_f1"],
            "candidate/oof_accuracy": veto_oof["accuracy"],
            "candidate/oof_macro_f1_delta_vs_c002b": screen["pooled_macro_f1_delta_vs_c002b"],
            "candidate/oof_unknown_to_known_reduction": screen["unknown_to_known_reduction"],
            "candidate/oof_known_to_unknown_increase": screen["known_to_unknown_increase"],
            "screen/passed": int(screen["passed"]), "encoder_updates": 0,
        }, sync=True, strict=True)
        tracking = _finish_tracker(tracker)
        report["tracking"] = tracking
        write_json(output / "terminal_receipt.json", {"status": "complete", **tracking})
        return {
            "output": str(output), "run_id": tracker.run_id,
            "baseline_macro_f1": baseline_oof["macro_f1"], "candidate_macro_f1": veto_oof["macro_f1"],
            "screen_passed": screen["passed"], "local_transfer_performed": False,
        }
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION, "status": "failed", "run_id": tracker.run_id,
            "error_type": type(error).__name__, "error": tracker.redactor.text(str(error)),
            "elapsed_seconds": time.monotonic() - started, "local_transfer_performed": False,
        }
        write_json(output / "failure.json", failure)
        tracker.write_report(failure)
        tracker.add_artifact(output / "failure.json", "reports/failure.json")
        try:
            tracker.finish("FAILED", strict=False)
        except Exception:
            pass
        raise
