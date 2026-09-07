"""Validate all experiment inputs before loading a GPU model or opening a run."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from speaker_id.data.splits import truth
from speaker_id.evaluation.metrics import validate_labels
from speaker_id.models.campp import file_sha256, validate_model_config
from speaker_id.training.schedules import validate_adaptation_schedule


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_fit_settings(fit: dict) -> None:
    required = {"epochs", "steps_per_epoch", "batch_size", "crop_seconds", "encoder_lr", "head_lr",
                "weight_decay", "margin", "scale", "freeze_batchnorm", "trainable_prefixes",
                "gradient_clip_norm", "mixed_precision", "checkpoint_every_steps", "epoch_selection", "augmentation"}
    if (not isinstance(fit, dict) or not required.issubset(fit)
            or set(fit) - required - {"adaptation_schedule"}):
        raise ValueError("Missing or unsupported fit settings")
    for key in ("epochs", "steps_per_epoch", "batch_size", "checkpoint_every_steps"):
        if type(fit[key]) is not int or fit[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("crop_seconds", "encoder_lr", "head_lr", "gradient_clip_norm", "scale"):
        value = fit[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("weight_decay", "margin"):
        value = fit[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if fit["margin"] >= math.pi / 2:
        raise ValueError("AAM margin must be smaller than pi/2")
    if fit["freeze_batchnorm"] is not True or fit["epoch_selection"] != "fixed_steps_no_outer_selection":
        raise ValueError("Initial recipes require fixed BN and fixed-step selection")
    if type(fit["mixed_precision"]) is not bool:
        raise ValueError("mixed_precision must be a boolean")
    prefixes = fit["trainable_prefixes"]
    if (not isinstance(prefixes, list) or not prefixes
            or any(not isinstance(item, str) or not item for item in prefixes)
            or len(set(prefixes)) != len(prefixes)):
        raise ValueError("Trainable prefixes must be a nonempty unique list of strings")
    if fit["augmentation"] != "none_initial_control":
        raise ValueError("Unimplemented augmentation cannot silently change an experiment")
    validate_adaptation_schedule(fit)


def load_contract(config_path: Path, root: Path, *, verify_audio: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("mode") not in {"frozen_baseline", "fine_tune"}:
        raise ValueError("Unsupported experiment schema or mode")
    if not config.get("tracking_required"):
        raise ValueError("Live MLflow tracking is mandatory for every execution")
    if config.get("known_classes") != 446 or config.get("evaluation_classes") != 447:
        raise ValueError("Experiment requires 446 known identities and 447 evaluation labels")
    if config.get("device") != "cuda":
        raise ValueError("Learning and full experiments must run on the remote CUDA server")
    if config["scoring"].get("method") != "normalized_mean_prototype" or config["scoring"].get("calibration") != "global_max_cosine":
        raise ValueError("Unsupported scoring policy")
    if config["scoring"]["threshold_candidates"] < 2 or config["scoring"]["probability_temperature"] <= 0:
        raise ValueError("Invalid calibration configuration")
    if config["inference"]["seconds"] <= 0 or config["inference"]["maximum_windows"] < 1:
        raise ValueError("Invalid inference windows")
    validate_fit_settings(config["fit"])
    inputs = {key: (root / config[key]).resolve() for key in ("model_config", "manifest", "folds", "roles", "label_map")}
    hashes = {key: file_sha256(path) for key, path in inputs.items()}
    # Resume/caches are invalidated by implementation changes, not just config.
    code_paths = sorted((root / "src").rglob("*.py")) + [root / "scripts/train.py"]
    code_hashes = {path.relative_to(root).as_posix(): file_sha256(path) for path in code_paths}
    model = json.loads(inputs["model_config"].read_text(encoding="utf-8"))
    validate_model_config(model)
    labels = validate_labels(json.loads(inputs["label_map"].read_text(encoding="utf-8"))["labels"])
    manifest, folds, roles = (read_csv(inputs[key]) for key in ("manifest", "folds", "roles"))
    by_name = {row["audio_file"]: row for row in manifest}
    fold_by_name = {row["audio_file"]: row for row in folds}
    if len(by_name) != len(manifest) or len(manifest) != config["expected_source_files"]:
        raise ValueError("Unexpected or duplicated source files")
    if len(fold_by_name) != len(folds) or by_name.keys() != fold_by_name.keys():
        raise ValueError("Fold coverage differs from the audio manifest")
    configured_folds = config["fold_ids"]
    if sorted(configured_folds) != sorted({int(row["fold"]) for row in folds}) or len(set(configured_folds)) != len(configured_folds):
        raise ValueError("All original outer folds must be evaluated exactly once")
    role_folds = defaultdict(list)
    for row in roles:
        role_folds[int(row["outer_fold"])].append(row)
    summaries = []
    for outer in configured_folds:
        selected = role_folds[outer]
        indexed = {row["audio_file"]: row for row in selected}
        if len(indexed) != len(selected) or indexed.keys() != by_name.keys():
            raise ValueError("Each outer fold needs one role per original file")
        enrollment_labels, fit_labels = set(), set()
        group_roles = defaultdict(set)
        for name, row in indexed.items():
            source, split = by_name[name], fold_by_name[name]
            if row["speaker_id"] != source["speaker_id"] or row["speaker_id"] != split["speaker_id"]:
                raise ValueError("Label mismatch between manifest, split and calibration roles")
            if row["group_id"] != split["group_id"]:
                raise ValueError("Content group mismatch")
            is_outer = int(split["fold"]) == outer
            fitting, enrolling, query, evaluating = (truth(row[key]) for key in ("encoder_fit_allowed", "enrollment_allowed", "calibration_query", "outer_evaluation_included"))
            if evaluating != is_outer or (is_outer and (fitting or enrolling or query)):
                raise ValueError("Outer validation leakage detected")
            if query and (fitting or enrolling):
                raise ValueError("Calibration query leaked into encoder fit or gallery")
            if (fitting or enrolling or query) and not truth(split["train_eligible"]):
                raise ValueError("Ineligible signal entered fit, gallery or calibration")
            if enrolling and row["speaker_id"] == "unknown":
                raise ValueError("Unknown identities cannot share a known prototype")
            if enrolling:
                enrollment_labels.add(row["speaker_id"])
            if fitting and row["speaker_id"] != "unknown":
                fit_labels.add(row["speaker_id"])
            group_roles[row["group_id"]].add((fitting, enrolling, query, evaluating))
        if any(len(values) != 1 for values in group_roles.values()):
            raise ValueError("A content group crosses experiment roles")
        if enrollment_labels != set(labels[1:]) or fit_labels != set(labels[1:]):
            raise ValueError("A known class lacks independent fit/enrollment support")
        summaries.append({"outer_fold": outer, "role_counts": dict(Counter(r["role"] for r in selected)),
                          "known_fit_files": sum(truth(r["encoder_fit_allowed"]) and r["speaker_id"] != "unknown" for r in selected),
                          "known_classes": len(fit_labels)})
    if verify_audio:
        for name, row in by_name.items():
            path = (root / config["data_dir"] / name).resolve()
            if not path.is_file() or file_sha256(path) != row["input_sha256"]:
                raise ValueError(f"Raw audio integrity check failed: {name}")
    resolved = {"config": config, "model": model, "input_hashes": hashes, "code_hashes": code_hashes,
                "labels": labels, "manifest": manifest, "roles": roles, "folds": folds,
                "summary": {"files": len(manifest), "evaluation_classes": len(labels),
                            "folds": summaries, "audio_hashes_checked": verify_audio}}
    resolved["signature"] = hashlib.sha256(json.dumps({"config": config, "model": model, "input_hashes": hashes, "code_hashes": code_hashes}, sort_keys=True).encode()).hexdigest()
    return resolved
