"""P001 build orchestration; this module is never included in the submission ZIP."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import uuid
import zipfile

import numpy as np

from speaker_id.infrastructure.data import confined_path
from speaker_id.models.campp import file_sha256
from speaker_id.training.contracts import load_contract
from speaker_id.training.final_references import prepare_final_references
from speaker_id.tracking.snapshot import write_json


SOURCE_FILES = (
    "src/speaker_id/__init__.py",
    "src/speaker_id/inference/__init__.py",
    "src/speaker_id/inference/scoring.py",
    "src/speaker_id/inference/runtime.py",
    "src/speaker_id/models/__init__.py",
    "src/speaker_id/models/campp.py",
    "src/speaker_id/models/vendor/__init__.py",
    "src/speaker_id/models/vendor/campplus/__init__.py",
    "src/speaker_id/models/vendor/campplus/DTDNN.py",
    "src/speaker_id/models/vendor/campplus/layers.py",
    "src/speaker_id/models/vendor/campplus/LICENSE",
    "src/speaker_id/models/vendor/campplus/PROVENANCE.json",
)


def _relative_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
        raise ValueError("Build input must be a relative project path")
    path = confined_path(root, Path(name))
    if not path.is_file():
        raise ValueError(f"Required build input is missing: {name}")
    return path


def load_build_contract(root: Path, config_path: Path, *, verify_audio=False) -> tuple[dict, dict, dict]:
    root = root.resolve(strict=True)
    path = confined_path(root, config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    fixed = {"schema_version": 1, "experiment_code": "P001", "method": "max_reference",
             "output_root": "artifacts/releases", "expected_vast_instance_id": 50079023,
             "tracking_required": True, "inference": {"seconds": 180.0, "maximum_windows": 1},
             "unknown_weights": [0.0, .25, .5, .75, 1.0], "margin_weights": [0.0, .5],
             "threshold_candidates": 201, "probability_temperature": .05,
             "expected_calibration_queries": 4440, "expected_known_references": 2217,
             "expected_unknown_references": 2223, "selection_recipe": "S002f"}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("P001 must preserve the preregistered S002f extraction/scoring policy")
    baseline = _relative_file(root, config["baseline_config"])
    contract = load_contract(baseline, root, verify_audio=verify_audio)
    if contract["config"]["mode"] != "frozen_baseline" or contract["config"]["inference"] != config["inference"]:
        raise ValueError("P001 requires the public frozen B002 full-utterance contract")
    source = confined_path(root, Path(config["source_run"]))
    selection = confined_path(root, Path(config["selection_run"]))
    if not source.is_relative_to(root / "artifacts/training") or not selection.is_relative_to(root / "artifacts/training"):
        raise ValueError("Source/selection evidence must remain in training artifacts")
    source_state = json.loads((source / "experiment_state.json").read_text())
    selection_state = json.loads((selection / "experiment_state.json").read_text())
    selection_report = json.loads((selection / "experiment_report.json").read_text())
    recipe_report = json.loads((selection / "S002f/experiment_report.json").read_text())
    if (source_state.get("status") != "complete" or source_state.get("parent_run_id") != config["source_parent_run_id"]
            or selection_state.get("status") != "complete" or selection_state.get("parent_run_id") != config["selection_parent_run_id"]
            or selection_report.get("status") != "complete"
            or selection_report.get("parent_run_id") != config["selection_parent_run_id"]):
        raise ValueError("Completed B002/S002 evidence does not match the package configuration")
    recipe = recipe_report["recipe"]
    if (recipe.get("id") != "S002f" or recipe.get("method") != "max_reference"
            or recipe.get("calibration_protocol") != "leave_content_group_out"
            or recipe.get("unknown_weights") != config["unknown_weights"]
            or recipe.get("margin_weights") != config["margin_weights"]
            or abs(recipe_report["oof"]["macro_f1"] - config["selection_oof_macro_f1_447"]) > 1e-12):
        raise ValueError("S002f evidence differs from the selected fixed recipe")
    inputs = {"package_config": path, "baseline_config": baseline,
              "source_config": source / "resolved_config.json", "source_report": source / "experiment_report.json",
              "selection_report": selection / "experiment_report.json",
              "selection_recipe_report": selection / "S002f/experiment_report.json",
              "selection_cache_provenance": selection / "cache_provenance.json",
              "submission_entrypoint": _relative_file(root, "submission.py"),
              "build_entrypoint": _relative_file(root, "scripts/package_frozen.py"),
              "public_weights": _relative_file(root, contract["model"]["weights_path"]),
              **{key: root / contract["config"][key] for key in ("manifest", "folds", "roles", "label_map", "model_config")}}
    for name in SOURCE_FILES:
        _relative_file(root, name)
    return config, contract, inputs


def release_manifest(package: Path, *, release_id: str, provenance: dict) -> dict:
    """Inventory exactly the portable payload, rejecting links and unsafe paths."""
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("A portable package cannot contain linked paths")
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("A portable package contains a special file")
        name = path.relative_to(package).as_posix()
        if name == "manifest.json":
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in PurePosixPath(name).parts):
            raise ValueError("Hidden/generated files cannot enter the portable payload")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if not files or len({name.casefold() for name in files}) != len(files):
        raise ValueError("Portable payload is empty or has case-colliding paths")
    return {"schema_version": 1, "release_id": release_id, "files": files,
            "leaderboard_validation": "pending", "encoder_updates": 0,
            "provenance": provenance}


def write_portable_zip(package: Path, archive_path: Path, manifest: dict) -> dict:
    """Write only the manifest allowlist, with submission.py at ZIP root."""
    names = sorted([*manifest["files"], "manifest.json"])
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual != set(names):
        raise ValueError("Package file set changed after manifest construction")
    if archive_path.exists():
        raise FileExistsError("A release archive already exists")
    temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
    with temporary.open("xb") as output:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                    raise ValueError("Unsafe portable archive path")
                source = package / name
                if source.is_symlink() or not source.resolve().is_relative_to(package.resolve()):
                    raise ValueError("Portable file escaped the package")
                payload = source.read_bytes()
                if name != "manifest.json" and (len(payload) != manifest["files"][name]["bytes"]
                        or hashlib.sha256(payload).hexdigest() != manifest["files"][name]["sha256"]):
                    raise ValueError("Portable file changed after manifest construction")
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(names):
            raise ValueError("Portable ZIP readback failed")
    if archive_path.exists():
        raise FileExistsError("A release archive appeared during construction")
    temporary.rename(archive_path)
    return {"archive_name": archive_path.name, "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": file_sha256(archive_path), "member_count": len(names), "crc_verified": True}


def build_payload(root: Path, package: Path, model: dict, labels: list[str], final: dict,
                  *, release_id: str, provenance: dict) -> dict:
    package.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        destination = package / name.removeprefix("src/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_relative_file(root, name), destination)
    shutil.copyfile(_relative_file(root, "submission.py"), package / "submission.py")
    assets = package / "assets"
    assets.mkdir()
    portable_model = {**model, "weights_path": "assets/campplus_voxceleb.bin"}
    weights = _relative_file(root, model["weights_path"])
    if file_sha256(weights) != model["weights_sha256"]:
        raise ValueError("Pinned public weight changed before packaging")
    shutil.copyfile(weights, assets / "campplus_voxceleb.bin")
    write_json(assets / "model_config.json", portable_model)
    write_json(assets / "labels.json", {"labels": labels, "unknown_index": 0})
    write_json(assets / "calibration.json", final["calibration"])
    np.savez_compressed(assets / "gallery.npz", **final["gallery"])
    (package / "README.md").write_text(
        "# P001 CAM++ speaker identification\n\n"
        "Run from the extracted ZIP directory:\n\n"
        "    python submission.py --data-dir /path/to/audio --predictions-file-path /path/to/predictions.csv\n\n"
        "Output columns are audio_file,speaker_id. The label map has unknown first and 446 known UUIDs.\n"
        "The supplied public frozen CAM++ weight and all reference/calibration assets are local.\n"
        "No network, hub cache, credentials, MLflow, source training audio, or project checkout is required.\n"
        "Inference uses content-based SoundFile decoding, 16 kHz Kaldi80 FBank and a single centered window capped at 180 seconds.\n"
        "Zero signal or an audio decoding failure produces unknown; model/configuration integrity errors fail the run.\n\n"
        "Runtime: Python 3.12; NumPy 2.x below 2.3; SciPy below 1.16; SoundFile 0.13.x; "
        "matched Torch/Torchaudio 2.10.x. CUDA 12.8 was used for development; inference also supports CPU.\n"
        "Tested development pins: NumPy 2.2.6, SciPy 1.15.3, SoundFile 0.13.1, Torch/Torchaudio 2.10.0+cu128.\n"
        "Installed evaluator versions, GPU/time/memory limits and final leaderboard execution are not certified by this bundle.\n\n"
        "S002f achieved development OOF Macro-F1 0.9386383080637496. The final scorer was separately calibrated "
        "on all available training data with each query content group excluded, then all references were restored. "
        "Its training calibration score is not an OOF or hidden-test result. Encoder optimizer updates: zero.\n\n"
        "CAM++ code/weights provenance and Apache-2.0 license are in speaker_id/models/vendor/campplus/.\n",
        encoding="utf-8")
    manifest = release_manifest(package, release_id=release_id, provenance=provenance)
    write_json(package / "manifest.json", manifest)
    return manifest


def execute_build(root: Path, config_path: Path, binding_path: Path) -> dict:
    """Only an explicit remote invocation may fit the final scorer and publish its run."""
    if os.environ.get("VAST_INSTANCE_ID") != "50079023":
        raise RuntimeError("P001 execution requires the authorized Vast 50079023 RTX3090 CUDA server")
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.inference.scoring import score_embeddings
    from speaker_id.inference.runtime import verify_payload
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.training.frozen_suite import validated_cache
    from speaker_id.training.reference_scoring import known_scores, reference_probabilities

    root = root.resolve(strict=True)
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("P001 execution requires the authorized Vast 50079023 RTX3090 CUDA server")
    config, contract, inputs = load_build_contract(root, config_path, verify_audio=True)
    ready = validate_readiness_for_execution(root, contract)
    torch.set_num_threads(contract["config"]["cpu_threads"])
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    source_hashes = {name: file_sha256(root / name) for name in SOURCE_FILES}
    binding = ExperimentBinding(**json.loads(confined_path(root, binding_path).read_text())["binding"])
    release_id = "P001_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = confined_path(root, Path(config["output_root"]) / release_id)
    output.mkdir(parents=True, exist_ok=False)
    tracker = DurableMLflowRun.prepare(project_root=root, spool_dir=output / "tracking", binding=binding,
        run_name=release_id + "-frozen-final-reference-package", run_kind="frozen_reference_packaging",
        training_started=False, config={"package": config, "baseline_signature": contract["signature"],
                                       "input_hashes": input_hashes, "portable_source_hashes": source_hashes}, input_paths=inputs)
    try:
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.verify_remote_metadata()
        write_json(output / "build_state.json", {"status": "running", "parent_run_id": tracker.run_id, "encoder_updates": 0})
        for name in ("submission.py", "scripts/package_frozen.py"):
            tracker.add_artifact(root / name, "build_source/" + name)
        source = root / config["source_run"]
        embeddings, valid, cache_provenance = validated_cache(root, source, contract, config["source_parent_run_id"])
        selected_cache = json.loads(inputs["selection_cache_provenance"].read_text())
        if any(cache_provenance[key] != selected_cache.get(key)
               for key in ("source_signature", "source_parent_run_id", "files")):
            raise ValueError("Verified B002 cache differs from the exact cache used by completed S002")
        write_json(output / "cache_provenance.json", cache_provenance)
        final = prepare_final_references(embeddings, valid, contract["manifest"], contract["folds"], contract["labels"],
            unknown_weights=config["unknown_weights"], margin_weights=config["margin_weights"],
            candidates=config["threshold_candidates"], temperature=config["probability_temperature"])
        for report_key, config_key in (("calibration_query_files", "expected_calibration_queries"),
                                       ("known_reference_files", "expected_known_references"),
                                       ("unknown_reference_files", "expected_unknown_references")):
            if final["report"][report_key] != config[config_key]:
                raise ValueError("Final reference/query count differs from the pinned EDA contract")
        write_json(output / "final_calibration_curve.json", {"metric_scope": "training calibration only; not OOF", "curve": final["curve"]})
        np.savez_compressed(output / "final_calibration_scores.npz", calibration_indices=final["calibration_indices"],
            known_scores=final["calibration_known_scores"], unknown_similarity=final["calibration_unknown_similarity"],
            reference_indices=final["reference_indices"])
        # Compare the portable scorer against the established training implementation.
        sample = np.unique(np.r_[np.linspace(0, len(embeddings) - 1, 16, dtype=int), np.flatnonzero(~valid)[:1]])
        normalized = embeddings[sample].copy()
        normalized[valid[sample]] /= np.linalg.norm(normalized[valid[sample]], axis=1, keepdims=True)
        gallery = final["gallery"]
        expected_known = known_scores(normalized, gallery["known_embeddings"], gallery["known_targets"], "max_reference")
        expected_unknown = (normalized @ gallery["unknown_embeddings"].T).max(axis=1)
        np.clip(expected_known, -1, 1, out=expected_known)
        np.clip(expected_unknown, -1, 1, out=expected_unknown)
        expected = reference_probabilities(expected_known, expected_unknown, final["calibration"], valid[sample], config["probability_temperature"])
        observed = score_embeddings(embeddings[sample], valid[sample], gallery, final["calibration"])
        if not np.allclose(expected, observed, rtol=0, atol=1e-6) or not np.array_equal(expected.argmax(axis=1), observed.argmax(axis=1)):
            raise ValueError("Portable cached scorer does not reproduce the established scorer")
        provenance = {"source_parent_run_id": config["source_parent_run_id"],
            "source_signature": cache_provenance["source_signature"],
            "selection_parent_run_id": config["selection_parent_run_id"], "selection_recipe": "S002f",
            "development_oof_macro_f1_447": config["selection_oof_macro_f1_447"],
            "final_metric_scope": "training group-excluded calibration; not OOF", "build_git_commit": ready["git_commit"],
            "model_weight_sha256": contract["model"]["weights_sha256"],
            "data_manifest_sha256": contract["input_hashes"]["manifest"], "encoder_updates": 0}
        package = output / "package"
        manifest = build_payload(root, package, contract["model"], contract["labels"], final,
                                 release_id=release_id, provenance=provenance)
        verify_payload(package)
        for name, path in inputs.items():
            if file_sha256(path) != input_hashes[name]:
                raise ValueError("Package input changed during build")
        if any(file_sha256(root / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Portable source changed during build")
        archive_path = output / (release_id + ".zip")
        archive = write_portable_zip(package, archive_path, manifest)
        report = {"status": "built_pending_offline_qa", "release_id": release_id, "parent_run_id": tracker.run_id,
            "encoder_updates": 0, "leaderboard_validation": "pending", "final_fit": final["report"],
            "selected_calibration": final["calibration"], "provenance": provenance,
            "cached_score_parity_samples": len(sample), "cached_score_parity_max_abs_error": float(np.max(np.abs(expected - observed))),
            "archive": archive, "portable_files": len(manifest["files"]),
            "command": "python submission.py --data-dir INPUT --predictions-file-path OUTPUT.csv",
            "required_next_check": "Extract ZIP outside the checkout and run real audio with network blocked and empty caches; no hidden leaderboard execution has occurred."}
        write_json(output / "build_report.json", report)
        for name in ("cache_provenance.json", "final_calibration_curve.json", "final_calibration_scores.npz", "build_report.json"):
            tracker.add_artifact(output / name, name)
        for name in ("assets/gallery.npz", "assets/calibration.json", "assets/labels.json", "manifest.json"):
            tracker.add_artifact(package / name, "release/" + name)
        tracker.add_artifact(archive_path, "release/" + archive_path.name)
        tracker.log_metrics({"final_calibration/training_group_excluded_macro_f1_447": final["report"]["training_group_excluded_macro_f1_447"],
            "references/known_files": final["report"]["known_reference_files"], "references/unknown_files": final["report"]["unknown_reference_files"],
            "calibration/query_files": final["report"]["calibration_query_files"], "encoder/optimizer_updates": 0,
            "release/archive_bytes": archive["archive_bytes"], "validation/cached_scorer_max_abs_error": report["cached_score_parity_max_abs_error"]})
        tracker.write_report(report, markdown="# P001 frozen CAM++ release\n\nFinal scorer calibrated on all training data with whole-query-group exclusion. Encoder updates: zero.\n\nThe final calibration score is a training fit metric. S002f OOF performance remains separate.\n\nPortable package built; independent offline audio QA and actual leaderboard execution are pending.\n")
        tracker.flush(strict=True)
        roundtrip = tracker.verify_artifacts()
        metadata = tracker.verify_remote_metadata()
        write_json(output / "mlflow_roundtrip.json", {"artifacts": roundtrip, "metadata": metadata})
        tracker.finish("FINISHED", strict=True)
        write_json(output / "build_state.json", {"status": "complete", "parent_run_id": tracker.run_id,
                                                  "leaderboard_validation": "pending", "encoder_updates": 0})
        return {"output": str(output), **report}
    except BaseException as error:
        failure = tracker.redactor({"status": "failed", "error_type": type(error).__name__, "error": str(error), "encoder_updates": 0})
        write_json(output / "failure.json", failure)
        tracker.write_report(failure)
        tracker.finish("FAILED", strict=False)
        write_json(output / "build_state.json", {**failure, "parent_run_id": tracker.run_id})
        raise
