"""Aggregate measured server evidence and reject stale launch authorization.

Readiness never authorizes training by itself. The training entry point still
requires its explicit execution flag and a fresh live MLflow roundtrip.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import subprocess
import sys

from speaker_id.infrastructure.data import confined_path, sha256_file, write_json_atomic


DEFAULT_EVIDENCE = {
    "data": "artifacts/infrastructure/data_readiness.json",
    "runtime": "artifacts/infrastructure/runtime_readiness.json",
    "campp": "artifacts/infrastructure/campp_probe.json",
    "mlflow": "artifacts/infrastructure/mlflow_probe/preflight_result.json",
    "instance": "artifacts/infrastructure/instance.json",
}


class ReadinessError(ValueError):
    """Training is blocked by missing, failed, or stale measured evidence."""


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                            text=True, timeout=15, check=True)
    return result.stdout.strip()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _package_versions() -> dict:
    return {distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")}


def _normalise_device_name(value: object) -> str:
    """Return a stable form for a user-facing GPU name substring."""
    return "".join(str(value).casefold().split())


def _normalise_workspace(value: object) -> str:
    """Canonicalize either a native absolute path or a Linux server path.

    C002 is intentionally a Linux-only execution recipe, but accepting its
    explicit POSIX path during a local static check gives a useful target
    mismatch instead of misclassifying the configuration as malformed.
    """
    text = str(value).strip()
    posix = PurePosixPath(text)
    if posix.is_absolute():
        return posix.as_posix()
    path = Path(text)
    _assert(path.is_absolute(), "Training configuration expected_workspace must be absolute")
    return str(path.resolve())


def _configured_target(config: dict, root: Path) -> dict:
    """Resolve the server identity that this selected recipe is allowed to use.

    Older recipes predate explicit workspace/GPU fields.  Their historical
    target is retained only as a compatibility default; C002 records every
    value in its selected configuration so a copied readiness report cannot
    authorize a different checkout or GPU.
    """
    instance_id = config.get("expected_vast_instance_id")
    _assert(type(instance_id) is int and instance_id > 0,
            "Training configuration must contain a positive expected_vast_instance_id")

    workspace_text = config.get("expected_workspace", str(root))
    _assert(isinstance(workspace_text, str) and workspace_text.strip(),
            "Training configuration must contain a nonempty expected_workspace")
    workspace = _normalise_workspace(workspace_text)

    expected_gpu = config.get("expected_gpu", "RTX 3090")
    _assert(isinstance(expected_gpu, str) and _normalise_device_name(expected_gpu),
            "Training configuration must contain a nonempty expected_gpu")

    minimum_gpu_memory_gib = config.get("minimum_gpu_memory_gib", 20)
    _assert(type(minimum_gpu_memory_gib) in (int, float)
            and math.isfinite(minimum_gpu_memory_gib) and minimum_gpu_memory_gib > 0,
            "Training configuration minimum_gpu_memory_gib must be positive and finite")
    return {
        "instance_id": instance_id,
        "workspace": workspace,
        "expected_gpu": expected_gpu,
        "minimum_gpu_memory_gib": minimum_gpu_memory_gib,
    }


def _data_verification_mode(config: dict) -> str:
    """Select a pinned data-evidence policy without weakening legacy checks."""
    policy = config.get("data_verification", {"mode": "archive_crc"})
    _assert(isinstance(policy, dict) and set(policy) == {"mode"},
            "Training configuration data_verification must contain only mode")
    mode = policy["mode"]
    _assert(mode in {"archive_crc", "installed_manifest_sha256"},
            "Training configuration data_verification mode is unsupported")
    return mode


def _mlflow_state_path(config: dict, root: Path) -> Path:
    """Resolve the experiment binding receipt selected by this contract."""
    value = config.get("mlflow_state_path", "artifacts/infrastructure/mlflow_state.json")
    _assert(isinstance(value, str) and value.strip(),
            "Training configuration mlflow_state_path must be a nonempty string")
    path = confined_path(root, value)
    _assert(path.is_relative_to(root / "artifacts/infrastructure"),
            "MLflow binding state must stay under artifacts/infrastructure")
    return path


def validate_archive_source(root: Path, data: dict) -> dict:
    """Bind alternate container evidence without changing the trusted payload.

    Old local-upload reports retain their original validation path. The official
    original ZIP needs a separately committed container identity; an absent or
    placeholder hash cannot produce ready evidence.
    """
    source_id = data.get("archive_source_id", "local_upload")
    if source_id == "local_upload":
        return {"archive_source_id": source_id}
    _assert(source_id == "official_original", "Unrecognized dataset archive source ID")
    config_path = confined_path(root, "configs/infra/archive_sources.json")
    source = json.loads(config_path.read_text(encoding="utf-8"))["official_original"]
    digest, labels_digest = source.get("archive_sha256"), source.get("labels_sha256")
    for name, value in (("archive", digest), ("labels", labels_digest)):
        _assert(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
                and value != "0" * 64, f"Official {name} source SHA256 is missing or a placeholder")
    expected_archive = confined_path(root, source["archive_path"])
    _assert(expected_archive.is_relative_to(root / "data/incoming") and expected_archive.suffix.lower() == ".zip",
            "Official archive must be an incoming ZIP within this workspace")
    _assert(confined_path(root, data["archive"]) == expected_archive, "Official archive path differs from its source identity")
    _assert(data.get("archive_sha256") == digest.lower(), "Official ZIP SHA256 differs from its separately pinned source identity")
    _assert(type(source.get("archive_size_bytes")) is int and source["archive_size_bytes"] > 0
            and data.get("archive_size_bytes") == source["archive_size_bytes"],
            "Official ZIP byte size differs from its source identity")
    prefix = source.get("member_prefix")
    _assert(isinstance(prefix, str) and re.fullmatch(r"[A-Za-z0-9_-]+", prefix) is not None,
            "Official ZIP source has an invalid explicit member prefix")
    _assert(data.get("archive_prefix") == prefix, "Official ZIP layout prefix differs from its source identity")
    _assert(data.get("labels_byte_sha256_verified") is True
            and data.get("expected_labels_sha256") == labels_digest.lower()
            and data.get("labels_sha256") == labels_digest.lower(),
            "Official ZIP labels.csv was not checked against the exact original CSV bytes")
    return {"archive_source_id": source_id, "archive_sha256": digest.lower(),
            "archive_source_config_sha256": sha256_file(config_path),
            "labels_byte_sha256_verified": True, "archive_prefix": prefix}


def _read_evidence(root: Path, paths: dict) -> tuple[dict, dict, list]:
    reports, metadata, checks = {}, {}, []
    for name, value in paths.items():
        try:
            path = confined_path(root, value)
            if not path.is_relative_to(root / "artifacts/infrastructure"):
                raise ReadinessError("Evidence must live under artifacts/infrastructure")
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ReadinessError("Evidence root must be a JSON object")
            reports[name] = report
            metadata[name] = {"path": str(path.relative_to(root).as_posix()), "sha256": sha256_file(path)}
            checks.append({"check": f"evidence_{name}", "status": "passed"})
        except Exception as error:
            checks.append({"check": f"evidence_{name}", "status": "blocked", "reason": str(error)})
    return reports, metadata, checks


def check_readiness(root: Path, config_path: Path, *, evidence_paths: dict | None = None,
                    contract: dict | None = None, write_report: bool = True,
                    report_path: Path = Path("artifacts/infrastructure/readiness.json")) -> dict:
    """Check all prerequisites without fitting, calibration, or network writes."""
    root = root.resolve(strict=True)
    config_path = confined_path(root, config_path)
    paths = {**DEFAULT_EVIDENCE, **(evidence_paths or {})}
    if set(paths) != set(DEFAULT_EVIDENCE):
        raise ReadinessError("Unsupported readiness evidence key")
    reports, metadata, checks = _read_evidence(root, paths)
    config = None
    commit = None
    target = None

    def run_check(name, function, *, evidence=None, requires_contract=False):
        try:
            if evidence is not None:
                _assert(evidence in reports, f"Required {evidence} evidence is unavailable; see evidence_{evidence}")
            if requires_contract:
                _assert(config is not None and contract is not None, "Training contract did not validate")
            detail = function()
            checks.append({"check": name, "status": "passed", **({"detail": detail} if detail else {})})
        except Exception as error:
            checks.append({"check": name, "status": "blocked", "reason": str(error)})

    def git_check():
        nonlocal commit
        commit = _git(root, "rev-parse", "HEAD")
        _assert(not _git(root, "status", "--porcelain", "--untracked-files=all"),
                "Workspace has modified or untracked files; commit locally, push, then pull the server")
        return {"commit": commit, "worktree_clean": True}

    run_check("git_checkout", git_check)

    def contract_check():
        nonlocal contract, config, target
        from speaker_id.training.contracts import load_contract
        observed = load_contract(config_path, root)
        if contract is not None:
            _assert(observed["signature"] == contract["signature"], "In-memory training contract is stale")
        contract = observed
        config = contract["config"]
        _assert(bool(contract.get("code_hashes")), "Training contract lacks source-code fingerprints")
        _assert(config["device"] == "cuda", "Ready execution must target CUDA")
        target = _configured_target(config, root)
        data_verification_mode = _data_verification_mode(config)
        return {"signature": contract["signature"], "input_hashes": contract["input_hashes"],
                "target": {"instance_id": target["instance_id"],
                           "workspace": str(target["workspace"]),
                           "expected_gpu": target["expected_gpu"],
                           "minimum_gpu_memory_gib": target["minimum_gpu_memory_gib"],
                           "data_verification_mode": data_verification_mode}}

    run_check("training_contract", contract_check)

    def configured_target_check():
        _assert(target is not None, "Training configuration target did not validate")
        _assert(_normalise_workspace(root) == target["workspace"],
                "Current workspace differs from configuration expected_workspace")
        return {"instance_id": target["instance_id"], "workspace": str(target["workspace"]),
                "expected_gpu": target["expected_gpu"],
                "minimum_gpu_memory_gib": target["minimum_gpu_memory_gib"]}

    run_check("configured_target", configured_target_check, requires_contract=True)

    def instance_check():
        marker = reports["instance"]
        _assert(target is not None, "Training configuration target did not validate")
        _assert(marker.get("status") == "verified", "Vast/SSH instance identity has not been verified")
        _assert(int(marker.get("instance_id", -1)) == target["instance_id"], "Unexpected Vast instance ID")
        _assert(marker.get("verified_via") == "vast_api_and_ssh", "Instance needs Vast API and SSH verification")
        _assert(platform.system() == "Linux", "Only the remote Linux instance can be marked ready")
        _assert(marker.get("hostname") == socket.gethostname(), "Instance marker is from another hostname")
        marker_workspace = _normalise_workspace(marker.get("remote_workspace", ""))
        _assert(marker_workspace == target["workspace"], "Instance marker workspace differs from configuration")
        _assert(marker_workspace == _normalise_workspace(root), "Instance workspace differs")
        expected_instance_id = str(target["instance_id"])
        _assert(os.environ.get("VAST_INSTANCE_ID") == expected_instance_id,
                f"VAST_INSTANCE_ID={expected_instance_id} must be set in the execution environment")
        return {"instance_id": target["instance_id"], "hostname": socket.gethostname(),
                "workspace": str(target["workspace"])}

    run_check("remote_instance", instance_check, evidence="instance", requires_contract=True)

    def runtime_check():
        runtime = reports["runtime"]
        _assert(target is not None, "Training configuration target did not validate")
        _assert(runtime.get("status") == "passed", "Runtime preflight has not passed")
        _assert(runtime.get("training_started") is False, "Runtime evidence must not describe a training run")
        _assert(sys.version_info[:2] == (3, 12), "Python 3.12 required")
        _assert(runtime.get("git_commit") == commit and bool(commit), "Runtime preflight refers to an older commit")
        _assert(runtime.get("git_worktree_clean") is True, "Runtime preflight did not verify a clean checkout")
        _assert(Path(runtime.get("workspace", "")).resolve() == root, "Runtime probe belongs to another workspace")
        _assert(runtime.get("packages") == _package_versions(), "Installed packages changed since runtime preflight")
        cuda = runtime["cuda"]
        _assert(cuda.get("available") is True and cuda.get("arithmetic_check_passed") is True,
                "CUDA arithmetic probe has not passed")
        expected_gpu = _normalise_device_name(target["expected_gpu"])
        minimum_memory = target["minimum_gpu_memory_gib"] * 1024**3
        _assert(any(expected_gpu in _normalise_device_name(device.get("name", ""))
                    and device.get("total_memory_bytes", 0) >= minimum_memory
                    for device in cuda.get("devices", [])),
                f"{target['expected_gpu']} with at least {target['minimum_gpu_memory_gib']} GiB was not verified")
        _assert(runtime["checks"]["pip_check"]["returncode"] == 0, "pip dependency check failed")
        _assert(runtime["checks"]["audio_decode"]["status"] == "passed", "Real WAV/MP3 audio decoding was skipped/failed")
        _assert(runtime["checks"]["audio_decode"]["manifest_sha256"] == contract["input_hashes"]["manifest"],
                "Runtime audio probe used a different manifest")
        versions = runtime["checks"]["leaderboard_core_versions"]
        _assert(versions["source_sha256"] == sha256_file(root / "Competition-Guide/leaderbordpakage.txt"),
                "Leaderboard guide changed after runtime preflight")
        _assert({item["guide_distribution"] for item in versions["comparisons"]}
                == {"numpy", "scipy", "soundfile", "torch", "torchaudio", "mlflow"},
                "Runtime report omitted required core version comparisons")
        _assert(all(item["passed"] for item in versions["comparisons"]),
                "Core packages did not meet leaderboard ranges")
        minimum = runtime["disk"]["minimum_free_gb"] * 1024**3
        _assert(shutil.disk_usage(root).free >= minimum, "Free disk fell below the runtime threshold")
        return {"cuda": True, "packages_unchanged": True, "live_free_bytes": shutil.disk_usage(root).free}

    run_check("runtime", runtime_check, evidence="runtime", requires_contract=True)

    def data_check():
        data = reports["data"]
        _assert(data.get("status") == "passed", "Full transferred-data verification has not passed")
        mode = _data_verification_mode(config)
        if mode == "archive_crc":
            archive_identity = validate_archive_source(root, data)
            _assert(data.get("archive_crc_verified_members") == config["expected_source_files"] + 1,
                    "ZIP CRC was not checked for every audio file and labels.csv")
            _assert(data.get("archive_deleted") is True, "Verified incoming server ZIP has not been deleted")
        else:
            _assert(data.get("verification_scope") == "all_installed_raw_files_manifest_sha256",
                    "Installed-data report did not rehash every raw file against the manifest")
            _assert(data.get("installed_file_sha256_verified") is True,
                    "Installed-data report did not verify raw SHA256 values")
            _assert(data.get("output_bytes_verified") is True,
                    "Installed-data report did not verify raw file byte sizes")
            _assert(data.get("output_files_verified") == config["expected_source_files"] + 1,
                    "Installed-data inventory verification is incomplete")
            archive_identity = {"data_verification_mode": mode,
                                "installed_file_sha256_verified": True,
                                "output_bytes_verified": True}
        _assert(data.get("training_started") is False, "Data evidence must not describe training")
        _assert(data.get("manifest_sha256") == contract["input_hashes"]["manifest"], "Data manifest changed after verification")
        _assert(data.get("audio_files_verified") == config["expected_source_files"], "Not all audio files were verified")
        _assert(data.get("class_count") == config["evaluation_classes"], "Verified class count differs")
        _assert(data.get("output_files_verified") == config["expected_source_files"] + 1,
                "Extracted output verification is incomplete")
        _assert(data.get("labels_verified") is True, "labels.csv was not verified")
        _assert(Path(data["output"]).resolve() == root / "data/raw", "Data verification belongs to another output path")
        _assert(sha256_file(root / "data/raw/labels.csv") == data["labels_sha256"], "labels.csv changed after verification")
        expected_names = {row["audio_file"] for row in contract["manifest"]} | {"labels.csv"}
        actual_names = {path.name for path in (root / "data/raw").iterdir()}
        _assert(expected_names == actual_names, "Extracted dataset inventory changed after verification")
        for row in contract["manifest"]:
            audio = confined_path(root, root / "data/raw" / row["audio_file"])
            _assert(audio.is_file() and audio.stat().st_size == int(row["file_bytes"]),
                    f"Raw audio missing/size changed: {row['audio_file']}")
        return {"audio_files": data["audio_files_verified"], "data_verification_mode": mode,
                **archive_identity,
                "launch_policy": "scripts/train.py rehashes every audio file before execution"}

    run_check("full_data_verification", data_check, evidence="data", requires_contract=True)

    def model_check():
        model = contract["model"]
        _assert(target is not None, "Training configuration target did not validate")
        weights = confined_path(root, model["weights_path"])
        _assert(sha256_file(weights) == model["weights_sha256"], "CAM++ checkpoint changed or is missing")
        probe = reports["campp"]
        _assert(probe.get("status") == "passed_forward_only", "Real CAM++ forward probe did not pass")
        _assert(probe.get("device") == "cuda"
                and _normalise_device_name(target["expected_gpu"]) in _normalise_device_name(probe.get("gpu", "")),
                f"CAM++ probe was not on {target['expected_gpu']} CUDA")
        _assert(probe.get("training_started") is False and probe.get("optimizer_steps") == 0
                and probe.get("backward_calls") == 0, "Infrastructure probe must not fit model weights")
        _assert(probe.get("model_weight_sha256") == model["weights_sha256"], "Probe used different model weights")
        _assert(probe.get("model_config_sha256") == contract["input_hashes"]["model_config"], "Probe model configuration is stale")
        _assert(probe.get("input_hashes") == contract["input_hashes"], "Probe dataset/roles/model inputs are stale")
        _assert(probe.get("code_hashes") == contract["code_hashes"], "Probe source code changed")
        _assert(probe.get("contract_signature") == contract["signature"], "Probe experiment contract changed")
        _assert(probe.get("git_commit") == commit and bool(commit), "Probe commit changed")
        _assert(probe.get("gradient_graph_constructed") is True, "Trainable forward graph was not checked")
        _assert(probe.get("embedding_shape") == [512] and probe.get("fit_forward_logits_shape") == [2, 446],
                "CAM++ embedding/head tensor shape probe failed")
        return {"model_weight_sha256": model["weights_sha256"], "optimizer_steps": 0}

    run_check("campp_forward", model_check, evidence="campp", requires_contract=True)

    def mlflow_check():
        probe = reports["mlflow"]
        _assert(probe.get("status") == "passed" and probe.get("training_started") is False,
                "MLflow infrastructure roundtrip failed or is missing")
        _assert(probe.get("base_model") == "CAM++", "MLflow probe refers to another base model")
        _assert(str(probe.get("experiment_id", "0")) != "0" and bool(probe.get("experiment_name")),
                "MLflow needs a new explicit experiment")
        _assert(probe["final_artifact_roundtrip"]["status"] == "passed"
                and probe["final_artifact_roundtrip"]["files_verified"] >= 7,
                "MLflow artifact upload/download verification incomplete")
        _assert(probe["final_metadata_readback"]["status"] == "passed"
                and probe["final_metadata_readback"]["remote_run_status"] == "FINISHED",
                "MLflow metric/parameter/status readback failed")
        directory = confined_path(root, probe["local_run_directory"])
        _assert(directory.is_relative_to(root / "artifacts/infrastructure"), "MLflow probe spool is outside infrastructure artifacts")
        recorded = json.loads((directory / "artifacts/inputs_manifest.json").read_text(encoding="utf-8"))
        for name, digest in {**contract["input_hashes"], "weights": contract["model"]["weights_sha256"],
                             "resolved_config_input": sha256_file(config_path)}.items():
            _assert(recorded.get(name, {}).get("sha256") == digest,
                    f"MLflow preflight did not fingerprint the current {name}")
        source = json.loads((directory / "artifacts/source_manifest.json").read_text(encoding="utf-8"))
        _assert(source.get("git_commit") == commit and source.get("src_dirty") is False,
                "MLflow source snapshot is not the current clean commit")
        excluded = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
        current_source = {path.relative_to(root).as_posix(): sha256_file(path)
                          for path in (root / "src").rglob("*") if path.is_file()
                          and not any(part in excluded for part in path.relative_to(root).parts)
                          and path.suffix not in {".pyc", ".pyo"}}
        archived_source = {item["path"]: item["sha256"] for item in source["files"]}
        _assert(current_source == archived_source, "MLflow uploaded source snapshot is stale")
        _assert(source.get("archive_format") == "zip" and source.get("archive_name") == "source_snapshot.zip",
                "MLflow source snapshot must use the transport-stable ZIP format")
        _assert(sha256_file(directory / "artifacts/source_snapshot.zip") == source["archive_sha256"],
                "Local source snapshot differs from its recorded hash")
        run_state = json.loads((directory / "run_state.json").read_text(encoding="utf-8"))
        _assert(str(run_state["run_id"]) == str(probe["run_id"]), "MLflow report/run state differ")
        _assert(run_state["remote_status"] == "FINISHED" and not run_state["last_sync_error"],
                "MLflow probe has unsynchronized terminal state")
        uploaded = run_state["uploaded_artifacts"]
        _assert(set(probe["final_artifact_roundtrip"]["artifact_paths"]) == set(uploaded),
                "Not every uploaded artifact was included in final readback")
        for name, digest in uploaded.items():
            artifact = confined_path(root, directory / "artifacts" / name)
            _assert(sha256_file(artifact) == digest, f"MLflow probe artifact changed: {name}")
        binding_path = _mlflow_state_path(config, root)
        binding = json.loads(binding_path.read_text(encoding="utf-8"))["binding"]
        _assert(binding == run_state["binding"], "MLflow binding changed after probe")
        _assert(str(binding["experiment_id"]) == str(probe["experiment_id"]), "MLflow report refers to another experiment")
        return {"experiment_id": binding["experiment_id"], "run_id": probe["run_id"],
                "binding_state_path": binding_path.relative_to(root).as_posix(),
                "fresh_live_roundtrip_required_at_execution": True}

    run_check("mlflow_roundtrip", mlflow_check, evidence="mlflow", requires_contract=True)
    blocked = [check for check in checks if check["status"] != "passed"]
    target_summary = (None if target is None else {
        "instance_id": target["instance_id"], "workspace": str(target["workspace"]),
        "expected_gpu": target["expected_gpu"], "minimum_gpu_memory_gib": target["minimum_gpu_memory_gib"],
    })
    execution_command = ("Target identity did not validate; do not execute training"
                         if target is None else
                         f"VAST_INSTANCE_ID={target['instance_id']} .venv/bin/python "
                         "scripts/infra/with_project_env.py .venv/bin/python scripts/train.py "
                         f"--config {config_path.relative_to(root).as_posix()} --execute-training")
    result = {
        "schema_version": 1, "status": "blocked" if blocked else "ready",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "training_started": False,
        "user_start_instruction_required": True,
        "instance_id": target["instance_id"] if target is not None else None,
        "configured_target": target_summary,
        "workspace": str(root), "git_commit": commit,
        "config_path": str(config_path.relative_to(root).as_posix()), "config_sha256": sha256_file(config_path),
        "contract_signature": contract.get("signature") if contract else None,
        "code_hashes": contract.get("code_hashes") if contract else None,
        "input_hashes": contract.get("input_hashes") if contract else None,
        "model_weight_sha256": contract["model"]["weights_sha256"] if contract else None,
        "evidence": metadata, "checks": checks, "blocked_checks": len(blocked),
        "next_command_after_user_start": execution_command,
        "limits": ["This report performs no model training or calibration.",
                   "Execution rehashes all source audio and repeats a live MLflow artifact and metadata roundtrip.",
                   "Leaderboard package ranges are checked; exact offline submission compatibility requires the final bundle test."],
    }
    if write_report:
        destination = confined_path(root, report_path)
        _assert(destination.is_relative_to(root / "artifacts/infrastructure"), "Readiness report must stay under infrastructure artifacts")
        _assert(destination not in {confined_path(root, item) for item in paths.values()}, "Readiness output cannot overwrite evidence")
        write_json_atomic(destination, result)
    return result


def validate_readiness_for_execution(root: Path, contract: dict,
                                     report_path: Path = Path("artifacts/infrastructure/readiness.json")) -> dict:
    """Fail closed if any input, artifact, environment, or source changed."""
    root = root.resolve(strict=True)
    path = confined_path(root, report_path)
    if not path.is_file():
        raise ReadinessError("Run scripts/infra/check_readiness.py successfully before execution")
    stored = json.loads(path.read_text(encoding="utf-8"))
    _assert(stored.get("status") == "ready", "Infrastructure readiness is blocked")
    _assert(stored.get("contract_signature") == contract["signature"], "Readiness belongs to a different training contract")
    evidence = stored["evidence"]
    _assert(set(evidence) == set(DEFAULT_EVIDENCE), "Readiness report is missing required evidence")
    for item in evidence.values():
        evidence_path = confined_path(root, item["path"])
        _assert(sha256_file(evidence_path) == item["sha256"], "Readiness evidence changed; rerun aggregate checks")
    current = check_readiness(root, Path(stored["config_path"]),
                              evidence_paths={key: item["path"] for key, item in evidence.items()},
                              contract=contract, write_report=False)
    _assert(current["status"] == "ready", "Readiness recheck failed: " + "; ".join(
        check.get("reason", check["check"]) for check in current["checks"] if check["status"] != "passed"))
    return current
