"""C002's fresh, GPU-only paired frontend contract.

C002 is deliberately independent from the interrupted CPU-only C001 attempt and
from S010's historical CUDA cache.  It may use S008c only as a scored historical
control.  New identity and gain embeddings are extracted into a fresh C002
namespace and are never required to be bitwise-equal to historical embeddings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform

from speaker_id.audio.gain import GAIN_POLICY, IDENTITY_POLICY
from speaker_id.models.campp import file_sha256
from speaker_id.training.gain_suite import (
    ALPHAS,
    DECISION,
    SELECTION,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA,
    load_gain_inputs,
    require,
)
from speaker_id.training.fusion_suite import project_path


TARGET_IDENTITY = {
    "vast_instance_id": 50288952,
    "gpu_name_contains": "RTX 3090",
    "minimum_gpu_memory_gib": 24,
    "minimum_free_memory_mib": 10240,
}
EXECUTION = {
    "device": "cuda",
    "worker_count": 1,
    "tensor_dtype": "float32",
    "maximum_extraction_seconds": 28800,
    "progress_every_pairs": 50,
    "no_cpu_fallback": True,
    "no_resume": True,
}
FIXED = {
    "schema_version": 1,
    "experiment_code": "C002",
    "run_name": "C002-campp-fresh-cuda-fixed-rms-boost",
    "readiness_config": "configs/train/campp_coverage_c002.json",
    "output_root": "artifacts/training/cuda_gain_c002",
    "source_release_config": SOURCE_CONFIG,
    "source_release_config_sha256": SOURCE_CONFIG_SHA,
    "gain_policy": GAIN_POLICY,
    "identity_policy": IDENTITY_POLICY,
    "alphas": list(ALPHAS),
    "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
    "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
    "margin_weights": [0.0, 0.5],
    "threshold_candidates": 201,
    "probability_temperature": 0.05,
    "selection_policy": SELECTION,
    "decision_rule": DECISION,
    "primary_contrast": "C002d minus C002b; C002a historical control is never a selectable frontend",
    "recipes": [
        "C002a_historical_s008c_control",
        "C002b_fresh_cuda_identity",
        "C002c_fresh_cuda_gain",
        "C002d_inner_frontend_choice",
    ],
    "target_identity": TARGET_IDENTITY,
    "execution": EXECUTION,
    "mlflow_payload": "configs_source_hashes_reports_and_scalar_metrics_no_embeddings",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def validate_cuda_gain_config(suite):
    """Reject substitutions, including an old instance or output namespace."""
    require(canonical(suite) == canonical(FIXED),
            "C002 requires its exact fresh-CUDA identity, fixed policy and output namespace")


def require_c002_path(root, path, *, require_existing=False):
    """Constrain mutable evidence to C002 and reject C001/cache reuse by construction."""
    root = Path(root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.exists() else (candidate.parent.resolve() / candidate.name)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("C002 path escapes project root") from error
    parts = tuple(part.lower() for part in relative.parts)
    require("c001" not in "/".join(parts), "C002 must not read, write or resume any C001 path/cache")
    require("cpu_gain" not in "/".join(parts), "C002 must not use CPU gain cache namespaces")
    require(not require_existing or resolved.exists(), "Required C002 path is missing")
    return resolved


def load_cuda_gain_inputs(root, suite):
    """Metadata-only validation; this path does not load a model or start a run."""
    validate_cuda_gain_config(suite)
    root = Path(root).resolve()
    require_c002_path(root, root / suite["output_root"])
    source_path = project_path(root, suite["source_release_config"], "configs/package")
    require(file_sha256(source_path) == suite["source_release_config_sha256"], "Selected S008 source config changed")
    source_config = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        source_config["selection"]["family"] == "public_advanced"
        and source_config["selection"]["recipe_id"] == "S008c",
        "C002 requires the established S008c source only as historical control",
    )
    from speaker_id.packaging.selected_sources import verify_selection
    from speaker_id.training.contracts import load_contract

    selected = verify_selection(root, source_config["selection"])
    contract = load_contract(project_path(root, suite["readiness_config"], "configs/train"), root)
    require(
        contract["config"]["inference"] == {"seconds": 180.0, "maximum_windows": 1}
        and contract["config"]["mode"] == "frozen_baseline",
        "C002 preserves full-utterance frozen inputs",
    )
    return contract, source_config, selected


def capture_cuda_backend(suite):
    """Fail closed before any encoder is loaded; CUDA is the only permitted device."""
    validate_cuda_gain_config(suite)
    import torch

    target = suite["target_identity"]
    require(os.environ.get("VAST_INSTANCE_ID") == str(target["vast_instance_id"]),
            "C002 must run on its declared Vast instance")
    require(torch.cuda.is_available(), "C002 requires CUDA; CPU fallback is prohibited")
    require(torch.cuda.device_count() >= 1, "C002 requires a visible CUDA device")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    free_bytes, visible_total_bytes = (int(value) for value in torch.cuda.mem_get_info(device))
    require(target["gpu_name_contains"].lower() in name.lower(), "C002 GPU model does not match its declared target")
    require(total_bytes >= target["minimum_gpu_memory_gib"] * 1024 ** 3
            and visible_total_bytes >= target["minimum_gpu_memory_gib"] * 1024 ** 3,
            "C002 GPU capacity is below its declared target")
    require(free_bytes >= target["minimum_free_memory_mib"] * 1024 ** 2,
            "C002 free GPU memory is below the preflight floor")
    require(torch.get_default_dtype() == torch.float32, "C002 requires FP32 default tensors")
    backend = {
        "schema_version": 1,
        "device": "cuda",
        "device_index": int(device),
        "device_name": name,
        "cuda_runtime": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "tensor_dtype": "float32",
        "total_memory_bytes": total_bytes,
        "visible_total_memory_bytes": visible_total_bytes,
        "free_memory_bytes": free_bytes,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "no_cpu_fallback": True,
        "encoder_updates": 0,
    }
    validate_cuda_backend(backend, suite)
    return backend


def validate_cuda_backend(backend, suite):
    validate_cuda_gain_config(suite)
    fields = {
        "schema_version", "device", "device_index", "device_name", "cuda_runtime", "torch_version",
        "python_version", "tensor_dtype", "total_memory_bytes", "visible_total_memory_bytes",
        "free_memory_bytes", "cudnn_enabled", "no_cpu_fallback", "encoder_updates",
    }
    require(type(backend) is dict and set(backend) == fields, "Incomplete C002 CUDA backend evidence")
    require(backend["schema_version"] == 1 and backend["device"] == "cuda"
            and type(backend["device_index"]) is int and backend["device_index"] >= 0
            and backend["tensor_dtype"] == "float32" and backend["no_cpu_fallback"] is True
            and backend["encoder_updates"] == 0, "C002 must use one frozen FP32 CUDA backend")
    target = suite["target_identity"]
    require(target["gpu_name_contains"].lower() in backend["device_name"].lower()
            and backend["total_memory_bytes"] >= target["minimum_gpu_memory_gib"] * 1024 ** 3
            and backend["visible_total_memory_bytes"] >= target["minimum_gpu_memory_gib"] * 1024 ** 3
            and backend["free_memory_bytes"] >= target["minimum_free_memory_mib"] * 1024 ** 2,
            "C002 backend no longer meets the declared GPU identity")
    require(all(isinstance(backend[name], str) and backend[name]
                for name in ("device_name", "cuda_runtime", "torch_version", "python_version")),
            "Incomplete CUDA version evidence")
    require(all(type(backend[name]) is int and backend[name] > 0
                for name in ("total_memory_bytes", "visible_total_memory_bytes", "free_memory_bytes")),
            "CUDA memory evidence must be positive integer bytes")
    require(type(backend["cudnn_enabled"]) is bool, "CUDA backend flags must be booleans")


def prepare_cuda_execution(root, suite):
    """Validate the C002-only output namespace and live GPU identity before models."""
    validate_cuda_gain_config(suite)
    root = Path(root).resolve()
    output_root = require_c002_path(root, root / suite["output_root"])
    require(output_root.as_posix().endswith("artifacts/training/cuda_gain_c002"),
            "C002 output root changed")
    return {"backend": capture_cuda_backend(suite), "output_root": output_root.as_posix()}


def build_cuda_identity(root, suite, contract, sources, frontend, backend):
    """Bind a new frontend cache to source/model/runtime identity without historical parity gates."""
    validate_cuda_gain_config(suite)
    validate_cuda_backend(backend, suite)
    require(frontend in ("identity", "gain"), "Only the two preregistered C002 frontends are allowed")
    policy = suite["identity_policy"] if frontend == "identity" else suite["gain_policy"]
    body = {
        "schema_version": 1,
        "experiment_code": "C002",
        "frontend": frontend,
        "frontend_policy": policy,
        "inference": contract["config"]["inference"],
        "data_input_hashes": contract["input_hashes"],
        "labels": contract["labels"],
        "embedding_dims": {"public": 512, "advanced": 192},
        "model_sources": {name: asset["source_record"] for name, asset in sources["assets"].items()},
        "model_configs": {name: asset["config"] for name, asset in sources["assets"].items()},
        "code_hashes": {
            **contract["code_hashes"],
            "scripts/score_gain_cuda.py": file_sha256(Path(root) / "scripts/score_gain_cuda.py"),
        },
        "backend": backend,
        "historical_embedding_parity_required": False,
        "base_feature_implementation_unchanged": True,
        "encoder_updates": 0,
    }
    return {**body, "signature": hashlib.sha256(canonical(body)).hexdigest()}
