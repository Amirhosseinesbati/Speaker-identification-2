"""Strict offline loader for the official Chinese/English CAM++ 192d candidate.

This candidate reuses the unchanged vendored backbone and audio frontend. It
does not alter the historical 512d loader or cache contracts, download weights,
or implement model fitting. A real comparison remains conditional on the F004
decision and the separately authorized server execution gate.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np

from speaker_id.models.campp import (
    EXPECTED_FRONTEND, crop_waveform, file_sha256, make_fbank, read_mono,
)


PINNED_METADATA = {
    "schema_version": 1,
    "architecture": "CAMPPlus",
    "embedding_dim": 192,
    "sample_rate": 16000,
    "fbank_bins": 80,
    "weights_bytes": 28044640,
    "weights_sha256": "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
    "public_model": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
    "public_revision": "v1.0.0",
    "checkpoint_revision": "73001f7ab0bbf7a18739f6e0a48bc1bc74f7a271",
    "source_commit": "065629c313eaf1a01c65c640c46d77e61e9607b4",
    "license": "Apache-2.0",
}
ARCHITECTURE_KWARGS = {
    "feat_dim": 80, "embedding_size": 192, "growth_rate": 32, "bn_size": 4,
    "init_channels": 128, "config_str": "batchnorm-relu",
    # Recompute-for-backward is unnecessary for a frozen inference encoder.
    "memory_efficient": False,
}
INFERENCE_POLICY = {"seconds": 180.0, "maximum_windows": 1}
SOURCE_REFERENCES = {
    "model_card": "https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/repo?Revision=v1.0.0&FilePath=README.md",
    "checkpoint_metadata": "https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/repo/files?Revision=v1.0.0&Recursive=true",
    "registry": "https://raw.githubusercontent.com/modelscope/3D-Speaker/065629c313eaf1a01c65c640c46d77e61e9607b4/speakerlab/bin/infer_sv.py",
}


def _exact(value, expected) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(_exact(value[key], item) for key, item in expected.items())
    return value == expected


def validate_advanced_config(config: dict) -> None:
    """Refuse a changed checkpoint, frontend, architecture or inference policy."""
    fields = set(PINNED_METADATA) | {"weights_path", "architecture_kwargs", "frontend", "inference", "source_references"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("Advanced CAM++ requires the complete versioned model configuration")
    if any(type(config[key]) is not type(value) or config[key] != value for key, value in PINNED_METADATA.items()):
        raise ValueError("Advanced CAM++ checkpoint identity or architecture differs from the pinned candidate")
    if (not _exact(config["architecture_kwargs"], ARCHITECTURE_KWARGS) or not _exact(config["frontend"], EXPECTED_FRONTEND)
            or not _exact(config["inference"], INFERENCE_POLICY) or not _exact(config["source_references"], SOURCE_REFERENCES)):
        raise ValueError("Advanced CAM++ architecture/frontend/inference metadata is fixed")
    relative = config["weights_path"]
    if not isinstance(relative, str):
        raise ValueError("Advanced CAM++ weights require a relative local artifact path")
    part = PurePosixPath(relative)
    if (not relative.startswith("artifacts/models/") or part.is_absolute() or "\\" in relative or ":" in relative
            or ".." in part.parts or part.as_posix() != relative or part.name != "campplus_cn_en_common.pt"):
        raise ValueError("Advanced CAM++ weights require a confined local artifact path")


def _local_weights(config: dict, root: Path) -> Path:
    root = Path(root).resolve()
    unresolved = root / config["weights_path"]
    path = unresolved.resolve()
    if (not path.is_relative_to(root / "artifacts/models") or unresolved.is_symlink() or not path.is_file()
            or any(parent.is_symlink() for parent in unresolved.parents if parent.is_relative_to(root))):
        raise ValueError("Advanced CAM++ checkpoint must already exist as a confined regular local file")
    if path.stat().st_size != config["weights_bytes"] or file_sha256(path) != config["weights_sha256"]:
        raise ValueError("Advanced CAM++ checkpoint size or SHA256 differs from the official pinned identity")
    return path


def load_advanced(config: dict, root: Path, device: str = "cpu"):
    """Load only verified local weights; return an FP32, eval, fully frozen model."""
    validate_advanced_config(config)
    path = _local_weights(config, root)
    import torch
    from speaker_id.models.vendor.campplus.DTDNN import CAMPPlus

    state = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   or not torch.isfinite(value).all() for key, value in state.items())):
        raise ValueError("Advanced CAM++ requires a finite plain tensor state_dict")
    encoder = CAMPPlus(**ARCHITECTURE_KWARGS)
    encoder.load_state_dict(state, strict=True)
    encoder.to(device=device, dtype=torch.float32)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def extract_advanced_embedding(encoder, path: Path, *, device: str,
                               seconds: float = 180.0, maximum_windows: int = 1) -> tuple[np.ndarray, dict]:
    """One centered full-utterance view capped at 180 seconds, with zero fallback."""
    if (type(seconds) not in (float, int) or seconds != 180.0
            or type(maximum_windows) is not int or maximum_windows != 1):
        raise ValueError("Advanced CAM++ candidate requires exactly the preregistered 180s single view")
    path = Path(path)
    signal = read_mono(path)
    info = {"nonzero_signal": bool(np.any(signal)), "seconds": len(signal) / 16000, "window_count": 0}
    if not info["nonzero_signal"]:
        return np.zeros(192, dtype=np.float32), info

    import torch
    if encoder.training:
        raise ValueError("Advanced CAM++ extraction requires the frozen eval encoder")
    with torch.inference_mode():
        features = make_fbank(crop_waveform(signal, seconds, position=0.5))
        output = encoder(features.unsqueeze(0).to(device=device, dtype=torch.float32))
        if output.shape != (1, 192) or not torch.isfinite(output).all():
            raise ValueError(f"Invalid advanced CAM++ embedding for {path.name}")
        vector = output[0].float().cpu().numpy()
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"Zero or nonfinite advanced CAM++ embedding for nonzero signal: {path.name}")
    # Match the original extract_embedding single-view arithmetic exactly:
    # normalize in FP32, aggregate the view, then normalize again in FP32.
    aggregate = np.mean([vector / norm], axis=0)
    final_norm = np.linalg.norm(aggregate)
    if not np.isfinite(final_norm) or final_norm <= 1e-12:
        raise ValueError("Advanced CAM++ view has an undefined normalized reference")
    info["window_count"] = 1
    return (aggregate / final_norm).astype(np.float32), info
