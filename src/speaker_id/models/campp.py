"""Offline CAM++ loading and one shared, deterministic audio frontend.

Feature policy follows the public ModelScope CAM++ wrapper: Kaldi 80-bin
FBank and per-utterance mean normalization. Decoding is content-based SoundFile;
torchaudio is used only for the Kaldi feature implementation, never audio.load.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EXPECTED_FRONTEND = {
    "decoder": "soundfile", "channel_policy": "maximum_mean_square_energy",
    "resampling": "scipy.signal.resample_poly", "fbank": "torchaudio.compliance.kaldi.fbank",
    "frame_length_ms": 25.0, "frame_shift_ms": 10.0, "window_type": "povey",
    "dither": 0.0, "snip_edges": True, "mean_normalization": True,
    "vad": False, "gain_normalization": False, "zero_signal_policy": "unknown",
}


def validate_model_config(config: dict):
    if (config.get("architecture") != "CAMPPlus" or config.get("embedding_dim") != 512
            or config.get("sample_rate") != 16000 or config.get("fbank_bins") != 80
            or config.get("frontend") != EXPECTED_FRONTEND):
        raise ValueError("Unsupported CAM++ architecture/frontend; the initial control policy is fixed")
    if config.get("weights_sha256") != "5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de":
        raise ValueError("The initial CAM++ base must be the pinned public VoxCeleb checkpoint")


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_mono(path: Path, sample_rate: int = 16000) -> np.ndarray:
    signal, original_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Empty or nonfinite waveform: {Path(path).name}")
    # EDA found identical stereo channels. Strongest-channel selection also avoids
    # cancellation if future test files have phase-inverted or asymmetric channels.
    energy = np.mean(signal.astype(np.float64) ** 2, axis=0)
    mono = np.ascontiguousarray(signal[:, int(np.argmax(energy))])
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, original_rate // divisor).astype(np.float32)
    return mono


def crop_waveform(signal: np.ndarray, seconds: float, *, position: float = 0.5,
                  sample_rate: int = 16000, minimum_seconds: float = 1.0) -> np.ndarray:
    if seconds <= 0 or not 0 <= position <= 1 or minimum_seconds <= 0:
        raise ValueError("Invalid crop configuration")
    length = max(1, round(seconds * sample_rate))
    start = round(max(0, len(signal) - length) * position)
    crop = signal[start:start + length]
    minimum = min(length, round(minimum_seconds * sample_rate))
    if len(crop) < minimum:
        crop = np.pad(crop, (0, minimum - len(crop)))
    return np.ascontiguousarray(crop, dtype=np.float32)


def make_fbank(signal: np.ndarray):
    import torch
    from torchaudio.compliance.kaldi import fbank
    values = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)).unsqueeze(0)
    if values.shape[1] < 400:
        raise ValueError("FBank requires at least one 25 ms frame; pad short audio first")
    features = fbank(values, num_mel_bins=80, sample_frequency=16000.0,
                     dither=0.0, frame_length=25.0, frame_shift=10.0,
                     window_type="povey", snip_edges=True, use_energy=False)
    features = features - features.mean(dim=0, keepdim=True)
    if features.ndim != 2 or features.shape[1] != 80 or not torch.isfinite(features).all():
        raise ValueError("Nonfinite or malformed Kaldi FBank features")
    return features


def load_campp(model_config: dict, root: Path, device: str = "cpu"):
    import torch
    from speaker_id.models.vendor.campplus.DTDNN import CAMPPlus
    validate_model_config(model_config)
    path = (root / model_config["weights_path"]).resolve()
    actual = file_sha256(path)
    if actual != model_config["weights_sha256"]:
        raise ValueError("Public CAM++ weight checksum does not match the model contract")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain CAM++ state_dict of tensors")
    encoder = CAMPPlus(feat_dim=80, embedding_size=512, memory_efficient=False)
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    return encoder


def extract_embedding(encoder, path: Path, *, device: str, seconds: float = 6.0,
                      maximum_windows: int = 3) -> tuple[np.ndarray, dict]:
    import torch
    if maximum_windows < 1:
        raise ValueError("At least one inference window is required")
    signal = read_mono(path)
    info = {"nonzero_signal": bool(np.any(signal)), "seconds": len(signal) / 16000,
            "window_count": 0}
    if not info["nonzero_signal"]:
        return np.zeros(512, dtype=np.float32), info
    windows = min(maximum_windows, max(1, math.ceil(len(signal) / (seconds * 16000))))
    positions = [0.5] if windows == 1 else np.linspace(0, 1, windows)
    vectors = []
    with torch.inference_mode():
        for position in positions:
            features = make_fbank(crop_waveform(signal, seconds, position=float(position)))
            output = encoder(features.unsqueeze(0).to(device))
            if output.shape != (1, 512) or not torch.isfinite(output).all():
                raise ValueError(f"Invalid CAM++ embedding for {path.name}")
            vector = output[0].float().cpu().numpy()
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                raise ValueError(f"Zero CAM++ embedding for nonzero signal: {path.name}")
            vectors.append(vector / norm)
    aggregate = np.mean(vectors, axis=0)
    norm = np.linalg.norm(aggregate)
    if norm <= 1e-12:
        raise ValueError("Window embeddings cancel to an undefined reference")
    info["window_count"] = windows
    return (aggregate / norm).astype(np.float32), info
