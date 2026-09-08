"""Isolated frozen dual-encoder frontend for the fixed S010 gain hypothesis.

Historical loaders/frontends remain unchanged. Both encoders consume the same
single computed FBank view; the identity branch preserves original arithmetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from speaker_id.audio.gain import apply_gain, validate_gain_policy
from speaker_id.models.campp import crop_waveform, make_fbank, read_mono


INFERENCE_POLICY = {"seconds": 180.0, "maximum_windows": 1}
EMBEDDING_DIMS = {"public": 512, "advanced": 192}


def _frozen_fp32(encoder, torch):
    if encoder.training or any(module.training for module in encoder.modules()):
        raise ValueError("Gain extraction requires every encoder module in eval mode")
    for value in encoder.parameters():
        if value.requires_grad or value.dtype != torch.float32:
            raise ValueError("Gain extraction requires frozen FP32 encoder parameters")
    for value in encoder.buffers():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError("Gain extraction requires FP32 floating encoder buffers")


def _embedding(output, dimension, torch):
    if output.shape != (1, dimension) or not torch.isfinite(output).all():
        raise ValueError("Gain encoder returned an invalid embedding shape/value")
    vector = output[0].float().cpu().numpy()
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Gain encoder returned an undefined nonzero-signal embedding")
    # Preserve the exact historical one-view norm/mean/norm FP32 arithmetic.
    aggregate = np.mean([vector / norm], axis=0)
    final_norm = np.linalg.norm(aggregate)
    if not np.isfinite(final_norm) or final_norm <= 1e-12:
        raise ValueError("Gain encoder returned an undefined normalized embedding")
    return (aggregate / final_norm).astype(np.float32)


def extract_gain_pair(public_encoder, advanced_encoder, path: Path, *, device: str,
                      policy: dict, seconds: float = 180.0,
                      maximum_windows: int = 1) -> tuple[dict[str, np.ndarray], dict]:
    """Decode once, gain before crop/pad, compute one FBank, infer two frozen models."""
    validate_gain_policy(policy)
    if (type(seconds) not in (int, float) or seconds != 180.0
            or type(maximum_windows) is not int or maximum_windows != 1):
        raise ValueError("Gain candidate requires exactly one centered 180-second view")
    signal = read_mono(Path(path))
    waveform, gain = apply_gain(signal, policy)
    info = {"nonzero_signal": bool(np.any(signal)), "seconds": len(signal) / 16000,
            "window_count": 0, "gain": gain}
    if not info["nonzero_signal"]:
        return {name: np.zeros(dim, dtype=np.float32) for name, dim in EMBEDDING_DIMS.items()}, info
    import torch
    encoders = {"public": public_encoder, "advanced": advanced_encoder}
    for encoder in encoders.values():
        _frozen_fp32(encoder, torch)
    with torch.inference_mode():
        features = make_fbank(crop_waveform(waveform, seconds, position=0.5))
        batch = features.unsqueeze(0).to(device=device, dtype=torch.float32)
        vectors = {name: _embedding(encoder(batch), EMBEDDING_DIMS[name], torch)
                   for name, encoder in encoders.items()}
    info["window_count"] = 1
    return vectors, info
