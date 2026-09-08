"""One fixed, per-waveform gain hypothesis; no dataset statistics or clipping.

The caller supplies the original strongest-channel, resampled float32 waveform.
Measurement precedes cropping/padding. The identity policy preserves its bytes.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


IDENTITY_POLICY = {"schema_version": 1, "name": "identity_v1"}
GAIN_POLICY = {
    "schema_version": 1,
    "name": "boost_rms_v1",
    "target_rms_dbfs": -20.0,
    "maximum_gain_db": 60.0,
    "peak_ceiling": 0.95,
    "position": "after_channel_selection_and_resampling_before_crop_or_pad",
    "statistics_dtype": "float64",
    "multiplication_dtype": "float32",
    "gain_rounding": "float32_toward_zero",
    "attenuate": False,
    "clip": False,
    "zero_signal": "unchanged",
}
_PINNED = (deepcopy(IDENTITY_POLICY), deepcopy(GAIN_POLICY))


def _exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(_exact(value[key], item) for key, item in expected.items())
    return value == expected


def validate_gain_policy(policy: dict) -> None:
    if not any(_exact(policy, expected) for expected in _PINNED):
        raise ValueError("Waveform gain policy must be the exact identity or preregistered fixed boost policy")


def _statistics(signal):
    values = signal.astype(np.float64)
    rms = float(np.sqrt(np.mean(values * values)))
    peak = float(np.max(np.abs(values)))
    return {"rms": rms, "rms_dbfs": None if rms == 0 else 20.0 * math.log10(rms), "peak": peak}


def apply_gain(signal: np.ndarray, policy: dict) -> tuple[np.ndarray, dict]:
    """Return input unchanged or one FP32 multiplication; never mutate the input."""
    validate_gain_policy(policy)
    if (not isinstance(signal, np.ndarray) or signal.dtype != np.float32
            or signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all()):
        raise ValueError("Gain expects a nonempty finite one-dimensional float32 waveform")
    before = _statistics(signal)
    gain = np.float32(1.0)
    limit = 1.0
    reason = "identity"
    if before["peak"] == 0:
        reason = "zero_signal"
    elif policy["name"] != "identity_v1":
        target = 10.0 ** (policy["target_rms_dbfs"] / 20.0)
        if before["rms"] >= target:
            reason = "rms_at_or_above_target"
        elif before["peak"] >= policy["peak_ceiling"]:
            reason = "peak_at_or_above_ceiling"
        else:
            bounds = {"target_rms": target / before["rms"],
                      "maximum_gain": 10.0 ** (policy["maximum_gain_db"] / 20.0),
                      "peak_ceiling": policy["peak_ceiling"] / before["peak"]}
            reason = min(bounds, key=bounds.get)
            limit = bounds[reason]
            gain = np.float32(limit)
            if float(gain) > limit:
                gain = np.nextafter(gain, np.float32(0.0))
            if gain < np.float32(1.0):
                raise ValueError("Fixed boost must never attenuate")
    result = signal if gain == np.float32(1.0) else np.multiply(signal, gain, dtype=np.float32)
    after = before.copy() if result is signal else _statistics(result)
    if not np.isfinite(result).all():
        raise ValueError("Gain produced a nonfinite waveform")
    if result is not signal and (after["peak"] > policy["peak_ceiling"]
            or float(gain) > 10.0 ** (policy["maximum_gain_db"] / 20.0)):
        raise ValueError("FP32 gain violated the fixed peak/gain bound")
    info = {"policy": policy["name"], "applied": result is not signal,
            "gain": float(gain), "gain_db": 20.0 * math.log10(float(gain)),
            "gain_limit_float64": limit, "limiting_reason": reason,
            "sample_count": len(signal), "before": before, "after": after}
    return result, info
