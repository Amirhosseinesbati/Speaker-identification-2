"""Deterministic nested real-waveform crops, without feature extraction or padding.

The caller supplies the sampler Generator and explicit durations. Saved crop
metadata or per-step counter seeds should pair experiment arms; degenerate
``integers(0, 1)`` calls need not advance every NumPy bit generator equally.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _requested_samples(seconds: float, sample_rate: int, name: str) -> tuple[float, int]:
    if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, Real):
        raise ValueError(f"{name} must be a positive finite duration")
    try:
        duration = float(seconds)
        scaled = duration * sample_rate
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} cannot be represented as a sample count") from error
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(scaled):
        raise ValueError(f"{name} must be a positive finite duration")
    samples = round(scaled)
    if not 1 <= samples <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must round to a positive int64 sample count")
    return duration, samples


def paired_waveform_views(
    signal: np.ndarray,
    *,
    rng: np.random.Generator,
    short_seconds: float,
    long_seconds: float,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return independent ``(student, teacher, metadata)`` FP32 waveform copies.

    Sample the teacher start uniformly among complete real crops up to the long
    duration, then sample the student start uniformly within that teacher crop.
    Half-open offsets refer to the supplied waveform, after any caller-owned
    decoding/resampling. Durations use Python round-to-nearest/ties-to-even.

    No real samples are discarded because a recording is short: each nonempty
    input produces both views. ``consistency_mask`` only records strictly more
    real teacher samples; it is not a speech-quality, signal-validity or role
    check. The caller owns those checks, loss eligibility and any minimum padding
    needed downstream. Real lengths must not be replaced by padded tensor sizes.

    All arguments are validated before either of the two Generator integer calls.
    No global RNG, labels, paths, model, normalization or dataset state is used.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    short_duration, short_requested = _requested_samples(short_seconds, sample_rate, "short_seconds")
    long_duration, long_requested = _requested_samples(long_seconds, sample_rate, "long_seconds")
    if short_duration >= long_duration or short_requested >= long_requested:
        raise ValueError("The requested long view must exceed the short view in real sample units")
    if (not isinstance(signal, np.ndarray) or np.ma.isMaskedArray(signal)
            or signal.ndim != 1 or signal.dtype != np.dtype("float32")
            or signal.size == 0 or not np.isfinite(signal).all()):
        raise ValueError("signal must be a nonempty finite one-dimensional float32 waveform")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")

    input_samples = len(signal)
    teacher_samples = min(input_samples, long_requested)
    student_samples = min(teacher_samples, short_requested)
    teacher_start = int(rng.integers(0, input_samples - teacher_samples + 1))
    student_offset = int(rng.integers(0, teacher_samples - student_samples + 1))
    student_start = teacher_start + student_offset
    teacher_stop = teacher_start + teacher_samples
    student_stop = student_start + student_samples

    teacher = signal[teacher_start:teacher_stop].copy(order="C")
    student = signal[student_start:student_stop].copy(order="C")
    metadata = {
        "schema_version": 1,
        "algorithm": "nested_real_waveform_crops_v1",
        "sample_rate": sample_rate,
        "input_samples": input_samples,
        "requested_short_seconds": short_duration,
        "requested_long_seconds": long_duration,
        "requested_short_samples": short_requested,
        "requested_long_samples": long_requested,
        "sample_rounding": "python_round_nearest_ties_even",
        "rng_bit_generator": type(rng.bit_generator).__name__,
        "student": {"start_sample": student_start, "stop_sample": student_stop,
                    "real_samples": student_samples, "offset_in_teacher": student_offset,
                    "padding_samples": 0},
        "teacher": {"start_sample": teacher_start, "stop_sample": teacher_stop,
                    "real_samples": teacher_samples, "padding_samples": 0},
        "consistency_mask": teacher_samples > student_samples,
    }
    return student, teacher, metadata
