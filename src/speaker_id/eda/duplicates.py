"""Bounded acoustic duplication audit; does not infer speaker/session identity.

Spectral landmarks propose alignments. Only exact decoded/file hashes or multiple
aligned high-correlation waveform windows establish a duplicate edge. Silence is
reported separately and never joins otherwise unrelated recordings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage, signal
import soundfile as sf


@dataclass(frozen=True)
class FingerprintConfig:
    sample_rate: int = 4000
    n_fft: int = 512
    hop: int = 128
    peaks_per_second: int = 8
    fanout: int = 5
    max_pair_seconds: float = 2.0
    min_pair_seconds: float = 0.128
    min_rms: float = 1e-5
    max_hash_files: int = 20
    max_hash_occurrences: int = 80
    offset_bin_frames: int = 4
    min_votes: int = 10
    min_vote_span_seconds: float = 2.0
    max_candidates: int = 1000
    correlation_threshold: float = 0.985
    min_overlap_seconds: float = 4.0


def strongest_channel(audio: np.ndarray) -> np.ndarray:
    """Select energetic channel, preserving anti-phase stereo without averaging."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    energy = np.einsum("ij,ij->j", audio, audio, dtype=np.float64)
    return audio[:, int(np.argmax(energy))]


def prepare_waveform(audio: np.ndarray, sample_rate: int, config: FingerprintConfig) -> np.ndarray:
    mono = strongest_channel(audio)
    if sample_rate != config.sample_rate:
        divisor = gcd(sample_rate, config.sample_rate)
        mono = signal.resample_poly(mono, config.sample_rate // divisor, sample_rate // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_waveform(path: Path, config: FingerprintConfig) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return prepare_waveform(samples, sample_rate, config)


def spectral_landmarks(waveform: np.ndarray, config: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return uint32 hashes and anchor frame positions for gain-resilient peaks."""
    empty = np.empty(0, dtype=np.uint32)
    if len(waveform) < config.sample_rate or np.max(np.abs(waveform), initial=0) < config.min_rms:
        return empty, empty.copy()
    _, _, transform = signal.stft(
        waveform, fs=config.sample_rate, window="hann", nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop, boundary=None, padded=False,
    )
    magnitude = np.abs(transform)
    # Remove DC/very low frequencies. Frame threshold makes silence uninformative.
    magnitude[:8] = 0
    maxima = ndimage.maximum_filter(magnitude, size=(9, 7), mode="constant")
    floor = np.maximum(np.median(magnitude, axis=0) * 6.0, config.min_rms)
    frequencies, times = np.nonzero((magnitude == maxima) & (magnitude > floor[None, :]))
    if len(times) < 2:
        return empty, empty.copy()
    strength = magnitude[frequencies, times]
    seconds = (times * config.hop) // config.sample_rate
    order = np.lexsort((-strength, seconds))
    selected = []
    counts: Counter = Counter()
    for index in order:
        second = int(seconds[index])
        if counts[second] < config.peaks_per_second:
            selected.append(index)
            counts[second] += 1
    selected = sorted(selected, key=lambda index: (times[index], frequencies[index]))
    peak_times = times[selected]
    peak_freq = frequencies[selected]
    min_delta = int(round(config.min_pair_seconds * config.sample_rate / config.hop))
    max_delta = int(round(config.max_pair_seconds * config.sample_rate / config.hop))
    hashes, anchors = [], []
    for index in range(len(selected)):
        count = 0
        for target in range(index + 1, len(selected)):
            delta = int(peak_times[target] - peak_times[index])
            if delta > max_delta:
                break
            if delta < min_delta:
                continue
            # Coarse temporal quantization tolerates crop offsets between STFT frames.
            delta_bin = (delta + 1) // 2
            value = (int(peak_freq[index]) << 15) | (int(peak_freq[target]) << 6) | delta_bin
            hashes.append(value)
            anchors.append(peak_times[index])
            count += 1
            if count >= config.fanout:
                break
    return np.asarray(hashes, dtype=np.uint32), np.asarray(anchors, dtype=np.uint32)


def find_candidates(
    hashes: np.ndarray, files: np.ndarray, anchors: np.ndarray, config: FingerprintConfig,
) -> tuple[list[dict], dict]:
    """Inverted hash index with stop-hash/candidate caps; never all-pairs audio."""
    if len(hashes) == 0:
        return [], {"landmarks": 0, "eligible_candidates": 0, "retained_candidates": 0}
    order = np.argsort(hashes, kind="stable")
    ordered_hashes = hashes[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_hashes)) + 1, len(order)]
    # First count distinct hash support in a compact 82 MB matrix for 4,529
    # files. Only supported pairs enter the more expensive offset dictionary.
    # This prevents millions of incidental single-hash collisions consuming RAM.
    file_count = int(np.max(files)) + 1
    if file_count > 10000:
        raise ValueError("This bounded audit supports at most 10,000 indexed files")
    pair_support = np.zeros((file_count, file_count), dtype=np.uint32)
    eligible_ranges = []
    skipped_frequent, used_hashes, skipped_same_file = 0, 0, 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        length = int(right - left)
        if length < 2:
            continue
        if length > config.max_hash_occurrences:
            skipped_frequent += 1
            continue
        indices = order[left:right]
        hash_files = files[indices]
        unique_files = np.unique(hash_files)
        if len(unique_files) > config.max_hash_files:
            skipped_frequent += 1
            continue
        if np.all(hash_files == hash_files[0]):
            skipped_same_file += 1
            continue
        used_hashes += 1
        pair_first, pair_second = np.triu_indices(len(unique_files), k=1)
        pair_support[unique_files[pair_first], unique_files[pair_second]] += 1
        eligible_ranges.append((left, right))
    supported_pairs = pair_support >= config.min_votes
    supported_pair_count = int(np.count_nonzero(supported_pairs))
    del pair_support
    # pair/offset -> [votes, earliest reference anchor, latest reference anchor]
    votes: dict[tuple[int, int, int], list[int]] = {}
    for left, right in eligible_ranges:
        indices = order[left:right]
        length = len(indices)
        hash_files = files[indices]
        hash_anchors = anchors[indices].astype(np.int64)
        seen = set()
        for first in range(length):
            for second in range(first + 1, length):
                a, b = int(hash_files[first]), int(hash_files[second])
                if a == b:
                    continue
                ta, tb = int(hash_anchors[first]), int(hash_anchors[second])
                if a > b:
                    a, b, ta, tb = b, a, tb, ta
                if not supported_pairs[a, b]:
                    continue
                offset = int(np.floor((tb - ta) / config.offset_bin_frames + 0.5))
                key = (a, b, offset)
                # Repeated tone/peak occurrences cannot inflate one hash's vote.
                if key in seen:
                    continue
                seen.add(key)
                if key not in votes:
                    votes[key] = [1, ta, ta]
                else:
                    record = votes[key]
                    record[0] += 1
                    record[1] = min(record[1], ta)
                    record[2] = max(record[2], ta)
    best_by_pair: dict[tuple[int, int], dict] = {}
    for (first, second, offset), (count, earliest, latest) in votes.items():
        span = (latest - earliest) * config.hop / config.sample_rate
        if count < config.min_votes or span < config.min_vote_span_seconds:
            continue
        candidate = {
            "file_index_a": first, "file_index_b": second, "landmark_votes": count,
            "vote_span_seconds": span,
            "vote_start_seconds_a": earliest * config.hop / config.sample_rate,
            "vote_end_seconds_a": latest * config.hop / config.sample_rate,
            "candidate_offset_seconds": offset * config.offset_bin_frames * config.hop / config.sample_rate,
        }
        pair = (first, second)
        if pair not in best_by_pair or count > best_by_pair[pair]["landmark_votes"]:
            best_by_pair[pair] = candidate
    candidates = sorted(best_by_pair.values(), key=lambda item: (-item["landmark_votes"], item["file_index_a"], item["file_index_b"]))
    return candidates[:config.max_candidates], {
        "landmarks": int(len(hashes)), "distinct_hashes": len(boundaries) - 1,
        "hashes_used": used_hashes, "frequent_hashes_skipped": skipped_frequent,
        "single_file_hashes_skipped": skipped_same_file,
        "pairs_with_minimum_hash_support": supported_pair_count,
        "evaluated_pair_offset_bins": len(votes),
        "eligible_candidates": len(candidates), "retained_candidates": min(len(candidates), config.max_candidates),
        "candidate_cap_reached": len(candidates) > config.max_candidates,
    }


def _window_correlation(reference: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Find best Pearson correlation over all valid target lags (sign invariant)."""
    reference = reference.astype(np.float64)
    target = target.astype(np.float64)
    reference -= reference.mean()
    norm = np.linalg.norm(reference)
    if norm < 1e-8 or len(target) < len(reference):
        return 0.0, 0
    dots = signal.correlate(target, reference, mode="valid", method="fft")
    cumulative = np.r_[0.0, np.cumsum(target)]
    squares = np.r_[0.0, np.cumsum(target * target)]
    length = len(reference)
    sums = cumulative[length:] - cumulative[:-length]
    energy = np.maximum(0, squares[length:] - squares[:-length] - sums * sums / length)
    correlations = np.abs(dots) / np.maximum(norm * np.sqrt(energy), 1e-12)
    best = int(np.argmax(correlations))
    return float(min(1.0, correlations[best])), best


def verify_alignment(
    a: np.ndarray, b: np.ndarray, offset_seconds: float, config: FingerprintConfig,
    region_start_seconds: float | None = None, region_end_seconds: float | None = None,
) -> dict:
    """Confirm multiple aligned bandlimited waveform windows, allowing gain/sign."""
    rate = config.sample_rate
    expected = int(round(offset_seconds * rate))
    start = max(0, -expected)
    end = min(len(a), len(b) - expected)
    # Restrict verification to landmark-supported audio. Shared excerpts may sit
    # inside different recordings with unrelated introductions and endings.
    if region_start_seconds is not None:
        start = max(start, int(region_start_seconds * rate))
    if region_end_seconds is not None:
        end = min(end, int(region_end_seconds * rate) + config.n_fft)
    overlap = max(0, end - start)
    result = {"verified": False, "overlap_seconds": overlap / rate, "verification_region_start_seconds_a": start / rate, "verification_region_end_seconds_a": end / rate, "verification_correlations": [], "verification_offsets_seconds": []}
    if overlap < config.min_overlap_seconds * rate:
        result["verification_reason"] = "overlap_too_short"
        return result
    window = min(4 * rate, overlap // 2)
    margin = int(0.20 * rate)
    positions = np.unique(np.linspace(start, end - window, num=3, dtype=int))
    offsets, correlations = [], []
    for position in positions:
        reference = a[position:position + window]
        if np.std(reference, dtype=np.float64) < config.min_rms:
            continue
        target_start = max(0, position + expected - margin)
        target_end = min(len(b), position + expected + window + margin)
        correlation, lag = _window_correlation(reference, b[target_start:target_end])
        correlations.append(correlation)
        offsets.append(target_start + lag - int(position))
    passed = [index for index, value in enumerate(correlations) if value >= config.correlation_threshold]
    consistent = len(passed) >= 2 and np.ptp([offsets[index] for index in passed]) <= 8
    result.update({
        "verified": bool(consistent), "verification_correlations": correlations,
        "verification_offsets_seconds": [value / rate for value in offsets],
        "verification_reason": "multiple_consistent_windows" if consistent else "waveform_similarity_insufficient",
        "verified_offset_seconds": float(np.median([offsets[index] for index in passed]) / rate) if consistent else None,
    })
    return result


def exact_groups(rows: Iterable[dict]) -> list[dict]:
    """Hash groups include no-signal conflicts but mark them unsuitable for splits."""
    rows = list(rows)
    output = []
    for field in ("file_sha256", "decoded_sha256"):
        grouped = defaultdict(list)
        for row in rows:
            digest = row.get(field)
            if digest:
                grouped[digest].append(row)
        for digest, members in sorted(grouped.items()):
            if len(members) < 2:
                continue
            no_signal = all(bool(row.get("_no_signal", False)) for row in members)
            output.append({
                "group_id": f"{field}_{digest[:16]}", "kind": field,
                "hash": digest, "no_signal": no_signal, "use_for_split_grouping": not no_signal,
                "label_conflict": len({row["speaker_id"] for row in members}) > 1,
                "audio_files": [row["audio_file"] for row in members],
                "speaker_ids": sorted({row["speaker_id"] for row in members}),
            })
    return output


def acoustic_components(names: list[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Connected components only for already verified ordinary signal edges."""
    parents = {name: name for name in names}
    def find(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name
    for first, second in edges:
        a, b = find(first), find(second)
        parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    return [sorted(members) for _, members in sorted(groups.items()) if len(members) > 1]


def cache_signature(rows: list[dict], config: FingerprintConfig) -> str:
    payload = {"algorithm_version": 1, "config": asdict(config), "files": [(row["audio_file"], row.get("file_sha256", ""), row.get("decoded_sha256", ""), row.get("_no_signal", False), row.get("status", "ok")) for row in rows]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
