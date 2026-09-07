"""Bounded waveform follow-up of embedding similarity candidates.

Embedding similarity selects pairs only. Recording overlap requires aligned
waveform evidence; speaker identities and labels are never modified here.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import signal
import soundfile as sf

from speaker_id.audio.io import file_sha256
from speaker_id.eda.duplicates import FingerprintConfig, load_waveform, verify_alignment


def healthy(row: dict) -> bool:
    return row["status"] == "ok" and float(row["duration_seconds"]) >= 8 and float(row["max_channel_rms_dbfs"]) >= -50 and float(row["mono_zero_fraction"]) < .99


def same_exact_group(a: dict, b: dict) -> bool:
    return any(a.get(key) and a.get(key) == b.get(key) for key in ("input_sha256", "pcm_sha256"))


def probe_positions(length: int, rate: int) -> list[int]:
    """Up to three nonoverlapping 4-second windows at start/middle/end."""
    width = 4 * rate
    if length < width:
        return []
    proposed = np.unique(np.linspace(0, length - width, 3, dtype=int))
    selected = []
    for position in proposed:
        if not selected or position - selected[-1] >= width:
            selected.append(int(position))
    return selected


def global_correlation(reference: np.ndarray, target: np.ndarray, min_rms: float) -> tuple[float, int]:
    """Normalized global correlation with an explicit target energy guard.

    Zero-energy target spans must not become perfect matches when FFT roundoff
    is divided by a tiny denominator. Recompute the selected Pearson directly.
    """
    reference = reference.astype(np.float64)
    target = target.astype(np.float64)
    reference -= reference.mean()
    length = len(reference)
    norm = np.linalg.norm(reference)
    if length == 0 or len(target) < length or norm < min_rms * np.sqrt(length):
        return 0.0, 0
    dots = signal.correlate(target, reference, mode="valid", method="fft")
    sums = np.r_[0.0, np.cumsum(target)]
    squares = np.r_[0.0, np.cumsum(target * target)]
    window_sums = sums[length:] - sums[:-length]
    energy = np.maximum(0, squares[length:] - squares[:-length] - window_sums * window_sums / length)
    valid = energy >= min_rms ** 2 * length
    scores = np.zeros(len(energy), dtype=np.float64)
    scores[valid] = np.abs(dots[valid]) / (norm * np.sqrt(energy[valid]))
    if not np.any(valid):
        return 0.0, 0
    best = int(np.argmax(scores))
    matched = target[best:best + length].copy()
    matched -= matched.mean()
    target_norm = np.linalg.norm(matched)
    if target_norm < min_rms * np.sqrt(length):
        return 0.0, 0
    return float(min(1.0, abs(np.dot(reference, matched)) / (norm * target_norm))), best


def verify_pair(a: np.ndarray, b: np.ndarray, config: FingerprintConfig) -> dict:
    """Find global probe matches, then verify aligned local windows independently."""
    swapped = len(a) > len(b)
    reference, target = (b, a) if swapped else (a, b)
    width = 4 * config.sample_rate
    probes = []
    for start in probe_positions(len(reference), config.sample_rate):
        probe = reference[start:start + width]
        if float(np.std(probe, dtype=np.float64)) < config.min_rms:
            probes.append({"start_seconds": start / config.sample_rate, "status": "low_energy_skipped"})
            continue
        correlation, lag = global_correlation(probe, target, config.min_rms)
        probes.append({"start_seconds": start / config.sample_rate, "status": "checked", "absolute_correlation": correlation,
                       "offset_seconds_target_minus_reference": (lag - start) / config.sample_rate})
    checked = [p for p in probes if p["status"] == "checked"]
    high = [p for p in checked if p["absolute_correlation"] >= config.correlation_threshold]
    attempted = []
    for anchor in high:
        offset = anchor["offset_seconds_target_minus_reference"]
        cluster = [p for p in high if abs(p["offset_seconds_target_minus_reference"] - offset) * config.sample_rate <= 8.000001]
        region_start = min(p["start_seconds"] for p in cluster)
        region_end = max(p["start_seconds"] + 4 for p in cluster)
        # Existing helper adds n_fft to the supplied endpoint; pass its inverse
        # so the verification region is exactly the globally supported interval.
        verification = verify_alignment(reference, target, offset, config, region_start, region_end - config.n_fft / config.sample_rate)
        verification["consistent_global_probe_count"] = len(cluster)
        attempted.append(verification)
        if verification["verified"]:
            break
    verified = next((v for v in attempted if v["verified"]), None)
    return {"verified": verified is not None, "reference_file_is_b": swapped, "global_probes": probes,
            "checked_global_probes": len(checked), "high_correlation_global_probes": len(high),
            "maximum_global_absolute_correlation": max((p["absolute_correlation"] for p in checked), default=0),
            "verification_attempts": attempted,
            "verified_offset_seconds_b_minus_a": ((-1 if swapped else 1) * verified["verified_offset_seconds"]) if verified else None,
            "verified_overlap_seconds": verified["overlap_seconds"] if verified else None,
            "verification_reason": "multiple_consistent_waveform_windows" if verified else "no_verified_overlap_in_probed_regions"}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def select_pairs(manifest: list[dict], candidates: list[dict], neighbors: list[dict]) -> tuple[list[dict], dict]:
    lookup = {row["audio_file"]: row for row in manifest}
    selected, seen = [], set()
    ordered = sorted(candidates, key=lambda r: (-float(r["cosine"]), r["audio_file_a"], r["audio_file_b"]))
    def eligible(a: str, b: str) -> bool:
        return a in lookup and b in lookup and a != b and lookup[a]["speaker_id"] != lookup[b]["speaker_id"] and healthy(lookup[a]) and healthy(lookup[b]) and not same_exact_group(lookup[a], lookup[b])
    healthy_candidates = [r for r in ordered if eligible(r["audio_file_a"], r["audio_file_b"])]
    for row in healthy_candidates:
        key = tuple(sorted((row["audio_file_a"], row["audio_file_b"])))
        if key in seen:
            continue
        selected.append({**row, "selection_reason": "highest_cosine_quality_eligible_cross_label"})
        seen.add(key)
        if len(selected) >= 80:
            break
    negative_count = 0
    for row in sorted((r for r in neighbors if r["speaker_id"] != "unknown" and r.get("same_vs_other_margin") and float(r["same_vs_other_margin"]) < 0), key=lambda r: (float(r["same_vs_other_margin"]), r["audio_file"])):
        a, b = row["audio_file"], row["nearest_other_known_file"]
        key = tuple(sorted((a, b)))
        if key in seen or not eligible(a, b):
            continue
        selected.append({"audio_file_a": a, "speaker_a": row["speaker_id"], "audio_file_b": b, "speaker_b": lookup[b]["speaker_id"],
                         "cosine": row["nearest_other_known_cosine"], "candidate_type": "negative_margin_known", "selection_reason": "negative_margin_known_additional",
                         "query_same_vs_other_margin": float(row["same_vs_other_margin"])})
        seen.add(key)
        negative_count += 1
        if negative_count >= 40:
            break
    distribution = {}
    for name, subset in [("all_candidates", ordered), ("top20", ordered[:20]), ("top100", ordered[:100])]:
        distribution[name] = {"pairs": len(subset), "both_quality_eligible": sum(healthy(lookup[r["audio_file_a"]]) and healthy(lookup[r["audio_file_b"]]) for r in subset),
                              "either_rms_below_minus50": sum(min(float(lookup[r["audio_file_a"]]["max_channel_rms_dbfs"]), float(lookup[r["audio_file_b"]]["max_channel_rms_dbfs"])) < -50 for r in subset),
                              "either_zero_fraction_at_least_099": sum(max(float(lookup[r["audio_file_a"]]["mono_zero_fraction"]), float(lookup[r["audio_file_b"]]["mono_zero_fraction"])) >= .99 for r in subset),
                              "either_duration_below8s": sum(min(float(lookup[r["audio_file_a"]]["duration_seconds"]), float(lookup[r["audio_file_b"]]["duration_seconds"])) < 8 for r in subset)}
    return selected, {"candidate_rows": len(candidates), "eligible_cross_label_candidate_rows": len(healthy_candidates),
                      "eligible_cross_label_cosine_at_least_09": sum(float(r["cosine"]) >= .9 for r in healthy_candidates),
                      "eligible_cross_label_cosine_at_least_095": sum(float(r["cosine"]) >= .95 for r in healthy_candidates),
                      "eligible_cross_label_cosine_max": max((float(r["cosine"]) for r in healthy_candidates), default=None),
                      "selected_highest_cosine_pairs": len(selected) - negative_count, "selected_additional_negative_margin_pairs": negative_count,
                      "selected_pairs": len(selected), "extreme_candidate_quality": distribution}


def run(manifest_path: Path, candidates_path: Path, neighbors_path: Path, data_dir: Path, output: Path, summary_path: Path) -> dict:
    started = time.monotonic()
    manifest, candidates, neighbors = _read(manifest_path), _read(candidates_path), _read(neighbors_path)
    selected, selection = select_pairs(manifest, candidates, neighbors)
    lookup = {r["audio_file"]: r for r in manifest}
    config = FingerprintConfig()
    @lru_cache(maxsize=24)
    def waveform(name: str) -> np.ndarray:
        path = data_dir / name
        if file_sha256(path) != lookup[name]["input_sha256"]:
            raise ValueError("Raw file SHA256 differs from manifest")
        return load_waveform(path, config)
    results = []
    for row in selected:
        result = {**row, "status": "error", "error": ""}
        try:
            result.update(verify_pair(waveform(row["audio_file_a"]), waveform(row["audio_file_b"]), config), status="ok")
            for suffix in ("a", "b"):
                result[f"input_sha256_{suffix}"] = lookup[row[f"audio_file_{suffix}"]]["input_sha256"]
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        if result.get("verified"):
            print(f"VERIFIED WAVEFORM OVERLAP: {row['audio_file_a']} / {row['audio_file_b']}", flush=True)
        if len(results) % 20 == 0:
            print(f"Embedding waveform follow-up: {len(results)}/{len(selected)}; {time.monotonic() - started:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["audio_file_a", "speaker_a", "audio_file_b", "speaker_b", "status", "error", "verified"]
    columns += sorted(set().union(*(r.keys() for r in results)) - set(columns))
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in results)
    good = [r for r in results if r["status"] == "ok"]
    verified = [r for r in good if r["verified"]]
    summary = {"audit_kind": "embedding_candidates_waveform_followup", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "selection": selection, "successful_pairs": len(good), "failed_pairs": len(results) - len(good), "verified_pairs": len(verified),
               "maximum_global_absolute_correlation": max((r["maximum_global_absolute_correlation"] for r in good), default=None),
               "pairs_with_global_probe_correlation_at_least_0985": sum(r["high_correlation_global_probes"] > 0 for r in good),
               "verified_pair_details": verified, "failures": [r for r in results if r["status"] != "ok"],
               "config": asdict(config), "elapsed_seconds": time.monotonic() - started,
               "input_sha256": {p.name: file_sha256(p) for p in (manifest_path, candidates_path, neighbors_path)}, "output_csv_sha256": file_sha256(output),
               "code_sha256": {Path(__file__).name: file_sha256(Path(__file__)), "duplicates.py": file_sha256(Path(__file__).with_name("duplicates.py")),
                               "io.py": file_sha256(Path(__file__).parents[1] / "audio/io.py")},
               "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
               "definitions": {"quality": "Both files decode OK, duration>=8s, strongest RMS>=-50dBFS, mono zero fraction<0.99; distinct supplied labels and nonidentical exact file/PCM hashes.",
                               "selection": "Up to80 unique highest-cosine eligible candidate pairs, then up to40 additional unique eligible negative-margin known query/nearest-other-known pairs ordered by smallest margin.",
                               "waveform": "Full-file strongest channel, polyphase resampling to4kHz via existing duplicate helpers; at most24 cached waves; one worker.",
                               "global_search": "Shorter file supplies up to3 nonoverlapping4s probes at start/middle/end. Each probe searches all valid positions of the complete other file with absolute normalized Pearson cross-correlation, allowing gain and polarity inversion. Both probe and target-window standard deviation must exceed1e-5; winning correlation is recomputed directly to guard FFT/variance roundoff in silence.",
                               "verification": "For >=0.985 global matches, cluster offsets within8samples and recheck the supported interval with existing verify_alignment: >=4s region, >=2 aligned local windows each abs Pearson>=0.985, offsets within8samples."},
               "limitations": ["Fixed probes can miss shared excerpts between probe locations, shorter than4s, or with low probe energy; no verified edge is not proof of no overlap.",
                               "Bandwidth is limited to2kHz by the4kHz resampling. Tempo/pitch shifts, mixing, nonlinear processing and substantial noise can defeat this method.",
                               "Highly periodic or stereotyped signals can correlate without proving common recording provenance; waveform verification is an operational signal-overlap criterion.",
                               "Embedding cosine is similarity only: neither speaker identity, label error nor recording-session leakage is established by cosine.",
                               "This is selected-candidate follow-up, not exhaustive all-pairs waveform comparison. No labels, split groups, folds or eligibility were changed."]}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"successful_pairs": len(good), "failed_pairs": len(results) - len(good), "verified_pairs": len(verified), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return summary
