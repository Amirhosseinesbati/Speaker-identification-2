"""Resumable full-content audio audit with bounded streaming workers."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import struct
import time

import numpy as np
import soundfile as sf

from speaker_id.audio.io import file_sha256, inspect_wave_payload, open_audio
from speaker_id.eda.signal import CLIP_LEVEL, DB_FLOOR, FRAME_SECONDS, SignalAccumulator

AUDIT_VERSION = "signal-v1"
BLOCK_FRAMES = 65536
PCM_HASH_PREFIX = b"speaker-id-pcm-f32le-v1\0"


def audit_audio(path: Path, speaker_id: str, block_frames: int = BLOCK_FRAMES) -> dict:
    """Read a single file completely. No model, resampling or audio writes occur."""
    result = {"audio_file": path.name, "speaker_id": speaker_id,
              "status": "error", "error": "", "usable_for_training": False,
              "quality_flags": [], "audit_version": AUDIT_VERSION}
    flags = result["quality_flags"]
    try:
        before = path.stat()
        result.update(file_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                      input_sha256=file_sha256(path))
        result.update(inspect_wave_payload(path))
        with open_audio(path) as stream:
            sr, channels, expected_frames = stream.samplerate, stream.channels, stream.frames
            result.update(detected_format=stream.format, subtype=stream.subtype,
                          sample_rate_hz=sr, channels=channels, header_frames=expected_frames)
            valid_extensions = {"WAV": {".wav", ".wave"}, "WAVEX": {".wav", ".wave"},
                                "RF64": {".wav"}, "MP3": {".mp3"}, "FLAC": {".flac"},
                                "OGG": {".ogg", ".oga"}}
            result["extension_matches_container"] = path.suffix.lower() in valid_extensions.get(stream.format, set())
            if not result["extension_matches_container"]:
                flags.append("extension_container_mismatch")
            pcm_hash = hashlib.sha256(PCM_HASH_PREFIX + struct.pack("<II", sr, channels))
            stats = SignalAccumulator(sr, channels)
            while True:
                samples = stream.read(block_frames, dtype="float32", always_2d=True)
                if not len(samples):
                    break
                pcm_hash.update(samples.astype("<f4", copy=False).tobytes(order="C"))
                stats.update(samples)
            metrics = stats.finalize()
        result.update(metrics)
        pcm_hash.update(struct.pack("<Q", metrics["decoded_frames"]))
        result["pcm_sha256"] = pcm_hash.hexdigest()
        result["header_frames_match"] = expected_frames == metrics["decoded_frames"]
        wave_frames = result["wave_payload_frames"]
        if wave_frames is not None and wave_frames != metrics["decoded_frames"]:
            flags.append("wave_payload_frames_mismatch")
        if not result["header_frames_match"]:
            flags.append("decoder_header_frames_mismatch")
        if result["wave_integrity"] == "error":
            flags.append("wave_container_integrity_error")
        if metrics["nonfinite_samples"]:
            flags.append("nonfinite_samples")
        if not metrics["has_nonzero_signal"]:
            flags.append("no_signal")
        if metrics["duration_seconds"] < 0.25:
            flags.append("tiny_duration_lt_025s")
        elif metrics["duration_seconds"] < 1:
            flags.append("short_duration_lt_1s")
        if metrics["max_channel_rms_dbfs"] < -50:
            flags.append("low_rms_below_minus50_dbfs")
        if metrics["max_channel_clip_fraction"] > 0.001:
            flags.append("near_full_scale_fraction_gt_0001")
        if metrics["channel_rms_imbalance_db"] > 6:
            flags.append("channel_imbalance_gt_6db")
        if metrics["downmix_attenuation_db"] < -6:
            flags.append("downmix_loss_gt_6db")
        if metrics["channel_correlation"] is not None and metrics["channel_correlation"] < 0:
            flags.append("negative_channel_correlation")
        if metrics["max_channel_abs_dc"] > 0.01:
            flags.append("high_dc_offset_gt_001")
        if (metrics["below_minus50_dbfs_fraction"] or 0) > 0.95:
            flags.append("mostly_low_energy_frames_gt_095")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Input changed during read; retry with stable raw inputs")
        integrity_ok = (result["header_frames_match"] and result["wave_integrity"] != "error"
                        and "wave_payload_frames_mismatch" not in flags and not metrics["nonfinite_samples"])
        result["status"] = "ok" if integrity_ok else "error"
        if not integrity_ok:
            result["error"] = "Decoded, but header/payload/finite-sample integrity failed"
        # Eligibility is signal/integrity only, not evidence of speech or speaker-label validity.
        # Short, low-energy, clipped or imbalanced signals are flagged, never silently dropped.
        result["usable_for_training"] = integrity_ok and metrics["has_nonzero_signal"]
    except Exception as exc:
        result["status"] = "error"
        result["usable_for_training"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        flags.append("decode_or_read_error")
    return result


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "p01": float(np.quantile(array, .01)),
            "p10": float(np.quantile(array, .1)), "median": float(np.median(array)),
            "p90": float(np.quantile(array, .9)), "p99": float(np.quantile(array, .99)),
            "max": float(array.max()), "mean": float(array.mean()), "sum": float(array.sum())}


def build_summary(rows: list[dict], labels_path: Path, signature: str, packages: dict,
                  code_hashes: dict, elapsed: float, resumed_count: int) -> dict:
    successes = [r for r in rows if r["status"] == "ok"]
    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["audio_file"]):
        fingerprint.update(json.dumps([row["audio_file"], row["speaker_id"], row.get("input_sha256")], separators=(",", ":")).encode())
        fingerprint.update(b"\n")
    numeric_fields = ["duration_seconds", "max_channel_rms_dbfs", "mono_rms_dbfs", "max_channel_clip_fraction",
                      "mono_zero_fraction", "channel_correlation", "channel_rms_imbalance_db", "downmix_attenuation_db",
                      "analysis_zcr", "frame_energy_p10_dbfs", "frame_energy_p50_dbfs", "frame_energy_p90_dbfs",
                      "below_minus50_dbfs_fraction", "relative_low_energy_fraction", "energy_above_minus50_dbfs_seconds",
                      "vad_mode1_speech_seconds", "vad_mode1_speech_fraction", "vad_mode3_speech_seconds", "vad_mode3_speech_fraction"]
    numeric = {}
    for group, group_rows in (("all", successes), ("known", [r for r in successes if r["speaker_id"] != "unknown"]),
                              ("unknown", [r for r in successes if r["speaker_id"] == "unknown"])):
        numeric[group] = {key: _summarize([r[key] for r in group_rows if r.get(key) is not None]) for key in numeric_fields}
    return {
        "audit_kind": "full_signal_streaming", "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": {"labeled_files": len(rows), "fully_decoded_and_integrity_passed": len(successes),
                     "failures": len(rows) - len(successes), "file_sha256_count": sum(bool(r.get("input_sha256")) for r in rows),
                     "pcm_sha256_count": sum(bool(r.get("pcm_sha256")) for r in rows),
                     "no_signal_count": sum("no_signal" in r["quality_flags"] for r in rows),
                     "usable_for_training_count": sum(r["usable_for_training"] for r in rows),
                     "all_expected_rows_processed": True, "resumed_from_checkpoint_count": resumed_count,
                     "current_invocation_seconds": elapsed,
                     "total_decoded_seconds": sum(r.get("duration_seconds", 0) for r in rows),
                     "formats": dict(Counter(r.get("detected_format", "failed") for r in rows)),
                     "sample_rates_hz": dict(Counter(str(r.get("sample_rate_hz", "failed")) for r in rows)),
                     "channel_counts": dict(Counter(str(r.get("channels", "failed")) for r in rows)),
                     "wave_payload_integrity": dict(Counter(r.get("wave_integrity", "failed") for r in rows)),
                     "vad_status": dict(Counter(r.get("vad_status", "failed") for r in rows))},
        "quality_flag_counts": dict(sorted(Counter(flag for r in rows for flag in r["quality_flags"]).items())),
        "numeric_summary": numeric,
        "failures": [{"audio_file": r["audio_file"], "error": r["error"]} for r in rows if r["status"] != "ok"],
        "input_fingerprint": {"labeled_audio_sha256": fingerprint.hexdigest(), "labels_csv_sha256": file_sha256(labels_path),
                              "definition": "SHA256 over filename-sorted compact JSON [audio_file,speaker_id,input_sha256] lines; UTF-8, LF"},
        "audit_signature": signature, "code_sha256": code_hashes, "packages": packages,
        "config": {"block_frames": BLOCK_FRAMES, "energy_frame_seconds": FRAME_SECONDS, "db_floor": DB_FLOOR,
                   "near_full_scale_threshold_absolute_amplitude": CLIP_LEVEL,
                   "energy_thresholds_dbfs": [-60, -50, -40], "relative_low_energy_db_below_p90": 30,
                   "vad_modes": [1, 3], "vad_frame_seconds": .030},
        "units_and_definitions": {
            "samples": "float32 decoder output; float64 accumulation; normalized PCM full scale is amplitude 1",
            "dbfs": "20*log10(amplitude); floor -160 dBFS used for zeros; RMS and peak references are both amplitude 1",
            "channel_arrays": "JSON arrays in original channel order; no resampling, normalization, cropping or downmix before hashing",
            "channel_correlation": "Pearson correlation of first two full-length channels; null if fewer than two or zero variance",
            "channel_identical": "Every decoded sample equal across ALL channels; null for mono",
            "channel_rms_imbalance_db": "largest channel RMS dBFS minus smallest; 160 dB floor applies",
            "downmix_attenuation_db": "arithmetic mean downmix RMS dBFS minus strongest original channel RMS dBFS",
            "analysis_channel_index": "0-based index of strongest full-file RMS channel, used only for EDA energy/ZCR; not a model preprocessing decision",
            "energy": "Non-overlapping 20 ms RMS frames on analysis channel; final partial frame included; not VAD, speech duration or SNR",
            "energy_above_minus50_dbfs_seconds": "Sum actual durations of RMS frames >= -50dBFS on analysis channel; signal-energy heuristic only",
            "relative_low_energy_fraction": "Fraction of RMS frames below per-file p90 minus30dB; silence p90 uses -160dB floor",
            "vad": "WebRTC VAD modes1&3 independently on every original channel; 30ms frames quantized to PCM16 with nearest rounding and saturation, no gain adjustment; tail <30ms not analyzed. Speech fractions use analyzed duration. Analysis-channel outputs use strongest full-file RMS channel. Predictions are not ground truth and never filter training eligibility.",
            "zcr": "Fraction of adjacent sample pairs with differing (sample < 0) booleans; zeros count as nonnegative",
            "channel_clip_fraction": "Fraction abs(sample)>=32767/32768; near-full-scale proxy, not confirmed clipping/distortion",
            "dc": "Arithmetic waveform mean in normalized amplitude units",
            "usable_for_training": "Successful complete decode and integrity checks, finite samples, at least one nonzero sample. Does not certify usable speech. Quality flags alone do not remove signals.",
            "short_duration": "<0.25s tiny flag; >=0.25s and <1s short flag; does not automatically exclude nonzero audio",
            "pcm_sha256": "SHA256(prefix b'speaker-id-pcm-f32le-v1\\0', uint32 little endian samplerate/channels, full interleaved little endian float32 decoded samples, uint64 little endian actual frame count)",
            "resume": "Reuse requires same code/config/package signature, filename/label, file size and mtime_ns; assumes immutable raw inputs; --no-resume forces byte reads",
        },
        "limitations": ["Energy thresholds are diagnostics, not calibrated voice activity, speech quality or SNR estimates.",
                        "MP3 decoded to EOF but bitstream/frame CRC completeness is not independently established.",
                        "This scan does not identify speakers, overlapping speakers, near duplicates or perceptual audio quality.",
                        "Archive CRC/decompressed-to-disk content comparison is outside this scan.",
                        "Decoded PCM hashes are decoder-version dependent for lossy formats; runtime versions are recorded."],
    }


def run_audit(data_dir: Path, output_dir: Path, report_path: Path, workers: int = 4,
              resume: bool = True, limit: int | None = None) -> dict:
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    started = time.monotonic()
    labels_path = data_dir / "labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"audio_file", "speaker_id"}:
            raise ValueError("labels.csv must contain exactly audio_file,speaker_id")
        labels = sorted(list(reader), key=lambda r: r["audio_file"])
    if len({r["audio_file"] for r in labels}) != len(labels):
        raise ValueError("Duplicate filenames in labels.csv")
    for row in labels:
        if Path(row["audio_file"]).name != row["audio_file"] or not row["speaker_id"]:
            raise ValueError("Labels require plain filenames and nonempty speaker_id")
    if limit is not None:
        labels = labels[:limit]
    packages = {"python": platform.python_version(), "numpy": np.__version__, "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__, "scipy": version("scipy")}
    try:
        packages["webrtcvad-wheels"] = version("webrtcvad-wheels")
    except PackageNotFoundError:
        packages["webrtcvad-wheels"] = None
    root = Path(__file__).resolve().parents[3]
    code_paths = [Path(__file__), Path(__file__).with_name("signal.py"), Path(__file__).parents[1] / "audio/io.py",
                  root / "scripts/eda/audit_audio.py"]
    code_hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in code_paths}
    signature = hashlib.sha256(json.dumps({"code": code_hashes, "packages": packages,
                                          "block_frames": BLOCK_FRAMES}, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "audio_audit.checkpoint.jsonl"
    cache = {}
    if resume and checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    cached = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Interrupted final line cannot invalidate earlier completed files.
                if cached.get("audit_signature") == signature:
                    cache[cached["row"]["audio_file"]] = cached["row"]
    results, pending = [], []
    for label in labels:
        candidate = cache.get(label["audio_file"])
        path = data_dir / label["audio_file"]
        stat = path.stat() if path.exists() else None
        if (candidate and stat and candidate.get("status") == "ok" and candidate["speaker_id"] == label["speaker_id"]
                and candidate.get("file_bytes") == stat.st_size and candidate.get("mtime_ns") == stat.st_mtime_ns):
            results.append(candidate)
        else:
            pending.append(label)
    resumed_count = len(results)
    print(f"Audio audit: {len(labels)} files; {resumed_count} resumed; {len(pending)} to decode; workers={workers}", flush=True)
    with checkpoint.open("a" if resume else "w", encoding="utf-8") as stream:
        # Terminate any partial final JSONL line left by interruption before appending.
        if resume and checkpoint.stat().st_size:
            stream.write("\n")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio-eda") as pool:
            futures = {pool.submit(audit_audio, data_dir / row["audio_file"], row["speaker_id"]): row for row in pending}
            last_progress = time.monotonic()
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps({"audit_signature": signature, "row": result}, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if time.monotonic() - last_progress >= 15 or len(results) == len(labels):
                    seconds = time.monotonic() - started
                    print(f"Audio audit: {len(results)}/{len(labels)} files; elapsed={seconds:.1f}s; errors={sum(r['status'] != 'ok' for r in results)}", flush=True)
                    last_progress = time.monotonic()
    results.sort(key=lambda r: r["audio_file"])
    preferred = ["audio_file", "speaker_id", "status", "error", "usable_for_training", "quality_flags"]
    columns = preferred + sorted(set().union(*(r.keys() for r in results)) - set(preferred))
    manifest = output_dir / "audio_manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    temporary.replace(manifest)
    summary = build_summary(results, labels_path, signature, packages, code_hashes,
                            time.monotonic() - started, resumed_count)
    summary["coverage"]["limit_applied"] = limit
    summary["outputs"] = {"manifest": str(manifest), "checkpoint": str(checkpoint)}
    _json_write(report_path, summary)
    print(f"Wrote {manifest} and {report_path}", flush=True)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/eda_v1")
    parser.add_argument("--report", type=Path, default=root / "reports/eda/signal_summary.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; use separate output paths")
    args = parser.parse_args()
    run_audit(args.data_dir.resolve(), args.output_dir.resolve(), args.report.resolve(),
              workers=args.workers, resume=not args.no_resume, limit=args.limit)


if __name__ == "__main__":
    main()
