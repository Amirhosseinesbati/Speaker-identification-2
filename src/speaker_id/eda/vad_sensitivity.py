"""Controlled-gain VAD diagnostic; this does not prescribe preprocessing."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from speaker_id.audio.io import file_sha256, open_audio
from speaker_id.eda.signal import SignalAccumulator

TARGET_RMS_DBFS = -20.0
MAX_GAIN_DB = 40.0
PEAK_CEILING = .95


def controlled_gain(rms: float, peak: float, target_dbfs: float = TARGET_RMS_DBFS,
                    max_gain_db: float = MAX_GAIN_DB, peak_ceiling: float = PEAK_CEILING) -> float:
    """Bound amplification by RMS target, maximum gain and peak headroom.

    Silence is unchanged. Already-loud samples are never attenuated, including
    originals whose peak already exceeds the proposed amplification ceiling.
    """
    if not all(math.isfinite(x) for x in (rms, peak, target_dbfs, max_gain_db, peak_ceiling)):
        raise ValueError("Gain inputs must be finite")
    if rms < 0 or peak < 0 or max_gain_db < 0 or not 0 < peak_ceiling <= 1:
        raise ValueError("Invalid gain bounds")
    if rms == 0 or peak == 0:
        return 1.0
    return max(1.0, min(10 ** (target_dbfs / 20) / rms,
                        10 ** (max_gain_db / 20), peak_ceiling / peak))


def select_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    class_speech = defaultdict(float)
    for row in rows:
        if row["speaker_id"] != "unknown":
            class_speech[row["speaker_id"]] += float(row.get("vad_mode3_speech_seconds") or 0)
    low_class = {speaker for speaker, seconds in class_speech.items() if seconds < 5}
    selected = []
    reason_counts = Counter()
    for row in rows:
        if row["status"] != "ok" or str(row["has_nonzero_signal"]).lower() != "true":
            continue
        reasons = []
        if float(row["max_channel_rms_dbfs"]) < -50:
            reasons.append("rms_below_minus50_dbfs")
        fraction = row.get("vad_mode3_speech_fraction")
        if fraction not in (None, "") and float(fraction) < .1:
            reasons.append("baseline_mode3_fraction_below_01")
        if row["speaker_id"] in low_class:
            reasons.append("known_class_total_mode3_below_5s")
        if reasons:
            selected.append({**row, "selection_reasons": reasons})
            reason_counts.update(reasons)
    return selected, {"source_files": len(rows), "selected_files": len(selected),
                      "selected_seconds": sum(float(r["duration_seconds"]) for r in selected),
                      "selected_known_files": sum(r["speaker_id"] != "unknown" for r in selected),
                      "selection_reason_counts": dict(reason_counts),
                      "known_classes_total_baseline_mode3_below_5s": sorted(low_class)}


def diagnose_file(path: Path, row: dict) -> dict:
    result = {"audio_file": row["audio_file"], "speaker_id": row["speaker_id"], "status": "error", "error": "",
              "selection_reasons": row.get("selection_reasons", []), "input_sha256": row["input_sha256"]}
    try:
        if file_sha256(path) != row["input_sha256"]:
            raise ValueError("Raw input hash differs from baseline manifest")
        channel = int(row["analysis_channel_index"])
        peaks_db = json.loads(row["channel_peak_dbfs"]) if isinstance(row["channel_peak_dbfs"], str) else row["channel_peak_dbfs"]
        rms_db = float(row["max_channel_rms_dbfs"])
        rms = 10 ** (rms_db / 20)
        peak = 10 ** (float(peaks_db[channel]) / 20)
        gain = controlled_gain(rms, peak)
        with open_audio(path) as stream:
            if stream.samplerate != int(row["sample_rate_hz"]) or stream.channels != int(row["channels"]):
                raise ValueError("Decoder format differs from baseline manifest")
            accumulator = SignalAccumulator(stream.samplerate, 1)
            while True:
                block = stream.read(65536, dtype="float32", always_2d=True)
                if not len(block):
                    break
                amplified = (block[:, channel:channel + 1].astype(np.float64) * gain).astype(np.float32)
                accumulator.update(amplified)
        post = accumulator.finalize()
        if post["decoded_frames"] != int(row["decoded_frames"]) or post["nonfinite_samples"]:
            raise ValueError("Frame-count mismatch or nonfinite samples in controlled-gain decode")
        if post["vad_status"] != "ok":
            raise ValueError("WebRTC VAD unavailable or sample rate unsupported")
        result.update({"status": "ok", "sample_rate_hz": int(row["sample_rate_hz"]),
                       "analysis_channel_index": channel, "duration_seconds": post["duration_seconds"],
                       "gain_factor": gain, "gain_db": 20 * math.log10(gain), "baseline_rms_dbfs": rms_db,
                       "post_gain_rms_dbfs": post["max_channel_rms_dbfs"], "baseline_peak_dbfs": float(peaks_db[channel]),
                       "post_gain_peak_dbfs": post["mono_peak_dbfs"], "post_gain_exact_zero_fraction": post["mono_zero_fraction"],
                       "rms_target_reached": post["max_channel_rms_dbfs"] >= TARGET_RMS_DBFS - 1e-4,
                       "max_gain_reached": gain >= 10 ** (MAX_GAIN_DB / 20) - 1e-6,
                       "peak_ceiling_already_exceeded_before_gain": peak > PEAK_CEILING,
                       "vad_analyzed_seconds": post["vad_analyzed_seconds"],
                       "vad_unprocessed_tail_seconds": post["vad_unprocessed_tail_seconds"]})
        for mode in (1, 3):
            for unit in ("seconds", "fraction"):
                key = f"vad_mode{mode}_speech_{unit}"
                baseline = float(row[key]) if row.get(key) not in (None, "") else None
                result[f"baseline_{key}"] = baseline
                result[f"post_gain_{key}"] = post[key]
                result[f"delta_{key}"] = post[key] - baseline if baseline is not None and post[key] is not None else None
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def run(manifest: Path, data_dir: Path, output_csv: Path, summary_path: Path) -> dict:
    started = time.monotonic()
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected, selection = select_rows(rows)
    print(f"VAD gain diagnostic: {len(selected)} selected files; {selection['selected_seconds']:.2f} seconds", flush=True)
    results = []
    last_progress = time.monotonic()
    for row in selected:
        results.append(diagnose_file(data_dir / row["audio_file"], row))
        if time.monotonic() - last_progress >= 15:
            print(f"VAD gain diagnostic: {len(results)}/{len(selected)}; elapsed={time.monotonic() - started:.1f}s", flush=True)
            last_progress = time.monotonic()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["audio_file", "speaker_id", "status", "error", "selection_reasons"]
    columns += sorted(set().union(*(r.keys() for r in results)) - set(columns))
    temporary = output_csv.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in results)
    temporary.replace(output_csv)
    good = [r for r in results if r["status"] == "ok"]
    per_class = {}
    for speaker in sorted({r["speaker_id"] for r in good}):
        members = [r for r in good if r["speaker_id"] == speaker]
        per_class[speaker] = {"selected_files": len(members), "selected_seconds": sum(r["duration_seconds"] for r in members),
                              "post_gain_rms_dbfs_min": min(r["post_gain_rms_dbfs"] for r in members),
                              "post_gain_rms_dbfs_max": max(r["post_gain_rms_dbfs"] for r in members)}
        for mode in (1, 3):
            for kind in ("baseline", "post_gain"):
                key = f"{kind}_vad_mode{mode}_speech_seconds"
                per_class[speaker][key] = sum(r[key] for r in members)
    summary = {"audit_kind": "controlled_gain_vad_sensitivity", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "selection": selection, "coverage": {"successful_files": len(good), "failed_files": len(results) - len(good),
                                                       "elapsed_seconds": time.monotonic() - started},
               "config": {"target_rms_dbfs": TARGET_RMS_DBFS, "maximum_gain_db": MAX_GAIN_DB, "peak_ceiling": PEAK_CEILING,
                          "attenuation_allowed": False, "modes": [1, 3], "frame_seconds": .030},
               "gain_statistics": {"max_gain_reached_files": sum(r["max_gain_reached"] for r in good),
                                   "rms_target_reached_files": sum(r["rms_target_reached"] for r in good)},
               "totals": {f"{kind}_vad_mode{mode}_speech_seconds": sum(r[f"{kind}_vad_mode{mode}_speech_seconds"] for r in good)
                          for mode in (1, 3) for kind in ("baseline", "post_gain")},
               "per_selected_class": per_class,
               "failures": [{"audio_file": r["audio_file"], "error": r["error"]} for r in results if r["status"] != "ok"],
               "input_manifest_sha256": file_sha256(manifest), "output_csv_sha256": file_sha256(output_csv),
               "code_sha256": {"vad_sensitivity.py": file_sha256(Path(__file__)),
                               "signal.py": file_sha256(Path(__file__).with_name("signal.py")),
                               "io.py": file_sha256(Path(__file__).parents[1] / "audio/io.py")},
               "packages": {"numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                            "webrtcvad-wheels": version("webrtcvad-wheels")},
               "definitions": {"selection": "Nonzero, status=ok AND (strongest-channel RMS<-50dBFS OR baseline mode3 fraction<0.1 OR known class total baseline mode3 speech<5seconds). Reasons can overlap.",
                               "gain": "max(1,min(10^((-20-RMSdBFS)/20),100,0.95/peak)); silence remains unchanged; no stored normalized originals",
                               "channel": "Strongest full-file RMS original channel index from baseline manifest; no resampling or downmix",
                               "quantization_and_tail": "Same SignalAccumulator as baseline: float32 audio, nearest-rounded x32768, saturated PCM16, independent WebRTC modes1/3 per file, complete30ms frames only, tail excluded; fractions use analyzed seconds",
                               "exact_zero_fraction": "Fraction of amplified float waveform samples exactly zero; gain cannot recreate values already quantized to zero"},
               "limitations": ["VAD predictions are not speech ground truth; sensitivity or insensitivity to amplification does not prove speech presence or absence.",
                               "40dB cap and peak constraint can leave RMS far below target; a failed target is reported explicitly.",
                               "Amplification also raises noise and quantization artifacts. This diagnostic does not recommend blanket normalization or exclusion.",
                               "Selection focuses on weak baseline VAD/energy, so aggregate effects are not representative of the entire dataset.",
                               "No training, model scoring or raw-audio modification occurred."]}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(f"Wrote {output_csv} and {summary_path}; errors={len(results) - len(good)}", flush=True)
    return summary
