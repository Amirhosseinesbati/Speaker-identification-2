"""Prepare a bounded listening queue; do not claim that a human reviewed it."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import soundfile as sf


def prepare_review(manifest: Path, data_dir: Path, output: Path, limit: int = 24) -> dict:
    frame = pd.read_csv(manifest)
    frame = frame[frame.status == "ok"].copy()
    selection: dict[str, list[str]] = {}

    def take(rows, reason, count=2):
        for name in rows.head(count).audio_file:
            selection.setdefault(name, []).append(reason)

    valid = frame[frame.duration_seconds >= 0.25]
    # Prioritize sparse/quiet known classes; file-count support can hide low speech evidence.
    if "vad_mode3_speech_seconds" in valid:
        known_vad = valid[valid.speaker_id != "unknown"].groupby("speaker_id").vad_mode3_speech_seconds.sum()
        for speaker in known_vad[known_vad < 5].sort_values().index:
            take(valid[(valid.speaker_id == speaker) & valid.usable_for_training].sort_values("duration_seconds", ascending=False), "known_class_under5s_vad3_prediction", 1)
    take(valid.sort_values("duration_seconds"), "shortest_nontrivial")
    take(valid.sort_values("duration_seconds", ascending=False), "longest")
    take(valid.sort_values("max_channel_rms_dbfs"), "lowest_rms")
    take(valid.sort_values("max_channel_clip_fraction", ascending=False), "highest_near_full_scale_fraction")
    take(valid.sort_values("relative_low_energy_fraction", ascending=False), "largest_relative_low_energy_fraction")
    take(valid[valid.detected_format == "MP3"], "genuine_mp3", 1)
    for label_group in ("known", "unknown"):
        subset = valid[(valid.speaker_id == "unknown") == (label_group == "unknown")].copy()
        subset["median_distance"] = (subset.duration_seconds - subset.duration_seconds.median()).abs()
        take(subset.sort_values(["median_distance", "audio_file"]), f"representative_{label_group}", 4)
    stereo = valid[valid.channels > 1]
    take(stereo.sort_values("downmix_attenuation_db"), "largest_downmix_attenuation", 2)
    if "vad_mode3_speech_fraction" in valid:
        take(valid.sort_values("vad_mode3_speech_fraction"), "lowest_vad3_prediction", 2)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    by_name = frame.set_index("audio_file")
    for name, reasons in list(selection.items())[:limit]:
        info = by_name.loc[name]
        with sf.SoundFile(data_dir / name) as audio:
            channel = int(info.analysis_channel_index)
            length = min(len(audio), int(8 * audio.samplerate))
            start = max(0, (len(audio) - length) // 2)
            audio.seek(start)
            samples = audio.read(length, dtype="float32", always_2d=True)[:, channel]
            output_name = Path(name).stem + ".wav"
            sf.write(output / output_name, samples, audio.samplerate, subtype="PCM_16")
            rows.append({"audio_file": name, "speaker_id": info.speaker_id,
                         "selection_reasons": "|".join(reasons), "source_start_seconds": start / audio.samplerate,
                         "clip_seconds": len(samples) / audio.samplerate, "source_channel_index": channel,
                         "clip_file": output_name, "listening_status": "pending", "notes": ""})
    csv_path = output.parent / "listening_queue.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["audio_file", "listening_status"])
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        cards.append(f'<article><h3 dir="ltr">{html.escape(row["audio_file"])}</h3><p dir="ltr">{html.escape(row["speaker_id"])} · {html.escape(row["selection_reasons"])}</p><p>کانال {row["source_channel_index"]}، شروع {row["source_start_seconds"]:.2f} ثانیه، مدت {row["clip_seconds"]:.2f} ثانیه؛ وضعیت بررسی: انجام نشده</p><audio controls preload="none" src="{html.escape(row["clip_file"])}"></audio></article>')
    (output / "index.html").write_text('''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>صف بررسی شنیداری</title><style>body{font:16px/1.8 Tahoma,Arial,sans-serif;max-width:1040px;margin:32px auto;padding:0 20px;background:#f5f7fb;color:#182942}article{background:white;border:1px solid #dde4ef;border-radius:12px;margin:16px 0;padding:20px}h3{font-size:13px;overflow-wrap:anywhere}audio{width:100%}p[dir=ltr]{font-size:13px}</style><h1>صف نمونه‌های بررسی شنیداری</h1><p>این صفحه نمونه‌ها را برای بررسی آماده می‌کند. پخش یا شنیدن آن‌ها در این مرحله تأیید نشده است و هیچ نتیجه‌ای دربارهٔ زبان، لهجه، موسیقی یا چندگویندگی ثبت نشده است. هر نمونه حداکثر ۸ ثانیه از مرکز فایل و از کانال دارای RMS بیشتر است؛ بدون نرمال‌سازی شدت. این انتخاب تمام بخش‌های فایل را پوشش نمی‌دهد.</p>''' + "\n".join(cards) + "</html>", encoding="utf-8")
    summary = {"selected_files": len(rows), "listening_completed": False,
               "selection": "Deterministic representative and anomaly strata; maximum 24 clips of 8 seconds",
               "channel_policy": "Highest-RMS original channel selected by full-file audit; no normalization",
               "location": "Centered in each original file; not exhaustive coverage of a recording",
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (output.parent / "listening_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
