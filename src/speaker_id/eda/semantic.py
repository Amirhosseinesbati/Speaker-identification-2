"""Bounded, offline audio-event and language screening; never a human listening label.

AST sigmoid outputs are model scores, not calibrated probabilities or truth. Whisper
can hallucinate; sparse/near-zero inputs remain inconclusive even if a model fires.
No demographic or identity-related model outputs are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

EVENTS = ("Speech", "Conversation", "Narration, monologue", "Speech synthesizer",
          "Singing", "Music", "Silence", "Noise", "White noise", "Pink noise",
          "Static", "Hum", "Hiss", "Inside, small room", "Inside, large room or hall",
          "Inside, public space", "Outside, urban or manmade", "Echo",
          "Wind noise (microphone)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, field, default=0.0):
    try:
        result = float(row.get(field, default))
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def plan_windows(duration, length=10.0):
    """At most two non-overlapping, deterministic windows in original file time."""
    if duration <= 0:
        return []
    if duration <= length:
        return [(0.0, duration)]
    if duration < 2 * length:
        return [((duration - length) / 2, length)]
    return [(0.0, length), (duration - length, length)]


def select_files(manifest, listening, sensitivity, controls_per_stratum=10):
    by_name = {row["audio_file"]: row for row in manifest}
    reasons = {}
    for source, reason in ((listening, "listening_queue"), (sensitivity, "vad_sensitivity")):
        for row in source:
            if row["audio_file"] not in by_name:
                raise ValueError("Selected file missing from manifest")
            reasons.setdefault(row["audio_file"], set()).add(reason)
    controls = []
    for label in ("known", "unknown"):
        candidates = [r for r in manifest if r["audio_file"] not in reasons
                      and (r["speaker_id"] == "unknown") == (label == "unknown")
                      and r["status"] == "ok" and number(r, "max_channel_rms_dbfs", -999) > -40
                      and number(r, "vad_mode3_speech_fraction") >= .6
                      and number(r, "duration_seconds") >= 10]
        candidates.sort(key=lambda r: (number(r, "duration_seconds"), r["audio_file"]))
        seen = set()
        for quartile in range(4):
            part = candidates[len(candidates) * quartile // 4:len(candidates) * (quartile + 1) // 4]
            part.sort(key=lambda r: hashlib.sha256(("semantic-v1:" + r["audio_file"]).encode()).hexdigest())
            chosen = 0
            for row in part:
                if label == "known" and row["speaker_id"] in seen:
                    continue
                reasons.setdefault(row["audio_file"], set()).add(f"control_{label}_duration_q{quartile + 1}")
                controls.append(row["audio_file"])
                seen.add(row["speaker_id"])
                chosen += 1
                if chosen == controls_per_stratum:
                    break
    selected = [{**by_name[name], "selection_reasons": "|".join(sorted(reasons[name]))}
                for name in sorted(reasons)]
    return selected, controls


def gain_factor(rms, peak, target_db=-20, cap_db=40, ceiling=.95):
    """Diagnostic gain only. Never attenuate or amplify exact silence."""
    if rms <= 0 or peak <= 0:
        return 1.0
    return max(1.0, min(10 ** (cap_db / 20), 10 ** (target_db / 20) / rms, ceiling / peak))


def model_provenance(directory, repository):
    files = {p.relative_to(directory).as_posix(): sha256(p)
             for p in sorted(directory.iterdir()) if p.is_file()}
    revisions = set()
    metadata = {}
    for path in sorted((directory / ".cache").rglob("*.metadata")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0])
        metadata[path.relative_to(directory).as_posix()] = sha256(path)
    return {"repository": repository, "local_directory": str(directory.resolve()),
            "revision_from_huggingface_download_metadata": sorted(revisions),
            "files_sha256": files, "download_metadata_sha256": metadata}


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.extra_site_packages:
        sys.path.insert(0, str(args.extra_site_packages.resolve()))
    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import transformers
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, WhisperProcessor, WhisperForConditionalGeneration

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.30, device)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    listening = read_csv(args.listening)
    sensitivity = read_csv(args.sensitivity)
    selected, controls = select_files(manifest, listening, sensitivity, args.controls_per_stratum)
    model_info = {
        "ast": model_provenance(args.ast, "MIT/ast-finetuned-audioset-10-10-0.4593"),
        "whisper": model_provenance(args.whisper, "openai/whisper-base"),
    }
    provenance = {"version": "semantic-v1", "manifest_sha256": sha256(args.manifest),
                  "listening_sha256": sha256(args.listening), "sensitivity_sha256": sha256(args.sensitivity),
                  "code_sha256": sha256(Path(__file__)), "models": model_info,
                  "configuration": {"window_seconds": 10, "max_windows": 2,
                                    "controls_per_stratum": args.controls_per_stratum,
                                    "gain_files": args.gain_files, "whisper_files": args.whisper_files,
                                    "max_new_tokens": 96, "gain_cap_db": 40,
                                    "torch_dtype": "float32", "device": str(device)}}
    run_key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cached = {}
    if args.checkpoint.exists():
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("run_key") == run_key:
                cached[item["key"]] = item["row"]

    def persist(key, row):
        cached[key] = row
        with args.checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_key": run_key, "key": key, "row": row}, ensure_ascii=False) + "\n")

    def read_window(info, start_s, length_s):
        with sf.SoundFile(args.data / info["audio_file"]) as audio:
            start = round(start_s * audio.samplerate)
            audio.seek(start)
            x = audio.read(round(length_s * audio.samplerate), dtype="float32", always_2d=True)[:, int(info["analysis_channel_index"])]
            rate = audio.samplerate
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            x = resample_poly(x, 16000 // divisor, rate // divisor).astype(np.float32)
        return x

    quiet = [r for r in selected if -120 < number(r, "max_channel_rms_dbfs", -999) < -50
             and number(r, "duration_seconds") >= .25 and number(r, "has_nonzero_signal", 1)]
    # Include the pathological class first; a score after gain still isn't evidence of speech.
    quiet.sort(key=lambda r: (r["speaker_id"] != "44ea12a5-af34-418e-a9c2-fa6b93b66fce",
                              number(r, "max_channel_rms_dbfs"), r["audio_file"]))
    gain_names = {r["audio_file"] for r in quiet[:args.gain_files]}
    segments = []
    print(f"Selected {len(selected)} files, {len(controls)} healthy controls, {len(gain_names)} gain comparators.", flush=True)
    feature = ASTFeatureExtractor.from_pretrained(str(args.ast), local_files_only=True)
    ast = ASTForAudioClassification.from_pretrained(str(args.ast), local_files_only=True).eval().to(device)
    event_ids = {name: ast.config.label2id[name] for name in EVENTS if name in ast.config.label2id}
    for index, info in enumerate(selected):
        for segment_index, (start_s, length_s) in enumerate(plan_windows(number(info, "duration_seconds"))):
            for variant in (["original", "diagnostic_gain"] if info["audio_file"] in gain_names else ["original"]):
                key = f"ast:{info['audio_file']}:{segment_index}:{variant}"
                if key in cached:
                    segments.append(cached[key])
                    continue
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Time cap reached; checkpoint preserved. Re-run same command to resume.")
                row = {"audio_file": info["audio_file"], "speaker_id": info["speaker_id"],
                       "selection_reasons": info["selection_reasons"], "input_sha256": info["input_sha256"],
                       "segment_index": segment_index, "start_seconds": start_s, "duration_seconds": length_s,
                       "source_channel_index": int(info["analysis_channel_index"]), "variant": variant,
                       "human_listening_completed": False, "status": "ok"}
                try:
                    x = read_window(info, start_s, length_s)
                    peak, rms = float(np.max(np.abs(x))), float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                    zero_fraction = float(np.mean(x == 0))
                    gain = gain_factor(rms, peak) if variant == "diagnostic_gain" else 1.0
                    row.update({"input_rms_dbfs": 20 * math.log10(max(rms, 1e-15)),
                                "exact_zero_fraction": zero_fraction, "gain_db": 20 * math.log10(gain),
                                "inconclusive_signal": rms < 1e-4 or zero_fraction > .99,
                                "padded_input_seconds": max(0, 10 - len(x) / 16000)})
                    if peak == 0 or len(x) < 400:
                        row["status"] = "inconclusive_zero_or_under25ms"
                    else:
                        inputs = feature(x * gain, sampling_rate=16000, return_tensors="pt")
                        with torch.inference_mode():
                            scores = ast(**{k: v.to(device) for k, v in inputs.items()}).logits[0].sigmoid().cpu().numpy()
                        row.update({"ast_" + name.lower().replace(" ", "_").replace(",", ""): float(scores[idx])
                                    for name, idx in event_ids.items()})
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                persist(key, row)
                segments.append(row)
        if (index + 1) % 20 == 0:
            print(f"AST {index + 1}/{len(selected)} files; elapsed {time.monotonic() - started:.1f}s", flush=True)
    del ast, feature
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Sixteen queue entries plus eight controls; never interpret this anomaly-enriched
    # sample as a dataset language distribution. All-zero/sparse entries are retained.
    whisper_names = [r["audio_file"] for r in listening][:max(0, args.whisper_files - 8)]
    by_name = {r["audio_file"]: r for r in selected}
    for unknown in (False, True):
        candidates = [n for n in controls if (by_name[n]["speaker_id"] == "unknown") == unknown]
        whisper_names += candidates[::max(1, len(candidates) // 4)][:4]
    whisper_names = list(dict.fromkeys(whisper_names))[:args.whisper_files]
    language_rows = []
    processor = WhisperProcessor.from_pretrained(str(args.whisper), local_files_only=True)
    whisper = WhisperForConditionalGeneration.from_pretrained(str(args.whisper), local_files_only=True).eval().to(device)
    language_ids = whisper.generation_config.lang_to_id
    language_order = list(language_ids)
    lang_tokens = torch.tensor([language_ids[key] for key in language_order], device=device)
    for index, name in enumerate(whisper_names):
        key = "whisper:" + name
        if key in cached:
            language_rows.append(cached[key])
            continue
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Time cap reached; checkpoint preserved. Re-run to resume.")
        info = by_name[name]
        originals = [s for s in segments if s["audio_file"] == name and s["variant"] == "original"]
        chosen = max(originals, key=lambda r: (r.get("ast_speech", -1), -r["start_seconds"]))
        row = {"audio_file": name, "speaker_id": info["speaker_id"], "input_sha256": info["input_sha256"],
               "selection_reasons": info["selection_reasons"], "start_seconds": chosen["start_seconds"],
               "duration_seconds": chosen["duration_seconds"], "source_channel_index": int(info["analysis_channel_index"]),
               "window_selection": "highest AST Speech score among up to two preselected original windows",
               "human_listening_completed": False, "status": "ok", "is_transcription_verified": False}
        if chosen.get("inconclusive_signal", True) or chosen["status"] != "ok":
            row["status"] = "inconclusive_sparse_or_quiet_signal"
        elif chosen.get("ast_speech", 0) < .1:
            row["status"] = "inconclusive_low_ast_speech_score"
        else:
            try:
                x = read_window(info, chosen["start_seconds"], chosen["duration_seconds"])
                inputs = processor(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    decoder = torch.tensor([[whisper.config.decoder_start_token_id]], device=device)
                    logits = whisper(input_features=inputs["input_features"], decoder_input_ids=decoder).logits[0, -1]
                    probs = torch.softmax(logits[lang_tokens], dim=0)
                    values, indices = probs.topk(3)
                    top_languages = [{"language": language_order[i].replace("<|", "").replace("|>", ""),
                                      "conditional_model_probability": float(v)} for v, i in zip(values.cpu(), indices.cpu())]
                    lang = top_languages[0]["language"]
                    generated = whisper.generate(**inputs, language=lang, task="transcribe", max_new_tokens=96,
                                                 do_sample=False, num_beams=1, return_timestamps=False)
                row.update({"top_language": lang, "language_probability": top_languages[0]["conditional_model_probability"],
                            "top3_language_model_probabilities": json.dumps(top_languages),
                            "transcript_unverified": processor.batch_decode(generated, skip_special_tokens=True)[0],
                            "generated_tokens": int(generated.shape[1]),
                            "token_cap_may_have_truncated": int(generated.shape[1]) >= 96})
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        persist(key, row)
        language_rows.append(row)
        print(f"Whisper {index + 1}/{len(whisper_names)} {row['status']}; elapsed {time.monotonic() - started:.1f}s", flush=True)
    del whisper, processor
    csv_write(args.output / "semantic_segments.csv", segments)
    csv_write(args.output / "semantic_language.csv", language_rows)
    file_rows = []
    for info in selected:
        original = [r for r in segments if r["audio_file"] == info["audio_file"] and r["variant"] == "original"]
        scored = [r for r in original if r["status"] == "ok"]
        file_rows.append({"audio_file": info["audio_file"], "speaker_id": info["speaker_id"],
                          "selection_reasons": info["selection_reasons"], "file_duration_seconds": number(info, "duration_seconds"),
                          "sampled_original_seconds": sum(r["duration_seconds"] for r in original),
                          "original_segments": len(original), "scored_segments": len(scored),
                          "all_segments_inconclusive_signal": all(r.get("inconclusive_signal", True) for r in original),
                          "max_ast_speech": max((r["ast_speech"] for r in scored), default=""),
                          "max_ast_music": max((r["ast_music"] for r in scored), default=""),
                          "max_ast_noise": max((r["ast_noise"] for r in scored), default=""),
                          "max_ast_silence": max((r["ast_silence"] for r in scored), default=""),
                          "human_listening_completed": False})
    csv_write(args.output / "semantic_files.csv", file_rows)
    original = [r for r in segments if r["variant"] == "original"]
    scored = [r for r in original if r["status"] == "ok"]
    interpretable = [r for r in scored if not r["inconclusive_signal"]]
    language_counts = {}
    for row in language_rows:
        if row.get("top_language"):
            language_counts[row["top_language"]] = language_counts.get(row["top_language"], 0) + 1
    strata = {}
    for name, match in (("healthy_controls", lambda r: "control_" in r["selection_reasons"]),
                        ("anomaly_and_listening_union", lambda r: "control_" not in r["selection_reasons"])):
        part = [r for r in interpretable if match(r)]
        strata[name] = {"interpretable_original_segments": len(part),
                        "ast_speech_ge_05_segments": sum(r["ast_speech"] >= .5 for r in part),
                        "ast_music_ge_05_segments": sum(r["ast_music"] >= .5 for r in part),
                        "ast_speech_and_music_ge_05_segments": sum(r["ast_speech"] >= .5 and r["ast_music"] >= .5 for r in part),
                        "ast_noise_ge_05_segments": sum(r["ast_noise"] >= .5 for r in part)}
    summary = {**provenance, "run_key": run_key, "status": "completed", "human_listening_completed": False,
               "runtime": {"seconds": time.monotonic() - started, "python": sys.version, "torch": torch.__version__,
                           "transformers": transformers.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                           "soundfile": sf.__version__, "lib_sndfile": sf.__libsndfile_version__},
               "coverage": {"dataset_files": len(manifest), "dataset_seconds": sum(number(r, "duration_seconds") for r in manifest),
                            "selected_files": len(selected), "healthy_control_files": len(controls),
                            "selected_known_files": sum(r["speaker_id"] != "unknown" for r in selected),
                            "selected_unknown_files": sum(r["speaker_id"] == "unknown" for r in selected),
                            "selected_file_seconds": sum(number(r, "duration_seconds") for r in selected),
                            "original_windows": len(original), "sampled_unique_original_seconds": sum(r["duration_seconds"] for r in original),
                            "ast_scored_original_windows": len(scored), "inconclusive_signal_original_windows": sum(r.get("inconclusive_signal", True) for r in original),
                            "gain_comparator_files": len(gain_names), "gain_comparator_windows": len(segments) - len(original),
                            "whisper_selected_files": len(language_rows), "whisper_scored_files": sum(r["status"] == "ok" for r in language_rows),
                            "whisper_scored_seconds": sum(r["duration_seconds"] for r in language_rows if r["status"] == "ok")},
               "ast_error_windows": sum(r["status"] == "error" for r in segments),
               "whisper_error_files": sum(r["status"] == "error" for r in language_rows),
               "descriptive_threshold": .5, "strata": strata,
               "whisper_top_language_counts_selected_sample_only": language_counts,
               "interpretation": ["Automated model screening only; no human auditory or transcript verification.",
                                  "Selection enriches anomalies; model-score counts are not dataset prevalence estimates.",
                                  "AST sigmoid scores are multilabel, uncalibrated, and threshold 0.5 is descriptive only.",
                                  "Raw original audio is used; diagnostic gain is a separate paired sensitivity condition.",
                                  "All-zero, RMS below -80 dBFS, or over 99% exact zero windows remain signal-inconclusive.",
                                  "Whisper language probabilities are conditional over language tokens; they are not calibrated confidence.",
                                  "No demographic inference, speaker identity inference, or verified multi-speaker claim is made.",
                                  "Short audio is padded by each model's feature extractor; skipped windows do not imply absent speech.",
                                  "No source data, labels, training masks, or folds were changed."],
               "output_sha256": {name: sha256(args.output / name) for name in
                                 ("semantic_segments.csv", "semantic_files.csv", "semantic_language.csv")}}
    (args.output / "semantic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/eda_v1/audio_manifest.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--listening", type=Path, default=Path("reports/eda/listening_queue.csv"))
    parser.add_argument("--sensitivity", type=Path, default=Path("reports/eda/vad_sensitivity.csv"))
    parser.add_argument("--ast", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controls-per-stratum", type=int, default=10)
    parser.add_argument("--gain-files", type=int, default=16)
    parser.add_argument("--whisper-files", type=int, default=24)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/eda/semantic_checkpoint.jsonl"))
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({k: result[k] for k in ("status", "coverage", "ast_error_windows", "whisper_error_files", "strata")}, indent=2))
