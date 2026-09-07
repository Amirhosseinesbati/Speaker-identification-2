"""Read-only sample-code and spectral diagnostics for low-level recordings.

These measurements describe stored waveforms. They do not establish intelligible
speech, speaker identity, recording provenance, or an ADC's effective bit depth.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import signal
import soundfile as sf

from speaker_id.audio.io import file_sha256, open_audio
from speaker_id.eda.vad_sensitivity import select_rows

TARGET_SPEAKER = "44ea12a5-af34-418e-a9c2-fa6b93b66fce"


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of contiguous True runs, including boundary runs."""
    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[1::2] - changes[::2]


def sample_metrics(codes: np.ndarray, sample_rate: int) -> tuple[dict, dict]:
    """Metrics of exact signed PCM16 codes; zero-valued files are supported."""
    if codes.ndim != 1 or len(codes) == 0 or sample_rate <= 0:
        raise ValueError("Expected nonempty mono PCM codes and positive sample rate")
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() < -32768 or codes.max() > 32767:
        raise ValueError("Expected signed PCM16 integer codes")
    values, counts = np.unique(codes, return_counts=True)
    probabilities = counts / len(codes)
    nonzero = codes != 0
    nruns, zruns = run_lengths(nonzero), run_lengths(~nonzero)
    x = codes.astype(np.float64) / 32768
    power = float(np.mean(x * x))
    peak = float(np.max(np.abs(x)))
    nz_count = int(nonzero.sum())
    peak_gain = max(1.0, min(.1 / math.sqrt(power), .95 / peak)) if power and peak else 1.0
    centered = x - x.mean()
    denom = np.dot(centered[:-1], centered[:-1]) * np.dot(centered[1:], centered[1:])
    lag1 = float(np.dot(centered[:-1], centered[1:]) / np.sqrt(denom)) if denom > 0 else None
    frame_n = max(1, round(.020 * sample_rate))
    frames = np.pad(x, (0, (-len(x)) % frame_n)).reshape(-1, frame_n)
    frame_energy = np.sum(frames ** 2, axis=1)
    energy_sum = float(frame_energy.sum())
    top_k = max(1, math.ceil(len(frame_energy) * .01))
    metrics = {
        "samples": len(codes), "sample_rate_hz": sample_rate,
        "duration_seconds": len(codes) / sample_rate,
        "nonzero_samples": nz_count, "zero_fraction": float(1 - nonzero.mean()),
        "nonzero_sample_support_seconds": nz_count / sample_rate,
        "pcm_min": int(values[0]), "pcm_max": int(values[-1]),
        "unique_pcm_codes": len(values), "observed_code_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "observed_code_alphabet_bits": float(np.log2(len(values))),
        "observed_code_span_bits": float(np.log2(int(values[-1]) - int(values[0]) + 1)),
        "nonzero_abs_one_fraction": float(np.mean(np.abs(codes[nonzero].astype(np.int32)) == 1)) if nz_count else None,
        "nonzero_negative_fraction": float(np.mean(codes[nonzero] < 0)) if nz_count else None,
        "pcm_absolute_p999": float(np.quantile(np.abs(codes.astype(np.int32)), .999)),
        "nonzero_runs": len(nruns), "nonzero_run_median_samples": float(np.median(nruns)) if len(nruns) else 0,
        "nonzero_run_p95_samples": float(np.quantile(nruns, .95)) if len(nruns) else 0,
        "nonzero_run_max_samples": int(nruns.max()) if len(nruns) else 0,
        "nonzero_run_max_seconds": float(nruns.max() / sample_rate) if len(nruns) else 0,
        "isolated_nonzero_sample_fraction": float(np.sum(nruns == 1) / nz_count) if nz_count else None,
        "zero_run_median_samples": float(np.median(zruns)) if len(zruns) else 0,
        "zero_run_p95_samples": float(np.quantile(zruns, .95)) if len(zruns) else 0,
        "zero_run_max_seconds": float(zruns.max() / sample_rate) if len(zruns) else 0,
        "lag1_correlation": lag1,
        "nonzero_20ms_frame_fraction": float(np.mean(frame_energy > 0)),
        "top_one_percent_20ms_frames_energy_fraction": float(np.sort(frame_energy)[-top_k:].sum() / energy_sum) if energy_sum else None,
        "rms_dbfs": 10 * math.log10(power) if power else None,
        "peak_dbfs": 20 * math.log10(peak) if peak else None,
        "diagnostic_peak_safe_gain_db": 20 * math.log10(peak_gain),
        "diagnostic_post_gain_rms_dbfs": 10 * math.log10(power) + 20 * math.log10(peak_gain) if power else None,
        "diagnostic_target_minus20_dbfs_reached": bool(power and power * peak_gain ** 2 >= .01 * (1 - 1e-10)),
    }
    nperseg = min(512, len(x))
    freqs, psd = signal.welch(x, sample_rate, nperseg=nperseg, noverlap=nperseg // 2,
                             detrend="constant", scaling="density")
    denominator = float(psd.sum())
    for lo, hi, name in [(0, 80, "below80"), (80, 300, "80to300"), (300, 3400, "300to3400"), (3400, sample_rate / 2 + 1, "above3400")]:
        metrics[f"spectral_power_fraction_{name}_hz"] = float(psd[(freqs >= lo) & (freqs < hi)].sum() / denominator) if denominator else None
    positive = psd[1:]
    metrics["spectral_centroid_hz"] = float(np.dot(freqs, psd) / denominator) if denominator else None
    metrics["spectral_flatness_excluding_dc"] = float(np.exp(np.mean(np.log(np.maximum(positive, 1e-30)))) / np.mean(positive)) if np.any(positive) else None
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return metrics, histogram


def select_forensics(rows: list[dict]) -> tuple[list[dict], dict]:
    suspicious, details = select_rows(rows)
    selected = [{**row, "forensic_group": "selected_anomaly", "matched_to": "", "control_to_target_duration_ratio": None} for row in suspicious]
    ids = {r["audio_file"] for r in suspicious}
    targets = sorted((r for r in suspicious if "known_class_total_mode3_below_5s" in r["selection_reasons"]), key=lambda r: r["audio_file"])
    candidates = [r for r in rows if r["audio_file"] not in ids and r["status"] == "ok" and r["subtype"] == "PCM_16"
                  and r["speaker_id"] != "unknown" and -35 <= float(r["max_channel_rms_dbfs"]) <= -15
                  and float(r.get("vad_mode3_speech_fraction") or 0) >= .8]
    for target in targets:
        if not candidates:
            break
        match = min(candidates, key=lambda r: (abs(math.log(float(r["duration_seconds"]) / float(target["duration_seconds"]))), r["audio_file"]))
        candidates.remove(match)
        selected.append({**match, "forensic_group": "duration_matched_control", "matched_to": target["audio_file"],
                         "control_to_target_duration_ratio": float(match["duration_seconds"]) / float(target["duration_seconds"]),
                         "selection_reasons": ["duration_match_to_known_class_total_mode3_below_5s"]})
    ratios = [r["control_to_target_duration_ratio"] for r in selected if r["matched_to"]]
    return selected, {**details, "anomaly_files": len(suspicious), "control_files": len(selected) - len(suspicious),
                      "control_to_target_duration_ratio": {"min": min(ratios), "median": float(np.median(ratios)), "max": max(ratios), "within_twenty_percent_files": sum(.8 <= r <= 1.2 for r in ratios)},
                      "control_definition": "Known, PCM16, RMS in [-35,-15]dBFS, mode3 fraction>=0.8. Unique greedy nearest available log-duration match for every low-class-VAD file, UUID order. Actual ratios reported: many short targets lack close matches. Algorithmic controls, not verified speech ground truth."}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["audio_file", "speaker_id", "status", "error", "forensic_group", "matched_to"]
    columns += sorted(set().union(*(row.keys() for row in rows)) - set(columns))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def verify_sparse_coverage(source_rows: list[dict], results: list[dict], threshold: int = 64) -> dict:
    """Use complete-manifest zero counts to verify exact-audit sparse coverage.

    Original channels are identical in this dataset, but strongest-channel zero
    fractions are used explicitly rather than assuming that fact downstream.
    """
    expected = {}
    for row in source_rows:
        if row["status"] != "ok" or str(row["has_nonzero_signal"]).lower() != "true":
            continue
        zero_fraction = json.loads(row["channel_zero_fraction"])[int(row["analysis_channel_index"])]
        estimated = int(row["decoded_frames"]) * (1 - zero_fraction)
        count = round(estimated)
        if abs(count - estimated) > 1e-5:
            raise ValueError("Manifest zero fraction cannot reconstruct an integer sample count")
        if count <= threshold:
            expected[row["audio_file"]] = count
    observed = {row["audio_file"]: row["nonzero_samples"] for row in results if row["status"] == "ok"}
    missing = sorted(set(expected) - set(observed))
    mismatch = sorted(name for name, count in expected.items() if name in observed and observed[name] != count)
    return {"threshold_nonzero_samples": threshold, "source_rows": len(source_rows), "eligible_nonzero_source_rows": sum(str(r["has_nonzero_signal"]).lower() == "true" and r["status"] == "ok" for r in source_rows),
            "matching_global_files": len(expected), "all_matched_files_exactly_audited": not missing and not mismatch,
            "missing_files": missing, "count_mismatch_files": mismatch,
            "definition": "Reconstruct strongest-channel nonzero sample count from full-manifest channel_zero_fraction and decoded_frames, round with <1e-5 tolerance, then verify against exact PCM16 forensic counts. Review threshold only, not automatic exclusion or speech label."}


def _plot(selected: list[dict], rows: list[dict], data_dir: Path, figures_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    good = [r for r in rows if r["status"] == "ok"]
    colors = {"selected_anomaly": "#b55936", "duration_matched_control": "#287984"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for group, color in colors.items():
        members = [r for r in good if r["forensic_group"] == group]
        axes[0].scatter([r["rms_dbfs"] for r in members], [r["unique_pcm_codes"] for r in members], s=22, alpha=.7, color=color, label="selected anomaly" if group == "selected_anomaly" else "nearest available duration control")
        axes[1].scatter([r["zero_fraction"] for r in members], [r["isolated_nonzero_sample_fraction"] for r in members], s=22, alpha=.7, color=color)
    axes[0].set(xlabel="Full-file RMS (dBFS)", ylabel="Distinct stored PCM16 codes", yscale="log")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Exact zero sample fraction", ylabel="Isolated single-sample fraction among nonzero samples", ylim=(-.03, 1.03))
    fig.suptitle("Selected low-level files and algorithmic comparison controls")
    paths = [figures_dir / "forensic_quantization.png"]
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    target = next(r for r in selected if r["speaker_id"] == TARGET_SPEAKER)
    control = next(r for r in selected if r["matched_to"] == target["audio_file"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for index, row in enumerate([target, control]):
        with open_audio(data_dir / row["audio_file"]) as stream:
            x = stream.read(dtype="float64", always_2d=True)[:, int(row["analysis_channel_index"])]
            sr = stream.samplerate
        frame = max(1, round(.1 * sr))
        energy = np.mean(np.pad(x, (0, (-len(x)) % frame)).reshape(-1, frame) ** 2, axis=1)
        axes[index, 0].plot(np.arange(len(energy)) * .1, 10 * np.log10(np.maximum(energy, 1e-20)), lw=.8)
        axes[index, 0].set(xlabel="Time (seconds)", ylabel="100 ms power (dBFS)", ylim=(-140, 0))
        segment_len = min(len(x), 8 * sr)
        # Select the loudest 8-second region on a 1-second grid, then normalize only for display.
        energies = [np.sum(x[start:start + segment_len] ** 2) for start in range(0, max(1, len(x) - segment_len + 1), sr)]
        start = int(np.argmax(energies)) * sr
        clip = x[start:start + segment_len]
        scale = max(float(np.max(np.abs(clip))), 1e-12)
        f, t, power = signal.spectrogram(clip / scale, sr, nperseg=min(400, len(clip)), noverlap=min(240, len(clip) // 2), nfft=512)
        db = 10 * np.log10(np.maximum(power, 1e-15))
        axes[index, 1].pcolormesh(t + start / sr, f / 1000, db - db.max(), vmin=-70, vmax=0, shading="auto", cmap="magma")
        axes[index, 1].set(xlabel="Time in file (seconds)", ylabel="Frequency (kHz)")
        axes[index, 0].set_title(("Sparse low-level file " if index == 0 else "Duration-matched control ") + row["audio_file"][:8])
        axes[index, 1].set_title("Peak-normalized spectrogram, relative dB (display only)")
    paths.append(figures_dir / "forensic_target_spectrogram.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    return paths


def run(manifest: Path, data_dir: Path, output: Path, summary_path: Path, figures_dir: Path) -> dict:
    started = time.monotonic()
    with manifest.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    selected, selection = select_forensics(source_rows)
    results, histograms = [], {}
    for row in selected:
        result = {key: row[key] for key in ("audio_file", "speaker_id", "forensic_group", "matched_to", "control_to_target_duration_ratio", "selection_reasons", "input_sha256")}
        result.update(status="error", error="")
        try:
            path = data_dir / row["audio_file"]
            if file_sha256(path) != row["input_sha256"]:
                raise ValueError("Raw SHA256 differs from manifest")
            with open_audio(path) as stream:
                if stream.subtype != "PCM_16":
                    raise ValueError("Sample-code diagnostics require exact PCM16 source")
                codes = stream.read(dtype="int16", always_2d=True)[:, int(row["analysis_channel_index"])]
                sr = stream.samplerate
            if len(codes) != int(row["decoded_frames"]):
                raise ValueError("Frame count differs from manifest")
            metrics, histogram = sample_metrics(codes, sr)
            result.update(metrics, status="ok")
            histograms[row["audio_file"]] = histogram
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        if len(results) % 25 == 0:
            print(f"Forensics: {len(results)}/{len(selected)}; {time.monotonic() - started:.1f}s", flush=True)
    _write_csv(output, results)
    histogram_path = output.with_name("forensics_code_histograms.json")
    histogram_path.write_text(json.dumps(histograms, separators=(",", ":")), encoding="utf-8")
    figures = _plot(selected, results, data_dir, figures_dir)
    good = [r for r in results if r["status"] == "ok"]
    sparse_coverage = verify_sparse_coverage(source_rows, results)
    near_empty = [r for r in good if r["nonzero_samples"] <= sparse_coverage["threshold_nonzero_samples"]]
    near_empty_path = output.with_name("forensics_near_empty.csv")
    _write_csv(near_empty_path, near_empty)
    metrics = ["unique_pcm_codes", "zero_fraction", "isolated_nonzero_sample_fraction", "nonzero_run_max_samples", "observed_code_entropy_bits", "rms_dbfs", "lag1_correlation", "spectral_flatness_excluding_dc", "spectral_power_fraction_above3400_hz", "diagnostic_peak_safe_gain_db", "diagnostic_post_gain_rms_dbfs"]
    groups = {name: [r for r in good if r["forensic_group"] == name] for name in ("selected_anomaly", "duration_matched_control")}
    groups["target_44ea12a5"] = [r for r in good if r["speaker_id"] == TARGET_SPEAKER]
    for speaker in selection["known_classes_total_baseline_mode3_below_5s"]:
        if speaker != TARGET_SPEAKER:
            groups[f"low_vad_class_{speaker}"] = [r for r in good if r["speaker_id"] == speaker]
    summary = {
        "audit_kind": "stored_pcm_code_forensics", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection, "successful_files": len(good), "failed_files": len(results) - len(good),
        "elapsed_seconds": time.monotonic() - started,
        "groups": {name: {"files": len(members), "seconds": sum(r["duration_seconds"] for r in members),
                           "metrics": {key: {"min": min(r[key] for r in members if r[key] is not None),
                                             "median": float(np.median([r[key] for r in members if r[key] is not None])),
                                             "max": max(r[key] for r in members if r[key] is not None)} for key in metrics},
                           "files_with_at_most_3_codes": sum(r["unique_pcm_codes"] <= 3 for r in members),
                           "target_rms_reached_by_peak_safe_diagnostic_files": sum(r["diagnostic_target_minus20_dbfs_reached"] for r in members)}
                   for name, members in groups.items()},
        "target_files": [r for r in good if r["speaker_id"] == TARGET_SPEAKER],
        "near_empty_global_coverage": sparse_coverage,
        "near_empty_files": [{key: r[key] for key in ("audio_file", "speaker_id", "duration_seconds", "nonzero_samples", "nonzero_sample_support_seconds", "unique_pcm_codes", "zero_fraction")} for r in near_empty],
        "failures": [r for r in results if r["status"] != "ok"],
        "input_manifest_sha256": file_sha256(manifest), "output_csv_sha256": file_sha256(output),
        "histogram_sha256": file_sha256(histogram_path), "figure_sha256": {p.name: file_sha256(p) for p in figures},
        "near_empty_csv_sha256": file_sha256(near_empty_path),
        "code_sha256": {Path(__file__).name: file_sha256(Path(__file__)), "vad_sensitivity.py": file_sha256(Path(__file__).with_name("vad_sensitivity.py")),
                        "io.py": file_sha256(Path(__file__).parents[1] / "audio/io.py")},
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
        "definitions": {"codes": "Exact PCM16 codes from strongest original channel; all samples of selected files, no resampling or denoising.",
                        "entropy": "Empirical Shannon entropy over stored amplitude codes, including zero. Alphabet/span bits are descriptive logarithms, NOT ADC ENOB, speech information or recoverability.",
                        "run_length": "Contiguous exact-zero/nonzero samples; isolated fraction counts nonzero samples with zero-valued neighbors (file boundaries treated as zero).",
                        "spectra": "Welch PSD, Hann 512 sample segments/256 overlap, per-segment mean removed. Fractions over PSD bins; flatness excludes DC; gain-invariant except numerical floor.",
                        "energy_concentration": "Share of total energy in ceil(1% of 20ms frames), including zero-padding of last frame.",
                        "gain": "Diagnostic only: max(1,min(0.1/RMS,0.95/peak)); no 40dB cap. Applied algebraically to RMS/peak. Raw files unchanged and zero samples remain zero.",
                        "controls": selection["control_definition"]},
        "limitations": ["No auditory judgement, intelligibility label, speaker verification or speech presence ground truth was obtained.",
                        "Sparse single-sample events are structurally impulse-like. Their cause and presence of encoded speech cannot be established from these statistics alone.",
                        "Normalization cannot recreate amplitude values that were already quantized to zero; it also increases stored noise and artifacts.",
                        "Selected anomalies and controls are not a random sample; findings cannot be extrapolated as prevalence across the full dataset.",
                        "Controls use nearest available duration, not exact matching; many short targets have much longer controls. Duration ratios are explicit. Controls are algorithmically normal-level/high-VAD and not matched on speaker, recording device, session or source.",
                        "Spectral similarity or broadband power is not sufficient to establish a source, editing operation or recording defect."]}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"successful_files": len(good), "failed_files": len(results) - len(good), "seconds": summary["elapsed_seconds"]}), flush=True)
    return summary
