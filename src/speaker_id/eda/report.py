"""Render a local, evidence-based Persian EDA report from audit artifacts.

This module does not decode audio, train models, contact services or infer speech
content. Signal heuristics retain the definitions of the waveform audit.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"known": "#2563a6", "unknown": "#d57537"}
GROUP_NAMES = {"known": "شناخته‌شده", "unknown": "unknown"}
FIGURES = [
    ("duration", "توزیع مدت و صوت‌های کوتاه", "مدت از تعداد نمونه‌های واقعاً decodeشده محاسبه می‌شود؛ هر فایل وزن برابر دارد."),
    ("class_support", "پشتیبانی کلاس‌های شناخته‌شده", "واحد این نمودار گویندهٔ شناخته‌شده است. کلاس unknown به‌دلیل تجمیع هویت‌های متعدد در این نمودار نیست."),
    ("quality_flags", "پرچم‌های کیفیت", "یک فایل ممکن است چند پرچم داشته باشد؛ جمع ستون‌ها برابر تعداد فایل‌ها نیست. درصد هر گروه بر تعداد کل فایل‌های همان گروه تقسیم شده است."),
    ("signal_levels", "سطح سیگنال و انرژی فریم‌ها", "انرژی روی کانال اصلی با بیشترین RMS سنجیده شده است. سهم انرژی کمتر از −50 dBFS، VAD یا برآورد SNR نیست. در کسر و صدک انرژی، همهٔ فریم‌های ۲۰ میلی‌ثانیه‌ای و فریم ناقص انتهایی وزن برابر دارند؛ مجموع ثانیه‌های انرژی، طول واقعی فریم آخر را حساب می‌کند. RMS صفر در کف −160 dBFS نمایش داده می‌شود."),
    ("channels", "اثر کانال‌ها بر تبدیل به mono", "همبستگی فقط در فایل‌های چندکاناله با مقدار تعریف‌شده نمایش داده می‌شود. اختلاف کانال‌ها و افت downmix پیش از انتخاب سیاست تبدیل باید بررسی شوند."),
    ("duration_level", "رابطهٔ مدت و سطح سیگنال", "هر نقطه یک فایل decodeشده با مدت مثبت و RMS متناهی است؛ محور مدت لگاریتمی است. سطح کم، به‌تنهایی نشانهٔ نبود گفتار نیست."),
]
VAD_FIGURE = ("vad_predictions", "پیش‌بینی فعالیت گفتار با WebRTC VAD", "دو حالت مستقل WebRTC روی فریم‌های ۳۰ میلی‌ثانیه‌ای کانال با بیشترین RMS اجرا شده‌اند. کسر گفتار از مدت تحلیل‌شده محاسبه می‌شود؛ tail کوتاه‌تر از یک فریم و فایل فاقد فریم کامل از نمودار حذف شده‌اند. این خروجی برچسب واقعی گفتار یا تأیید کیفیت نیست.")
VAD_COLUMNS = {"vad_status", "vad_analyzed_seconds", "vad_mode1_speech_seconds", "vad_mode3_speech_seconds", "vad_mode1_speech_fraction", "vad_mode3_speech_fraction"}


def _escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def _relative(path: Path, output_dir: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output_dir.resolve()).replace("\\", "/"), safe="/:.-_")


def _number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def _finite(values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def _flags(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("quality_flags must be a JSON array of strings")
    return parsed


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"audio_file": str, "speaker_id": str, "input_sha256": str, "pcm_sha256": str})
    required = {
        "audio_file", "speaker_id", "status", "quality_flags", "usable_for_training", "duration_seconds",
        "mono_rms_dbfs", "max_channel_rms_dbfs", "channels", "channel_correlation", "channel_rms_imbalance_db",
        "downmix_attenuation_db", "below_minus50_dbfs_fraction", "energy_above_minus50_dbfs_seconds",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame["audio_file"].duplicated().any() or frame[["audio_file", "speaker_id"]].isna().any().any():
        raise ValueError("Manifest requires exactly one labeled row per audio_file")
    if not set(frame["status"]).issubset({"ok", "error"}):
        raise ValueError("Unrecognized waveform status")
    frame["usable_for_training"] = _boolean(frame["usable_for_training"])
    frame["flag_list"] = frame["quality_flags"].map(_flags)
    frame["group"] = np.where(frame["speaker_id"].eq("unknown"), "unknown", "known")
    return frame


def per_speaker(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for speaker_id, subset in frame.groupby("speaker_id", sort=True):
        good = subset[subset["status"].eq("ok")]
        usable = good[good["usable_for_training"]]
        all_flags = Counter(flag for flags in subset["flag_list"] for flag in set(flags))
        rows.append({
            "speaker_id": speaker_id,
            "label_group": "unknown" if speaker_id == "unknown" else "known",
            "file_count": len(subset),
            "decoded_file_count": len(good),
            "technically_usable_file_count": len(usable),
            "decoded_duration_seconds": good["duration_seconds"].sum(min_count=1),
            "technically_usable_duration_seconds": usable["duration_seconds"].sum(),
            "minimum_duration_seconds": good["duration_seconds"].min(),
            "median_duration_seconds": good["duration_seconds"].median(),
            "maximum_duration_seconds": good["duration_seconds"].max(),
            "median_max_channel_rms_dbfs": good["max_channel_rms_dbfs"].replace([-np.inf, np.inf], np.nan).median(),
            "energy_above_minus50_dbfs_seconds": good["energy_above_minus50_dbfs_seconds"].sum(min_count=1),
            "below_1_second_file_count": int(good["duration_seconds"].lt(1).sum()),
            "below_5_seconds_file_count": int(good["duration_seconds"].lt(5).sum()),
            "flagged_file_count": int(subset["flag_list"].map(bool).sum()),
            "signal_quality_flagged_file_count": int(subset["flag_list"].map(lambda flags: bool(set(flags) - {"extension_container_mismatch"})).sum()),
            "extension_container_mismatch_count": all_flags.get("extension_container_mismatch", 0),
            "quality_flag_counts": json.dumps(dict(sorted(all_flags.items())), ensure_ascii=False),
        })
        if VAD_COLUMNS.issubset(frame.columns):
            vad = good[good["vad_status"].eq("ok") & good["vad_analyzed_seconds"].gt(0)]
            rows[-1]["vad_analyzed_seconds"] = vad["vad_analyzed_seconds"].sum(min_count=1)
            for mode in [1, 3]:
                rows[-1][f"vad_mode{mode}_predicted_speech_seconds"] = vad[f"vad_mode{mode}_speech_seconds"].sum(min_count=1)
                rows[-1][f"vad_mode{mode}_median_speech_fraction"] = vad[f"vad_mode{mode}_speech_fraction"].median()
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5bfc9",
        "axes.labelcolor": "#243447", "xtick.color": "#445464", "ytick.color": "#445464",
        "grid.alpha": 0.25, "savefig.facecolor": "#ffffff",
    })


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def _empty_axis(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color="#556477")


def figures(frame: pd.DataFrame, speakers: pd.DataFrame, output_dir: Path) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    good = frame[frame["status"].eq("ok")].copy()
    grouped = {group: good[good["group"].eq(group)] for group in COLORS}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    upper = max(1.0, float(good["duration_seconds"].max()))
    boundaries = np.linspace(0, upper * 1.001, 45)
    bins = [0, 0.001, 1, 5, 15, 30, 60, 120, np.inf]
    bin_labels = ["<1 ms", "1 ms–1 s", "1–5 s", "5–15 s", "15–30 s", "30–60 s", "60–120 s", "≥120 s"]
    for index, (group, subset) in enumerate(grouped.items()):
        values = _finite(subset["duration_seconds"])
        axes[0].hist(values, bins=boundaries, histtype="step", linewidth=2, color=COLORS[group], label=f"{group} (n={len(values):,})")
        counts = np.histogram(values, bins=bins)[0]
        axes[1].bar(np.arange(8) + (index - 0.5) * 0.36, counts, width=0.36, color=COLORS[group], label=group)
    axes[0].set(title="Decoded duration", xlabel="Seconds", ylabel="Files")
    axes[0].legend()
    axes[1].set(title="Duration bins [left, right)", ylabel="Files (symlog scale)", yscale="symlog", xticks=np.arange(8), xticklabels=bin_labels)
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend()
    _save(fig, output_dir, "duration")

    known_speakers = speakers[speakers["label_group"].eq("known")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    counts = known_speakers["technically_usable_file_count"].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color=COLORS["known"])
    for index, count in enumerate(counts.values):
        axes[0].text(index, count, str(count), ha="center", va="bottom")
    axes[0].set(title="Files per known speaker", xlabel="Technically usable files", ylabel="Known speakers")
    durations = np.sort(known_speakers["technically_usable_duration_seconds"].fillna(0).to_numpy()) / 60
    axes[1].plot(np.arange(1, len(durations) + 1), durations, color=COLORS["known"])
    axes[1].set(title="Available duration per known speaker", xlabel="Speaker rank by usable duration", ylabel="Minutes (technical eligibility only)")
    axes[1].grid(axis="y")
    _save(fig, output_dir, "class_support")

    flag_counts = {group: Counter(flag for flags in frame.loc[frame["group"].eq(group), "flag_list"] for flag in set(flags)) for group in COLORS}
    flags = sorted(set().union(*[set(counts) for counts in flag_counts.values()]), key=lambda flag: sum(counts[flag] for counts in flag_counts.values()))
    fig, axis = plt.subplots(figsize=(12, max(3, len(flags) * 0.4 + 1.3)), layout="constrained")
    for index, group in enumerate(COLORS):
        denominator = int(frame["group"].eq(group).sum())
        values = [100 * flag_counts[group][flag] / max(1, denominator) for flag in flags]
        axis.barh(np.arange(len(flags)) + (index - 0.5) * 0.38, values, height=0.38, label=f"{group} (n={denominator:,})", color=COLORS[group])
        for position, flag, value in zip(np.arange(len(flags)) + (index - 0.5) * 0.38, flags, values):
            if flag_counts[group][flag]:
                axis.text(value + 0.08, position, str(flag_counts[group][flag]), va="center", fontsize=8, color=COLORS[group])
    if not flags:
        _empty_axis(axis, "No quality flags")
    axis.set(title="Quality flags (not mutually exclusive)", xlabel="Percent of all files in each label group", yticks=np.arange(len(flags)), yticklabels=flags)
    axis.legend()
    axis.grid(axis="x")
    axis.margins(x=0.2)
    _save(fig, output_dir, "quality_flags")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    all_levels = _finite(good["max_channel_rms_dbfs"])
    lower = min(-60.0, float(np.min(all_levels))) if len(all_levels) else -160.0
    upper = max(0.0, float(np.max(all_levels))) if len(all_levels) else 0.0
    level_bins = np.linspace(lower - 0.01, upper + 0.01, 51)
    for group, subset in grouped.items():
        levels = _finite(subset["max_channel_rms_dbfs"])
        low_energy = _finite(subset["below_minus50_dbfs_fraction"])
        axes[0].hist(levels, bins=level_bins, histtype="step", linewidth=2, color=COLORS[group], label=f"{group} (finite n={len(levels):,})")
        axes[1].hist(low_energy, bins=np.linspace(0, 1, 26), histtype="step", linewidth=2, color=COLORS[group], label=f"{group} (n={len(low_energy):,})")
    axes[0].set(title="RMS of strongest original channel", xlabel="dBFS (nonfinite values excluded)", ylabel="Files")
    axes[1].set(title="Low-energy frame share (energy proxy)", xlabel="Fraction of frames below −50 dBFS", ylabel="Files")
    for axis in axes:
        axis.legend()
    _save(fig, output_dir, "signal_levels")

    stereo = good[good["channels"].gt(1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    correlations = _finite(stereo["channel_correlation"])
    axes[0].hist(correlations, bins=np.linspace(-1.001, 1.001, 45), color=COLORS["known"])
    axes[0].set(title=f"Channel correlation (finite n={len(correlations):,})", xlabel="Correlation coefficient", ylabel="Files (symlog scale)", yscale="symlog")
    attenuation = _finite(stereo["downmix_attenuation_db"])
    if len(attenuation):
        # A fixed range would hide unusual cancellation, so retain all finite values.
        axes[1].hist(attenuation, bins=35, color=COLORS["known"])
    else:
        _empty_axis(axes[1], "No finite stereo downmix measurements")
    axes[1].set(title=f"Downmix attenuation (finite n={len(attenuation):,})", xlabel="dB relative to strongest channel", ylabel="Files (symlog scale)", yscale="symlog")
    _save(fig, output_dir, "channels")

    fig, axis = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for group, subset in grouped.items():
        selected = subset[subset["duration_seconds"].gt(0) & np.isfinite(subset["max_channel_rms_dbfs"])]
        axis.scatter(selected["duration_seconds"], selected["max_channel_rms_dbfs"], s=9, alpha=0.38, color=COLORS[group], label=f"{group} (n={len(selected):,})", rasterized=True)
    axis.set(title="Duration versus strongest-channel RMS", xlabel="Decoded seconds (log scale)", ylabel="RMS, dBFS", xscale="log")
    axis.grid()
    axis.legend()
    _save(fig, output_dir, "duration_level")

    if VAD_COLUMNS.issubset(frame.columns):
        analyzed = good[good["vad_status"].eq("ok") & good["vad_analyzed_seconds"].gt(0)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for axis, mode in zip(axes, [1, 3]):
            for group in COLORS:
                values = _finite(analyzed.loc[analyzed["group"].eq(group), f"vad_mode{mode}_speech_fraction"])
                axis.hist(values, bins=np.linspace(0, 1, 26), histtype="step", linewidth=2, color=COLORS[group], label=f"{group} (n={len(values):,})")
            axis.set(title=f"WebRTC VAD mode {mode}: predicted speech", xlabel="Predicted speech fraction / analyzed duration", ylabel="Files")
            axis.legend()
        _save(fig, output_dir, "vad_predictions")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    return '<div class="table-scroll"><table><thead><tr>' + "".join(f"<th>{_escaped(item)}</th>" for item in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _file_table(frame: pd.DataFrame, output_dir: Path, data_dir: Path, limit: int = 12) -> str:
    rows = []
    for _, row in frame.head(limit).iterrows():
        file_link = f'<a class="mono" href="{_relative(data_dir / row.audio_file, output_dir)}">{_escaped(row.audio_file)}</a>'
        rows.append([file_link, f'<span class="mono">{_escaped(row.speaker_id)}</span>', _number(row.duration_seconds, 4), _number(row.max_channel_rms_dbfs), _escaped(", ".join(row.flag_list)) or "—"])
    return _table(["فایل اصلی", "برچسب", "مدت (s)", "RMS کانال قوی‌تر (dBFS)", "پرچم"], rows) if rows else '<p class="muted">موردی در این دسته وجود ندارد.</p>'


def _statistics(frame: pd.DataFrame, speakers: pd.DataFrame) -> dict:
    by_group = {}
    for group in ["all", "known", "unknown"]:
        subset = frame if group == "all" else frame[frame["group"].eq(group)]
        good = subset[subset["status"].eq("ok")]
        usable = good[good["usable_for_training"]]
        by_group[group] = {
            "files": len(subset), "decoded_files": len(good), "decode_errors": len(subset) - len(good),
            "technically_usable_files": len(usable), "decoded_hours": float(good["duration_seconds"].sum() / 3600),
            "technically_usable_hours": float(usable["duration_seconds"].sum() / 3600),
            "median_duration_seconds": float(good["duration_seconds"].median()) if len(good) else None,
            "below_one_second_files": int(good["duration_seconds"].lt(1).sum()),
            "below_five_seconds_files": int(good["duration_seconds"].lt(5).sum()),
            "quality_flag_counts": dict(Counter(flag for flags in subset["flag_list"] for flag in set(flags))),
        }
        if VAD_COLUMNS.issubset(frame.columns):
            vad = good[good["vad_status"].eq("ok") & good["vad_analyzed_seconds"].gt(0)]
            by_group[group]["vad"] = {
                "files_with_at_least_one_analyzed_frame": len(vad),
                "analyzed_hours": float(vad["vad_analyzed_seconds"].sum() / 3600),
                "mode1_predicted_speech_hours": float(vad["vad_mode1_speech_seconds"].sum() / 3600),
                "mode3_predicted_speech_hours": float(vad["vad_mode3_speech_seconds"].sum() / 3600),
                "mode1_median_speech_fraction": float(vad["vad_mode1_speech_fraction"].median()) if len(vad) else None,
                "mode3_median_speech_fraction": float(vad["vad_mode3_speech_fraction"].median()) if len(vad) else None,
            }
    known = speakers[speakers["label_group"].eq("known")]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_scope": "Full available waveform manifest including WebRTC VAD predictions, duplicate audit and available supplemental archive/gain/split artifacts; no human listening, speech ground truth, speaker embeddings or verified recording-session identities.",
        "full_eda_completed": False,
        "known_speakers": len(known), "label_values": len(speakers),
        "known_speakers_with_zero_technically_usable_files": int(known["technically_usable_file_count"].eq(0).sum()),
        "known_support_distribution": {str(key): int(value) for key, value in known["technically_usable_file_count"].value_counts().sort_index().items()},
        "by_label_group": by_group,
    }


def build_report(manifest_path: Path, summary_path: Path, output_dir: Path, data_dir: Path) -> dict:
    frame = read_manifest(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("coverage", {}).get("limit_applied") is not None:
        raise ValueError("A limited audit cannot be used to build the full-dataset report")
    expected_rows = summary.get("coverage", {}).get("labeled_files")
    if expected_rows is not None and expected_rows != len(frame):
        raise ValueError("Manifest row count differs from signal summary")
    expected_fingerprint = summary.get("input_fingerprint", {}).get("labeled_audio_sha256")
    if expected_fingerprint and "input_sha256" in frame:
        digest = hashlib.sha256()
        for _, row in frame.sort_values("audio_file").iterrows():
            input_hash = None if pd.isna(row.input_sha256) else row.input_sha256
            digest.update(json.dumps([row.audio_file, row.speaker_id, input_hash], separators=(",", ":")).encode())
            digest.update(b"\n")
        if digest.hexdigest() != expected_fingerprint:
            raise ValueError("Manifest input fingerprint differs from signal summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    speakers = per_speaker(frame)
    speakers.to_csv(output_dir / "per_speaker.csv", index=False, encoding="utf-8")
    figures(frame, speakers, output_dir / "figures")
    stats = _statistics(frame, speakers)
    stats["report_version"] = "eda-report-v1"
    stats["report_builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stats["renderer_packages"] = {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    stats["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stats["signal_summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    stats["input_fingerprint"] = summary.get("input_fingerprint")
    duplicate_path = output_dir / "duplicate_summary.json"
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8")) if duplicate_path.exists() else None
    if duplicate is not None:
        if duplicate.get("file_count") != len(frame):
            raise ValueError("Duplicate audit row count differs from full waveform manifest")
        if duplicate.get("input_manifest_sha256") not in {None, stats["manifest_sha256"]}:
            raise ValueError("Duplicate audit used a different waveform manifest")
        stats["duplicate_summary_sha256"] = hashlib.sha256(duplicate_path.read_bytes()).hexdigest()
        stats["duplicate_audit"] = {key: duplicate.get(key) for key in ["file_count", "no_signal_files", "exact_file_groups", "exact_decoded_groups", "verified_acoustic_pairs", "verified_cross_label_pairs", "unverified_candidates"]}
    supplements = {}
    for name in ["listening_summary.json", "split_summary.json", "archive_anomalies_summary.json", "vad_sensitivity_summary.json"]:
        path = output_dir / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = data.get("manifest_sha256", data.get("input_manifest_sha256"))
            if expected not in {None, stats["manifest_sha256"]}:
                raise ValueError(f"{name} used a different waveform manifest")
            supplements[name] = data
    stats["supplementary_summary_sha256"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in supplements}
    (output_dir / "report_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    total = stats["by_label_group"]["all"]
    known = speakers[speakers["label_group"].eq("known")]
    good = frame[frame["status"].eq("ok")]
    exclusions = frame[~frame["usable_for_training"]]
    findings = []
    mismatch_count = total["quality_flag_counts"].get("extension_container_mismatch", 0)
    no_signal_count = total["quality_flag_counts"].get("no_signal", 0)
    findings.append(f"{mismatch_count:,} فایل دارای ناسازگاری پسوند و فرمت شناسایی‌شده است؛ انتخاب decoder باید از محتوای واقعی فایل پیروی کند.")
    findings.append(f"{no_signal_count:,} فایل فاقد هر نمونهٔ غیرصفر است و {total['decode_errors']:,} فایل status=error دارد. این دو شمارش الزاماً دسته‌های مستقل نیستند؛ همهٔ موارد در manifest باقی مانده‌اند.")
    if no_signal_count and "decoded_frames" in frame:
        zero = frame[frame["flag_list"].map(lambda flags: "no_signal" in flags)]
        single_frame = int(zero["decoded_frames"].eq(1).sum())
        longer = int(zero["decoded_frames"].gt(1).sum())
        findings.append(f"نبود سیگنال به فایل‌های تک‌فریم محدود نیست: {single_frame} مورد تک‌فریم و {longer} مورد طولانی‌تر هستند؛ طولانی‌ترین فایل تمام‌صفر {_number(zero['duration_seconds'].max())} ثانیه است.")
    if "channel_identical" in good:
        multichannel = good[good["channels"].gt(1)]
        identical_count = int(_boolean(multichannel["channel_identical"]).sum())
        findings.append(f"در {identical_count:,} مورد از {len(multichannel):,} فایل چندکاناله، همهٔ نمونه‌های کانال‌ها دقیقاً برابرند. برای همین موارد، میانگین‌گیری کانال‌ها سیگنال را عوض نمی‌کند؛ موارد دیگر باید جداگانه ارزیابی شوند.")
    if len(known):
        minimum = int(known["technically_usable_file_count"].min())
        maximum = int(known["technically_usable_file_count"].max())
        findings.append(f"پشتیبانی فنی هر گویندهٔ شناخته‌شده بین {minimum} تا {maximum} فایل است. استقلال ضبط‌ها از شمارش فایل‌ها نتیجه نمی‌شود و اعتبارسنجی باید گروه‌های مرتبط را کنار هم نگه دارد.")
        if minimum < 5:
            findings.append("تقسیم پنج‌fold که در هر بخش ارزیابی از همهٔ گویندگان نمونهٔ غیرصفر داشته باشد، با این پشتیبانی ممکن نیست. تعداد fold باید پس از ممیزی گروه‌های مرتبط انتخاب شود.")
    findings.append(f"{total['below_five_seconds_files']:,} فایل decodeشده کوتاه‌تر از پنج ثانیه است؛ سیاست قطعه‌بندی و ورودی کوتاه باید برای این دامنه رفتار مشخص داشته باشد.")
    provenance = {key: summary[key] for key in ["input_fingerprint", "config", "units_and_definitions", "coverage", "packages", "code_sha256", "limitations"] if key in summary}
    links = [("manifest کامل فایل‌ها", manifest_path), ("جدول کامل هر کلاس", output_dir / "per_speaker.csv"), ("خلاصهٔ ممیزی موج", summary_path), ("آمار همین گزارش", output_dir / "report_stats.json")]
    for name in ["duplicate_pairs.csv", "duplicate_groups.csv", "duplicate_summary.json", "listening_queue.csv", "listening_summary.json", "audio_samples/index.html", "split_support.csv", "split_summary.json", "archive_anomalies.csv", "archive_anomalies_summary.json", "vad_sensitivity.csv", "vad_sensitivity_summary.json"]:
        if (output_dir / name).exists():
            links.append((name, output_dir / name))
    if (manifest_path.parent / "folds.csv").exists():
        links.append(("تقسیم‌های پیشنهادی فایل‌ها", manifest_path.parent / "folds.csv"))
    cards = [("فایل ورودی", f"{total['files']:,}"), ("decode موفق", f"{total['decoded_files']:,}"), ("ساعت صوت decodeشده", _number(total["decoded_hours"])), ("کلاس شناخته‌شده", f"{stats['known_speakers']:,}"), ("عبور از کنترل فنی", f"{total['technically_usable_files']:,}"), ("خطای decode", f"{total['decode_errors']:,}")]
    group_rows = []
    for group in ["known", "unknown", "all"]:
        item = stats["by_label_group"][group]
        group_rows.append([GROUP_NAMES.get(group, "کل"), f"{item['files']:,}", _number(item["decoded_hours"]), _number(item["median_duration_seconds"]), f"{item['technically_usable_files']:,}", str(item["below_one_second_files"]), str(item["below_five_seconds_files"])])
    class_rows = []
    for _, row in known.sort_values(["technically_usable_duration_seconds", "speaker_id"]).head(15).iterrows():
        class_rows.append([f'<span class="mono">{_escaped(row.speaker_id)}</span>', str(row.file_count), str(row.technically_usable_file_count), _number(row.technically_usable_duration_seconds / 60), _number(row.minimum_duration_seconds, 3), str(row.signal_quality_flagged_file_count)])
    unsupported = known[known["technically_usable_file_count"].eq(0)]
    unsupported_text = ''
    if len(unsupported):
        unsupported_rows = [[f'<span class="mono">{_escaped(row.speaker_id)}</span>', str(row.file_count), _number(row.decoded_duration_seconds, 4), _escaped(row.quality_flag_counts)] for _, row in unsupported.iterrows()]
        unsupported_text = '<section class="notice"><h2>مانع مهم: کلاس‌های شناخته‌شده بدون سیگنال قابل استفاده</h2><p>' + str(len(unsupported)) + ' کلاس شناخته‌شده هیچ فایل عبورکرده از کنترل فنی ندارند. از صوت فعلیِ این کلاس‌ها نمی‌توان نمایندهٔ صوتی قابل اتکایی برای هویت ساخت. تقسیم و ارزیابی نباید ادعای پشتیبانی آموزشی همهٔ UUIDها کند؛ این موارد باید با دلیل ثبت و دربارهٔ دادهٔ مفقود یا اصلاح آن پیگیری شوند.</p>' + _table(["speaker_id", "کل فایل برچسب‌دار", "مدت decodeشده (s)", "تعداد پرچم‌ها"], unsupported_rows) + '</section>'
    format_rows = [[_escaped(format_name), f"{sample_rate:,.0f}", str(int(channels)), _escaped(subtype), f"{len(subset):,}"] for (format_name, sample_rate, channels, subtype), subset in good.groupby(["detected_format", "sample_rate_hz", "channels", "subtype"], dropna=False)] if "detected_format" in good else []
    sections = []
    for name, title, caption in FIGURES + ([VAD_FIGURE] if VAD_COLUMNS.issubset(frame.columns) else []):
        sections.append(f'<figure id="{name}"><h2>{title}</h2><img src="figures/{name}.png" alt="{_escaped(title)}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
    vad_text = ''
    if VAD_COLUMNS.issubset(frame.columns):
        vad_rows = []
        for group in ["known", "unknown", "all"]:
            vad = stats["by_label_group"][group]["vad"]
            vad_rows.append([GROUP_NAMES.get(group, "کل"), f"{vad['files_with_at_least_one_analyzed_frame']:,}", _number(vad["analyzed_hours"]), _number(vad["mode1_predicted_speech_hours"]), _number(vad["mode3_predicted_speech_hours"])])
        vad_text = '<section><h2>پوشش اجرای VAD</h2>' + _table(["گروه", "فایل دارای فریم کامل", "ساعت تحلیل‌شده", "ساعت پیش‌بینی گفتار mode 1", "ساعت پیش‌بینی گفتار mode 3"], vad_rows) + '<p class="muted">این ساعات، مجموع پیش‌بینی طبقه‌بند فعالیت گفتار هستند و ساعت گفتارِ تأییدشده نیستند. هر دو حالت روی همان صوت اجرا شده‌اند؛ خروجی آن‌ها با هم جمع نمی‌شود و تفاوتشان نشانهٔ حساسیت به حالت طبقه‌بند است. هیچ پیش‌بینی VAD شرط کنارگذاری از آموزش نیست.</p></section>'
        vad_class_rows = []
        for _, row in known.nsmallest(8, "vad_mode3_predicted_speech_seconds").iterrows():
            vad_class_rows.append([f'<span class="mono">{_escaped(row.speaker_id)}</span>', str(row.technically_usable_file_count), _number(row.technically_usable_duration_seconds), _number(row.vad_mode1_predicted_speech_seconds), _number(row.vad_mode3_predicted_speech_seconds)])
        vad_text += '<section><h2>کلاس‌های دارای کمترین پیش‌بینی گفتار؛ اولویت بازبینی</h2>' + _table(["speaker_id", "فایل عبور فنی", "مدت عبور فنی (s)", "پیش‌بینی mode 1 (s)", "پیش‌بینی mode 3 (s)"], vad_class_rows) + '<p class="muted">فایل غیرصفر می‌تواند برای استخراج هویت صوتی ضعیف باشد. در مقابل، VAD ممکن است گفتار کم‌دامنه را از دست بدهد؛ بنابراین پایین‌بودن این عدد به‌تنهایی نبود گفتار را ثابت نمی‌کند. این رتبه‌بندی فقط اولویت شنیدن و بررسی موج را تعیین می‌کند.</p></section>'
    status_text = '<section id="review_status"><h2>صف شنیداری و وضعیت تقسیم</h2><p>استخراج کلیپ یا تشکیل صف بررسی به معنی شنیده‌شدن فایل نیست. همچنین فایل split باید با دامنه و محدودیت‌های ممیزی گروه‌ها تفسیر شود.</p>'
    if listening := supplements.get("listening_summary.json"):
        status_text += '<p>فایل‌های انتخاب‌شده برای صف شنیداری: ' + _escaped(listening.get("selected_files", "—")) + '؛ وضعیت تکمیل شنیدن: ' + ('ثبت‌شده' if listening.get("listening_completed") else 'تکمیل نشده') + '.</p>'
    if split := supplements.get("split_summary.json"):
        status_text += '<p>تعداد fold ساخته‌شده: ' + _escaped(split.get("actual_folds", "—")) + '؛ تعداد فایل کنارگذاشته‌شده از آموزش: ' + _escaped(split.get("training_excluded_files", "—")) + '. سیاست ارزیابی همهٔ فایل‌های اصلی را نگه می‌دارد. این تقسیم‌ها موقت‌اند و مستقل‌بودن جلسه‌ها و هویت‌های unknown هنوز تأیید نشده است.</p>'
        fold_rows = [[str(row["fold"]), str(row["validation_files"]), str(row["training_eligible_files"]), str(row["validation_known_classes"]), str(row["training_known_classes"])] for row in split.get("folds", [])]
        if fold_rows:
            status_text += _table(["fold", "فایل ارزیابی", "فایل مجاز آموزش", "کلاس شناخته‌شدهٔ ارزیابی", "کلاس شناخته‌شدهٔ آموزش"], fold_rows)
    for name, label in [("listening_summary.json", "پوشش و وضعیت صف شنیداری"), ("split_summary.json", "وضعیت و محدودیت‌های تقسیم")]:
        if name in supplements:
            status_text += '<details><summary>' + label + '</summary><pre>' + _escaped(json.dumps(supplements[name], ensure_ascii=False, indent=2)) + '</pre></details>'
        else:
            status_text += '<p class="muted">' + label + ': خروجی در زمان ساخت گزارش موجود نیست.</p>'
    status_text += '</section>'
    supplementary_text = ''
    if archive := supplements.get("archive_anomalies_summary.json"):
        archive_rows = [["فایل انتخاب‌شده", str(archive["selected_files"])], ["CRC تأییدشده", str(archive["crc_verified_files"])], ["SHA256 یکسان با فایل استخراج‌شده", str(archive["matches_extracted_files"])], ["خطا", str(len(archive.get("errors", [])))]]
        supplementary_text += '<section><h2>راستی‌آزمایی موارد مشکوک در آرشیو اصلی</h2><p>این بررسی هدفمند فایل‌های تمام‌صفر، RMS کمتر از −50 dBFS یا کسر VAD mode 3 کمتر از ۰٫۱ را پوشش می‌دهد؛ راستی‌آزمایی کامل همهٔ اعضای آرشیو انجام نشده است.</p>' + _table(["سنجه", "مقدار"], archive_rows) + '<p>یکسانی hash در فایل‌های تأییدشده نشان می‌دهد همین محتوای مشاهده‌شده در آرشیو ارسالی هم وجود دارد؛ درست‌بودن صوت مورد انتظارِ منبع را ثابت نمی‌کند.</p><details><summary>دامنه و شواهد بررسی آرشیو</summary><pre>' + _escaped(json.dumps(archive, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    if sensitivity := supplements.get("vad_sensitivity_summary.json"):
        gain_rows = [[str(mode), _number(sensitivity["totals"][f"baseline_vad_mode{mode}_speech_seconds"]), _number(sensitivity["totals"][f"post_gain_vad_mode{mode}_speech_seconds"])] for mode in [1, 3]]
        selection = sensitivity["selection"]
        gain_config = sensitivity["config"]
        compact_sensitivity = {key: value for key, value in sensitivity.items() if key != "per_selected_class"}
        supplementary_text += '<section><h2>حساسیت VAD به افزایش دامنه</h2><p>روی ' + str(selection["selected_files"]) + ' فایل غیرصفرِ منتخب با مجموع ' + _number(selection["selected_seconds"]) + ' ثانیه، افزایش دامنه با هدف RMS برابر ' + _number(gain_config["target_rms_dbfs"], 0) + ' dBFS، سقف gain برابر ' + _number(gain_config["maximum_gain_db"], 0) + ' dB و محدودیت peak برابر ' + _number(gain_config["peak_ceiling"]) + ' آزمایش شد. انتخاب بر اساس انرژی یا VAD پایین بوده و نمایندهٔ تصادفی کل داده نیست.</p>' + _table(["حالت VAD", "پیش‌بینی گفتار اولیه (s)", "پیش‌بینی پس از gain (s)"], gain_rows) + '<p>این تفاوت فقط حساسیت خروجی طبقه‌بند به دامنه را نشان می‌دهد. افزایش gain نویز و آثار کوانتیزه‌سازی را نیز تقویت می‌کند و گفتار واقعیِ بازیابی‌شده را اثبات نمی‌کند؛ نتیجه، توصیه به نرمال‌سازی عمومی یا حذف فایل نیست. دادهٔ خام تغییر نکرده است.</p><details><summary>تنظیمات، دامنه و محدودیت آزمون حساسیت</summary><pre>' + _escaped(json.dumps(compact_sensitivity, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    duplicate_text = '<p>خروجی ممیزی تکرار هنوز در زمان ساخت این گزارش موجود نبوده است؛ نبود خروجی به معنی نبود تکرار نیست.</p>'
    if duplicate is not None:
        duplicate_keys = [
            ("file_count", "فایل در ممیزی تکرار"), ("no_signal_files", "فایل فاقد سیگنال"),
            ("exact_file_groups", "گروه hash یکسان فایل"), ("exact_decoded_groups", "گروه hash یکسان PCM"),
            ("signal_exact_file_groups", "گروه hash فایل با سیگنال غیرصفر"),
            ("no_signal_exact_file_groups", "گروه hash فایل فاقد سیگنال"),
            ("verified_acoustic_pairs", "زوج آکوستیکی تأییدشده طبق آزمون موج"),
            ("verified_nonexact_acoustic_pairs", "از زوج‌های تأییدشده: بدون hash کامل یکسان"),
            ("verified_cross_label_pairs", "زوج تأییدشده با برچسب متفاوت"),
            ("unverified_candidates", "نامزد تأییدنشده"),
        ]
        duplicate_rows = [[label, _escaped(duplicate.get(key, "—"))] for key, label in duplicate_keys]
        duplicate_text = '<p>خلاصهٔ زیر مستقیماً از ممیزی تکرار خوانده شده است. گروه‌های hash فایل و PCM هم‌پوشانی دارند و نباید با هم جمع شوند. یکسانی hash با شباهت fingerprint یکسان نیست؛ زوج‌های احتمالی نیاز به راستی‌آزمایی دارند.</p>' + _table(["سنجه", "مقدار"], duplicate_rows) + '<p>تأیید آکوستیکی به معنی گذر از آزمون‌های همبستگی پنجره‌های موج است؛ این خروجی اثبات استقلال جلسهٔ ضبط یا کشف همهٔ هم‌پوشانی‌ها نیست.</p><details><summary>نتیجه و دامنهٔ کامل ممیزی تکرار</summary><pre>' + _escaped(json.dumps(duplicate, ensure_ascii=False, indent=2)) + '</pre></details>'
        if coverage := duplicate.get("fingerprint_coverage"):
            duplicate_text += '<p>از ' + str(coverage["analyzed_signal_files"]) + ' فایل غیرصفرِ تحلیل‌شده، ' + str(coverage["indexed_files_with_landmarks"]) + ' فایل landmark قابل نمایه‌سازی داشتند و ' + str(coverage["signal_files_without_landmarks_count"]) + ' فایل بدون landmark باقی ماندند. نبود تطابق برای این دسته تأیید استقلال نیست؛ حساسیت جست‌وجو به مدت کوتاه و دامنهٔ بسیار کم محدود است.</p>'
    document = f'''<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ممیزی دادهٔ تشخیص گوینده — IAAA 2026</title>
<style>
:root{{--ink:#203247;--muted:#596c7d;--blue:#2563a6;--line:#dce3e9;--paper:#fff;--bg:#f3f5f7;--warm:#fff5e6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.95 Tahoma,Arial,sans-serif}}main{{max-width:1240px;margin:auto;padding:32px 28px 64px}}header{{padding:26px 32px;background:#183a58;color:white;border-radius:16px}}h1{{font-size:30px;line-height:1.6;margin:0 0 10px}}h2{{font-size:22px;margin:0 0 14px}}h3{{font-size:18px}}p{{margin:10px 0}}.eyebrow{{font-size:13px;letter-spacing:.08em;color:#b9d8ef}}.lead{{color:#e5eef5;max-width:1000px}}a{{color:var(--blue);text-underline-offset:4px}}header a{{color:#d7eafd}}nav{{display:flex;flex-wrap:wrap;gap:8px 20px;padding:16px 0}}nav a{{font-size:14px}}section,figure{{margin:22px 0;padding:26px;background:var(--paper);border:1px solid var(--line);border-radius:13px}}.cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:22px}}.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:17px 14px}}.card strong{{display:block;font:700 29px/1.5 Arial,sans-serif;color:#244f74}}.card span{{font-size:12px;color:var(--muted)}}.notice{{border-right:5px solid #d99835;background:var(--warm)}}.muted,figcaption{{color:var(--muted);font-size:14px}}img{{display:block;width:100%;height:auto;direction:ltr}}figcaption{{padding-top:12px}}.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 12px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#edf2f6;white-space:nowrap}}tr:nth-child(even){{background:#fafbfd}}.mono,code{{font:12px/1.7 Consolas,monospace;direction:ltr;unicode-bidi:isolate;display:inline-block}}pre{{direction:ltr;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:18px;border-radius:8px;font:12px/1.65 Consolas,monospace}}details{{margin:15px 0}}summary{{cursor:pointer;color:var(--blue)}}ul{{padding-right:23px}}li{{padding:4px 0}}.files{{display:flex;flex-wrap:wrap;gap:10px 22px}}footer{{font-size:12px;color:var(--muted);margin-top:26px}}@media(max-width:900px){{.cards{{grid-template-columns:repeat(3,1fr)}}main{{padding:18px 12px}}section,figure,header{{padding:20px}}h1{{font-size:24px}}}}@media(max-width:520px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white}}main{{padding:0}}section,figure{{break-inside:avoid}}nav{{display:none}}a{{text-decoration:none}}details{{display:block}}}}
</style></head><body><main>
<header><div class="eyebrow">IAAA 2026 · SPEAKER IDENTIFICATION · EDA v1</div><h1>ممیزی صوت، کیفیت داده و پشتیبانی کلاس‌ها</h1><p class="lead">گزارش محلی و قابل بازتولید از دادهٔ موجود. اعداد از manifest ممیزی محاسبه شده‌اند؛ دادهٔ خام تغییر نکرده و در این مرحله آموزشی اجرا نشده است.</p><p class="lead">وضعیت: ممیزی فنی موج و خروجی‌های موجود تکرار؛ بررسی شنیداری، ساختار embedding و استقلال منابع هنوز تکمیل نشده‌اند.</p></header>
<nav><a href="#overview">پوشش و آمار</a><a href="#duration">مدت</a><a href="#class_support">کلاس‌ها</a><a href="#quality_flags">کیفیت</a><a href="#channels">کانال‌ها</a><a href="#duplicates">تکرار</a><a href="#outliers">موارد بررسی</a><a href="#provenance">بازتولید</a></nav>
<div class="cards">{''.join(f'<div class="card"><strong>{value}</strong><span>{label}</span></div>' for label, value in cards)}</div>
<section><h2>یافته‌های عددی و اثر آن‌ها بر طراحی</h2><ul>{''.join('<li>' + _escaped(item) + '</li>' for item in findings)}</ul></section>
{unsupported_text}
<section id="overview"><h2>پوشش و تفاوت شناخته‌شده / unknown</h2>{_table(["گروه برچسب", "فایل", "ساعت decodeشده", "میانهٔ مدت (s)", "عبور از کنترل فنی", "کمتر از ۱ ثانیه", "کمتر از ۵ ثانیه"], group_rows)}<p class="muted">مدت‌ها فقط از فایل‌های با status=ok محاسبه شده‌اند. ستون‌های کمتر از ۱ و ۵ ثانیه هم‌پوشانی دارند. unknown یک برچسب تجمیعی است و تعداد هویت‌های واقعی آن از این CSV معلوم نمی‌شود.</p><h3>فرمت تشخیص‌داده‌شده پس از خواندن</h3>{_table(["فرمت", "نرخ نمونه (Hz)", "کانال", "subtype", "فایل decodeشده"], format_rows)}<p>پسوند فایل مبنای انتخاب decoder نیست. عبور از کنترل فنی به معنی تأیید گفتار، برچسب یا کیفیت شنیداری نیست؛ شرط دقیق آن در تنظیمات ممیزی پایین گزارش آمده است.</p></section>
<section class="notice"><h2>تفسیر درست سنجه‌ها</h2><ul><li>انرژی پایین، انرژی بالاتر از آستانه و تغییرات انرژی فقط ویژگی‌های موج‌اند؛ هیچ‌کدام «ثانیهٔ گفتار»، VAD معتبر یا SNR اندازه‌گیری‌شده نیستند.</li><li>نسبت نمونه‌های نزدیک دامنهٔ کامل، یک پرچم برای بازبینی است و به‌تنهایی اعوجاج یا clipping شنیداری را ثابت نمی‌کند.</li><li>کلاس‌های کم‌فایل به تقسیم در سطح ضبط نیاز دارند. hash و fingerprint، استقلال همهٔ جلسه‌ها و هویت‌های unknown را اثبات نمی‌کنند.</li><li>فایل‌های فاقد سیگنال یا مردود فنی در شمارش کل گزارش حفظ شده‌اند؛ حذف بی‌سروصدای آن‌ها از ارزیابی مجاز نیست.</li></ul></section>
{''.join(sections)}
{vad_text}
{supplementary_text}
<section><h2>۱۵ کلاس شناخته‌شده با کمترین مدت قابل استفادهٔ فنی</h2>{_table(["speaker_id", "کل فایل", "عبور فنی", "دقیقهٔ عبور فنی", "کوتاه‌ترین فایل (s)", "فایل با پرچم موج"], class_rows)}<p class="muted">رتبه بر اساس مجموع مدت عبورکرده از شرط فنی ممیزی است. جدول کامل {len(speakers):,} برچسب در per_speaker.csv ذخیره شده است. کلاس‌های شناخته‌شده بدون هیچ فایل عبورکرده از کنترل فنی: {stats['known_speakers_with_zero_technically_usable_files']}.</p></section>
<section id="duplicates"><h2>تکرار و خطر نشت</h2>{duplicate_text}<p>فایل یا قطعات مرتبط آن باید در یک گروه باقی بمانند. اختلاف برچسب در محتوای فاقد سیگنال، نشانهٔ قابل حل با طبقه‌بند گوینده نیست؛ برای محتوای عادیِ با برچسب متناقض، بازبینی جداگانه لازم است.</p></section>
{status_text}
<section id="outliers"><h2>موارد مشخص برای بازبینی</h2><p>این فهرست‌ها صف بررسی هستند، نه حکم حذف. پیوند هر فایل به نسخهٔ اصلی داده در همین پروژه اشاره دارد.</p><h3>کوتاه‌ترین فایل‌های عبورکرده از کنترل فنی</h3>{_file_table(good[good['usable_for_training']].sort_values(['duration_seconds','audio_file']), output_dir, data_dir)}<h3>طولانی‌ترین فایل‌ها</h3>{_file_table(good.sort_values(['duration_seconds','audio_file'], ascending=[False,True]), output_dir, data_dir)}<h3>ضعیف‌ترین RMS در فایل‌های غیرصفرِ عبورکرده از کنترل فنی</h3>{_file_table(good[good['usable_for_training'] & np.isfinite(good['max_channel_rms_dbfs'])].sort_values(['max_channel_rms_dbfs','audio_file']), output_dir, data_dir)}<h3>فایل‌های مردود فنی؛ نمایش حداکثر ۱۲ مورد از {len(exclusions):,} فایل</h3>{_file_table(exclusions.sort_values('audio_file'), output_dir, data_dir)}</section>
<section class="notice"><h2>بخش‌های باز پیش از نهایی‌کردن EDA</h2><ul><li>شنیدن نمونه‌های نماینده و موارد مشکوک، ثبت موسیقی، نویز، چندگویندگی و کیفیت واقعی گفتار.</li><li>تأیید گروه‌های ضبط و بررسی حساسیت split به هم‌پوشانی‌ها و تکرارهای احتمالی؛ عدم کشف یک زوج دلیل استقلال نیست.</li><li>بررسی embedding ثابت، پراکندگی هر UUID و ساختار چندگوینده‌ای unknown در مرحلهٔ محاسبات مدل.</li><li>انتخاب VAD، crop، sampling و سیاست نهایی کانال پس از کنار هم گذاشتن این شواهد و اعتبارسنجی.</li></ul><p>در نتیجه، این گزارش ادعای تکمیل همهٔ ابعاد EDA یا آماده‌بودن داده برای fine-tuning ندارد.</p></section>
<section id="provenance"><h2>خروجی‌ها و بازتولید</h2><div class="files">{''.join(f'<a href="{_relative(path, output_dir)}">{_escaped(label)}</a>' for label, path in links)}</div><p>زمان ساخت گزارش به UTC: <span class="mono">{_escaped(stats['generated_at_utc'])}</span></p><p>SHA256 فایل manifest: <span class="mono">{stats['manifest_sha256']}</span></p><details><summary>اثر انگشت داده، تنظیمات، واحدها و پوشش ممیزی</summary><pre>{_escaped(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details><p class="muted">آستانه‌ها، تعریف ویژگی‌ها و محیط اجرا از signal_summary.json نقل شده‌اند. نمودارهای علمی به انگلیسی رسم شده‌اند تا نمایش حروف و واحدها مستقل از فونت فارسی باقی بماند. برای نمایش کامل، index.html و پوشهٔ figures کنار هم نگه داشته شوند.</p></section>
<footer>گزارش از روی دادهٔ ممیزی تولید شده است؛ هیچ سرویس خارجی یا محتوای شبکه‌ای برای مشاهدهٔ نمودارها لازم نیست.</footer>
</main></body></html>'''
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/eda_v1/audio_manifest.csv"))
    parser.add_argument("--signal-summary", type=Path, default=Path("reports/eda/signal_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    stats = build_report(args.manifest, args.signal_summary, args.output_dir, args.data_dir)
    print(json.dumps({"report": str(args.output_dir / "index.html"), "files": stats["by_label_group"]["all"]["files"], "full_eda_completed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
