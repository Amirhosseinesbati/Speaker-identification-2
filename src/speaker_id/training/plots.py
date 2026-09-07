"""Export compact scientific diagnostics alongside live MLflow metric curves."""
from __future__ import annotations

import json
from pathlib import Path


def evaluation_plots(output: Path, report: dict, curve: list[dict]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    files = []
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot([r["threshold"] for r in curve], [r["inner_macro_f1_447"] for r in curve], color="#166a77")
    ax.axvline(report["threshold"], color="#b45c37", linestyle="--", label="Selected on inner queries")
    ax.set(xlabel=report.get("threshold_axis_label", "Global maximum-cosine rejection threshold"), ylabel="Inner Macro-F1 (447 labels)", title="CAM++ calibration; outer labels withheld")
    ax.legend()
    path = output / "calibration_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path)
    rows = report["outer"]["per_class"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist([r["f1"] for r in rows[1:]], bins=20, range=(0, 1), color="#166a77", edgecolor="white")
    ax.set(xlabel="Per-speaker F1", ylabel="Known speakers", title="Outer validation: all 446 known classes")
    path = output / "known_f1_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path)
    history = output / "fit_history.jsonl"
    if history.exists():
        data = [json.loads(line) for line in history.read_text().splitlines()]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot([r["step"] for r in data], [r["fit/loss_aam"] for r in data], linewidth=.8)
        axes[0].set(xlabel="Committed optimizer step", ylabel="AAM loss", title="Known-only training")
        axes[1].plot([r["step"] for r in data], [r["fit/encoder_lr"] for r in data], label="Encoder")
        axes[1].plot([r["step"] for r in data], [r["fit/head_lr"] for r in data], label="Head")
        axes[1].set(xlabel="Committed optimizer step", ylabel="Learning rate", yscale="log", title="Fixed-step learning schedule")
        axes[1].legend()
        path = output / "fit_curves.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)
    return files
