"""Plot bounded model-screening results without implying prevalence or verified labels."""
from pathlib import Path
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/eda"
frame = pd.read_csv(BASE / "semantic_segments.csv")
original = frame[(frame.variant == "original") & (frame.status == "ok")]
interpretable = original[~original.inconclusive_signal]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = {"Controls": "#1b8b8d", "Anomalies / queue": "#dc813a"}
for label, mask in (("Controls", interpretable.selection_reasons.str.contains("control_")),
                    ("Anomalies / queue", ~interpretable.selection_reasons.str.contains("control_"))):
    part = interpretable[mask]
    axes[0].scatter(part.ast_speech, part.ast_music, s=23, alpha=.6,
                    label=f"{label} (n={len(part)} windows)", color=colors[label])
axes[0].set(xlabel="AST Speech score", ylabel="AST Music score", xlim=(-.02, 1.02), ylim=(-.02, 1.02),
            title="Original audio; signal-inconclusive windows excluded")
axes[0].legend(loc="upper right", fontsize=8)
gain = frame[(frame.variant == "diagnostic_gain") & (frame.status == "ok")]
paired = gain.merge(original, on=["audio_file", "segment_index"], suffixes=("_gain", "_original"))
axes[1].plot([0, 1], [0, 1], ls="--", color="#91a0aa", lw=1)
axes[1].scatter(paired.ast_speech_original, paired.ast_speech_gain, c=paired.gain_db_gain,
                cmap="viridis", vmin=0, vmax=40, s=40, edgecolor="#26374a", linewidth=.4)
axes[1].set(xlabel="AST Speech score: original", ylabel="AST Speech score: diagnostic gain",
            xlim=(-.02, 1.02), ylim=(-.02, 1.02), title=f"Paired gain sensitivity ({len(paired)} windows)")
axes[1].text(.02, .98, "Includes sparse / near-zero signals.\nHigher score is not recovered speech.",
             transform=axes[1].transAxes, va="top", fontsize=9)
all_original = frame[frame.variant == "original"]
categories = ["Scored, sufficient signal", "Scored, signal inconclusive", "Skipped zero / <25 ms"]
counts = [len(interpretable), len(original) - len(interpretable), int((all_original.status != "ok").sum())]
axes[2].barh(categories, counts, color=["#1b8b8d", "#dc813a", "#708090"])
for index, count in enumerate(counts):
    axes[2].text(count + 2, index, str(count), va="center")
axes[2].set(xlabel="Original windows", title="Coverage and evidence limitations", xlim=(0, max(counts) * 1.15))
axes[2].invert_yaxis()
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle("Automated semantic screening: bounded, anomaly-enriched sample", fontsize=15)
fig.text(.5, .015, "Uncalibrated model scores; no human listening or transcript verification. Counts are not dataset prevalence estimates.",
         ha="center", fontsize=10, color="#40536a")
fig.tight_layout(rect=(0, .05, 1, .92))
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures/semantic_screening.png", dpi=160)
plt.close(fig)
print(BASE / "figures/semantic_screening.png")
