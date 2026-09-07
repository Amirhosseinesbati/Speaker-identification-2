"""Descriptive geometry of frozen embeddings; labels are used for audit only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from speaker_id.eda.embeddings import digest


def distribution(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {"count": len(x), "mean": float(x.mean()) if len(x) else None,
            **{f"p{q}": float(np.percentile(x, q)) if len(x) else None for q in (1, 5, 25, 50, 75, 95, 99)}}


def nearest_excluding_groups(sim, groups):
    score = sim.copy()
    score[np.asarray(groups)[:, None] == np.asarray(groups)[None, :]] = -np.inf
    return np.argmax(score, axis=1), np.max(score, axis=1)


def mutual_components(sim, thresholds):
    """Mutual-nearest edges only: components are similarities, not person counts."""
    if len(sim) < 2:
        return []
    score = sim.copy()
    np.fill_diagonal(score, -np.inf)
    neighbor = score.argmax(axis=1)
    edges = [(i, int(j), float(score[i, j])) for i, j in enumerate(neighbor) if i < j and neighbor[j] == i]
    return [{"threshold": threshold, "mutual_nearest_pairs": sum(s >= threshold for _, _, s in edges)} for threshold in thresholds]


def run(manifest, folds_path, cache, output):
    frame = pd.read_csv(manifest).set_index("audio_file")
    folds = pd.read_csv(folds_path).set_index("audio_file")
    summary_path = output / "embedding_summary.json"
    embedding_summary = json.loads(summary_path.read_text())
    if embedding_summary["status"] != "complete" or embedding_summary["manifest_sha256"] != digest(manifest):
        raise ValueError("Complete matching embeddings required")
    if embedding_summary["embedding_npz_sha256"] != digest(cache/"embeddings.npz"):
        raise ValueError("Embedding array hash mismatch")
    data = np.load(cache/"embeddings.npz", allow_pickle=False)
    names, x = data["audio_file"], data["embeddings"]
    frame = frame.loc[names].copy()
    labels = frame.speaker_id.to_numpy()
    groups = folds.loc[names].group_id.to_numpy()
    sim = np.clip(x @ x.T, -1, 1)
    sim[groups[:, None] == groups[None, :]] = -np.inf
    known = np.flatnonzero(labels != "unknown")
    unknown = np.flatnonzero(labels == "unknown")
    known_scores = sim[:, known]
    best_idx = known[np.argmax(known_scores, axis=1)]
    best_score = np.max(known_scores, axis=1)
    rows, same_scores, candidates, speaker_rows = [], [], [], []
    segments = pd.read_csv(output/"embedding_files.csv").set_index("audio_file")
    for i, name in enumerate(names):
        same = np.flatnonzero(labels == labels[i]) if labels[i] != "unknown" else np.array([], dtype=int)
        positive = sim[i, same]
        positive = positive[np.isfinite(positive)]
        other = known[labels[known] != labels[i]]
        other_i = other[np.argmax(sim[i, other])]
        nearest_positive = float(positive.max()) if len(positive) else None
        row = {"audio_file": name, "speaker_id": labels[i], "fold": int(folds.loc[name, "fold"]),
               "nearest_known_file": names[best_idx[i]], "nearest_known_speaker": labels[best_idx[i]],
               "nearest_known_cosine": float(best_score[i]),
               "nearest_other_known_file": names[other_i], "nearest_other_known_cosine": float(sim[i, other_i]),
               "same_label_neighbor_count": len(positive), "max_same_label_cosine": nearest_positive,
               "mean_same_label_cosine": float(positive.mean()) if len(positive) else None,
               "same_vs_other_margin": nearest_positive-float(sim[i, other_i]) if nearest_positive is not None else None,
               "nearest_known_label_agrees": bool(labels[best_idx[i]] == labels[i]) if labels[i] != "unknown" else None,
               "minimum_segment_cosine": segments.loc[name, "minimum_segment_cosine"],
               "rms_dbfs": float(frame.loc[name, "max_channel_rms_dbfs"]),
               "duration_seconds": float(frame.loc[name, "duration_seconds"]),
               "zero_fraction": float(frame.loc[name, "mono_zero_fraction"])}
        rows.append(row)
        if len(same):
            same_scores.extend(sim[i, same[same > i]][np.isfinite(sim[i, same[same > i]])].tolist())
        if labels[i] != "unknown":
            candidates.append({"audio_file_a": name, "speaker_a": labels[i], "audio_file_b": names[other_i],
                               "speaker_b": labels[other_i], "cosine": float(sim[i, other_i]),
                               "candidate_type": "different_known_labels", "status": "similarity_only_not_confirmed"})
        else:
            candidates.append({"audio_file_a": name, "speaker_a": "unknown", "audio_file_b": names[best_idx[i]],
                               "speaker_b": labels[best_idx[i]], "cosine": float(best_score[i]),
                               "candidate_type": "unknown_to_known", "status": "similarity_only_not_confirmed"})
    table = pd.DataFrame(rows)
    table.to_csv(output/"embedding_neighbors.csv", index=False)
    candidates = pd.DataFrame(candidates).sort_values("cosine", ascending=False)
    candidates["pair_key"] = candidates.apply(lambda r: "|".join(sorted([r.audio_file_a, r.audio_file_b])), axis=1)
    candidates.drop_duplicates("pair_key").drop(columns="pair_key").to_csv(output/"embedding_candidate_pairs.csv", index=False)
    for speaker, subset in table[table.speaker_id != "unknown"].groupby("speaker_id"):
        speaker_rows.append({"speaker_id": speaker, "files": len(subset),
                             "nearest_known_agreement_fraction": float(subset.nearest_known_label_agrees.astype(bool).mean()),
                             "negative_margin_files": int((subset.same_vs_other_margin < 0).sum()),
                             "mean_same_label_cosine": float(subset.mean_same_label_cosine.mean()),
                             "median_same_vs_other_margin": float(subset.same_vs_other_margin.median()),
                             "minimum_within_file_cosine": float(subset.minimum_segment_cosine.min()),
                             "median_rms_dbfs": float(subset.rms_dbfs.median())})
    speakers = pd.DataFrame(speaker_rows).sort_values(["nearest_known_agreement_fraction", "median_same_vs_other_margin"])
    speakers.to_csv(output/"embedding_speakers.csv", index=False)
    rng = np.random.default_rng(20260907)
    left, right = rng.choice(known, size=(2, 200000), replace=True)
    different = (labels[left] != labels[right]) & np.isfinite(sim[left, right])
    negatives = sim[left[different], right[different]]
    ktable = table[table.speaker_id != "unknown"]
    healthy = ktable[(ktable.rms_dbfs >= -50) & (ktable.duration_seconds >= 5) & (ktable.zero_fraction < .99)]
    # PCA is visualization fit on all audit rows; it is not reusable model preprocessing.
    centered = x.astype(float) - x.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    projection = centered @ eigenvectors[:, -2:][:, ::-1]
    pca_variance = np.sum(projection ** 2, axis=0) / np.sum(centered ** 2)
    pd.DataFrame({"audio_file": names, "speaker_id": labels, "pc1": projection[:, 0], "pc2": projection[:, 1]}).to_csv(output/"embedding_projection.csv", index=False)
    figure_dir = output/"figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(negatives, bins=70, density=True, alpha=.6, label="Random different known labels")
    axes[0].hist(same_scores, bins=50, density=True, alpha=.6, label="Same known label (different content groups)")
    axes[0].set(xlabel="Cosine similarity", ylabel="Density", title="Frozen ECAPA: descriptive pair geometry")
    axes[0].legend(fontsize=8)
    for ids, color, label in ((unknown, "#d57537", "Unknown label"), (known, "#2563a6", "Known labels")):
        axes[1].scatter(projection[ids, 0], projection[ids, 1], s=6, alpha=.35, color=color, label=label, rasterized=True)
    axes[1].set(xlabel=f"PC1 ({pca_variance[0]:.1%})", ylabel=f"PC2 ({pca_variance[1]:.1%})", title="PCA of all audit embeddings; not identity clusters")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir/"embedding_geometry.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(ktable.rms_dbfs, ktable.same_vs_other_margin, s=9, alpha=.45)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(xlabel="Original RMS (dBFS)", ylabel="Best same-label minus best other-label cosine", title="Known-label consistency and signal level")
    axes[1].hist(table.minimum_segment_cosine.dropna(), bins=50)
    axes[1].set(xlabel="Minimum cosine between disjoint probes in a file", ylabel="Files", title="Temporal stability; not a diarization verdict")
    fig.tight_layout()
    fig.savefig(figure_dir/"embedding_stability.png", dpi=160)
    plt.close(fig)
    fold_comparison = []
    whole = pd.read_csv(manifest).merge(pd.read_csv(folds_path)[["audio_file", "fold"]], on="audio_file", validate="one_to_one")
    for label_group in ("known", "unknown"):
        subset = whole[(whole.speaker_id == "unknown") == (label_group == "unknown")]
        for feature in ("duration_seconds", "max_channel_rms_dbfs", "vad_mode3_speech_fraction", "mono_zero_fraction"):
            a = subset[subset.fold == 0][feature].dropna().to_numpy()
            b = subset[subset.fold == 1][feature].dropna().to_numpy()
            fold_comparison.append({"label_group": label_group, "feature": feature,
                                    "fold0_count": len(a), "fold1_count": len(b),
                                    "fold0_median": float(np.median(a)), "fold1_median": float(np.median(b)),
                                    "ks_statistic": float(ks_2samp(a, b).statistic)})
    pd.DataFrame(fold_comparison).to_csv(output/"fold_feature_comparison.csv", index=False)
    gain = pd.read_csv(output/"embedding_gain.csv")
    summary = {"status": "descriptive_audit_complete", "manifest_sha256": digest(manifest),
               "folds_sha256": digest(folds_path), "embedding_summary_sha256": digest(summary_path),
               "code_sha256": digest(Path(__file__)), "embedded_files": len(x), "known_files": len(known),
               "unknown_files": len(unknown), "known_classes": len(speakers),
               "same_label_pair_cosine": distribution(same_scores), "random_other_label_pair_cosine": distribution(negatives),
               "nearest_known_agreement_all_known": {"count": len(ktable), "agree": int(ktable.nearest_known_label_agrees.astype(bool).sum()), "fraction": float(ktable.nearest_known_label_agrees.astype(bool).mean())},
               "nearest_known_agreement_healthy_subset": {"count": len(healthy), "agree": int(healthy.nearest_known_label_agrees.astype(bool).sum()), "fraction": float(healthy.nearest_known_label_agrees.astype(bool).mean())},
               "known_negative_margin_files": int((ktable.same_vs_other_margin < 0).sum()),
               "unknown_nearest_known_cosine": distribution(best_score[unknown]),
               "within_file_minimum_cosine": distribution(table.minimum_segment_cosine),
               "known_classes_with_any_neighbor_disagreement": int((speakers.nearest_known_agreement_fraction < 1).sum()),
               "known_classes_with_no_neighbor_agreement": int((speakers.nearest_known_agreement_fraction == 0).sum()),
               "unknown_mutual_nearest_similarity_pairs": mutual_components(sim[np.ix_(unknown, unknown)], [.6, .7, .8, .9]),
               "gain_probe_original_similarity": distribution(gain.original_to_gain_cosine),
               "pca_explained_variance_fraction": pca_variance.tolist(),
               "candidate_pairs_are_confirmed": False, "split_changed": False,
               "limitations": ["Descriptive all-data neighbor agreement is not cross-validation accuracy or Macro-F1",
                               "Unknown mutual-nearest pairs are threshold-sensitive similarity links, not identities or session groups",
                               "Close embeddings alone never merge files, alter labels, or prove leakage",
                               "Low temporal cosine can reflect channel, phonetics, silence or noise as well as multiple speakers",
                               "PCA is descriptive on all rows and is not a trained transformation for evaluation"]}
    outputs = ["embedding_neighbors.csv", "embedding_speakers.csv", "embedding_candidate_pairs.csv", "embedding_projection.csv", "fold_feature_comparison.csv"]
    summary["artifact_sha256"] = {n: digest(output/n) for n in outputs}
    (output/"embedding_geometry_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
