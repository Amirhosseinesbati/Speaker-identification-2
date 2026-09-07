"""Traceable review flags, separate from training eligibility and verified labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from speaker_id.eda.embeddings import digest


def run(manifest: Path, output: Path):
    frame = pd.read_csv(manifest)
    neighbors = pd.read_csv(output/"embedding_neighbors.csv").set_index("audio_file")
    embeds = pd.read_csv(output/"embedding_files.csv").set_index("audio_file")
    near = pd.read_csv(output/"forensics_near_empty.csv").set_index("audio_file")
    semantics = pd.read_csv(output/"semantic_files.csv").set_index("audio_file")
    speakers = pd.read_csv(output/"embedding_speakers.csv")
    unsupported_geometry = set(speakers.loc[speakers.nearest_known_agreement_fraction == 0, "speaker_id"])
    rows = []
    for item in frame.itertuples(index=False):
        reasons = []
        action = "retain_default"
        priority = 3
        if not item.has_nonzero_signal:
            reasons.append("confirmed_all_zero")
            action, priority = "exclude_training_keep_evaluation", 0
        elif item.audio_file in near.index:
            reasons.append("confirmed_at_most_64_nonzero_samples")
            action, priority = "recommend_exclusion_ablation_keep_current_eligibility", 0
        if item.max_channel_rms_dbfs < -50 and item.has_nonzero_signal:
            reasons.append("low_rms")
        if item.mono_zero_fraction > .99 and item.has_nonzero_signal:
            reasons.append("over_99_percent_zero_samples")
            priority = min(priority, 1)
        if item.speaker_id in unsupported_geometry and item.has_nonzero_signal:
            reasons.append("known_class_no_nearest_neighbor_agreement")
            priority = min(priority, 1)
        if item.has_nonzero_signal and item.duration_seconds < 5:
            reasons.append("shorter_than_5_seconds")
        if item.audio_file in neighbors.index:
            n = neighbors.loc[item.audio_file]
            if item.speaker_id != "unknown" and n.same_vs_other_margin < 0:
                reasons.append("different_known_label_closer_than_own_label")
                priority = min(priority, 1)
            if pd.notna(n.minimum_segment_cosine) and n.minimum_segment_cosine < .2:
                reasons.append("low_between_probe_cosine")
                priority = min(priority, 2)
            if item.speaker_id == "unknown" and n.nearest_known_cosine >= .8:
                reasons.append("unknown_close_to_known_embedding")
                priority = min(priority, 1)
            if embeds.loc[item.audio_file, "silent_probe_count"] > 0:
                reasons.append("embedding_contains_silent_probe")
        if item.audio_file in semantics.index:
            s = semantics.loc[item.audio_file]
            if s.all_segments_inconclusive_signal:
                reasons.append("semantic_signal_inconclusive")
            if s.max_ast_music >= .5:
                reasons.append("model_music_score_at_least_0_5")
                priority = min(priority, 1)
        if reasons and priority == 3:
            priority = 2
        rows.append({"audio_file": item.audio_file, "speaker_id": item.speaker_id,
                     "review_priority": priority, "review_required": bool(reasons),
                     "review_reasons": "|".join(reasons), "suggested_action": action,
                     "current_technical_train_eligible": item.usable_for_training,
                     "model_flags_confirm_labels": False, "human_listening_status": "not_performed",
                     "source_input_sha256": item.input_sha256})
    table = pd.DataFrame(rows)
    table.to_csv(output/"quality_review.csv", index=False)
    counts = {}
    for row in rows:
        for reason in row["review_reasons"].split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    known_flags = table[(table.speaker_id != "unknown") & table.review_required]
    summary = {"status": "review_flags_frozen_no_label_changes", "manifest_sha256": digest(manifest),
               "source_files": len(table), "flagged_files": int(table.review_required.sum()),
               "unflagged_files": int((~table.review_required).sum()), "reason_file_counts": counts,
               "known_classes_with_flags": int(known_flags.speaker_id.nunique()),
               "known_classes_with_no_embedding_neighbor_agreement": sorted(unsupported_geometry),
               "priority_counts": {str(k):int(v) for k,v in table.review_priority.value_counts().sort_index().items()},
               "human_listening_completed": False, "labels_changed": 0, "fold_eligibility_changed": 0,
               "primary_training_exclusions": 89, "additional_near_empty_exclusion_ablation_candidates": len(near),
               "thresholds": {"near_empty_nonzero_samples":64,"low_rms_dbfs":-50,
                              "low_between_probe_cosine":.2,"unknown_near_known_cosine":.8,"ast_music_score":.5},
               "threshold_policy": "Fixed triage heuristics, not tuned rejection thresholds or ground-truth labels",
               "limitations": ["Unflagged does not mean verified clean; semantic screening covers selected windows only",
                               "Human auditory review unavailable in this model interface; audio forwarding explicitly returned unsupported audio input",
                               "An 8-file near-empty exclusion is recommended as a preregistered ablation, not silently applied to frozen folds",
                               "No true unknown identity, session, language or label correction is assigned from model similarity"],
               "input_sha256": {n:digest(output/n) for n in ("embedding_neighbors.csv","embedding_files.csv","embedding_speakers.csv","forensics_near_empty.csv","semantic_files.csv")},
               "code_sha256": digest(Path(__file__)), "quality_review_sha256": digest(output/"quality_review.csv")}
    (output/"quality_review_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
