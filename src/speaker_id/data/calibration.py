"""Deterministic inner holdouts, assigned entirely inside each outer training fold."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from speaker_id.data.splits import truth


def build_calibration_roles(folds: list[dict], seed: int = 20260907,
                            unknown_query_fraction: float = 0.5) -> tuple[list[dict], dict]:
    if not 0 < unknown_query_fraction < 1:
        raise ValueError("Unknown query fraction must be between zero and one")
    if not folds or len({r["audio_file"] for r in folds}) != len(folds):
        raise ValueError("Folds need exactly one row per original filename")
    fold_ids = sorted({int(r["fold"]) for r in folds})
    if len(fold_ids) < 2:
        raise ValueError("At least two outer folds are required")
    known = {r["speaker_id"] for r in folds} - {"unknown"}
    all_groups = defaultdict(list)
    for row in folds:
        if not truth(row["evaluation_included"]):
            raise ValueError("Every original file must remain in outer evaluation")
        all_groups[row["group_id"]].append(row)
    for members in all_groups.values():
        if len({int(r["fold"]) for r in members}) != 1:
            raise ValueError("A content group crosses outer folds")
        if any(truth(r["train_eligible"]) for r in members):
            if len({r["speaker_id"] for r in members}) != 1:
                raise ValueError("Eligible content group has conflicting labels")
            if not all(truth(r["train_eligible"]) for r in members):
                raise ValueError("Mixed eligibility inside a content group")
    output, outer_summaries = [], []
    for outer in fold_ids:
        training_groups = {gid: members for gid, members in all_groups.items()
                           if int(members[0]["fold"]) != outer and truth(members[0]["train_eligible"])}
        by_label = defaultdict(list)
        for gid, members in training_groups.items():
            by_label[members[0]["speaker_id"]].append(gid)
        if not known <= by_label.keys():
            raise ValueError("Outer training data lacks enrollment for a known label")

        def ranked(label):
            return sorted(by_label[label], key=lambda gid: hashlib.sha256(
                f"{seed}|{outer}|{label}|{gid}".encode()).hexdigest())

        query_groups, singletons = set(), []
        for label in sorted(known):
            candidates = ranked(label)
            if len(candidates) >= 2:
                query_groups.add(candidates[0])
            else:
                singletons.append(label)
        unknown_groups = ranked("unknown")
        unknown_queries = (max(1, min(len(unknown_groups) - 1,
                                     int(len(unknown_groups) * unknown_query_fraction)))
                           if len(unknown_groups) >= 2 else 0)
        query_groups.update(unknown_groups[:unknown_queries])
        outer_rows = []
        for source in sorted(folds, key=lambda row: row["audio_file"]):
            label, gid = source["speaker_id"], source["group_id"]
            is_outer = int(source["fold"]) == outer
            eligible = truth(source["train_eligible"])
            query = not is_outer and eligible and gid in query_groups
            enrollment = not is_outer and eligible and not query and label != "unknown"
            fit = not is_outer and eligible and not query
            if is_outer:
                role = "outer_validation"
            elif not eligible:
                role = "training_excluded"
            elif query:
                role = "unknown_calibration_query" if label == "unknown" else "known_calibration_query"
            else:
                role = "unknown_development" if label == "unknown" else "known_enrollment"
            outer_rows.append({"outer_fold": outer, "audio_file": source["audio_file"],
                               "speaker_id": label, "group_id": gid, "role": role,
                               "encoder_fit_allowed": fit, "enrollment_allowed": enrollment,
                               "calibration_query": query, "outer_evaluation_included": is_outer,
                               "source_train_eligible": eligible,
                               "source_exclusion_reasons": source.get("exclusion_reasons", "")})
        output.extend(outer_rows)
        enrollment_groups = Counter(training_groups[g][0]["speaker_id"] for g in training_groups
                                    if g not in query_groups and training_groups[g][0]["speaker_id"] != "unknown")
        counts = Counter(r["role"] for r in outer_rows)
        outer_summaries.append({"outer_fold": outer, "role_file_counts": dict(sorted(counts.items())),
                                "known_classes_with_enrollment": len(enrollment_groups),
                                "known_calibration_query_classes": len(known) - len(singletons),
                                "known_singleton_enrollment_only": singletons,
                                "known_enrollment_group_support_distribution": dict(sorted(Counter(enrollment_groups.values()).items())),
                                "unknown_training_content_groups": len(unknown_groups),
                                "unknown_calibration_query_groups": unknown_queries,
                                "unknown_development_groups": len(unknown_groups) - unknown_queries,
                                "calibration_query_groups": len(query_groups),
                                "encoder_fit_files": sum(r["encoder_fit_allowed"] for r in outer_rows)})
    return output, {"version": "calibration_roles_v1", "seed": seed,
                    "unknown_query_fraction": unknown_query_fraction,
                    "source_files": len(folds), "outer_fold_count": len(fold_ids), "role_rows": len(output),
                    "known_class_count": len(known), "folds": outer_summaries,
                    "status": "roles_frozen_no_model_or_threshold_fitted",
                    "policy": {
                        "known_query": "One independent eligible content group for classes with at least two outer-training groups; remaining groups enroll",
                        "singleton": "Enrollment only, never an inner query",
                        "unknown_query": "Seeded content-group holdout; floor(fraction * groups), bounded to preserve query and development groups when at least two exist",
                        "invalid_signal": "Excluded from inner queries, enrollment and encoder fitting; retained in outer evaluation",
                        "encoder_fit": "All inner calibration-query and outer-validation rows withheld from supervised encoder fitting",
                        "learned_transforms": "Fit only encoder_fit_allowed rows; queries also excluded from prototypes and learned score-normalization cohorts",
                        "outer_evaluation": "Every original row evaluated once across outer folds; all 447 labels retained",
                        "unknown_identity": "Content-group disjointness does not imply distinct unknown people or recording sessions",
                        "threshold_selection": "Fit rejection/calibration only from inner queries; never select thresholds from outer labels",
                        "duplicate_query_weighting": "Rows retain file-level evaluation weights; exact duplicates can yield multiple query rows in one independent group. Group-balanced calibration is an explicit future ablation",
                        "final_fit": "Refitting after development or using more enrollment files changes score distributions and requires a documented final calibration strategy",
                    }}
