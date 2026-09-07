"""Pure NumPy final enrollment and training-only calibration for frozen CAM++."""

from __future__ import annotations

import numpy as np

from speaker_id.training.crossfit_references import all_training_crossfit_scores
from speaker_id.training.reference_scoring import calibrate_gate


def prepare_final_references(embeddings, valid, manifest: list[dict], folds: list[dict],
                             labels: list[str], *, unknown_weights: list[float],
                             margin_weights: list[float], candidates: int = 201,
                             temperature: float = .05) -> dict:
    """Fit only a rejection scorer; no encoder optimization or OOF metric exists.

    Gallery references retain every eligible file in manifest order. Duplicates
    are intentionally preserved: maximum aggregation is unaffected and no
    approximate deduplication changes the evaluated recipe. Calibration queries
    remain file-weighted, while their entire group leaves both reference pools.
    """
    if (len(labels) < 3 or labels[0] != "unknown" or len(set(labels)) != len(labels)
            or labels[1:] != sorted(labels[1:]) or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Final reference labels/temperature are invalid")
    scores = all_training_crossfit_scores(embeddings, valid, manifest, folds,
                                          method="max_reference", classes=len(labels) - 1)
    if scores["known_labels"] != labels[1:]:
        raise ValueError("Final known score columns differ from the fixed label map")
    positions = {label: index for index, label in enumerate(labels)}
    query = scores["calibration_indices"]
    truth = np.asarray([positions[manifest[int(i)]["speaker_id"]] for i in query], dtype=np.int64)
    chosen, curve = calibrate_gate(scores["inner_known_scores"], truth,
                                   scores["inner_unknown_similarity"], unknown_weights,
                                   margin_weights, candidates, classes=len(labels))
    training_metric = chosen.pop("inner_macro_f1_447")
    selected = {**chosen, "temperature": float(temperature),
                "inference": {"seconds": 180.0, "maximum_windows": 1},
                "protocol": "all_training_leave_content_group_out_v1",
                "score_scope": "training calibration; not OOF or leaderboard evaluation",
                "training_group_excluded_macro_f1_447": training_metric}
    calibration_curve = [{**{k: v for k, v in row.items() if k != "inner_macro_f1_447"},
                          "training_group_excluded_macro_f1_447": row["inner_macro_f1_447"]}
                         for row in curve]
    references = np.asarray(scores["provenance"]["reference_indices"], dtype=np.int64)
    targets = np.asarray([positions[manifest[int(i)]["speaker_id"]] for i in references], dtype=np.int64)
    vectors = np.asarray(embeddings, dtype=np.float32)[references].copy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    gallery = {"known_embeddings": np.ascontiguousarray(vectors[targets > 0]),
               "known_targets": targets[targets > 0],
               "unknown_embeddings": np.ascontiguousarray(vectors[targets == 0])}
    counts = scores["reference_counts"]
    report = {
        "protocol": selected["protocol"], "encoder_updates": 0,
        "calibration_query_files": len(query), "source_files": len(manifest),
        "known_reference_files": int((targets > 0).sum()),
        "unknown_reference_files": int((targets == 0).sum()),
        "known_reference_groups": int(counts["known_groups_per_class"].sum()),
        "unknown_reference_groups": int(counts["unknown_groups"]),
        "known_class_count": len(labels) - 1,
        "known_reference_files_per_class": counts["known_files_per_class"].tolist(),
        "singleton_known_labels_skipped_as_queries": counts["singleton_known_labels"],
        "reference_deduplication": "none; all eligible references retained",
        "query_weighting": "one query per eligible file, excluding its full content group",
        "training_group_excluded_macro_f1_447": training_metric,
        "metric_scope": "Fitted training calibration score, not OOF or hidden-test performance",
        "support_limit": "Calibration removes the query group; inference restores all training references",
        "excluded_invalid_or_ineligible_files": len(manifest) - len(references),
    }
    return {"gallery": gallery, "calibration": selected, "curve": calibration_curve,
            "report": report, "calibration_indices": query,
            "calibration_known_scores": scores["inner_known_scores"],
            "calibration_unknown_similarity": scores["inner_unknown_similarity"],
            "reference_indices": references}
