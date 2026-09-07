"""Expanded galleries whose queries are exclusively original encoder holdouts."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from speaker_id.data.splits import truth


def heldout_reference_scores(contract: dict, embeddings, valid, outer: int) -> dict:
    """Mask the complete query group before any known/unknown maximum.

    Every query comes from the immutable original role assignment, never from
    encoder-fit rows. Outer labels are not read. Cached unit vectors are used
    directly, with no new normalization or feature transformation.
    """
    manifest, labels = contract["manifest"], contract["labels"]
    values, mask = np.asarray(embeddings), np.asarray(valid)
    names = [row["audio_file"] for row in manifest]
    folds = {row["audio_file"]: row for row in contract["folds"]}
    roles_list = [row for row in contract["roles"] if int(row["outer_fold"]) == outer]
    roles = {row["audio_file"]: row for row in roles_list}
    if (outer not in contract["config"]["fold_ids"] or len(set(names)) != len(names)
            or len(folds) != len(contract["folds"]) or len(roles) != len(roles_list)
            or set(folds) != set(names) or set(roles) != set(names)
            or values.ndim != 2 or len(values) != len(names) or values.dtype != np.float32
            or mask.shape != (len(names),) or mask.dtype != np.bool_
            or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values[mask], axis=1), 1, atol=1e-5)
            or np.any(values[~mask])):
        raise ValueError("Held-out scoring requires aligned unique rows and unchanged valid unit embeddings")
    positions = {name: i for i, name in enumerate(names)}
    groups, reference_indices, fit_groups, group_assignments = [], [], set(), defaultdict(set)
    for name in names:
        split, role = folds[name], roles[name]
        group, assigned = split["group_id"], int(split["fold"])
        if not isinstance(group, str) or not group or role["group_id"] != group:
            raise ValueError("Original role and fold content groups differ")
        fitting, enrolling, querying, evaluating = [truth(role[key]) for key in
            ("encoder_fit_allowed", "enrollment_allowed", "calibration_query", "outer_evaluation_included")]
        eligible, i = truth(split["train_eligible"]), positions[name]
        if (evaluating != (assigned == outer) or (evaluating and (fitting or enrolling or querying))
                or (querying and (fitting or enrolling)) or ((fitting or enrolling or querying) and (not eligible or not mask[i]))):
            raise ValueError("Original held-out query/fit/outer role leakage or ineligible signal")
        groups.append(group)
        group_assignments[group].add((assigned, fitting, enrolling, querying, evaluating))
        if fitting:
            fit_groups.add(group)
        if assigned != outer and eligible and mask[i]:
            reference_indices.append(i)
    if any(len(assignment) != 1 for assignment in group_assignments.values()):
        raise ValueError("A content group crosses original fit/query/outer roles")
    # Keep the exact original query and outer row order, as fixed_role_scores does.
    query = np.asarray([positions[row["audio_file"]] for row in roles_list if truth(row["calibration_query"])], dtype=np.int64)
    evaluation = np.asarray([positions[row["audio_file"]] for row in roles_list if truth(row["outer_evaluation_included"])], dtype=np.int64)
    refs, groups = np.asarray(reference_indices, dtype=np.int64), np.asarray(groups, dtype=object)
    if (not len(query) or not len(evaluation) or not set(query).issubset(set(refs))
            or set(groups[query]) & fit_groups or set(groups[refs]) & set(groups[evaluation])):
        raise ValueError("Only the original encoder-held-out queries may calibrate expanded galleries")
    # Read labels only after restricting to permitted outer-training references.
    reference_labels = np.asarray([manifest[int(i)]["speaker_id"] for i in refs], dtype=object)
    for group in set(groups[refs]):
        if len(set(reference_labels[groups[refs] == group])) != 1:
            raise ValueError("A reference content group has conflicting class labels")
    if labels[0] != "unknown" or sorted(set(reference_labels) - {"unknown"}) != labels[1:]:
        raise ValueError("Expanded gallery known-label columns differ from the fixed label map")
    known_refs, unknown_refs = refs[reference_labels != "unknown"], refs[reference_labels == "unknown"]
    known_labels = reference_labels[reference_labels != "unknown"]
    if not len(unknown_refs):
        raise ValueError("Expanded gallery requires an unknown reference cohort")
    inner_similarity = values[query] @ values[known_refs].T
    same_known_group = groups[query, None] == groups[known_refs][None, :]
    inner_similarity[same_known_group] = -np.inf
    outer_similarity = values[evaluation] @ values[known_refs].T
    inner_known, outer_known, counts, group_counts, inner_counts = [], [], [], [], []
    for label in labels[1:]:
        columns = known_labels == label
        inner_known.append(inner_similarity[:, columns].max(axis=1))
        outer_known.append(outer_similarity[:, columns].max(axis=1))
        counts.append(int(columns.sum()))
        group_counts.append(len(set(groups[known_refs][columns])))
        inner_counts.append(int(columns.sum()) - same_known_group[:, columns].sum(axis=1))
    inner_known, outer_known = np.column_stack(inner_known), np.column_stack(outer_known)
    inner_counts = np.column_stack(inner_counts)
    inner_unknown_scores = values[query] @ values[unknown_refs].T
    same_unknown_group = groups[query, None] == groups[unknown_refs][None, :]
    inner_unknown_scores[same_unknown_group] = -np.inf
    inner_unknown = inner_unknown_scores.max(axis=1)
    outer_unknown = (values[evaluation] @ values[unknown_refs].T).max(axis=1)
    if (any(not np.isfinite(array).all() for array in (inner_known, outer_known, inner_unknown, outer_unknown))
            or np.any(inner_counts < 1)):
        raise ValueError("Whole-group exclusion leaves an empty class or unknown cohort")
    counts = np.asarray(counts, dtype=np.int64)
    target_positions = {label: i for i, label in enumerate(labels[1:])}
    own_inner = np.asarray([inner_counts[position, target_positions[manifest[int(i)]["speaker_id"]]]
        if manifest[int(i)]["speaker_id"] != "unknown" else -1 for position, i in enumerate(query)])
    own_outer = np.asarray([counts[target_positions[manifest[int(i)]["speaker_id"]]]
        if manifest[int(i)]["speaker_id"] != "unknown" else -1 for i in query])
    return {"calibration_indices": query, "outer_indices": evaluation, "known_labels": labels[1:],
        "inner_known_scores": inner_known, "outer_known_scores": outer_known,
        "inner_unknown_similarity": inner_unknown, "outer_unknown_similarity": outer_unknown,
        "outer_valid": mask[evaluation].copy(),
        "reference_counts": {"known_files": len(known_refs), "unknown_files": len(unknown_refs),
            "known_groups": len(set(groups[known_refs])), "unknown_groups": len(set(groups[unknown_refs])),
            "known_files_per_class": counts, "known_groups_per_class": np.asarray(group_counts),
            "inner_known_files_per_class": inner_counts,
            "inner_unknown_files": len(unknown_refs) - same_unknown_group.sum(axis=1),
            "removed_query_group_files": same_known_group.sum(axis=1) + same_unknown_group.sum(axis=1),
            "known_query_own_class_inner_files": own_inner, "known_query_own_class_outer_files": own_outer,
            "known_query_own_class_inner_range": [int(own_inner[own_inner >= 0].min()), int(own_inner.max())],
            "known_query_own_class_outer_range": [int(own_outer[own_outer >= 0].min()), int(own_outer.max())]},
        "provenance": {"protocol": "original_heldout_queries_expanded_gallery_v1",
            "outer_fold": outer, "query_indices": query.tolist(), "reference_indices": refs.tolist(),
            "query_groups": sorted(set(groups[query])), "reference_groups": sorted(set(groups[refs])),
            "query_group_excluded_before_all_reference_maxima": True,
            "encoder_fit_rows_never_calibration_queries": True, "outer_labels_accessed": False,
            "embedding_transform": "none; exact cached float32 unit vectors",
            "support_caveat": "The known query's own group is absent internally and restored for outer evaluation; per-class support can increase by 50–100%."}}
