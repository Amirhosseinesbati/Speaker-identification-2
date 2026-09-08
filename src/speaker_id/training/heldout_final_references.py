"""Draft: global enrollment with only a fixed encoder's original held-out queries.

This is a future release helper, not an experiment or a deployment. It never
uses all training rows as calibration queries for an adapted encoder.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from speaker_id.data.splits import truth


def heldout_final_scores(contract, embeddings, valid, *, encoder_fold=0):
    """Use global references, excluding the complete original query group.

    Caller must separately attest the encoder/checkpoint-to-cache binding.
    The fixed fold index is chosen independently of outer performance. There is
    no outer evaluation here: all eligible supplied recordings are enrolled.
    """
    if type(encoder_fold) is not int or encoder_fold != 0:
        raise ValueError("The minimal adapted release fixes fold 0 before final fitting")
    manifest, labels = contract['manifest'], contract['labels']
    values, mask = np.asarray(embeddings), np.asarray(valid)
    names = [row['audio_file'] for row in manifest]
    fold_rows = contract['folds']
    role_rows = [row for row in contract['roles'] if int(row['outer_fold']) == encoder_fold]
    splits = {row['audio_file']: row for row in fold_rows}
    roles = {row['audio_file']: row for row in role_rows}
    if (len(set(names)) != len(names) or len(splits) != len(fold_rows)
            or len(roles) != len(role_rows) or set(splits) != set(names) or set(roles) != set(names)
            or len(labels) < 3 or labels[0] != 'unknown' or labels[1:] != sorted(set(labels[1:]))
            or 'unknown' in labels[1:] or values.ndim != 2 or len(values) != len(names)
            or values.shape[1] < 1 or values.dtype != np.float32 or mask.dtype != np.bool_
            or mask.shape != (len(names),) or not np.isfinite(values).all() or np.any(values[~mask])):
        raise ValueError("Final reference inputs must preserve unique aligned rows and label order")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms[mask], 1, atol=1e-5):
        raise ValueError("Final references require verified unit source vectors")
    positions = {name: i for i, name in enumerate(names)}
    groups, references, fit_groups, assignments = [], [], set(), defaultdict(set)
    for name in names:
        split, role, i = splits[name], roles[name], positions[name]
        group, assigned = split['group_id'], int(split['fold'])
        eligible = truth(split['train_eligible'])
        fitting, enrolling, querying, evaluating = [truth(role[key]) for key in
            ('encoder_fit_allowed', 'enrollment_allowed', 'calibration_query', 'outer_evaluation_included')]
        if (not isinstance(group, str) or not group or role['group_id'] != group
                or evaluating != (assigned == encoder_fold)
                or (evaluating and (fitting or enrolling or querying))
                or (querying and (fitting or enrolling))
                or ((fitting or enrolling or querying) and (not eligible or not mask[i]))):
            raise ValueError("Original query/fit/outer roles or eligibility have changed")
        groups.append(group)
        assignments[group].add((assigned, fitting, enrolling, querying, evaluating))
        if fitting:
            fit_groups.add(group)
        if eligible and mask[i]:
            references.append(i)
    if any(len(assignment) != 1 for assignment in assignments.values()):
        raise ValueError("A complete content group must preserve one original role")
    groups = np.asarray(groups, dtype=object)
    refs = np.asarray(references, dtype=np.int64)
    queries = np.asarray([positions[row['audio_file']] for row in role_rows if truth(row['calibration_query'])], dtype=np.int64)
    if (not len(queries) or not set(queries).issubset(set(refs)) or set(groups[queries]) & fit_groups):
        raise ValueError("Only original encoder-held-out queries may calibrate the final scorer")
    targets = np.asarray([labels.index(manifest[int(i)]['speaker_id']) for i in refs], dtype=np.int64)
    for group in set(groups[refs]):
        if len(set(targets[groups[refs] == group])) != 1:
            raise ValueError("Eligible reference content group has conflicting labels")
    if set(targets) != set(range(len(labels))):
        raise ValueError("Final enrollment must cover every known label and the unknown cohort")
    normalized = np.zeros_like(values)
    normalized[mask] = values[mask] / norms[mask, None]
    known_refs, unknown_refs = refs[targets > 0], refs[targets == 0]
    known_targets = targets[targets > 0]
    same_known = groups[queries, None] == groups[known_refs][None, :]
    same_unknown = groups[queries, None] == groups[unknown_refs][None, :]
    similarity = normalized[queries] @ normalized[known_refs].T
    similarity[same_known] = -np.inf
    known, counts, remaining = [], [], []
    for target in range(1, len(labels)):
        selected = known_targets == target
        known.append(similarity[:, selected].max(axis=1))
        counts.append(int(selected.sum()))
        remaining.append(int(selected.sum()) - same_known[:, selected].sum(axis=1))
    known = np.column_stack(known)
    remaining = np.column_stack(remaining)
    unknown = normalized[queries] @ normalized[unknown_refs].T
    unknown[same_unknown] = -np.inf
    unknown = unknown.max(axis=1)
    if not np.isfinite(known).all() or not np.isfinite(unknown).all() or np.any(remaining < 1):
        raise ValueError("Full query-group exclusion leaves an empty reference pool")
    # Match the portable reference scorer, including float32 cosine overshoot.
    known, unknown = np.clip(known, -1.0, 1.0), np.clip(unknown, -1.0, 1.0)
    gallery = {'known_embeddings': np.ascontiguousarray(normalized[known_refs]),
               'known_targets': known_targets.copy(),
               'unknown_embeddings': np.ascontiguousarray(normalized[unknown_refs])}
    return {'gallery': gallery, 'calibration_indices': queries, 'reference_indices': refs,
            'inner_known_scores': known, 'inner_unknown_similarity': unknown,
            'reference_counts': {'known_files_per_class': np.asarray(counts, dtype=np.int64),
                'inner_known_files_per_class': remaining,
                'inner_unknown_files': len(unknown_refs) - same_unknown.sum(axis=1),
                'removed_query_group_files': same_known.sum(axis=1) + same_unknown.sum(axis=1)},
            'provenance': {'protocol': 'fixed_fold0_encoder_holdout_global_gallery_v1',
                'encoder_fold': 0, 'query_indices': queries.tolist(), 'reference_indices': refs.tolist(),
                'query_groups': sorted(set(groups[queries])), 'fit_groups': sorted(fit_groups),
                'whole_query_group_excluded': True, 'encoder_fit_rows_never_queries': True,
                'embedding_transform': 'normalize each source vector once for gallery/query cosine',
                'metric_scope': 'Final scorer fitting on encoder-held-out queries; not OOF or leaderboard'}}
