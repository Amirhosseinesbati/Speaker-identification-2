"""Preregistered S011 scorers; selection APIs never consume outer labels.

Every calibration query's complete content group leaves BOTH the gallery and
normalization cohort, including reference-side cohort statistics. No encoder,
audio frontend, transductive query collection, or feature transform is fitted.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from speaker_id.training.candidate_fusion import ALPHAS, TIE_ORDER, weighted_encoder_pair, select_inner_alpha
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities
from speaker_id.training.scoring import macro_f1_indices

CANDIDATES = [
    {'id': 'top2_025', 'family': 'robust_pool', 'kind': 'top2', 'blend': .25},
    {'id': 'top2_050', 'family': 'robust_pool', 'kind': 'top2', 'blend': .5},
    *[{'id': f'lme_{int(tau*100):02d}_{int(blend*100):02d}', 'family': 'robust_pool',
       'kind': 'logmeanexp', 'temperature': tau, 'blend': blend}
      for tau in (.05, .1) for blend in (.25, .5)],
    *[{'id': f'unknown_top3_{int(blend*100):02d}', 'family': 'unknown_pool',
       'kind': 'unknown_top3', 'blend': blend} for blend in (.25, .5)],
    *[{'id': f'density_{k}_{int(beta*100)}', 'family': 'density', 'kind': 'density',
       'cohort_topk': k, 'beta': beta} for k in (50, 200) for beta in (.5, 1.)],
    *[{'id': f'asnorm_{k}', 'family': 'asnorm', 'kind': 'asnorm',
       'cohort_topk': k, 'std_floor': .02} for k in (50, 200)],
    {'id': 'late_fusion', 'family': 'late_fusion', 'kind': 'late_fusion'},
    *[{'id': f'quality_ridge_{penalty}', 'family': 'quality_gate', 'kind': 'quality_gate',
       'penalty': penalty, 'alphas': [.5, .75], 'group_folds': 3,
       'target': 'top_known_identity_correct', 'loss_weighting': 'binary_balanced',
       'meta_cv_scope': 'logistic_coefficients_only_shared_permitted_reference_gallery'}
      for penalty in (.01, .1, 1.)],
]
UNKNOWN_WEIGHTS = [0., .25, .5, .75, 1.]
MARGIN_WEIGHTS = [0., .5]
FEATURE_NAMES = ['known_top', 'unknown_top', 'known_margin', 'known_top10_mean',
                 'log1p_duration_capped180', 'rms_dbfs_clipped', 'encoder_winner_agreement']


def prepare_fold(public, advanced, valid, manifest, folds, labels, outer):
    """Use the original exact NumPy arithmetic for the S008 baseline control."""
    classes = len(labels) - 1
    scores, values = {}, {}
    for alpha in ALPHAS:
        values[alpha] = public if alpha == 0 else advanced if alpha == 1 else weighted_encoder_pair(public, advanced, valid, alpha)
        scores[alpha] = crossfit_scores(values[alpha], valid, manifest, folds, outer, 'max_reference', classes)
        if scores[alpha]['known_labels'] != labels[1:]:
            raise ValueError('Source label columns differ')
    anchor = scores[0.]
    inner_idx, outer_idx = anchor['calibration_indices'], anchor['outer_indices']
    by_name = {r['audio_file']: r for r in folds}
    groups = np.asarray([by_name[r['audio_file']]['group_id'] for r in manifest])
    refs = np.asarray(anchor['provenance']['reference_indices'], dtype=np.int64)
    positions = {label: i for i, label in enumerate(labels)}
    # All label reads below are explicitly confined to outer-training rows.
    targets = np.asarray([positions[manifest[i]['speaker_id']] for i in refs])
    inner_truth = np.asarray([positions[manifest[i]['speaker_id']] for i in inner_idx])
    quality = np.zeros((len(manifest), 2), dtype=np.float64)
    for i in np.r_[inner_idx, outer_idx]:
        duration = float(manifest[i]['duration_seconds'])
        rms = float(manifest[i]['mono_rms_dbfs'])
        quality[i] = [np.log1p(np.clip(duration, 0, 180)), np.clip(rms, -100, 0)]
    if not np.isfinite(quality).all():
        raise ValueError('Nonfinite audio quality metadata')
    return {'scores_by_alpha': scores, 'values_by_alpha': values, 'valid': valid,
            'inner_truth': inner_truth, 'groups': groups, 'references': refs,
            'reference_targets': targets, 'quality': quality, 'classes': classes,
            'outer': outer, 'labels': labels}


def _result(prepared, policy, calibration, curves, scores, *, probabilities=None):
    if probabilities is None:
        probabilities = reference_probabilities(scores['outer_known_scores'], scores['outer_unknown_similarity'],
                                                calibration, scores['outer_valid'], .05)
    return {'policy': policy, 'calibration': calibration,
            'inner_macro_f1_447': calibration['inner_macro_f1_447'],
            'scores': scores, 'probabilities': probabilities, 'curves': curves}


def select_baseline(prepared):
    selected, candidates = select_inner_alpha({a: {'known': s['inner_known_scores'], 'unknown': s['inner_unknown_similarity']}
                                               for a, s in prepared['scores_by_alpha'].items()},
                                              prepared['inner_truth'], classes=prepared['classes'] + 1)
    alpha = selected['advanced_weight']
    result = _result(prepared, {'id': 'baseline', 'family': 'baseline', 'kind': 'baseline', 'advanced_weight': alpha},
                     selected['calibration'], candidates[str(alpha)]['curve'], prepared['scores_by_alpha'][alpha])
    result['baseline_alpha_candidates'] = candidates
    return result


def _top_statistics(values, k, *, axis=-1):
    """Finite-only top-k statistics; singleton references are supported."""
    values = np.asarray(values, dtype=np.float32)
    k = min(int(k), values.shape[axis])
    if k < 1:
        raise ValueError('An empty reference population is forbidden')
    top = np.partition(values, values.shape[axis] - k, axis=axis)
    top = np.take(top, np.arange(values.shape[axis] - k, values.shape[axis]), axis=axis)
    finite = np.isfinite(top)
    n = finite.sum(axis=axis)
    if np.any(n == 0):
        raise ValueError('A query has no permitted reference')
    safe = np.where(finite, top, 0).astype(np.float64)
    mean = safe.sum(axis=axis) / n
    variance = np.maximum((safe * safe).sum(axis=axis) / n - mean * mean, 0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _matmul(left, right, device):
    if device == 'cpu':
        return left @ right.T
    if device != 'cuda':
        raise ValueError('Only explicit CPU/CUDA devices are allowed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Requested local CUDA is unavailable')
    with torch.inference_mode():
        return (torch.as_tensor(left, device='cuda') @ torch.as_tensor(right, device='cuda').T).cpu().numpy()


def pair_context(prepared, alpha, device='cpu'):
    source = np.asarray(prepared['values_by_alpha'][alpha], dtype=np.float32)
    lengths = np.linalg.norm(source, axis=1, keepdims=True)
    normalized = np.divide(source, lengths, out=np.zeros_like(source), where=lengths > 1e-8)
    scores = prepared['scores_by_alpha'][alpha]
    inner, outer, refs = scores['calibration_indices'], scores['outer_indices'], prepared['references']
    both = np.r_[inner, outer]
    groups = prepared['groups']
    similarities = np.clip(_matmul(normalized[both], normalized[refs], device), -1., 1.)
    similarities[groups[both, None] == groups[refs][None, :]] = -np.inf
    unknown_columns = np.flatnonzero(prepared['reference_targets'] == 0)
    ref_cohort = np.clip(_matmul(normalized[refs], normalized[refs[unknown_columns]], device), -1., 1.)
    cohort_groups = groups[refs[unknown_columns]]
    ref_cohort[groups[refs, None] == cohort_groups[None, :]] = -np.inf
    return {'similarities': similarities, 'ref_cohort': ref_cohort,
            'query_groups': groups[both], 'cohort_groups': cohort_groups,
            'reference_targets': prepared['reference_targets'], 'unknown_columns': unknown_columns,
            'inner_length': len(inner), 'device': device, 'normalization_cache': {}}


def cohort_statistics(context, topk):
    """Query-specific reference statistics remove the WHOLE query group.

    This extra exclusion matters when an inner unknown query itself belongs to
    the normalization cohort. Excluding self only in the final similarity
    matrix would still leak its representation into reference-side statistics.
    """
    if topk in context['normalization_cache']:
        return context['normalization_cache'][topk]
    sims = context['similarities']
    q_mean, q_std = _top_statistics(sims[:, context['unknown_columns']], topk)
    ref = context['ref_cohort']
    groups, counts = np.unique(context['cohort_groups'], return_counts=True)
    padding = int(counts.max())
    width = min(ref.shape[1], topk + padding)
    indices = np.argsort(ref, axis=1)[:, -width:][:, ::-1]
    top = np.take_along_axis(ref, indices, axis=1)
    top_groups = context['cohort_groups'][indices]
    base_mean, base_std = _top_statistics(top, topk)
    means = np.broadcast_to(base_mean, sims.shape).copy()
    stds = np.broadcast_to(base_std, sims.shape).copy()
    affected = np.flatnonzero(np.isin(context['query_groups'], groups))
    # Bounded CPU batches avoid a query x gallery x complete-cohort tensor.
    for start in range(0, len(affected), 16):
        rows = affected[start:start+16]
        subset = np.broadcast_to(top, (len(rows), *top.shape)).copy()
        excluded = top_groups[None, :, :] == context['query_groups'][rows, None, None]
        subset[excluded] = -np.inf
        means[rows], stds[rows] = _top_statistics(subset, topk)
    result = (q_mean, q_std, means, stds)
    context['normalization_cache'][topk] = result
    return result


def pool_scores(similarities, targets, classes, *, kind='max', blend=0., temperature=.05):
    """Known speakers pool their own references; unknown remains a background."""
    known = np.empty((len(similarities), classes), dtype=np.float32)
    for c in range(classes):
        values = similarities[:, targets == c + 1]
        if values.shape[1] == 0:
            raise ValueError('Missing enrolled speaker')
        maximum = values.max(axis=1)
        if kind == 'top2':
            alternative = _top_statistics(values, 2)[0]
        elif kind == 'logmeanexp':
            n = np.isfinite(values).sum(axis=1)
            if np.any(n == 0) or temperature <= 0:
                raise ValueError('Invalid log-mean-exp references')
            alternative = temperature * (logsumexp(values.astype(np.float64) / temperature, axis=1) - np.log(n))
        elif kind == 'max':
            alternative = maximum
        else:
            raise ValueError('Unimplemented known pooling')
        known[:, c] = (1. - blend) * maximum + blend * alternative
    unknown = similarities[:, targets == 0].max(axis=1)
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError('Scorer produced missing/nonfinite reference evidence')
    return known, unknown


def policy_scores(prepared, alpha, candidate, context):
    base = prepared['scores_by_alpha'][alpha]
    kind = candidate['kind']
    if kind == 'late_fusion':
        p, a = prepared['scores_by_alpha'][0.], prepared['scores_by_alpha'][1.]
        return {**base, **{key: ((1-alpha)*p[key] + alpha*a[key]).astype(np.float32)
                           for key in ('inner_known_scores', 'outer_known_scores', 'inner_unknown_similarity', 'outer_unknown_similarity')}}
    sims = context['similarities']
    if kind in {'density', 'asnorm'}:
        qm, qs, rm, rs = cohort_statistics(context, candidate['cohort_topk'])
        if kind == 'density':
            adjusted = sims - candidate['beta'] * .5 * (qm[:, None] + rm)
        else:
            floor = candidate['std_floor']
            adjusted = .5 * ((sims - qm[:, None]) / np.maximum(qs[:, None], floor)
                             + (sims - rm) / np.maximum(rs, floor))
        known, unknown = pool_scores(adjusted, context['reference_targets'], prepared['classes'])
    elif kind == 'unknown_top3':
        known, unknown = pool_scores(sims, context['reference_targets'], prepared['classes'])
        local_background = _top_statistics(sims[:, context['unknown_columns']], 3)[0]
        unknown = (1-candidate['blend'])*unknown + candidate['blend']*local_background
    else:
        known, unknown = pool_scores(sims, context['reference_targets'], prepared['classes'],
                                    kind=kind, blend=candidate['blend'], temperature=candidate.get('temperature', .05))
    cut = context['inner_length']
    # Normalized/density-corrected values are NOT cosine values: never clip them.
    return {**base, 'inner_known_scores': known[:cut], 'outer_known_scores': known[cut:],
            'inner_unknown_similarity': unknown[:cut], 'outer_unknown_similarity': unknown[cut:]}


def quality_features(prepared, scores, phase):
    known = scores[phase + '_known_scores'].astype(np.float64)
    top = known.max(axis=1)
    second = np.partition(known, -2, axis=1)[:, -2]
    k = min(10, known.shape[1])
    mean = np.partition(known, known.shape[1]-k, axis=1)[:, -k:].mean(axis=1)
    indices = scores['calibration_indices' if phase == 'inner' else 'outer_indices']
    views = prepared['scores_by_alpha']
    agreement = (views[0.][phase + '_known_scores'].argmax(axis=1) == views[1.][phase + '_known_scores'].argmax(axis=1))
    return np.column_stack((top, scores[phase + '_unknown_similarity'], top-second, mean,
                            prepared['quality'][indices], agreement.astype(float)))


def fit_logistic(features, target, truth, penalty):
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x = (features-mean) / scale
    if len(truth) != len(target) or len(np.unique(target)) != 2:
        raise ValueError('Confidence calibration requires both correct and incorrect candidates')
    counts = np.bincount(target.astype(np.int64), minlength=2)
    weights = len(target) / (2. * counts[target.astype(np.int64)])
    design = np.column_stack((np.ones(len(x)), x))
    def objective(coef):
        logits = design @ coef
        loss = np.mean(weights * (np.logaddexp(0., logits) - target*logits)) + .5*penalty*(coef[1:]@coef[1:])
        gradient = design.T @ (weights*(expit(logits)-target)) / len(x)
        gradient[1:] += penalty*coef[1:]
        return loss, gradient
    fit = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Quality calibration did not converge')
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': fit.x.tolist(),
            'feature_names': FEATURE_NAMES, 'penalty': penalty, 'loss_weighting': 'binary_balanced',
            'target': 'top_known_identity_correct', 'converged': True}


def apply_logistic(features, model):
    x = (features - np.asarray(model['mean'])) / np.asarray(model['scale'])
    coef = np.asarray(model['coef'])
    return coef[0] + x @ coef[1:]


def quality_gate(prepared, alpha, candidate):
    scores = prepared['scores_by_alpha'][alpha]
    features = quality_features(prepared, scores, 'inner')
    truth = prepared['inner_truth']
    guess = scores['inner_known_scores'].argmax(axis=1)+1
    target = (truth == guess).astype(float)
    groups = prepared['groups'][scores['calibration_indices']]
    assignment = np.asarray([int(hashlib.sha256(('S011-quality-v1/' + g).encode()).hexdigest()[:8], 16) % 3 for g in groups])
    heldout_logits = np.empty(len(truth))
    cv = []
    for fold in range(3):
        train, test = assignment != fold, assignment == fold
        if not np.any(test) or len(np.unique(target[train])) != 2:
            raise ValueError('Insufficient grouped calibration data for logistic gate')
        model = fit_logistic(features[train], target[train], truth[train], candidate['penalty'])
        heldout_logits[test] = apply_logistic(features[test], model)
        cv.append({'fold': fold, 'fit_rows': int(train.sum()), 'validation_rows': int(test.sum()),
                   'group_disjoint': not bool(set(groups[train]) & set(groups[test])), 'model': model})
    thresholds = np.unique(np.r_[heldout_logits.min()-1e-6, np.quantile(heldout_logits, np.linspace(0,1,201)), heldout_logits.max()+1e-6])
    curve = [{'unknown_weight': 0., 'margin_weight': 0., 'threshold': float(t),
              'inner_macro_f1_447': macro_f1_indices(truth, np.where(heldout_logits>t, guess, 0), prepared['classes']+1)} for t in thresholds]
    selected = max(curve, key=lambda r: (r['inner_macro_f1_447'], r['threshold']))
    model = fit_logistic(features, target, truth, candidate['penalty'])
    policy = {**candidate, 'advanced_weight': alpha, 'quality_model': model,
              'meta_cv': cv, 'threshold_scope': 'threefold_group_out_of_fit_meta_logits'}
    # No outer label is needed for this forward calibration application.
    logits = apply_logistic(quality_features(prepared, scores, 'outer'), model)
    relative = scores['outer_known_scores'].astype(float) - scores['outer_known_scores'].max(axis=1, keepdims=True)
    all_logits = np.column_stack((selected['threshold'] - logits, relative)) / .05
    all_logits -= all_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(all_logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~scores['outer_valid']] = 0; probabilities[~scores['outer_valid'], 0] = 1
    return _result(prepared, policy, dict(selected), curve, scores, probabilities=probabilities)


def _candidate_key(result):
    # Fixed list order breaks exact method ties; baseline gets first preference.
    identifiers = ['baseline'] + [c['id'] for c in CANDIDATES]
    return (result['inner_macro_f1_447'], -identifiers.index(result['policy']['id']),
            -TIE_ORDER.index(result['policy']['advanced_weight']))


def evaluate_policies(prepared, device='cuda', on_progress=None):
    baseline = select_baseline(prepared)
    families, overall, summary, curve_values, ids = {}, baseline, [], [], []
    def keep(result):
        nonlocal overall
        policy = result['policy']; family = policy['family']
        identity = f"{policy['id']}/alpha={policy['advanced_weight']}"
        idx = len(ids); ids.append(identity)
        curve_values.append(np.asarray([[idx, r['unknown_weight'], r['margin_weight'], r['threshold'], r['inner_macro_f1_447']]
                                        for r in result['curves']], dtype=np.float64))
        summary.append({'candidate': identity, 'family': family, 'inner_macro_f1_447': result['inner_macro_f1_447'],
                        'calibration': result['calibration']})
        if family not in families or _candidate_key(result) > _candidate_key(families[family]):
            families[family] = result
        if _candidate_key(result) > _candidate_key(overall):
            overall = result
        if on_progress:
            on_progress(summary[-1])
    keep(baseline)
    for alpha in ALPHAS:
        context = pair_context(prepared, alpha, device)
        for candidate in CANDIDATES:
            if alpha not in candidate.get('alphas', ALPHAS):
                continue
            if candidate['kind'] == 'quality_gate':
                result = quality_gate(prepared, alpha, candidate)
            else:
                scores = policy_scores(prepared, alpha, candidate, context)
                calibration, curve = calibrate_gate(scores['inner_known_scores'], prepared['inner_truth'],
                    scores['inner_unknown_similarity'], UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201, prepared['classes']+1)
                result = _result(prepared, {**candidate, 'advanced_weight': alpha}, calibration, curve, scores)
            keep(result)
        del context
    return {'families': families, 'overall': overall, 'candidate_summary': summary,
            'curves': {'columns': ['candidate_index','unknown_weight','margin_weight','threshold','inner_macro_f1_447'],
                       'values': np.concatenate(curve_values), 'candidate_ids': ids},
            'provenance': {'outer_labels_accessed': False, 'new_encoder_updates': 0,
                           'normalization_cohort': 'outer_training_unknowns_query_and_reference_groups_excluded',
                           'backend': device, 'learned_gate': 'threefold_grouped_meta_predictions_for_threshold_selection'}}
