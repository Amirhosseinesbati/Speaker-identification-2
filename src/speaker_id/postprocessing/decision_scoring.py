"""S012: nested decision heads over unchanged CAM++ identity rankings.

Only callers holding outer-training truth can build the meta cases. The decision
head sees numerical scores and optional audio quality, never names, labels,
enrollment counts or the original outer-fold truth.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from speaker_id.postprocessing.scoring import _matmul, _top_statistics
from speaker_id.training.reference_scoring import gate_scores
from speaker_id.training.scoring import macro_f1_indices


CORE_FEATURE_NAMES = [
    'baseline_margin', 'fused_known_top', 'fused_known_runner_up',
    'fused_known_gap', 'fused_known_third_gap', 'fused_known_top10_mean',
    'fused_known_all_mean', 'fused_known_all_std', 'fused_unknown_top',
    'fused_unknown_top3_mean', 'fused_unknown_top10_mean',
    'fused_unknown_top50_mean', 'fused_unknown_top50_std',
    'public_score_at_fused_winner', 'advanced_score_at_fused_winner',
    'public_known_top', 'advanced_known_top', 'public_gap', 'advanced_gap',
    'public_unknown_top', 'advanced_unknown_top',
    'public_known_minus_unknown', 'advanced_known_minus_unknown',
    'encoder_winner_agreement', 'public_matches_fused', 'advanced_matches_fused',
]
FEATURE_NAMES = CORE_FEATURE_NAMES + ['log1p_duration_capped180', 'rms_dbfs_clipped']
FEATURE_SETS = ['scores_only', 'scores_quality']
MODES = [
    {'mode': direction, 'band': band}
    for direction in ('veto', 'rescue', 'two_way') for band in (.025, .05)
] + [{'mode': 'full', 'band': None}]
META_MIN_GAIN = .001
META_MAX_FOLD_LOSS = .002
THRESHOLD_QUANTILES = 201
PROBABILITY_POSITIVE_MARGIN_FLOOR = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decision_features(prepared, baseline, scope, device='cuda'):
    """No truth access; baseline calibration must come from the fitting pool."""
    require(scope in ('inner', 'outer'), 'Unknown decision feature scope')
    alpha = baseline['policy']['advanced_weight']
    scores = prepared['scores_by_alpha'][alpha]
    key = 'calibration_indices' if scope == 'inner' else 'outer_indices'
    indices = scores[key]
    known = scores[scope + '_known_scores']
    unknown = scores[scope + '_unknown_similarity']
    require(known.ndim == 2 and known.shape[1] >= 2, 'At least two known classes are required')
    order = np.sort(known.astype(np.float64), axis=1)[:, ::-1]
    guess = known.argmax(axis=1)
    cal = baseline['calibration']
    margin = gate_scores(known, unknown, cal['unknown_weight'], cal['margin_weight']) - cal['threshold']
    source = prepared['values_by_alpha'][alpha]
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    normalized = np.divide(source, norms, out=np.zeros_like(source), where=norms > 1e-8)
    unknown_refs = prepared['references'][prepared['reference_targets'] == 0]
    similarities = np.clip(_matmul(normalized[indices], normalized[unknown_refs], device), -1., 1.)
    similarities[prepared['groups'][indices, None] == prepared['groups'][unknown_refs][None, :]] = -np.inf
    require(np.allclose(similarities.max(axis=1), unknown, rtol=0, atol=2e-6),
            'Background features differ from the fixed cosine scorer')
    mean3, _ = _top_statistics(similarities, 3)
    mean10, _ = _top_statistics(similarities, 10)
    mean50, std50 = _top_statistics(similarities, 50)
    public = prepared['scores_by_alpha'][0.0]
    advanced = prepared['scores_by_alpha'][1.0]
    p = public[scope + '_known_scores']
    a = advanced[scope + '_known_scores']
    pt, at = p.max(axis=1), a.max(axis=1)
    po = np.sort(p, axis=1)[:, -2]
    ao = np.sort(a, axis=1)[:, -2]
    pu, au = public[scope + '_unknown_similarity'], advanced[scope + '_unknown_similarity']
    pg, ag = p.argmax(axis=1), a.argmax(axis=1)
    row = np.arange(len(indices))
    features = np.column_stack([
        margin, order[:, 0], order[:, 1], order[:, 0] - order[:, 1],
        order[:, 0] - order[:, min(2, known.shape[1]-1)], order[:, :10].mean(axis=1),
        order.mean(axis=1), order.std(axis=1), unknown, mean3, mean10, mean50, std50,
        p[row, guess], a[row, guess], pt, at, pt-po, at-ao, pu, au, pt-pu, at-au,
        pg == ag, pg == guess, ag == guess, prepared['quality'][indices, 0],
        prepared['quality'][indices, 1],
    ]).astype(np.float64)
    require(features.shape == (len(indices), len(FEATURE_NAMES)) and np.isfinite(features).all(),
            'Nonfinite or misaligned decision features')
    valid = np.ones(len(indices), dtype=bool) if scope == 'inner' else scores['outer_valid'].copy()
    return {'features': features, 'feature_names': list(FEATURE_NAMES), 'guess': guess+1,
            'margin': margin, 'valid': valid, 'indices': indices.copy(),
            'known_scores': known, 'unknown_similarity': unknown}


def feature_columns(name):
    require(name in FEATURE_SETS, 'Unregistered decision feature set')
    return np.arange(len(CORE_FEATURE_NAMES) if name == 'scores_only' else len(FEATURE_NAMES))


def eligible_rows(margin, mode, band):
    margin = np.asarray(margin, dtype=float)
    require(np.isfinite(margin).all() and mode in ('full', 'veto', 'rescue', 'two_way'),
            'Invalid decision cascade')
    if mode == 'full':
        require(band is None, 'Full gate has no uncertainty band')
        return np.ones(len(margin), dtype=bool)
    require(band in (.025, .05), 'Use a preregistered uncertainty band')
    eligible = np.abs(margin) <= band
    if mode == 'veto':
        eligible &= margin > 0
    elif mode == 'rescue':
        eligible &= margin <= 0
    return eligible


def apply_cascade(margin, confidence, mode, band, threshold):
    """Zero margin rejects; one-direction cascades cannot reverse the other side."""
    margin, confidence = np.asarray(margin, dtype=float), np.asarray(confidence, dtype=float)
    require(margin.shape == confidence.shape and margin.ndim == 1
            and np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1))
            and np.isfinite(threshold), 'Invalid decision head probabilities')
    active = eligible_rows(margin, mode, band)
    output = margin.copy()
    learned = confidence - threshold
    if mode == 'veto':
        active &= learned <= 0
    elif mode == 'rescue':
        active &= learned > 0
    output[active] = learned[active]
    return output


def decision_probabilities(known, margin, valid, temperature=.05):
    known, margin, valid = np.asarray(known), np.asarray(margin), np.asarray(valid)
    require(known.ndim == 2 and margin.shape == (len(known),) and valid.shape == margin.shape
            and valid.dtype == np.bool_ and np.isfinite(known).all() and np.isfinite(margin).all()
            and temperature > 0, 'Invalid decision-score inputs')
    relative = known.astype(np.float64) - known.max(axis=1, keepdims=True)
    # Positive sub-ULP margins must keep the known decision after softmax; an
    # exact zero deliberately remains an unknown-first tie. These normalized
    # decision scores are not calibrated posterior probabilities.
    display_margin = np.where(margin > 0, np.maximum(margin, PROBABILITY_POSITIVE_MARGIN_FLOOR), margin)
    logits = np.column_stack((-display_margin, relative)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~valid] = 0
    probabilities[~valid, 0] = 1
    expected = np.where(valid & (margin > 0), known.argmax(axis=1) + 1, 0)
    require(np.array_equal(probabilities.argmax(axis=1), expected),
            'Normalized decision scores changed the selected identity')
    return probabilities


def payload_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def select_curve(truth, guess, margin, confidence, assignments, mode, band, classes):
    active = eligible_rows(margin, mode, band)
    population = confidence[active] if np.any(active) else confidence
    thresholds = np.unique(np.r_[population.min()-1e-6,
        np.quantile(population, np.linspace(0, 1, THRESHOLD_QUANTILES)), population.max()+1e-6])
    base_pred = np.where(margin > 0, guess, 0)
    rows = []
    for threshold in thresholds:
        score = apply_cascade(margin, confidence, mode, band, float(threshold))
        prediction = np.where(score > 0, guess, 0)
        row = {'threshold': float(threshold), 'meta_macro_f1_447': macro_f1_indices(truth, prediction, classes),
               'changed_from_meta_baseline': int(np.sum(prediction != base_pred))}
        row['meta_fold_macro_f1_447'] = [macro_f1_indices(truth[assignments == f], prediction[assignments == f], classes)
                                         for f in range(3)]
        rows.append(row)
    selected = max(rows, key=lambda x: (x['meta_macro_f1_447'], -x['changed_from_meta_baseline'], x['threshold']))
    return dict(selected), rows


def select_policy_summary(candidates, baseline_f1, baseline_fold_f1):
    """Family comparisons are exploratory; the robust selector has a fallback."""
    families = {}
    for i, row in enumerate(candidates):
        row['candidate_order'] = i
        row['meta_gain'] = row['meta_macro_f1_447'] - baseline_f1
        row['meta_fold_delta'] = [a-b for a, b in zip(row['meta_fold_macro_f1_447'], baseline_fold_f1)]
        row['eligible_for_overall'] = (row['meta_gain'] >= META_MIN_GAIN
                                      and min(row['meta_fold_delta']) >= -META_MAX_FOLD_LOSS)
        key = (row['meta_macro_f1_447'], -row['changed_from_meta_baseline'], -i)
        family = row['family']
        if family not in families or key > families[family][0]:
            families[family] = (key, row)
    admissible = [row for row in candidates if row['eligible_for_overall']]
    overall = max(admissible, key=lambda r: (r['meta_macro_f1_447'], -r['changed_from_meta_baseline'],
                                           -r['candidate_order'])) if admissible else None
    return {name: row for name, (_, row) in families.items()}, overall


def evaluate_decisions(prepared, baseline, nested, *, device='cuda', on_progress=None):
    from speaker_id.postprocessing.tree_models import MODEL_SPECS, fit_export, predict_export
    q = np.asarray(nested['query_global_indices'])
    require(np.array_equal(q, prepared['scores_by_alpha'][0.]['calibration_indices']),
            'Meta validation must cover the original eligible calibration queries')
    lookup = {int(index): i for i, index in enumerate(q)}
    truth = np.asarray(prepared['inner_truth'])
    guess = np.empty(len(q), dtype=np.int64)
    margin = np.empty(len(q), dtype=float)
    assigned = np.full(len(q), -1, dtype=np.int64)
    for case in nested['cases']:
        validation = case['validation']
        rows = np.asarray([lookup[int(i)] for i in validation['global_indices']])
        require(np.all(assigned[rows] == -1) and np.array_equal(truth[rows], validation['truth']),
                'Duplicate or mislabeled meta validation queries')
        assigned[rows] = case['meta_fold']
        guess[rows], margin[rows] = validation['guess'], validation['margin']
    require(np.all(assigned >= 0) and np.array_equal(assigned, nested['assignments']),
            'Incomplete or inconsistent nested meta folds')
    classes = prepared['classes'] + 1
    base_pred = np.where(margin > 0, guess, 0)
    base_f1 = macro_f1_indices(truth, base_pred, classes)
    base_folds = [macro_f1_indices(truth[assigned == f], base_pred[assigned == f], classes) for f in range(3)]
    confidence_by_model, models, candidates, all_curves = {}, {}, [], []
    for spec in MODEL_SPECS:
        for feature_set in FEATURE_SETS:
            columns = feature_columns(feature_set)
            model_key = spec['id'] + '/' + feature_set
            confidence = np.full(len(q), np.nan)
            models[model_key] = []
            for case in nested['cases']:
                fit, validation = case['fit'], case['validation']
                target = (fit['truth'] == fit['guess']).astype(np.int64)
                payload = fit_export(spec, fit['features'][:, columns], target,
                                     check_features=validation['features'][:, columns])
                rows = np.asarray([lookup[int(i)] for i in validation['global_indices']])
                confidence[rows] = predict_export(payload, validation['features'][:, columns])
                models[model_key].append({'meta_fold': case['meta_fold'], 'payload': payload})
            require(np.isfinite(confidence).all(), 'Missing out-of-fit decision probabilities')
            confidence_by_model[model_key] = confidence
            for mode in MODES:
                selected, curve = select_curve(truth, guess, margin, confidence, assigned,
                                                mode['mode'], mode['band'], classes)
                candidate_id = f"{model_key}/{mode['mode']}/{mode['band']}"
                row = {'id': candidate_id, 'model_key': model_key, 'model_spec': spec,
                       'feature_set': feature_set, 'family': spec['family'], **mode, **selected}
                candidates.append(row)
                for entry in curve:
                    all_curves.append([len(candidates)-1, entry['threshold'], entry['meta_macro_f1_447'],
                                       entry['changed_from_meta_baseline'], *entry['meta_fold_macro_f1_447']])
            if on_progress:
                best = max(candidates[-len(MODES):], key=lambda r: r['meta_macro_f1_447'])
                on_progress({'model_key': model_key, 'completed_model_feature_pairs': len(confidence_by_model),
                             'best_meta_macro_f1_447': best['meta_macro_f1_447'], 'meta_baseline_macro_f1_447': base_f1})
    winners, overall = select_policy_summary(candidates, base_f1, base_folds)
    fit = decision_features(prepared, baseline, 'inner', device)
    outer = decision_features(prepared, baseline, 'outer', device)
    final_models = {}
    results = {}
    for family, selected in {**winners, 'overall': overall}.items():
        if selected is None:
            results[family] = {**baseline, 'model': None, 'meta_selection': {
                'baseline_retained': True, 'meta_macro_f1_447': base_f1, 'meta_fold_macro_f1_447': base_folds,
                'reason': 'No candidate passed the preregistered pooled-gain and fold-loss requirements'}}
            continue
        key = selected['model_key']
        columns = feature_columns(selected['feature_set'])
        if key not in final_models:
            final_models[key] = fit_export(selected['model_spec'], fit['features'][:, columns],
                (truth == fit['guess']).astype(np.int64), check_features=outer['features'][:, columns])
        payload = final_models[key]
        confidence = predict_export(payload, outer['features'][:, columns])
        output_margin = apply_cascade(outer['margin'], confidence, selected['mode'], selected['band'], selected['threshold'])
        probabilities = decision_probabilities(outer['known_scores'], output_margin, outer['valid'])
        policy = {'kind': 'nested_tree_decision', 'id': selected['id'], 'family': selected['family'],
                  'mode': selected['mode'], 'band': selected['band'], 'feature_set': selected['feature_set'],
                  'feature_names': [FEATURE_NAMES[i] for i in columns], 'model_spec': selected['model_spec'],
                  'model_sha256': payload_sha256(payload), 'base_policy': baseline['policy'],
                  'base_calibration': baseline['calibration'], 'advanced_weight': baseline['policy']['advanced_weight'],
                  'probability_positive_margin_floor': PROBABILITY_POSITIVE_MARGIN_FLOOR,
                  'threshold': selected['threshold'], 'identity_ranking': 'unchanged_S008c_max_reference'}
        scores = {**baseline['scores'], 'outer_base_margin': outer['margin'],
                  'outer_decision_margin': output_margin, 'outer_tree_confidence': confidence}
        results[family] = {'policy': policy, 'calibration': {'kind': 'nested_tree_decision_confidence_threshold',
                'threshold': selected['threshold'],
                'unknown_weight': 0., 'margin_weight': 0., 'inner_macro_f1_447': selected['meta_macro_f1_447']},
                'inner_macro_f1_447': selected['meta_macro_f1_447'], 'meta_selection': selected,
                'probabilities': probabilities, 'scores': scores, 'model': payload}
    return {'results': results, 'candidate_summary': candidates, 'meta_models': models,
            'final_models': final_models, 'full_fit_features': fit, 'outer_features': outer,
            'meta_predictions': {'query_global_indices': q, 'truth': truth, 'guess': guess, 'baseline_margin': margin,
                                 'assignments': assigned, **{'confidence_' + k.replace('/', '__'): v for k, v in confidence_by_model.items()}},
            'curves': {'columns': ['candidate_index', 'threshold', 'meta_macro_f1_447', 'changed_from_meta_baseline',
                                    'fold0_meta_macro_f1_447', 'fold1_meta_macro_f1_447', 'fold2_meta_macro_f1_447'],
                       'values': np.asarray(all_curves, dtype=np.float64)},
            'selection': {'baseline_meta_macro_f1_447': base_f1, 'baseline_meta_fold_macro_f1_447': base_folds,
                          'overall': overall, 'family_ids': {k: v['id'] for k, v in winners.items()},
                          'minimum_pooled_gain': META_MIN_GAIN, 'maximum_meta_fold_loss': META_MAX_FOLD_LOSS,
                          'meta_gallery_and_baseline_refitted_per_split': True, 'outer_labels_used': False}}
