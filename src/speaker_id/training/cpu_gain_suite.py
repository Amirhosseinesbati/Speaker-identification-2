"""C001: historical control and matched, freshly extracted CPU frontends.

One in-process worker; no resume and no feature-vector uploads to MLflow.
Scoring and all historical source validators remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np

from speaker_id.training import gain_suite as gain

FRONTENDS = ('identity', 'gain')
RECIPES = ('C001a', 'C001b', 'C001c', 'C001d')
LIMITATIONS = gain.LIMITATIONS + [
    'Historical GPU versus fresh CPU identity differences are numerical controls, not gain effects.',
    'The gain contrast uses fresh CPU identity and fresh CPU gain with identical frozen models and threads.',
    'C001d chooses only between fresh CPU policies; historical GPU vectors are never a selection arm.',
    'Feature NPZs stay on the server and are transferred directly to local storage, never to MLflow.',
    'Single worker, no automatic retry or resume; the eight-hour extraction budget stops at a pair boundary.',
]


def _metadata(value):
    """Accept JSON metadata, never ndarray/bytes or other serialized tensors."""
    if type(value) is dict:
        gain.require(all(type(key) is str for key in value), 'Metadata keys must be strings')
        return {key: _metadata(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_metadata(item) for item in value]
    gain.require(value is None or type(value) in (str, int, float, bool), 'Only JSON metadata may be tracked')
    if type(value) is float:
        gain.require(np.isfinite(value), 'Nonfinite metadata')
    return value


def extraction_callback(output, tracker):
    """Persist immutable, JSON-only extraction evidence and strictly sync it."""
    directory = output / 'cpu_extraction_evidence'
    directory.mkdir(exist_ok=False)
    sequence = 0

    def record(stage, payload):
        nonlocal sequence
        gain.require(stage in ('identities', 'progress', 'complete', 'failure'), 'Unknown worker evidence stage')
        clean = _metadata(payload)
        gain.require(type(clean) is dict, 'Worker evidence must be a JSON object')
        sequence += 1
        path = directory / f'{sequence:04d}_{stage}.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'stage': stage, 'payload': clean}, stream, allow_nan=False, indent=2)
        tracker.add_artifact(path, 'cpu_extraction/' + path.name)
        if stage == 'progress':
            metrics = {key: clean[key] for key in ('completed_pairs', 'total', 'elapsed_seconds')}
            gain.require(all(type(v) in (int, float) and np.isfinite(v) for v in metrics.values()),
                         'Malformed scalar progress')
            tracker.log_metrics({'extraction/' + key: value for key, value in metrics.items()},
                                step=int(metrics['completed_pairs']), sync=False)
            print(json.dumps({'stage': 'cpu_paired_extraction', **metrics}), flush=True)
        tracker.flush(strict=True)
    return record


def verify_cpu_extraction(output, extracted, contract, original_valid):
    gain.require(set(extracted['vectors']) == set(FRONTENDS)
        and set(extracted['identities']) == set(FRONTENDS)
        and set(extracted['receipts']) == set(FRONTENDS), 'Exactly two fresh CPU caches required')
    gain.require(extracted['valid'].dtype == np.bool_
        and np.array_equal(extracted['valid'], original_valid), 'Original zero-signal rows changed')
    for frontend in FRONTENDS:
        values, valid = gain.verify_gain_cache(output / (frontend + '_embedding_cache'),
            extracted['identities'][frontend], contract['manifest'], extracted['receipts'][frontend])
        gain.require(np.array_equal(valid, original_valid)
            and set(extracted['vectors'][frontend]) == {'public', 'advanced'}
            and all(np.array_equal(values[name], extracted['vectors'][frontend][name])
                    for name in ('public', 'advanced')), 'Returned CPU arrays differ from attested files')
    gain.require(extracted['identities']['identity']['signature'] != extracted['identities']['gain']['signature'],
                 'CPU frontend identities must be distinct')


def historical_control(output, tracker, contract, sources):
    """Exactly reproduce S008c before constructing any new encoder."""
    historical = sources['selection']['directory'] / 'S008c'
    labels = {label: index for index, label in enumerate(contract['labels'])}
    predictions, folds, checks, query_indices, truths = [], [], {}, {}, {}
    for outer in (0, 1):
        scores = gain.family_scores(sources['vectors']['public'], sources['vectors']['advanced'],
                                    sources['valid'], contract, outer)
        query_indices[outer] = scores[0.0]['calibration_indices'].copy()
        truths[outer] = np.asarray([labels[contract['manifest'][int(i)]['speaker_id']]
                                   for i in query_indices[outer]])
        selected, curves = gain.select_inner_alpha(gain.inner_arrays(scores), truths[outer])
        fitted = {'selected': selected, 'candidates': curves}
        gain.require(gain.read(historical / f'fold_{outer}/inner_alpha_calibration.json') == fitted,
                     'S008 alpha/gate curves changed')
        alpha = selected['advanced_weight']
        rows, report, check = gain._record_fold(output / f'fold_{outer}', tracker, contract, outer,
            scores[alpha], sources['valid'], selected['calibration'], curves[str(alpha)]['curve'], selected, historical)
        check.update(gain.verify_saved_control_arrays(output / f'fold_{outer}', historical / f'fold_{outer}'))
        gain.write_json(output / f'fold_{outer}/inner_alpha_calibration.json', fitted)
        tracker.add_artifact(output / f'fold_{outer}/inner_alpha_calibration.json',
                             f'fold_{outer}/inner_alpha_calibration.json')
        predictions.extend(rows)
        folds.append(report)
        checks[str(outer)] = check
    gain.require(gain.score_predictions(contract['manifest'], predictions, contract['labels'])
        == gain.read(historical / 'experiment_report.json')['oof'], 'S008 pooled control changed')
    control = {**gain.verify_predictions(historical / 'oof_predictions.csv', predictions),
        'exact_pooled_metrics': True, 'exact_inner_alpha_curves': True,
        'exact_probability_and_support_arrays': True, 'folds': checks}
    report = gain.finish_recipe(output, tracker, contract, predictions, folds, 'C001a', control)
    tracker.verify_artifacts()
    tracker.verify_remote_metadata()
    return report, predictions, control, query_indices, truths


def freeze_cpu_choices(output, tracker, contract, extracted, query_indices, truths):
    """Seal BOTH fold policies before any fresh outer-label reporting."""
    scored, choices, fits = {}, {}, {}
    for outer in (0, 1):
        scored[outer] = {}
        for frontend in FRONTENDS:
            vectors = extracted['vectors'][frontend]
            scored[outer][frontend] = gain.family_scores(vectors['public'], vectors['advanced'],
                                                        extracted['valid'], contract, outer)
            gain.require(all(np.array_equal(item['calibration_indices'], query_indices[outer])
                             for item in scored[outer][frontend].values()), 'CPU inner query indices changed')
        choices[outer], fits[outer] = gain.select_inner_frontend(
            {name: gain.inner_arrays(scored[outer][name]) for name in FRONTENDS}, truths[outer])
    frozen = {'selected': choices, 'inner_fits': fits, 'candidate_frontends': list(FRONTENDS),
        'alphas_per_frontend': 5, 'no_fresh_outer_evaluation_performed_yet': True,
        'selection_policy': gain.SELECTION, 'historical_GPU_excluded_from_selection': True}
    gain.write_json(output / 'frozen_inner_choices.json', frozen)
    tracker.add_artifact(output / 'frozen_inner_choices.json')
    tracker.flush(strict=True)
    tracker.verify_artifacts()
    tracker.verify_remote_metadata()
    return scored, frozen


def selected_cpu_policy(recipe, outer, frozen):
    gain.require(set(frozen['selected']) == {0, 1} and set(frozen['inner_fits']) == {0, 1}
        and frozen['no_fresh_outer_evaluation_performed_yet'] is True
        and frozen['historical_GPU_excluded_from_selection'] is True, 'Both CPU fold choices must be sealed')
    gain.require(recipe in RECIPES[1:], 'Unknown CPU recipe')
    if recipe == 'C001d':
        selected = frozen['selected'][outer]
    else:
        frontend = 'identity' if recipe == 'C001b' else 'gain'
        selected = {'frontend': frontend, **frozen['inner_fits'][outer][frontend]['selected']}
    gain.require(selected['frontend'] in FRONTENDS, 'Only fresh CPU policies can be selected')
    return selected


def execute_cpu_gain_suite(root, config_path, suite, contract, source_config, binding_path):
    from speaker_id.training.cpu_pair_contract import validate_cpu_gain_config, prepare_cpu_execution
    from speaker_id.training.cpu_pair_worker import extract_cpu_pair_caches
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding

    validate_cpu_gain_config(suite)
    execution = prepare_cpu_execution(root, suite)
    binding = ExperimentBinding(**gain.read(binding_path)['binding'])
    output = root / suite['output_root'] / ('C001_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                           + '_' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {'suite_config': config_path, 'source_release_config': root / suite['source_release_config'],
              'launcher': root / 'scripts/score_gain_cpu.py'}
    inputs.update({key: root / contract['config'][key]
                   for key in ('manifest', 'folds', 'roles', 'label_map', 'model_config')})
    resolved = {'suite': suite, 'source_release_config': source_config, 'cpu_execution': execution,
        'data_readiness_contract': {key: contract[key] for key in ('config', 'model', 'input_hashes', 'code_hashes', 'signature')},
        'limitations': LIMITATIONS, 'embedding_artifacts_uploaded': False, 'encoder_updates': 0}
    gain.write_json(output / 'resolved_config.json', resolved)
    common = {'project_root': root, 'binding': binding, 'input_paths': inputs,
              'run_kind': 'matched_cpu_frozen_gain_comparison', 'training_started': False}
    parent = DurableMLflowRun.prepare(spool_dir=output / 'tracking', run_name=suite['run_name'], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        gain.write_json(output / 'experiment_state.json', {'status': 'running', 'parent_run_id': parent.run_id})
        sources = gain.load_sources(root, source_config, verify_audio=True)
        gain.require(sources['contract']['signature'] == contract['signature'], 'Source changed after CPU readiness')
        remote = gain._verify_remote_evidence(parent.client, binding, sources['remote_requests'], output / 'verified_remote_evidence')
        gain.write_json(output / 'source_provenance.json', {'historical_sources': sources['proof'],
            'remote_evidence': remote, 'raw_audio_sha_verified': True, 'cp001': execution['cp001_evidence']})
        parent.add_artifact(output / 'source_provenance.json')
        for name in ('suite_config', 'source_release_config', 'launcher'):
            parent.add_artifact(inputs[name], 'input_configs/' + name + inputs[name].suffix)
        parent.flush(strict=True)
        parent.verify_artifacts()
        results, prediction_sets = [], {}
        path = output / 'C001a'
        path.mkdir()
        child = DurableMLflowRun.prepare(spool_dir=path / 'tracking', parent_run_id=parent.run_id,
            run_name='C001a-exact-S008c-control', config={**resolved, 'recipe': 'C001a'}, **common)
        child.flush(strict=True)
        result, predictions, control, query_indices, truths = historical_control(path, child, contract, sources)
        results.append(result)
        prediction_sets['C001a'] = predictions
        child = None
        gain.write_json(output / 'source_control_checks.json', control)
        parent.add_artifact(output / 'source_control_checks.json')
        parent.flush(strict=True)
        extracted = extract_cpu_pair_caches(root, suite, contract, sources, output, control, execution['backend'],
                                             progress_callback=extraction_callback(output, parent))
        verify_cpu_extraction(output, extracted, contract, sources['valid'])
        scored, frozen = freeze_cpu_choices(output, parent, contract, extracted, query_indices, truths)
        for recipe in RECIPES[1:]:
            path = output / recipe
            path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=path / 'tracking', parent_run_id=parent.run_id,
                run_name=recipe + '-' + suite['recipes'][RECIPES.index(recipe)].split('_', 1)[1],
                config={**resolved, 'recipe': recipe, 'cpu_identities': extracted['identities']}, **common)
            child.flush(strict=True)
            for filename in ('source_control_checks.json', 'frozen_inner_choices.json',
                             'identity_cache_manifest.json', 'gain_cache_manifest.json', 'paired_execution_report.json'):
                child.add_artifact(output / filename)
            predictions, folds = [], []
            for outer in (0, 1):
                chosen = selected_cpu_policy(recipe, outer, frozen)
                frontend, alpha = chosen['frontend'], chosen['advanced_weight']
                curve = frozen['inner_fits'][outer][frontend]['candidates'][str(alpha)]['curve']
                rows, report, _ = gain._record_fold(path / f'fold_{outer}', child, contract, outer,
                    scored[outer][frontend][alpha], extracted['valid'], chosen['calibration'], curve, chosen)
                if recipe == 'C001d':
                    endpoint = 'C001b' if frontend == 'identity' else 'C001c'
                    gain.verify_predictions(output / endpoint / f'fold_{outer}/predictions.csv', rows)
                    gain.verify_saved_control_arrays(path / f'fold_{outer}', output / endpoint / f'fold_{outer}')
                child.log_metrics({f'fold_{outer}/selected_advanced_weight': alpha,
                    f'fold_{outer}/selected_gain_frontend': int(frontend == 'gain')}, sync=True)
                predictions.extend(rows)
                folds.append(report)
                print(json.dumps({'stage': 'cpu_gain_scoring', 'recipe': recipe, 'fold': outer,
                    'frontend': frontend, 'advanced_weight': alpha, 'macro_f1': report['outer']['macro_f1']}), flush=True)
            results.append(gain.finish_recipe(path, child, contract, predictions, folds, recipe, control))
            prediction_sets[recipe] = predictions
            child.verify_artifacts()
            child.verify_remote_metadata()
            child = None
        by_recipe = {row['recipe']: row for row in results}
        comparisons = {}
        for base, candidate in (('C001a', 'C001b'), ('C001b', 'C001c'), ('C001b', 'C001d')):
            comparisons[candidate + '_vs_' + base] = {
                'pooled_macro_f1_delta': by_recipe[candidate]['oof']['macro_f1'] - by_recipe[base]['oof']['macro_f1'],
                'paired_quality_slices': gain.paired_diagnostics(contract['manifest'], prediction_sets[base],
                                                                prediction_sets[candidate], contract['labels'])}
        deltas = [by_recipe['C001d']['folds'][fold]['outer']['macro_f1']
                  - by_recipe['C001b']['folds'][fold]['outer']['macro_f1'] for fold in (0, 1)]
        rule = suite['decision_rule']
        decision = {'candidate_recipe': 'C001d', 'baseline_recipe': 'C001b', 'rule': rule,
            'meets_preregistered_development_rule': comparisons['C001d_vs_C001b']['pooled_macro_f1_delta']
                >= rule['minimum_pooled_improvement'] and min(deltas) >= -rule['maximum_fold_decline'],
            'fold_macro_f1_deltas': deltas, 'P002_unchanged': True, 'leaderboard_validation': 'pending'}
        report = {'status': 'complete', 'parent_run_id': parent.run_id, 'results': results,
            'comparisons': comparisons, 'decision': decision, 'source_control_checks': control,
            'selection_policy': gain.SELECTION, 'gain_policy': suite['gain_policy'], 'cpu_execution': execution,
            'paired_execution_report': extracted['execution_report'], 'embedding_artifacts_uploaded': False,
            'encoder_updates': 0, 'elapsed_seconds': time.monotonic() - started, 'limitations': LIMITATIONS}
        gain.write_json(output / 'experiment_report.json', report)
        for name in ('experiment_report.json', 'resolved_config.json'):
            parent.add_artifact(output / name)
        parent.log_metrics({row['recipe'] + '/oof_macro_f1_447': row['oof']['macro_f1'] for row in results}, sync=False)
        parent.write_report(report, markdown='# C001 matched CPU gain comparison\n\n'
            'Exact historical S008c control precedes fresh CPU extraction. Both fold inner choices were saved '
            'before any fresh outer evaluation. GPU-to-CPU identity drift is reported separately from gain. '
            'Both encoders frozen; feature caches remain server/local only. Development results, not leaderboard scores.\n')
        parent.finish('FINISHED', strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        gain.write_json(output / 'experiment_state.json', {'status': 'complete', 'parent_run_id': parent.run_id})
        return {'output': str(output), 'parent_run_id': parent.run_id,
                'results': {row['recipe']: row['oof']['macro_f1'] for row in results}, 'decision': decision}
    except BaseException as error:
        failure = {'status': 'failed', 'parent_run_id': parent.run_id, 'error_type': type(error).__name__,
                   'error': parent.redactor.text(str(error)),
                   'embedding_artifacts_uploaded': False, 'encoder_updates': 0, 'automatic_retry': False}
        gain.write_json(output / 'failure.json', failure)
        gain.write_json(output / 'experiment_state.json', failure)
        for tracker in (child, parent):
            if tracker is not None:
                try:
                    tracker.write_report(failure)
                    tracker.finish('FAILED', strict=False)
                except Exception:
                    pass  # Preserve the original error and already durable evidence.
        raise
