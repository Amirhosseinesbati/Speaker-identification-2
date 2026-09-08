"""S010: one fixed waveform-gain hypothesis against the verified S008 procedure.

The historical extractors are unchanged. Candidate feature caches are separate,
and the frontend/encoder mixture/gate are selected only on inner queries.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.packaging.selected_sources import load_sources, verify_selection
from speaker_id.training.adaptation_comparison import _verify_remote_evidence, paired_diagnostics
from speaker_id.training.candidate_comparison import encoder_state_sha256
from speaker_id.training.candidate_fusion import (
    ALPHAS, TIE_ORDER, scores_for_alpha, select_inner_alpha, _record_fold,
)
from speaker_id.training.contracts import load_contract
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.fusion_suite import project_path, verify_predictions
from speaker_id.training.runner import write_csv, write_json
from speaker_id.evaluation.metrics import score_predictions

SOURCE_CONFIG = 'configs/package/campp_selected.json'
SOURCE_CONFIG_SHA = '1f0281a98f60ccfd7c309a67a2dec3936b1c787f3f8b26d43b53f743eb17ae2b'
SELECTION = 'Choose frontend, alpha and gate using group-excluded inner queries only; ties prefer identity.'
DECISION = {'minimum_pooled_improvement': .003, 'maximum_fold_decline': .005}
LIMITATIONS = [
    'Repeatedly observed development folds are not a hidden test or a leaderboard score.',
    'Gain is one fixed hypothesis motivated by observational error slices, not an established causal repair.',
    'Louder noise may hurt unknown rejection; no extra denoising, VAD or gain constants are searched.',
    'All 4529 outer files and 447 classes are evaluated, including 89 zero-signal unknown fallbacks.',
    'Both public encoders remain frozen; feature extraction and inner calibration are not encoder training.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_gain_suite(suite):
    from speaker_id.audio.gain import GAIN_POLICY, validate_gain_policy
    fields = {'schema_version', 'experiment_code', 'run_name', 'readiness_config', 'output_root',
              'source_release_config', 'source_release_config_sha256', 'gain_policy', 'alphas',
              'alpha_tie_order', 'unknown_weights', 'margin_weights', 'threshold_candidates',
              'probability_temperature', 'selection_policy', 'decision_rule', 'recipes'}
    require(isinstance(suite, dict) and set(suite) == fields, 'Incomplete or extra S010 configuration')
    require(type(suite['schema_version']) is int and suite['schema_version'] == 1 and suite['experiment_code'] == 'S010'
        and suite['run_name'] == 'S010-campp-fixed-rms-boost-controlled'
        and suite['readiness_config'] == 'configs/train/campp_coverage.json'
        and suite['output_root'] == 'artifacts/training'
        and suite['source_release_config'] == SOURCE_CONFIG
        and suite['source_release_config_sha256'] == SOURCE_CONFIG_SHA
        and suite['gain_policy'] == GAIN_POLICY and suite['alphas'] == list(ALPHAS)
        and suite['alpha_tie_order'] == list(TIE_ORDER)
        and suite['unknown_weights'] == [0.0, .25, .5, .75, 1.0]
        and suite['margin_weights'] == [0.0, .5] and type(suite['threshold_candidates']) is int
        and suite['threshold_candidates'] == 201 and suite['probability_temperature'] == .05
        and suite['selection_policy'] == SELECTION and suite['decision_rule'] == DECISION
        and suite['recipes'] == ['S010a_identity_control', 'S010b_gain_only', 'S010c_inner_frontend_choice'],
        'S010 must keep its fixed preregistered policy, grids, controls and decision rule')
    require(all(isinstance(suite[key], list) and all(type(value) is float for value in suite[key])
                for key in ('alphas', 'alpha_tie_order', 'unknown_weights', 'margin_weights'))
        and type(suite['probability_temperature']) is float,
        'S010 numeric grids require literal floats, not booleans or alternate numeric types')
    validate_gain_policy(suite['gain_policy'])


def load_gain_inputs(root, suite):
    """Metadata and existing source snapshot checks only; no cache scoring or model."""
    validate_gain_suite(suite)
    source_path = project_path(root, SOURCE_CONFIG, 'configs/package')
    require(file_sha256(source_path) == SOURCE_CONFIG_SHA, 'Selected S008 source config changed')
    source_config = read(source_path)
    require(source_config['selection']['family'] == 'public_advanced'
        and source_config['selection']['recipe_id'] == 'S008c', 'S010 requires the existing best S008c')
    selected = verify_selection(root, source_config['selection'])
    contract = load_contract(project_path(root, suite['readiness_config'], 'configs/train'), root)
    require(contract['config']['inference'] == {'seconds': 180.0, 'maximum_windows': 1}
        and contract['config']['mode'] == 'frozen_baseline', 'S010 preserves full-utterance frozen inputs')
    return contract, source_config, selected


def select_inner_frontend(candidates, truth, *, classes=447):
    """The selector never accepts outer scores, predictions, labels or diagnostics."""
    require(set(candidates) == {'identity', 'gain'}, 'Exactly baseline and one gain candidate are required')
    fitted = {}
    for frontend in ('identity', 'gain'):
        selected, curves = select_inner_alpha(candidates[frontend], truth, classes=classes)
        fitted[frontend] = {'selected': selected, 'candidates': curves}
    chosen = max(('identity', 'gain'), key=lambda name: fitted[name]['selected']['calibration']['inner_macro_f1_447'])
    return {'frontend': chosen, **fitted[chosen]['selected'], 'frontend_tie_order': ['identity', 'gain'],
            'selection_scope': 'group-excluded inner queries only'}, fitted


def family_scores(public, advanced, valid, contract, outer):
    endpoints = {name: crossfit_scores(values, valid, contract['manifest'], contract['folds'], outer, 'max_reference')
                 for name, values in (('public', public), ('advanced', advanced))}
    require(all(value['known_labels'] == contract['labels'][1:] for value in endpoints.values()), 'Known label order changed')
    return {alpha: scores_for_alpha(public, advanced, valid, contract['manifest'], contract['folds'], outer,
                                   alpha, endpoints) for alpha in ALPHAS}


def inner_arrays(scored):
    return {alpha: {'known': values['inner_known_scores'], 'unknown': values['inner_unknown_similarity']}
            for alpha, values in scored.items()}


def verify_saved_control_arrays(observed, historical):
    """Require the complete recorded probability/support matrices to match."""
    for filename in ('outer_probabilities.npz', 'reference_support.npz'):
        with np.load(observed / filename, allow_pickle=False) as actual, np.load(historical / filename, allow_pickle=False) as expected:
            require(set(actual.files) == set(expected.files), 'Control array fields changed')
            for key in actual.files:
                require(actual[key].dtype == expected[key].dtype and np.array_equal(actual[key], expected[key]),
                        'Exact control probability/support arrays changed: ' + filename + '/' + key)
    return {'exact_probability_arrays': True, 'exact_reference_support_arrays': True}


def gain_identity(root, suite, contract, sources):
    body = {'schema_version': 1, 'frontend_policy': suite['gain_policy'],
        'inference': contract['config']['inference'], 'data_input_hashes': contract['input_hashes'],
        'labels': contract['labels'], 'embedding_dims': {'public': 512, 'advanced': 192},
        'model_sources': {key: value['source_record'] for key, value in sources['assets'].items()},
        'code_hashes': {**contract['code_hashes'], 'scripts/score_gain.py': file_sha256(root / 'scripts/score_gain.py')},
        'base_feature_implementation_unchanged': True, 'encoder_updates': 0}
    return {**body, 'signature': hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()}


def verify_gain_cache(cache, identity, manifest, receipt):
    body = {key: value for key, value in identity.items() if key != 'signature'}
    require(identity['signature'] == hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        and identity['embedding_dims'] == {'public': 512, 'advanced': 192}
        and set(identity['model_sources']) == {'public', 'advanced'}, 'Candidate identity signature or model dimensions differ')
    require(receipt['identity'] == identity and receipt['file_count'] == len(manifest)
        and len(receipt['files']) == len(manifest), 'Gain cache identity or coverage differs')
    require(receipt['encoder_state_sha256_before'] == receipt['encoder_state_sha256_after']
        and receipt['weight_file_sha256_before'] == receipt['weight_file_sha256_after'], 'Frozen model changed')
    for field in ('encoder_state_sha256_before', 'encoder_state_sha256_after', 'weight_file_sha256_before', 'weight_file_sha256_after'):
        require(set(receipt[field]) == {'public', 'advanced'}
            and all(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                    for value in receipt[field].values()), 'Incomplete or malformed frozen model hashes')
    require(all(receipt['weight_file_sha256_before'][name] == identity['model_sources'][name]['weights_sha256']
                for name in ('public', 'advanced')), 'Cache weights differ from the signed model sources')
    records = {row['audio_file']: row for row in receipt['files']}
    require(len(records) == len(manifest), 'Duplicate candidate cache rows')
    values, mask = {'public': [], 'advanced': []}, []
    from speaker_id.data.splits import truth
    for source in manifest:
        row = records[source['audio_file']]
        path = cache / row['cache_file']
        require(path.is_file() and not path.is_symlink() and path.parent == cache
            and path.name == Path(source['audio_file']).stem + '.npz'
            and path.stat().st_size == row['bytes'] and file_sha256(path) == row['cache_sha256'], 'Changed candidate cache file')
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {'public', 'advanced', 'valid', 'signature', 'audio_file', 'audio_sha256'}
                and saved['valid'].shape == () and saved['valid'].dtype == np.bool_
                and type(row['valid']) is bool
                and all(saved[key].shape == () and saved[key].dtype.kind == 'U'
                        for key in ('signature', 'audio_file', 'audio_sha256')), 'Malformed cache fields or scalar types')
            require(str(saved['signature']) == identity['signature']
                and str(saved['audio_file']) == source['audio_file']
                and str(saved['audio_sha256']) == source['input_sha256'] == row['audio_sha256'], 'Candidate source binding differs')
            valid = bool(saved['valid'])
            require(valid == row['valid'] == truth(source['has_nonzero_signal']), 'Original signal eligibility changed')
            for name, dim in identity['embedding_dims'].items():
                vector = saved[name].copy()
                require(vector.shape == (dim,) and vector.dtype == np.float32 and np.isfinite(vector).all()
                    and (np.isclose(np.linalg.norm(vector), 1, atol=1e-5) if valid else not np.any(vector)), 'Malformed candidate embedding')
                values[name].append(vector)
            mask.append(valid)
    return {name: np.asarray(v) for name, v in values.items()}, np.asarray(mask, dtype=bool)


def extract_gain_cache(root, suite, contract, sources, output, tracker, control):
    """Fresh source-bound candidate cache; old files are never overwritten/resumed."""
    from speaker_id.models.campp import load_campp
    from speaker_id.candidates.campp_advanced import load_advanced
    from speaker_id.candidates.gain_frontend import extract_gain_pair
    require(control.get('exact_prediction_reproduction') and control.get('exact_pooled_metrics')
        and control.get('exact_inner_alpha_curves') and control.get('exact_probability_and_support_arrays'),
        'Exact S008 reproduction must precede candidate extraction')
    identity = gain_identity(root, suite, contract, sources)
    write_json(output / 'gain_identity.json', identity)
    tracker.add_artifact(output / 'gain_identity.json')
    tracker.flush(strict=True)
    cache = output / 'gain_embedding_cache'
    cache.mkdir(exist_ok=False)
    paths = {name: project_path(root, asset['config']['weights_path'], 'artifacts/models')
             for name, asset in sources['assets'].items()}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    require(all(hashes[name] == identity['model_sources'][name]['weights_sha256'] for name in paths), 'Public weights changed')
    encoders = {'public': load_campp(sources['assets']['public']['config'], root, 'cuda'),
                'advanced': load_advanced(sources['assets']['advanced']['config'], root, 'cuda')}
    for encoder in encoders.values():
        encoder.requires_grad_(False).eval()
    before = {name: encoder_state_sha256(encoder) for name, encoder in encoders.items()}
    from speaker_id.audio.gain import IDENTITY_POLICY
    from speaker_id.data.splits import truth
    manifest = contract['manifest']
    nonzero = [i for i, row in enumerate(manifest) if truth(row['has_nonzero_signal'])]
    zero = [i for i, row in enumerate(manifest) if not truth(row['has_nonzero_signal'])]
    # Metadata-only deterministic representatives, independent of labels/errors.
    probe_indices = sorted({0, len(manifest) // 2, len(manifest) - 1, zero[0],
        min(nonzero, key=lambda i: float(manifest[i]['duration_seconds'])),
        max(nonzero, key=lambda i: float(manifest[i]['duration_seconds'])),
        min(nonzero, key=lambda i: float(manifest[i]['mono_rms_dbfs']))})
    probes = []
    for index in probe_indices:
        row = manifest[index]
        vectors, info = extract_gain_pair(encoders['public'], encoders['advanced'],
            root / contract['config']['data_dir'] / row['audio_file'], device='cuda', policy=IDENTITY_POLICY,
            **contract['config']['inference'])
        require(info['nonzero_signal'] == bool(sources['valid'][index])
            and all(np.array_equal(vector, sources['vectors'][name][index]) for name, vector in vectors.items()),
            'Shared identity frontend differs from the original attested GPU cache')
        probes.append({'audio_file': row['audio_file'], 'audio_sha256': row['input_sha256'],
                       'exact_public_and_advanced_cache_vectors': True})
    write_json(output / 'identity_frontend_parity.json', {'status': 'passed', 'cases': probes,
        'policy': IDENTITY_POLICY, 'selection': 'fixed metadata extrema and index positions; no labels/errors'})
    tracker.add_artifact(output / 'identity_frontend_parity.json')
    tracker.flush(strict=True)
    records, started = [], time.monotonic()
    for i, row in enumerate(contract['manifest']):
        filename = row['audio_file']
        require(Path(filename).name == filename, 'Input filenames must remain flat')
        raw = root / contract['config']['data_dir'] / filename
        vectors, info = extract_gain_pair(encoders['public'], encoders['advanced'], raw,
            device='cuda', policy=suite['gain_policy'], **contract['config']['inference'])
        require(file_sha256(raw) == row['input_sha256'], 'Raw audio changed during extraction')
        path = cache / (Path(filename).stem + '.npz')
        temporary = path.with_suffix('.partial')
        require(not path.exists(), 'Fresh cache filename collision')
        with temporary.open('xb') as handle:
            np.savez_compressed(handle, **vectors, valid=bool(info['nonzero_signal']), audio_file=filename,
                                audio_sha256=row['input_sha256'], signature=identity['signature'])
        temporary.replace(path)
        records.append({'audio_file': filename, 'audio_sha256': row['input_sha256'], 'cache_file': path.name,
            'bytes': path.stat().st_size, 'cache_sha256': file_sha256(path), 'valid': bool(info['nonzero_signal']),
            'frontend_diagnostics': info})
        if (i + 1) % 50 == 0 or i + 1 == len(contract['manifest']):
            write_json(output / 'extraction_progress.json', {'status': 'extracting', 'completed_files': i + 1,
                       'total_files': len(contract['manifest']), 'signature': identity['signature']})
            tracker.log_metrics({'extraction/completed_files': i + 1, 'extraction/elapsed_seconds': time.monotonic() - started}, step=i + 1, sync=True)
            print(json.dumps({'stage': 'fixed_gain_pair_extraction', 'completed_files': i + 1, 'total_files': len(contract['manifest'])}), flush=True)
    after = {name: encoder_state_sha256(encoder) for name, encoder in encoders.items()}
    require(before == after and all(file_sha256(path) == hashes[name] for name, path in paths.items()), 'Frozen encoder changed')
    del encoders
    receipt = {'schema_version': 1, 'identity': identity, 'file_count': len(records), 'files': records,
        'encoder_state_sha256_before': before, 'encoder_state_sha256_after': after,
        'weight_file_sha256_before': hashes, 'weight_file_sha256_after': hashes,
        'elapsed_seconds': time.monotonic() - started, 'encoder_updates': 0}
    write_json(output / 'gain_cache_manifest.json', receipt)
    write_json(output / 'extraction_progress.json', {'status': 'complete', 'completed_files': len(records),
        'total_files': len(manifest), 'signature': identity['signature']})
    arrays, valid = verify_gain_cache(cache, identity, contract['manifest'], receipt)
    require(np.array_equal(valid, sources['valid']), 'Gain changed zero-signal eligibility')
    archive_path = output / 'gain_embedding_cache.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_STORED) as archive:
        for row in records:
            archive.write(cache / row['cache_file'], 'gain_embedding_cache/' + row['cache_file'])
    for name in ('gain_cache_manifest.json', 'gain_embedding_cache.zip', 'extraction_progress.json'):
        tracker.add_artifact(output / name)
    tracker.flush(strict=True)
    return arrays, valid, receipt


def finish_recipe(path, tracker, contract, predictions, folds, recipe, controls):
    metrics = score_predictions(contract['manifest'], predictions, contract['labels'])
    report = {'recipe': recipe, 'oof': metrics, 'folds': folds, 'source_control_checks': controls, 'limitations': LIMITATIONS}
    write_json(path / 'experiment_report.json', report)
    write_csv(path / 'oof_predictions.csv', predictions)
    write_csv(path / 'oof_per_class.csv', metrics['per_class'])
    for name in ('experiment_report.json', 'oof_predictions.csv', 'oof_per_class.csv'):
        tracker.add_artifact(path / name)
    tracker.log_metrics({'oof/macro_f1_447': metrics['macro_f1'], 'oof/accuracy': metrics['accuracy'],
                        **{'oof/' + key: value for key, value in metrics['errors'].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# {recipe}\n\nPooled development Macro-F1: {metrics['macro_f1']:.9f}. "
        'Both public encoders frozen. All frontend/alpha/gate choices use inner queries only. '
        'One fixed gain policy; all 4529 rows retained. This is not a leaderboard score.\n')
    tracker.finish('FINISHED', strict=True)
    return report


def execute_gain_suite(root, config_path, suite, contract, source_config, binding_path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    validate_gain_suite(suite)
    validate_readiness_for_execution(root, contract)
    require(os.environ.get('VAST_INSTANCE_ID') == '50079023' and torch.cuda.is_available()
        and '3090' in torch.cuda.get_device_name(0), 'S010 execution requires the authorized RTX3090 instance')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    binding = ExperimentBinding(**read(binding_path)['binding'])
    output = root / 'artifacts/training' / ('S010_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {'suite_config': config_path, 'source_release_config': root / SOURCE_CONFIG, 'launcher': root / 'scripts/score_gain.py'}
    inputs.update({key: root / contract['config'][key] for key in ('manifest', 'folds', 'roles', 'label_map', 'model_config')})
    resolved = {'suite': suite, 'source_release_config': source_config,
                'data_readiness_contract': {key: contract[key] for key in ('config', 'model', 'input_hashes', 'code_hashes', 'signature')},
                'limitations': LIMITATIONS}
    write_json(output / 'resolved_config.json', resolved)
    common = {'project_root': root, 'binding': binding, 'input_paths': inputs,
              'run_kind': 'frozen_encoder_gain_comparison', 'training_started': False}
    parent = DurableMLflowRun.prepare(spool_dir=output / 'tracking', run_name=suite['run_name'], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        write_json(output / 'experiment_state.json', {'status': 'running', 'parent_run_id': parent.run_id})
        sources = load_sources(root, source_config, verify_audio=True)
        require(sources['contract']['signature'] == contract['signature'], 'Source changed after initial readiness')
        remote = _verify_remote_evidence(parent.client, binding, sources['remote_requests'], output / 'verified_remote_evidence')
        proof = {'historical_sources': sources['proof'], 'remote_evidence': remote,
                 'raw_audio_sha_verified': True, 'gain_extraction_requires_exact_S008_control': True}
        write_json(output / 'source_provenance.json', proof)
        parent.add_artifact(output / 'source_provenance.json')
        for name in ('suite_config', 'source_release_config', 'launcher'):
            parent.add_artifact(inputs[name], 'input_configs/' + name + inputs[name].suffix)
        valid, manifest, labels = sources['valid'], contract['manifest'], contract['labels']
        historical = sources['selection']['directory'] / 'S008c'
        label_index = {label: i for i, label in enumerate(labels)}
        baseline, baseline_fits, inner_truth = {}, {}, {}
        results, prediction_sets = [], {}
        path = output / 'S010a'
        path.mkdir()
        child = DurableMLflowRun.prepare(spool_dir=path / 'tracking', parent_run_id=parent.run_id,
            run_name='S010a-exact-S008c-control', config={**resolved, 'recipe': 'S010a'}, **common)
        child.flush(strict=True)
        predictions, folds, checks = [], [], {}
        for outer in (0, 1):
            baseline[outer] = family_scores(sources['vectors']['public'], sources['vectors']['advanced'], valid, contract, outer)
            query = baseline[outer][0.0]['calibration_indices']
            inner_truth[outer] = np.asarray([label_index[manifest[int(i)]['speaker_id']] for i in query])
            selected, curves = select_inner_alpha(inner_arrays(baseline[outer]), inner_truth[outer])
            baseline_fits[outer] = {'selected': selected, 'candidates': curves}
            require(read(historical / f'fold_{outer}/inner_alpha_calibration.json') == baseline_fits[outer], 'S008 alpha/gate curves changed')
            alpha = selected['advanced_weight']
            rows, report, check = _record_fold(path / f'fold_{outer}', child, contract, outer, baseline[outer][alpha], valid,
                selected['calibration'], curves[str(alpha)]['curve'], selected, historical)
            check.update(verify_saved_control_arrays(path / f'fold_{outer}', historical / f'fold_{outer}'))
            write_json(path / f'fold_{outer}/inner_alpha_calibration.json', baseline_fits[outer])
            child.add_artifact(path / f'fold_{outer}/inner_alpha_calibration.json', f'fold_{outer}/inner_alpha_calibration.json')
            predictions.extend(rows); folds.append(report); checks[str(outer)] = check
        pooled = score_predictions(manifest, predictions, labels)
        require(pooled == read(historical / 'experiment_report.json')['oof'], 'S008 pooled control changed')
        control = {**verify_predictions(historical / 'oof_predictions.csv', predictions), 'exact_pooled_metrics': True,
                   'exact_inner_alpha_curves': True, 'exact_probability_and_support_arrays': True, 'folds': checks}
        write_json(output / 'source_control_checks.json', control)
        parent.add_artifact(output / 'source_control_checks.json')
        results.append(finish_recipe(path, child, contract, predictions, folds, 'S010a', control))
        prediction_sets['S010a'] = predictions
        child = None
        gain, gain_valid, cache_receipt = extract_gain_cache(root, suite, contract, sources, output, parent, control)
        candidate, choices, fits = {}, {}, {}
        # Freeze BOTH folds' choices before reading ANY gain outer labels/metrics.
        for outer in (0, 1):
            candidate[outer] = family_scores(gain['public'], gain['advanced'], gain_valid, contract, outer)
            require(np.array_equal(candidate[outer][0.0]['calibration_indices'], baseline[outer][0.0]['calibration_indices']), 'Inner queries changed')
            choices[outer], fits[outer] = select_inner_frontend({'identity': inner_arrays(baseline[outer]),
                'gain': inner_arrays(candidate[outer])}, inner_truth[outer])
            require(fits[outer]['identity'] == baseline_fits[outer], 'Baseline inner fit changed during joint choice')
        frozen_choices = {'selected': choices, 'inner_fits': fits, 'candidate_frontends': 2, 'alphas_per_frontend': 5,
                          'no_gain_outer_evaluation_performed_yet': True, 'selection_policy': SELECTION}
        write_json(output / 'frozen_inner_choices.json', frozen_choices)
        parent.add_artifact(output / 'frozen_inner_choices.json')
        parent.flush(strict=True)
        for recipe in ('S010b', 'S010c'):
            path = output / recipe
            path.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=path / 'tracking', parent_run_id=parent.run_id,
                run_name=recipe + ('-fixed-gain-only' if recipe == 'S010b' else '-inner-frontend-choice'),
                config={**resolved, 'recipe': recipe, 'gain_identity': cache_receipt['identity']}, **common)
            child.flush(strict=True)
            child.add_artifact(output / 'source_control_checks.json')
            child.add_artifact(output / 'gain_cache_manifest.json')
            child.add_artifact(output / 'frozen_inner_choices.json')
            predictions, folds = [], []
            for outer in (0, 1):
                chosen = {'frontend': 'gain', **fits[outer]['gain']['selected']} if recipe == 'S010b' else choices[outer]
                frontend, alpha = chosen['frontend'], chosen['advanced_weight']
                scored = candidate[outer] if frontend == 'gain' else baseline[outer]
                curve = fits[outer][frontend]['candidates'][str(alpha)]['curve']
                rows, report, _ = _record_fold(path / f'fold_{outer}', child, contract, outer, scored[alpha], valid,
                    chosen['calibration'], curve, chosen)
                child.log_metrics({f'fold_{outer}/selected_advanced_weight': alpha,
                    f'fold_{outer}/selected_gain_frontend': int(frontend == 'gain')}, sync=True)
                predictions.extend(rows); folds.append(report)
                print(json.dumps({'stage': 'gain_scoring', 'recipe': recipe, 'fold': outer,
                                  'frontend': frontend, 'advanced_weight': alpha, 'macro_f1': report['outer']['macro_f1']}), flush=True)
            results.append(finish_recipe(path, child, contract, predictions, folds, recipe, control))
            prediction_sets[recipe] = predictions
            child = None
        comparisons = {recipe: {'pooled_macro_f1_delta': results[i]['oof']['macro_f1'] - results[0]['oof']['macro_f1'],
            'paired_quality_slices': paired_diagnostics(manifest, prediction_sets['S010a'], prediction_sets[recipe], labels)}
            for i, recipe in ((1, 'S010b'), (2, 'S010c'))}
        deltas = [results[2]['folds'][outer]['outer']['macro_f1'] - results[0]['folds'][outer]['outer']['macro_f1'] for outer in (0, 1)]
        decision = {'candidate_recipe': 'S010c', 'rule': DECISION,
            'meets_preregistered_development_rule': comparisons['S010c']['pooled_macro_f1_delta'] >= DECISION['minimum_pooled_improvement']
                and min(deltas) >= -DECISION['maximum_fold_decline'],
            'fold_macro_f1_deltas': deltas, 'P002_unchanged': True, 'leaderboard_validation': 'pending'}
        report = {'status': 'complete', 'parent_run_id': parent.run_id, 'results': results, 'comparisons': comparisons,
            'decision': decision, 'source_control_checks': control, 'selection_policy': SELECTION,
            'gain_policy': suite['gain_policy'], 'encoder_updates': 0, 'elapsed_seconds': time.monotonic() - started,
            'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(), 'limitations': LIMITATIONS}
        write_json(output / 'experiment_report.json', report)
        for name in ('experiment_report.json', 'resolved_config.json'):
            parent.add_artifact(output / name)
        for row in results:
            parent.log_metrics({row['recipe'] + '/oof_macro_f1_447': row['oof']['macro_f1']}, sync=False)
        parent.write_report(report, markdown='# S010 controlled waveform gain\n\nExact S008c predictions and full inner alpha/gate curves reproduced before candidate extraction. '
            'One fixed RMS boost policy; both encoders frozen. Both folds frontend/alpha/gate choices were saved before gain outer evaluation. '
            'Development results only; P002 fallback preserved.\n')
        parent.finish('FINISHED', strict=True)
        write_json(output / 'experiment_state.json', {'status': 'complete', 'parent_run_id': parent.run_id})
        return {'output': str(output), 'parent_run_id': parent.run_id, 'results': {r['recipe']: r['oof']['macro_f1'] for r in results}, 'decision': decision}
    except BaseException as error:
        failure = {'status': 'failed', 'parent_run_id': parent.run_id, 'error_type': type(error).__name__, 'error': str(error)}
        write_json(output / 'failure.json', failure)
        if child is not None:
            child.write_report(failure); child.finish('FAILED', strict=False)
        parent.write_report(failure); parent.finish('FAILED', strict=False)
        write_json(output / 'experiment_state.json', failure)
        raise
