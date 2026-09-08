"""Selected P002 release: permitted final-query fitting and minimal offline assets.

This builder is not shipped. Execution never updates or extracts an encoder.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from speaker_id.infrastructure.data import confined_path
from speaker_id.inference.selected_policy import (
    ADAPTED_PROTOCOL, FROZEN_PROTOCOL, INFERENCE, F004_SOURCE, ARCHITECTURE_512,
    MODEL_PATHS, ROLES_SHA256, validate_policy, validate_adapted_config, state_dict_sha256)
from speaker_id.models.campp import EXPECTED_FRONTEND, file_sha256
from speaker_id.packaging.frozen import release_manifest, write_portable_zip
from speaker_id.packaging.selected_sources import FAMILIES, FIXED, load_sources, project_file, require, validate_config
from speaker_id.tracking.snapshot import write_json
from speaker_id.training.candidate_fusion import weighted_encoder_pair, select_inner_alpha
from speaker_id.training.crossfit_references import all_training_crossfit_scores
from speaker_id.training.heldout_final_references import heldout_final_scores
from speaker_id.training.reference_scoring import calibrate_gate, known_scores, reference_probabilities

BASE_SOURCES = ('speaker_id/__init__.py', 'speaker_id/inference/__init__.py',
    'speaker_id/inference/scoring.py', 'speaker_id/inference/selected_policy.py', 'speaker_id/inference/selected_runtime.py',
    'speaker_id/models/__init__.py', 'speaker_id/models/campp.py', 'speaker_id/models/vendor/__init__.py',
    'speaker_id/models/vendor/campplus/__init__.py', 'speaker_id/models/vendor/campplus/DTDNN.py',
    'speaker_id/models/vendor/campplus/layers.py', 'speaker_id/models/vendor/campplus/LICENSE',
    'speaker_id/models/vendor/campplus/PROVENANCE.json')
ADVANCED_SOURCES = ('speaker_id/candidates/__init__.py', 'speaker_id/candidates/campp_advanced.py')
BUILDER_SOURCES = ('src/speaker_id/packaging/selected.py', 'src/speaker_id/packaging/selected_sources.py',
    'src/speaker_id/training/heldout_final_references.py', 'scripts/package_selected.py', 'scripts/submission_selected.py')


def release_policy(family, alpha):
    """Prune physical components without changing the selected calibration family."""
    require(family in {entry[1] for entry in FAMILIES.values()}, 'Unknown selected procedure family')
    allowed = (0.0,) if family == 'adapted_only' else (1.0,) if family == 'advanced_only' else tuple(FIXED['alphas'])
    require(type(alpha) is float and alpha in allowed, 'Final alpha is outside this fixed procedure')
    adapted = family in ('adapted_only', 'adapted_advanced')
    if alpha == 0:
        kind, components, dim = ('adapted_f004_fold0_512', ('adapted',), 512) if adapted else ('public_voxceleb_512', ('public',), 512)
    elif alpha == 1:
        kind, components, dim = 'advanced_public_192', ('advanced',), 192
    else:
        kind, components, dim = ('paired_f004_advanced_704', ('adapted', 'advanced'), 704) if adapted else ('paired_public512_advanced192_704', ('public', 'advanced'), 704)
    return validate_policy({'schema_version': 1, 'kind': kind, 'embedding_dim': dim, 'advanced_weight': alpha,
        'inference': dict(INFERENCE), 'model_configs': {key: MODEL_PATHS[key] for key in components},
        'calibration_protocol': ADAPTED_PROTOCOL if adapted else FROZEN_PROTOCOL})


def family_vectors(vectors, valid, family, alpha):
    if alpha == 1:
        return vectors['advanced']
    left = 'adapted' if family in ('adapted_only', 'adapted_advanced') else 'public'
    if alpha == 0:
        return vectors[left]
    return weighted_encoder_pair(vectors[left], vectors['advanced'], valid, alpha)


def frozen_final_scores(contract, values, valid):
    result = all_training_crossfit_scores(values, valid, contract['manifest'], contract['folds'],
        method='max_reference', classes=len(contract['labels']) - 1)
    require(result['known_labels'] == contract['labels'][1:], 'Final score label columns differ')
    refs = np.asarray(result['provenance']['reference_indices'], dtype=np.int64)
    positions = {label: i for i, label in enumerate(contract['labels'])}
    targets = np.asarray([positions[contract['manifest'][int(i)]['speaker_id']] for i in refs], dtype=np.int64)
    normalized = values[refs].copy()
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    result['reference_indices'] = refs
    result['gallery'] = {'known_embeddings': np.ascontiguousarray(normalized[targets > 0]), 'known_targets': targets[targets > 0],
                         'unknown_embeddings': np.ascontiguousarray(normalized[targets == 0])}
    return result


def fit_final_scorer(contract, vectors, valid, family):
    """All alpha candidates use one family's exact query set, including endpoints."""
    alphas = [0.0] if family == 'adapted_only' else [1.0] if family == 'advanced_only' else FIXED['alphas']
    score_fn = heldout_final_scores if family in ('adapted_only', 'adapted_advanced') else frozen_final_scores
    scores, candidates, query, references = {}, {}, None, None
    for alpha in alphas:
        values = family_vectors(vectors, valid, family, alpha)
        item = score_fn(contract, values, valid)
        if query is None:
            query, references = item['calibration_indices'], item['reference_indices']
        require(np.array_equal(query, item['calibration_indices']) and np.array_equal(references, item['reference_indices']),
            'Alpha changed eligible calibration queries or final enrollment')
        scores[alpha] = item
        candidates[alpha] = {'known': item['inner_known_scores'], 'unknown': item['inner_unknown_similarity']}
    labels = {name: i for i, name in enumerate(contract['labels'])}
    truth = np.asarray([labels[contract['manifest'][int(i)]['speaker_id']] for i in query], dtype=np.int64)
    if len(alphas) == 1:
        alpha = alphas[0]
        calibration, curve = calibrate_gate(candidates[alpha]['known'], truth, candidates[alpha]['unknown'],
            FIXED['unknown_weights'], FIXED['margin_weights'], 201, len(labels))
        curves = {str(alpha): {'advanced_weight': alpha, 'selected': calibration, 'curve': curve}}
    else:
        chosen, curves = select_inner_alpha(candidates, truth, classes=len(labels))
        alpha, calibration = chosen['advanced_weight'], chosen['calibration']
    item, policy = scores[alpha], release_policy(family, float(alpha))
    metric = calibration['inner_macro_f1_447']
    calibration = {k: v for k, v in calibration.items() if k != 'inner_macro_f1_447'}
    calibration.update({'temperature': .05, 'inference': dict(INFERENCE)})
    report = {'protocol': policy['calibration_protocol'], 'calibration_query_files': len(query),
        'known_reference_files': len(item['gallery']['known_targets']), 'unknown_reference_files': len(item['gallery']['unknown_embeddings']),
        'source_files': len(valid), 'zero_signal_files': int((~valid).sum()), 'roles_sha256': ROLES_SHA256,
        'encoder_fit_queries_used': 0, 'whole_query_group_excluded': True, 'metric_scope': 'fitted_calibration_not_oof',
        'fitted_calibration_macro_f1_447': metric, 'alpha_tie_order': FIXED['alpha_tie_order'],
        'reference_helper_protocol': item['provenance']['protocol'],
        'family_preserved_after_pruning': True, 'new_encoder_updates': 0}
    # Curves retain the established helper's field names, with their scope explicit.
    return {'policy': policy, 'calibration': calibration, 'gallery': item['gallery'], 'report': report,
        'curves': {'metric_scope': 'fitted_calibration_not_oof', 'alphas': curves},
        'scores': item, 'vectors': family_vectors(vectors, valid, family, alpha)}


def export_encoder_only(checkpoint, destination, *, expected_sha, expected_signature,
                        loader=None, writer=None, tensor_predicate=None):
    """Export only checkpoint['encoder']; retain legitimate backbone head.* tensors."""
    if loader is None or writer is None or tensor_predicate is None:
        import torch
        loader = loader or (lambda path: torch.load(path, map_location='cpu', weights_only=True))
        writer = writer or torch.save
        tensor_predicate = tensor_predicate or (lambda value: isinstance(value, torch.Tensor))
    require(file_sha256(checkpoint) == expected_sha == F004_SOURCE['checkpoint_sha256'], 'Wrong fixed fold-0 checkpoint')
    saved = loader(checkpoint)
    require(saved.get('format_version') == 2 and saved.get('outer_fold') == 0 and saved.get('completed_steps') == 1100
        and saved.get('signature') == expected_signature, 'Wrong adapted source/fold/completion metadata')
    state = saved.get('encoder')
    require(isinstance(state, dict) and state and all(isinstance(name, str) and tensor_predicate(value)
        and np.isfinite(value.detach().cpu().contiguous().numpy()).all() for name, value in state.items()),
        'Checkpoint must contain a finite encoder tensor dictionary')
    digest = state_dict_sha256(state)
    require(not destination.exists(), 'Encoder export must not overwrite an earlier payload')
    with destination.open('xb') as output:
        writer(state, output)
    exported = loader(destination)
    require(isinstance(exported, dict) and set(exported) == set(state)
        and all(tensor_predicate(value) for value in exported.values()) and state_dict_sha256(exported) == digest,
        'Plain encoder export changed tensors or retained training state')
    require(file_sha256(checkpoint) == expected_sha, 'Original checkpoint changed during encoder export')
    return {'weights_bytes': destination.stat().st_size, 'weights_sha256': file_sha256(destination), 'encoder_state_sha256': digest}


def assert_final_counts(final):
    report, counts = final['report'], FIXED['expected_counts']
    expected_query = counts['q0_queries'] if final['policy']['calibration_protocol'] == ADAPTED_PROTOCOL else counts['frozen_queries']
    require(report['source_files'] == counts['source'] and report['zero_signal_files'] == counts['zeros']
        and report['known_reference_files'] == counts['known_references'] and report['unknown_reference_files'] == counts['unknown_references']
        and report['calibration_query_files'] == expected_query, 'Final family query/gallery coverage differs from original roles')


def build_payload(root, package, sources, final, config, release_id):
    package.mkdir(parents=True, exist_ok=False)
    policy = final['policy']
    components = set(policy['model_configs'])
    source_names = BASE_SOURCES + (ADVANCED_SOURCES if 'advanced' in components else ())
    for name in source_names:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_file(root, 'src/' + name), destination)
    shutil.copyfile(project_file(root, 'scripts/submission_selected.py'), package / 'submission.py')
    (package / 'assets').mkdir()
    configs, records = {}, {}
    for key in sorted(components):
        asset = sources['assets'][key]
        if key == 'adapted':
            exported = export_encoder_only(asset['checkpoint'], package / 'assets/f004_fold0_encoder.pt',
                expected_sha=asset['source']['folds']['0']['checkpoint_sha256'], expected_signature=asset['source']['signature'])
            model = validate_adapted_config({'schema_version': 1, 'encoder_kind': 'adapted_f004_fold0', 'architecture': 'CAMPPlus',
                'embedding_dim': 512, 'sample_rate': 16000, 'fbank_bins': 80, 'frontend': EXPECTED_FRONTEND,
                'architecture_kwargs': ARCHITECTURE_512, 'inference': INFERENCE, 'weights_path': 'assets/f004_fold0_encoder.pt',
                'source': F004_SOURCE, **exported})
            records[key] = {**F004_SOURCE, 'weights_sha256': exported['weights_sha256'], 'encoder_state_sha256': exported['encoder_state_sha256']}
        else:
            model = dict(asset['config'])
            if key == 'public':
                model['weights_path'] = 'assets/campplus_voxceleb.bin'
            require(file_sha256(asset['weights']) == model['weights_sha256'], 'Pinned frozen weight changed before packaging')
            destination = package / model['weights_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset['weights'], destination)
            require(file_sha256(destination) == model['weights_sha256'], 'Frozen weight copy failed')
            records[key] = asset['source_record']
        configs[key] = model
        if key == 'advanced':
            shutil.copyfile(asset['config_path'], package / MODEL_PATHS[key])
        else:
            write_json(package / MODEL_PATHS[key], model)
    selection = config['selection']
    provenance = {'schema_version': 1, 'release_id': release_id, 'policy': policy,
        'model_config_sha256': {key: file_sha256(package / path) for key, path in policy['model_configs'].items()},
        'model_sources': records, 'selection': {'family': selection['family'], 'experiment_code': FAMILIES[selection['recipe_id']][0],
            'recipe_id': selection['recipe_id'], 'parent_run_id': selection['source']['parent_run_id'],
            'report_sha256': selection['report_sha256'],
            'precursor_oof_macro_f1_447': sources['selection']['report']['oof']['macro_f1'],
            'score_scope': 'Development procedure only; final fitted payload has no matching OOF estimate'},
        'calibration': final['report'], 'new_encoder_updates': 0}
    if policy['calibration_protocol'] == ADAPTED_PROTOCOL:
        provenance['procedure_history'] = {'adaptation_source': F004_SOURCE, 'roles_sha256': ROLES_SHA256,
            'final_encoder_excludes_adapted': 'adapted' not in components}
    write_json(package / 'assets/policy.json', policy)
    write_json(package / 'assets/provenance.json', provenance)
    write_json(package / 'assets/calibration.json', final['calibration'])
    write_json(package / 'assets/labels.json', {'labels': sources['contract']['labels'], 'unknown_index': 0})
    np.savez_compressed(package / 'assets/gallery.npz', **final['gallery'])
    (package / 'README.md').write_text('# Selected CAM++ release\n\n'
        'python submission.py --data-dir INPUT --predictions-file-path OUTPUT.csv\n\n'
        'Output columns: audio_file,speaker_id. All selected encoder weights and references are bundled. '
        'Use the competition installed Python, NumPy, SciPy, SoundFile, Torch and Torchaudio environment. '
        'Inference is FP32 with the fixed 180-second single-view frontend; CPU or CUDA is supported.\n\n'
        'Selection procedure: ' + selection['recipe_id'] + '. Runtime kind: ' + policy['kind'] + '. '
        'The precursor development OOF score belongs to that procedure. Final calibration uses ' + str(final['report']['calibration_query_files'])
        + ' permitted queries with their complete content group excluded; references are restored for inference. '
        'The fitted calibration metric is not an OOF or leaderboard result. No encoder updates were made during this build. '
        'For adapted procedures, original fold-0 adaptation history and held-out roles remain in provenance even if endpoint pruning removes that encoder.\n\n'
        'Independent offline audio QA and hidden leaderboard execution remain pending. CAM++ Apache-2.0 code and source attribution '
        'are in speaker_id/models/vendor/campplus/.\n', encoding='utf-8')
    manifest = release_manifest(package, release_id=release_id, provenance={
        'policy_sha256': file_sha256(package / 'assets/policy.json'), 'provenance_sha256': file_sha256(package / 'assets/provenance.json')})
    manifest['encoder_updates'] = 500 if 'adapted' in components else 0
    manifest['new_encoder_updates'] = 0
    write_json(package / 'manifest.json', manifest)
    return manifest, provenance


def verify_cached_parity(final, valid):
    from speaker_id.inference.scoring import score_embeddings
    values, gallery = final['vectors'], final['gallery']
    sample = np.unique(np.r_[np.linspace(0, len(values) - 1, 16, dtype=int), np.flatnonzero(~valid)[:1]])
    normalized = values[sample].copy()
    normalized[valid[sample]] /= np.linalg.norm(normalized[valid[sample]], axis=1, keepdims=True)
    known = np.clip(known_scores(normalized, gallery['known_embeddings'], gallery['known_targets'], 'max_reference',
                                 classes=len(np.unique(gallery['known_targets']))), -1, 1)
    unknown = np.clip((normalized @ gallery['unknown_embeddings'].T).max(axis=1), -1, 1)
    expected = reference_probabilities(known, unknown, final['calibration'], valid[sample], .05)
    actual = score_embeddings(values[sample], valid[sample], gallery, final['calibration'], classes=known.shape[1])
    require(np.allclose(expected, actual, rtol=0, atol=1e-6) and np.array_equal(expected.argmax(1), actual.argmax(1)),
        'Portable scorer differs from the established cached scoring implementation')
    return {'sample_count': len(sample), 'maximum_absolute_probability_error': float(np.abs(expected - actual).max()),
            'argmax_exact': True}


def calibration_plot(final, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, result in final['curves']['alphas'].items():
        chosen = result['selected']
        rows = [row for row in result['curve'] if row['unknown_weight'] == chosen['unknown_weight']
                and row['margin_weight'] == chosen['margin_weight']]
        ax.plot([row['threshold'] for row in rows], [row['inner_macro_f1_447'] for row in rows], label='alpha=' + name)
    ax.set(xlabel='Rejection threshold', ylabel='Fitted calibration Macro-F1',
        title='Final permitted-query fit (not OOF or leaderboard)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def execute_build(root, config_path, binding_path):
    """Explicit manual server build, with fresh ownership/readback before final fitting."""
    require(os.environ.get('VAST_INSTANCE_ID') == '50079023', 'P002 requires the authorized Vast instance')
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.inference.selected_runtime import verify_payload
    from speaker_id.training.adaptation_comparison import _verify_remote_evidence
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    require(torch.cuda.is_available() and '3090' in torch.cuda.get_device_name(0), 'P002 requires the authorized RTX3090')
    root = root.resolve(strict=True)
    config_path = confined_path(root, config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    from speaker_id.training.contracts import load_contract
    contract = load_contract(project_file(root, config['readiness_config']), root, verify_audio=True)
    ready = validate_readiness_for_execution(root, contract)
    torch.set_num_threads(4)
    binding = ExperimentBinding(**json.loads(confined_path(root, binding_path).read_text(encoding='utf-8'))['binding'])
    release_id = 'P002_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = confined_path(root, Path(config['output_root']) / release_id)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {'build_config': config_path, 'verification': project_file(root, config['selection']['verification']['path']),
        **{name: project_file(root, contract['config'][name]) for name in ('manifest', 'folds', 'roles', 'label_map', 'model_config')},
        **{name: project_file(root, name) for name in BUILDER_SOURCES}}
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    code_hashes = {str(path.relative_to(root)).replace('\\', '/'): file_sha256(path) for path in (root / 'src').rglob('*')
                   if path.is_file() and path.suffix in ('.py', '.json')}
    tracker = DurableMLflowRun.prepare(project_root=root, spool_dir=output / 'tracking', binding=binding,
        run_name=release_id + '-selected-final-package', run_kind='selected_reference_packaging', training_started=False,
        config={'package': config, 'readiness': ready, 'input_hashes': hashes, 'code_hashes': code_hashes,
                'readiness_contract': {key: contract[key] for key in ('config', 'input_hashes', 'code_hashes', 'signature')}}, input_paths=inputs)
    try:
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.verify_remote_metadata()
        write_json(output / 'build_state.json', {'status': 'running', 'parent_run_id': tracker.run_id, 'new_encoder_updates': 0})
        sources = load_sources(root, config, verify_audio=True)
        asset_paths = {key + '/weights': asset.get('weights', asset.get('checkpoint')) for key, asset in sources['assets'].items()}
        asset_paths.update({key + '/model_config': asset['config_path'] for key, asset in sources['assets'].items() if 'config_path' in asset})
        asset_hashes = {name: file_sha256(path) for name, path in asset_paths.items()}
        live = _verify_remote_evidence(tracker.client, binding, sources['remote_requests'], output / 'verified_remote_evidence')
        write_json(output / 'source_provenance.json', sources['proof'])
        write_json(output / 'source_readbacks.json', live)
        for name, export in sources['exports'].items():
            write_json(output / ('audited_' + name + '_export.json'), export)
        for name, captured in sources['captured_configs'].items():
            write_json(output / ('captured_' + name + '_config.json'), captured)
        final = fit_final_scorer(sources['query_contract'], sources['vectors'], sources['valid'], config['selection']['family'])
        assert_final_counts(final)
        write_json(output / 'final_calibration_curve.json', final['curves'])
        calibration_plot(final, output / 'final_calibration_curve.png')
        write_json(output / 'final_reference_roles.json', final['scores']['provenance'])
        np.savez_compressed(output / 'final_calibration_scores.npz', calibration_indices=final['scores']['calibration_indices'],
            reference_indices=final['scores']['reference_indices'], known_scores=final['scores']['inner_known_scores'],
            unknown_similarity=final['scores']['inner_unknown_similarity'])
        parity = verify_cached_parity(final, sources['valid'])
        package = output / 'package'
        manifest, provenance = build_payload(root, package, sources, final, config, release_id)
        verify_payload(package)
        require(all(file_sha256(path) == hashes[name] for name, path in inputs.items())
            and all(file_sha256(root / name) == digest for name, digest in code_hashes.items())
            and all(file_sha256(path) == asset_hashes[name] for name, path in asset_paths.items()),
            'Input/source/model bytes changed during final build')
        write_json(output / 'model_assets_verification.json', {'sha256_before': asset_hashes,
            'sha256_after': {name: file_sha256(path) for name, path in asset_paths.items()},
            'new_encoder_updates': 0, 'all_original_assets_unchanged': True})
        archive_path = output / (release_id + '.zip')
        archive = write_portable_zip(package, archive_path, manifest)
        report = {'status': 'built_pending_offline_qa', 'release_id': release_id, 'parent_run_id': tracker.run_id,
            'new_encoder_updates': 0, 'bundled_prior_encoder_updates': manifest['encoder_updates'],
            'leaderboard_validation': 'pending', 'policy': final['policy'], 'final_fit': final['report'],
            'provenance': provenance, 'cached_scorer_parity': parity, 'archive': archive,
            'required_next_check': 'Independent extracted-package real-audio QA with networking blocked; leaderboard untested'}
        write_json(output / 'build_report.json', report)
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.npz', '.png') and path.name != 'build_state.json':
                tracker.add_artifact(path)
        for name in BUILDER_SOURCES:
            tracker.add_artifact(root / name, 'build_source/' + name)
        for name in manifest['files']:
            if name.startswith('assets/') or name.startswith('artifacts/models/'):
                tracker.add_artifact(package / name, 'release/' + name)
        tracker.add_artifact(package / 'manifest.json', 'release/manifest.json')
        tracker.add_artifact(archive_path, 'release/' + archive_path.name)
        tracker.log_metrics({'final_fit/calibration_macro_f1_447': final['report']['fitted_calibration_macro_f1_447'],
            'calibration/query_files': final['report']['calibration_query_files'], 'encoder/new_updates': 0,
            'encoder/bundled_prior_updates': manifest['encoder_updates'], 'release/archive_bytes': archive['archive_bytes'],
            'validation/cached_scorer_max_abs_error': parity['maximum_absolute_probability_error']})
        tracker.write_report(report, markdown='# Selected P002 package\n\nFinal calibration is a separate fitted metric, not OOF. '
            'Original selected procedure and encoder history are retained. No new encoder updates. Offline audio QA and leaderboard execution remain pending.\n')
        tracker.flush(strict=True)
        write_json(output / 'mlflow_roundtrip.json', {'artifacts': tracker.verify_artifacts(), 'metadata': tracker.verify_remote_metadata()})
        tracker.finish('FINISHED', strict=True)
        write_json(output / 'build_state.json', {'status': 'complete', 'parent_run_id': tracker.run_id, 'new_encoder_updates': 0})
        return {'output': str(output), **report}
    except BaseException as error:
        failure = tracker.redactor({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'new_encoder_updates': 0})
        write_json(output / 'failure.json', failure)
        tracker.write_report(failure)
        tracker.finish('FAILED', strict=False)
        write_json(output / 'build_state.json', {**failure, 'parent_run_id': tracker.run_id})
        raise
