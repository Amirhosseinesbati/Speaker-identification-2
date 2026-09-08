"""S009: own-fold F004 + public advanced192, original held-out queries only."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import uuid
import zipfile

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training.adapted_scoring import (
    assert_fixed_role_groups, verified_export_inventory, validate_adapted_identity, load_adapted_fold_cache,
)
from speaker_id.training.adaptation_comparison import (
    verify_source_snapshot, verify_final_schedule, _remote_requests as adapted_remote_requests,
    _verify_remote_evidence, paired_diagnostics,
)
from speaker_id.training.candidate_comparison import verify_candidate_cache, require_public_control
from speaker_id.training.candidate_fusion import (
    ALPHAS, TIE_ORDER, UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, weighted_encoder_pair, select_inner_alpha,
    validate_historical_candidate, _candidate_remote_requests, _record_fold,
)
from speaker_id.training.contracts import load_contract
from speaker_id.training.crossfit_references import crossfit_scores
from speaker_id.training.fusion import require_aligned_scores
from speaker_id.training.fusion_suite import project_path, verify_predictions
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import calibrate_gate
from speaker_id.training.runner import write_json, write_csv

PROTOCOL = 'original_heldout_queries_expanded_gallery'
SELECTION_POLICY = 'Select alpha and gate only on original encoder-held-out query groups; exact S006f and full-protocol S007b controls precede mixtures. The matched advanced endpoint is not S007b.'
EXECUTION_POLICY = 'Manual dispatch only after independently verified S008c, S007b and S006f each remain below pooled OOF Macro-F1 0.965; no automatic chaining.'
RECIPES = [
    {'id': 'S009a', 'source': 'adapted', 'protocol': PROTOCOL, 'control': 'S006f'},
    {'id': 'S009b', 'source': 'advanced', 'protocol': 'leave_content_group_out', 'control': 'S007b'},
    {'id': 'S009c', 'source': 'advanced', 'protocol': PROTOCOL, 'control': None},
    {'id': 'S009d', 'source': 'inner_selected_fusion', 'protocol': PROTOCOL, 'control': None},
]
LIMITATIONS = [
    'Development OOF is repeatedly consulted; it is not an untouched test or hidden leaderboard estimate.',
    'Each outer fold uses only its own F004 encoder; opposite-fold adapted ensembling is not evaluated.',
    'S007b is a full-crossfit source control, not the advanced held-out endpoint.',
    'Original query groups were never used in encoder fitting and leave both expanded reference pools.',
    'Only selected mixed policies receive outer evaluation; policy selection sees inner labels only.',
    'An all-training release needs separately recorded held-out-encoder calibration and offline QA.',
]
EXPORT_KEYS = {'run', 'parent_run_id', 'git_commit', 'export_manifest_paths', 'export_manifest_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hex_value(value, length):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{' + str(length) + '}', value)


def validate_export(source, prefix):
    require(isinstance(source, dict) and EXPORT_KEYS.issubset(source)
        and isinstance(source['run'], str) and re.fullmatch(r'artifacts/training/' + prefix + r'_\d{8}T\d{6}Z_[a-f0-9]{8}', source['run'])
        and hex_value(source['parent_run_id'], 32) and hex_value(source['git_commit'], 40)
        and hex_value(source['export_manifest_sha256'], 64)
        and isinstance(source['export_manifest_paths'], list) and source['export_manifest_paths']
        and all(isinstance(p, str) and p.startswith('artifacts/') for p in source['export_manifest_paths']),
        'A completed source needs actual immutable export identities, never placeholders')


def validate_verification(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and isinstance(pin['path'], str)
        and pin['path'].startswith('artifacts/infrastructure/') and hex_value(pin['sha256'], 64),
        'A complete independently verified receipt and SHA are mandatory')


def validate_suite(suite):
    fields = {'schema_version', 'experiment_code', 'run_name', 'readiness_config', 'output_root', 'source_adapted',
        'source_advanced', 'candidate_verification', 'control_adapted', 'prerequisite_fusion', 'recipes',
        'alphas', 'alpha_tie_order', 'unknown_weights', 'margin_weights', 'threshold_candidates',
        'probability_temperature', 'selection_policy', 'execution_policy'}
    require(isinstance(suite, dict) and set(suite) == fields and suite['schema_version'] == 1
        and suite['experiment_code'] == 'S009' and suite['run_name'] == 'S009-f004-advanced192-heldout-paired-reference'
        and suite['readiness_config'] == 'configs/train/campp_coverage.json' and suite['output_root'] == 'artifacts/training'
        and suite['recipes'] == RECIPES and suite['alphas'] == list(ALPHAS) and suite['alpha_tie_order'] == list(TIE_ORDER)
        and suite['unknown_weights'] == UNKNOWN_WEIGHTS and suite['margin_weights'] == MARGIN_WEIGHTS
        and type(suite['threshold_candidates']) is int and suite['threshold_candidates'] == 201
        and type(suite['probability_temperature']) is float and suite['probability_temperature'] == .05
        and suite['selection_policy'] == SELECTION_POLICY and suite['execution_policy'] == EXECUTION_POLICY,
        'S009 requires exactly its four preregistered recipes and original-held-out selection policy')
    source = suite['source_adapted']
    validate_export(source, 'F004')
    require(set(source) == EXPORT_KEYS | {'config', 'signature', 'completed_steps', 'folds'}
        and source['config'] == 'configs/train/campp_finetune_head600.json'
        and source['completed_steps'] == 1100 and hex_value(source['signature'], 64)
        and set(source['folds']) == {'0', '1'}, 'S009 requires the fixed final F004 600+500 source')
    for fold in source['folds'].values():
        require(set(fold) == {'child_run_id', 'checkpoint_sha256'} and hex_value(fold['child_run_id'], 32)
            and hex_value(fold['checkpoint_sha256'], 64), 'Each adapted fold needs its exact checkpoint and child')
    require(len({source['parent_run_id'], *(r['child_run_id'] for r in source['folds'].values())}) == 3
        and len({r['checkpoint_sha256'] for r in source['folds'].values()}) == 2, 'Adapted fold identities must be distinct')
    source = suite['source_advanced']
    validate_export(source, 'S007')
    require(set(source) == EXPORT_KEYS | {'source_signature', 'children'} and hex_value(source['source_signature'], 64)
        and set(source['children']) == {'S007a', 'S007b'}, 'Advanced source must retain its historical S007 signature')
    for name, codes in (('source_advanced', {'S007a', 'S007b'}), ('prerequisite_fusion', {'S008a', 'S008b', 'S008c'})):
        source = suite[name]
        validate_export(source, 'S007' if name == 'source_advanced' else 'S008')
        require(set(source['children']) == codes and all(hex_value(rid, 32) for rid in source['children'].values())
            and len({source['parent_run_id'], *source['children'].values()}) == len(codes) + 1, 'Invalid completed child identities')
    source = suite['prerequisite_fusion']
    require(set(source) == EXPORT_KEYS | {'children', 'verification'}, 'S008 prerequisite has unexpected fields')
    validate_verification(source['verification'])
    source = suite['control_adapted']
    validate_export(source, 'S006')
    require(set(source) == EXPORT_KEYS | {'recipe_id', 'child_run_id', 'verification'} and source['recipe_id'] == 'S006f'
        and hex_value(source['child_run_id'], 32) and source['child_run_id'] != source['parent_run_id'], 'Exact S006f control is required')
    validate_verification(source['verification'])
    validate_verification(suite['candidate_verification'])


def load_contracts(root, suite, *, verify_audio=False):
    validate_suite(suite)
    readiness = load_contract(project_path(root, suite['readiness_config'], 'configs/train'), root, verify_audio=verify_audio)
    adapted = load_contract(project_path(root, suite['source_adapted']['config'], 'configs/train'), root)
    for key in ('model', 'input_hashes', 'manifest', 'folds', 'roles', 'labels'):
        require(adapted[key] == readiness[key], 'F004 and readiness data/role/frontend identities differ')
    cfg, fit = adapted['config'], adapted['config']['fit']
    require(readiness['config']['experiment_code'] == 'B002' and cfg['experiment_code'] == 'F004'
        and cfg['fold_ids'] == [0, 1] and cfg['mode'] == 'fine_tune' and fit['mixed_precision'] is False
        and cfg['inference'] == {'seconds': 180.0, 'maximum_windows': 1}
        and cfg['expected_source_files'] == 4529 and cfg['evaluation_classes'] == 447
        and fit['adaptation_schedule']['head_only_steps'] == 600 and fit['epochs'] * fit['steps_per_epoch'] == 500
        and fit['epoch_selection'] == 'fixed_steps_no_outer_selection', 'S009 requires the original complete FP32 F004 contract')
    for outer in (0, 1):
        assert_fixed_role_groups(adapted, outer)
    for key in ('source_adapted', 'source_advanced', 'control_adapted', 'prerequisite_fusion'):
        project_path(root, suite[key]['run'], 'artifacts/training', exists=False)
        for path in suite[key]['export_manifest_paths']:
            project_path(root, path, 'artifacts', exists=False)
    return {'readiness': readiness, 'adapted': adapted}


def read_export(directory, relative, inventory):
    require(relative in inventory, 'Required captured evidence is absent from the audited export: ' + relative)
    return json.loads((directory / relative).read_text(encoding='utf-8'))


def verify_completed_audit(root, pin, source, report, code, expected_runs):
    path = project_path(root, pin['path'], 'artifacts/infrastructure')
    require(file_sha256(path) == pin['sha256'], 'Independent completion proof SHA differs')
    audit = json.loads(path.read_text(encoding='utf-8'))
    require(audit.get('status') == 'verified' and audit.get('parent_run_id') == source['parent_run_id']
        and audit.get('run_name') == Path(source['run']).name and audit.get('git_commit') == source['git_commit'],
        'Only a final independently verified matching source can permit S009')
    digest = audit.get('export_manifest_sha256', audit.get('archive_verification', {}).get('manifest_sha256'))
    require(digest == source['export_manifest_sha256'], 'Independent audit refers to a different export')
    runs = audit.get('runs', [])
    by_id = {r['run_id']: r for r in runs}
    require(len(runs) == len(by_id) == expected_runs and source['parent_run_id'] in by_id
        and all(r.get('status') == 'FINISHED' and r.get('git_commit') == source['git_commit'] for r in runs),
        'Every independent source parent/child run must have completed')
    pinned_children = source.get('children', {code: source.get('child_run_id')})
    require(all(rid in by_id and by_id[rid]['name'].split('-')[0] == name for name, rid in pinned_children.items()),
        'Independent completion proof child identities differ')
    matched = [r for r in report['results'] if (r['recipe'] if isinstance(r['recipe'], str) else r['recipe']['id']) == code]
    require(len(matched) == 1 and matched[0]['oof'] == audit.get('metrics', {}).get(code), 'Independent decision metrics differ')
    metric = matched[0]['oof']
    require(metric['row_count'] == 4529 and metric['class_count'] == 447 and type(metric['macro_f1']) is float
        and np.isfinite(metric['macro_f1']) and 0 <= metric['macro_f1'] < .965, 'S009 conditional target gate is false')
    return {'verification_path': pin['path'], 'verification_sha256': pin['sha256'], 'recipe': code,
        'parent_run_id': source['parent_run_id'], 'pooled_macro_f1': metric['macro_f1'], 'all_runs_finished': expected_runs}


def verify_scoring_export(root, directory, source, inventory, expected_children):
    state = read_export(directory, 'experiment_state.json', inventory)
    report = read_export(directory, 'experiment_report.json', inventory)
    original = read_export(directory, 'resolved_config.json', inventory)
    require(state.get('status') == report.get('status') == 'complete'
        and state.get('parent_run_id') == report.get('parent_run_id') == source['parent_run_id'], 'Incomplete source scorer')
    require(read_export(directory, 'tracking/artifacts/resolved_config.json', inventory) == original,
        'Original scorer configuration differs from its captured tracking snapshot')
    parent = read_export(directory, 'tracking/run_state.json', inventory)
    require(parent.get('run_id') == source['parent_run_id'] and parent.get('remote_status') == 'FINISHED'
        and parent.get('last_sync_error') is None and parent['tags'].get('mlflow.source.git.commit') == source['git_commit'],
        'Completed scorer parent receipt differs')
    metadata = read_export(directory, 'tracking/artifacts/source_manifest.json', inventory)
    archive_path = directory / 'tracking/artifacts/source_snapshot.zip'
    require('tracking/artifacts/source_snapshot.zip' in inventory and metadata.get('schema_version') == 2
        and metadata.get('archive_format') == 'zip' and metadata.get('src_dirty') is False
        and metadata.get('git_commit') == source['git_commit'] and metadata.get('archive_sha256') == file_sha256(archive_path),
        'Scorer source snapshot was not captured cleanly at the pinned revision')
    entries = {r['path']: r for r in metadata['files']}
    require(len(entries) == len(metadata['files']) == metadata['file_count'], 'Duplicate source snapshot inventory')
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.infolist()) == len(entries) and set(archive.namelist()) == set(entries), 'Incomplete source ZIP')
        for item in archive.infolist():
            path, entry = PurePosixPath(item.filename), entries[item.filename]
            require(item.filename.startswith('src/') and path.as_posix() == item.filename and '..' not in path.parts
                and '\\' not in item.filename and ':' not in item.filename and not item.is_dir()
                and not stat.S_ISLNK(item.external_attr >> 16) and item.file_size == entry['bytes']
                and hashlib.sha256(archive.read(item)).hexdigest() == entry['sha256'], 'Unsafe/changed source ZIP entry')
    for name in ('heldout_references.py', 'reference_scoring.py', 'scoring.py', 'crossfit_references.py'):
        relative = 'src/speaker_id/training/' + name
        require(entries.get(relative, {}).get('sha256') == file_sha256(root / relative), 'Historical shared scorer changed')
    input_manifest = read_export(directory, 'tracking/artifacts/inputs_manifest.json', inventory)
    launcher = directory / 'tracking/artifacts/input_configs/launcher.py'
    require('tracking/artifacts/input_configs/launcher.py' in inventory and input_manifest.get('launcher', {}).get('sha256') == file_sha256(launcher)
        and input_manifest['launcher']['bytes'] == launcher.stat().st_size, 'Captured scoring launcher differs from its input receipt')
    for code, identifier in expected_children.items():
        child = read_export(directory, code + '/tracking/run_state.json', inventory)
        config = read_export(directory, code + '/tracking/artifacts/resolved_config.json', inventory)
        recipe = config.get('recipe', {})
        recipe_id = recipe if isinstance(recipe, str) else recipe.get('id')
        require(child.get('run_id') == identifier and child.get('remote_status') == 'FINISHED' and child.get('last_sync_error') is None
            and child['tags'].get('mlflow.parentRunId') == source['parent_run_id']
            and child['tags'].get('mlflow.source.git.commit') == source['git_commit'] and config.get('suite') == original['suite']
            and recipe_id == code,
            'Completed scorer child identity/configuration differs')
    return original, report, {'git_commit': source['git_commit'], 'source_archive_sha256': metadata['archive_sha256'], 'source_file_count': len(entries)}


def require_controls(checks):
    require(set(checks) == {'S006f', 'S007b'}, 'Both exact historical controls must precede matched candidate/fusion evaluation')
    for code in checks:
        require_public_control(checks[code])
        require(all(row.get('exact_probabilities') is True and row.get('exact_full_calibration_curve') is True
            for row in checks[code]['folds'].values()), 'Control probability and full-curve equality are required')


def matched_scores(contract, adapted_by_fold, advanced, valid, outer, alpha, endpoints):
    """Only this fold's encoder participates; no adapted vector enters frozen crossfit."""
    require(outer in (0, 1) and alpha in ALPHAS and set(adapted_by_fold) == {0, 1}, 'Both explicitly keyed own-fold sources are required')
    if alpha == 0:
        return endpoints['adapted']
    if alpha == 1:
        return endpoints['advanced_matched']
    vectors, fold_valid = adapted_by_fold[outer]
    require(np.array_equal(fold_valid, valid), 'Own-fold and advanced validity masks differ')
    mixed = weighted_encoder_pair(vectors, advanced, valid, alpha)
    result = heldout_reference_scores(contract, mixed, valid, outer)
    require_aligned_scores(endpoints['adapted'], result)
    result['provenance'] = {**result['provenance'], 'adapted_outer_fold': outer, 'advanced_weight': float(alpha),
        'fusion': 'same_reference_fp32_sqrt_weighted_encoder_concatenation', 'post_concatenation_normalization': 'none'}
    return result


def verify_probability_control(directory, historical, outer):
    current_calibration = json.loads((directory / 'calibration.json').read_text(encoding='utf-8'))
    prior_calibration = json.loads((historical / f'fold_{outer}/calibration.json').read_text(encoding='utf-8'))
    require(current_calibration == prior_calibration, 'Historical complete calibration curve differs')
    with np.load(directory / 'outer_probabilities.npz', allow_pickle=False) as current, np.load(historical / f'fold_{outer}/outer_probabilities.npz', allow_pickle=False) as old:
        require(set(current.files) == set(old.files) and all(np.array_equal(current[k], old[k]) for k in current.files),
            'Historical control probabilities or their file/label order differ')


def scoring_remote_requests(directory, source, children):
    parent = [(name, directory / name) for name in ('experiment_report.json', 'source_control_checks.json')]
    parent += [(name, directory / 'tracking/artifacts' / name) for name in ('resolved_config.json', 'source_manifest.json', 'source_snapshot.zip', 'input_configs/launcher.py')]
    requests = [(source['parent_run_id'], None, parent)]
    for code, rid in children.items():
        files = [(name, directory / code / name) for name in ['experiment_report.json', 'oof_predictions.csv']
            + [f'fold_{outer}/{name}' for outer in (0, 1) for name in ('evaluation.json', 'calibration.json', 'predictions.csv', 'outer_probabilities.npz')]]
        files += [('resolved_config.json', directory / code / 'tracking/artifacts/resolved_config.json')]
        files += [(name, directory / 'tracking/artifacts' / name) for name in ('source_manifest.json', 'source_snapshot.zip')]
        requests.append((rid, source['parent_run_id'], files))
    return requests


def load_sources(root, suite, contracts):
    """Full local source attestation and completed-score gates; no model/scoring."""
    adapted = suite['source_adapted']
    directory, export, inventory = verified_export_inventory(root, adapted)
    original, folds = validate_adapted_identity(directory, adapted, contracts['adapted'], inventory)
    snapshot = verify_source_snapshot(directory, adapted, original, inventory)
    schedules = verify_final_schedule(directory, adapted, contracts['adapted'], inventory)
    arrays = {}
    for outer in (0, 1):
        vectors, valid, files = load_adapted_fold_cache(directory, contracts['adapted'], adapted['signature'], inventory, outer)
        arrays[outer] = vectors, valid
        folds[str(outer)]['files'] = files
    adapted_proof = {'parent_run_id': adapted['parent_run_id'], 'source_signature': adapted['signature'],
        'git_commit': adapted['git_commit'], 'export_manifest_sha256': adapted['export_manifest_sha256'],
        'verified_export_files': len(inventory), 'folds': folds}
    adapted_directory = directory
    manifests = {'F004': export}

    control = suite['control_adapted']
    directory, export, inventory = verified_export_inventory(root, control)
    children = {'S006f': control['child_run_id']}
    captured, report, source_proof = verify_scoring_export(root, directory, control, inventory, children)
    require(captured['suite']['sources']['F004'] == adapted and captured['suite']['expanded_arm'] is True
        and read_export(directory, 'source_provenance.json', inventory)['sources']['F004'] == adapted_proof,
        'Exact S006f control did not use these same F004 checkpoints/caches')
    condition = {'S006f': verify_completed_audit(root, control['verification'], control, report, 'S006f', 7)}
    control_directory = directory
    control_snapshot = source_proof
    manifests['S006'] = export

    source = suite['source_advanced']
    directory, export, inventory = verified_export_inventory(root, source)
    identity, source_proof = validate_historical_candidate(directory, source, contracts['readiness'], inventory, root)
    candidate_report = read_export(directory, 'experiment_report.json', inventory)
    condition['S007b'] = verify_completed_audit(root, suite['candidate_verification'], source, candidate_report, 'S007b', 3)
    receipt = read_export(directory, 'candidate_cache_manifest.json', inventory)
    advanced, valid = verify_candidate_cache(directory / 'candidate_embedding_cache', identity, contracts['readiness']['manifest'], receipt)
    require(all(np.array_equal(row[1], valid) for row in arrays.values()) and int((~valid).sum()) == 89,
        'Source encoders disagree on the original 89 zero-signal recordings')
    candidate_directory, candidate_proof = directory, source_proof
    manifests['S007'] = export

    prerequisite = suite['prerequisite_fusion']
    directory, export, inventory = verified_export_inventory(root, prerequisite)
    captured, report, prior_snapshot = verify_scoring_export(root, directory, prerequisite, inventory, prerequisite['children'])
    require(captured['suite']['source_advanced'] == source, 'Completed S008 used a different historical S007 source')
    checks = report.get('source_control_checks', {})
    require(set(checks) == {'public', 'advanced'}, 'S008 did not complete both preregistered controls')
    for check in checks.values():
        require_public_control(check)
    condition['S008c'] = verify_completed_audit(root, prerequisite['verification'], prerequisite, report, 'S008c', 4)
    manifests['S008'] = export
    requests = adapted_remote_requests({'sources': {'F004': adapted}, 'controls': {}}, {'F004': adapted_directory}, {})
    requests += scoring_remote_requests(control_directory, control, {'S006f': control['child_run_id']})
    requests += _candidate_remote_requests(candidate_directory, source)
    requests += scoring_remote_requests(directory, prerequisite, prerequisite['children'])
    proof = {'adapted': adapted_proof, 'adapted_snapshot': snapshot, 'adapted_final_schedules': schedules,
        'S006_snapshot': control_snapshot, 'historical_candidate': candidate_proof, 'S008_snapshot': prior_snapshot,
        'conditional_dispatch': condition, 'no_encoder_fitting_or_extraction': True,
        'own_outer_fold_only': True, 'source_audio_order_and_zero_mask_identical': True}
    return {'adapted': arrays, 'advanced': advanced, 'valid': valid, 'candidate_identity': identity,
        'proof': proof, 'manifests': manifests, 'remote_requests': requests,
        'controls': {'S006f': control_directory / 'S006f', 'S007b': candidate_directory / 'S007b'}}


def finish_recipe(directory, tracker, contract, predictions, folds, recipe, checks):
    pooled = score_predictions(contract['manifest'], predictions, contract['labels'])
    report = {'recipe': recipe, 'oof': pooled, 'folds': folds, 'source_control_checks': deepcopy(checks), 'limitations': LIMITATIONS}
    write_json(directory / 'experiment_report.json', report)
    write_csv(directory / 'oof_predictions.csv', predictions)
    write_csv(directory / 'oof_per_class.csv', pooled['per_class'])
    for name in ('experiment_report.json', 'oof_predictions.csv', 'oof_per_class.csv'):
        tracker.add_artifact(directory / name)
    tracker.log_metrics({'oof/macro_f1_447': pooled['macro_f1'], 'oof/accuracy': pooled['accuracy'],
        **{'oof/' + key: value for key, value in pooled['errors'].items()}}, sync=False)
    tracker.write_report(report, markdown=f"# {recipe['id']}\n\n{recipe['protocol']}. OOF Macro-F1: {pooled['macro_f1']:.6f}. No new encoder updates; each adapted query uses only its own outer-fold source.")
    tracker.finish('FINISHED', strict=True)
    return report


def execute(root, config_path, suite, contracts, binding_path):
    import torch
    from speaker_id.infrastructure.readiness import validate_readiness_for_execution
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding

    validate_suite(suite)
    validate_readiness_for_execution(root, contracts['readiness'])
    require(os.environ.get('VAST_INSTANCE_ID') == '50079023' and torch.cuda.is_available()
        and '3090' in torch.cuda.get_device_name(0), 'S009 requires the authorized RTX 3090 instance')
    torch.set_num_threads(4)
    # Failed prerequisites create no tracking run and perform no scoring.
    sources = load_sources(root, suite, contracts)
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding='utf-8'))['binding'])
    output = root / 'artifacts/training' / ('S009_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    inputs = {'suite_config': config_path, 'launcher': root / 'scripts/score_adapted_candidate_fusion.py',
        'adapted_config': root / suite['source_adapted']['config'],
        'candidate_verification': root / suite['candidate_verification']['path'],
        'S006_verification': root / suite['control_adapted']['verification']['path'],
        'S008_verification': root / suite['prerequisite_fusion']['verification']['path']}
    inputs.update({key: root / contracts['readiness']['config'][key] for key in ('manifest', 'folds', 'roles', 'label_map', 'model_config')})
    resolved = {'suite': suite, 'contracts': {name: {key: row[key] for key in ('config', 'model', 'input_hashes', 'code_hashes', 'signature')}
        for name, row in contracts.items()}, 'captured_candidate_identity': sources['candidate_identity'], 'limitations': LIMITATIONS}
    write_json(output / 'resolved_config.json', resolved)
    common = {'project_root': root, 'binding': binding, 'input_paths': inputs,
        'run_kind': 'adapted_frozen_original_heldout_fusion', 'training_started': False}
    parent = DurableMLflowRun.prepare(spool_dir=output / 'tracking', run_name=suite['run_name'], config=resolved, **common)
    child, started = None, time.monotonic()
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        remote = _verify_remote_evidence(parent.client, binding, sources['remote_requests'], output / 'verified_remote_evidence')
        proof = {**sources['proof'], 'remote_evidence': remote}
        write_json(output / 'source_provenance.json', proof)
        parent.add_artifact(output / 'source_provenance.json')
        for name, exported in sources['manifests'].items():
            path = output / (name + '_audited_export_manifest.json')
            write_json(path, exported)
            parent.add_artifact(path)
        for name in ('suite_config', 'launcher', 'adapted_config', 'candidate_verification', 'S006_verification', 'S008_verification'):
            parent.add_artifact(inputs[name], 'input_configs/' + name + inputs[name].suffix)
        write_json(output / 'experiment_state.json', {'status': 'running', 'parent_run_id': parent.run_id})
        contract, valid = contracts['adapted'], sources['valid']
        labels, manifest = contract['labels'], contract['manifest']
        label_index = {label: i for i, label in enumerate(labels)}
        checks, results, predictions_by_recipe = {}, [], {}
        endpoints = {0: {}, 1: {}}
        for recipe in RECIPES:
            if recipe['control'] is None:
                require_controls(checks)
            directory = output / recipe['id']
            directory.mkdir()
            child = DurableMLflowRun.prepare(spool_dir=directory / 'tracking', parent_run_id=parent.run_id,
                run_name=recipe['id'] + '-' + recipe['source'], config={**resolved, 'recipe': recipe}, **common)
            child.flush(strict=True)
            child.add_artifact(output / 'source_provenance.json')
            predictions, fold_reports, fold_checks = [], [], {}
            for outer in (0, 1):
                alpha = 0.0 if recipe['id'] == 'S009a' else 1.0
                if recipe['id'] == 'S009a':
                    scores = heldout_reference_scores(contract, sources['adapted'][outer][0], valid, outer)
                    endpoints[outer]['adapted'] = scores
                elif recipe['id'] == 'S009b':
                    # This one independent source control uses ONLY the public encoder.
                    scores = crossfit_scores(sources['advanced'], valid, manifest, contract['folds'], outer, 'max_reference')
                elif recipe['id'] == 'S009c':
                    scores = heldout_reference_scores(contract, sources['advanced'], valid, outer)
                    require_aligned_scores(endpoints[outer]['adapted'], scores)
                    endpoints[outer]['advanced_matched'] = scores
                else:
                    require_controls(checks)
                    candidates = {a: matched_scores(contract, sources['adapted'], sources['advanced'], valid, outer, a, endpoints[outer]) for a in ALPHAS}
                    query = endpoints[outer]['adapted']['calibration_indices']
                    inner_truth = np.asarray([label_index[manifest[int(i)]['speaker_id']] for i in query])
                    selected, curves = select_inner_alpha({a: {'known': s['inner_known_scores'], 'unknown': s['inner_unknown_similarity']}
                        for a, s in candidates.items()}, inner_truth)
                    alpha = selected['advanced_weight']
                    scores, calibration, curve = candidates[alpha], selected['calibration'], curves[str(alpha)]['curve']
                    inner_path = directory / f'fold_{outer}_inner_alpha_calibration.json'
                    write_json(inner_path, {'selected': selected, 'candidates': curves})
                    child.add_artifact(inner_path, f'fold_{outer}/inner_alpha_calibration.json')
                if recipe['id'] != 'S009d':
                    query = scores['calibration_indices']
                    inner_truth = np.asarray([label_index[manifest[int(i)]['speaker_id']] for i in query])
                    calibration, curve = calibrate_gate(scores['inner_known_scores'], inner_truth, scores['inner_unknown_similarity'],
                        UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, 201)
                require(scores['known_labels'] == labels[1:], 'S009 score columns differ from the original labels')
                historical = sources['controls'].get(recipe['control'])
                policy = {'recipe': recipe, 'advanced_weight': alpha, 'adapted_outer_fold': outer if alpha < 1 else None,
                    'role_proof': assert_fixed_role_groups(contract, outer),
                    'checkpoint_binding': suite['source_adapted']['folds'][str(outer)] if alpha < 1 else None,
                    'selection_scope': 'original held-out groups' if recipe['id'] != 'S009b' else 'frozen full-crossfit control only'}
                rows, report, reproduction = _record_fold(directory / f'fold_{outer}', child, contract, outer, scores, valid,
                    calibration, curve, policy, historical)
                if historical:
                    verify_probability_control(directory / f'fold_{outer}', historical, outer)
                    if recipe['id'] == 'S009a':
                        prior = json.loads((historical / f'fold_{outer}/evaluation.json').read_text(encoding='utf-8'))
                        require(prior['scoring_provenance'] == scores['provenance'], 'S006f original query/reference groups changed')
                    fold_checks[str(outer)] = {**reproduction, 'exact_probabilities': True, 'exact_full_calibration_curve': True}
                child.log_metrics({f'fold_{outer}/selected_advanced_weight': alpha}, sync=True)
                predictions.extend(rows)
                fold_reports.append(report)
                print(json.dumps({'stage': 'adapted_candidate_fusion', 'recipe': recipe['id'], 'fold': outer,
                    'advanced_weight': alpha, 'historical_control_exact': historical is not None}), flush=True)
            if recipe['control']:
                historical = sources['controls'][recipe['control']]
                pooled = score_predictions(manifest, predictions, labels)
                require(json.loads((historical / 'experiment_report.json').read_text(encoding='utf-8'))['oof'] == pooled,
                    'Historical pooled metrics differ')
                checks[recipe['control']] = {**verify_predictions(historical / 'oof_predictions.csv', predictions),
                    'exact_pooled_metrics': True, 'folds': fold_checks}
                write_json(output / 'source_control_checks.json', checks)
            child.add_artifact(output / 'source_control_checks.json')
            result = finish_recipe(directory, child, contract, predictions, fold_reports, recipe, checks)
            results.append(result)
            predictions_by_recipe[recipe['id']] = predictions
            child = None
        require_controls(checks)
        comparisons = {name + '_to_S009d': paired_diagnostics(manifest, predictions_by_recipe[name], predictions_by_recipe['S009d'], labels)
            for name in ('S009a', 'S009c')}
        report = {'status': 'complete', 'parent_run_id': parent.run_id, 'results': results,
            'source_control_checks': checks, 'comparisons': comparisons, 'conditional_dispatch': proof['conditional_dispatch'],
            'selection_policy': SELECTION_POLICY, 'execution_policy': EXECUTION_POLICY, 'new_encoder_updates': 0,
            'source_adaptation': 'own-fold F004; 600 head-only plus 500 tail steps',
            'limitations': LIMITATIONS, 'elapsed_seconds': time.monotonic() - started}
        write_json(output / 'experiment_report.json', report)
        for name in ('experiment_report.json', 'source_control_checks.json', 'resolved_config.json'):
            parent.add_artifact(output / name)
        parent.log_metrics({result['recipe']['id'] + '/oof_macro_f1_447': result['oof']['macro_f1'] for result in results}, sync=True)
        parent.write_report(report, markdown='# S009 original-held-out adapted/public fusion\n\nBoth historical controls passed before mixtures. The advanced matched endpoint has its own protocol; it is not S007b. No new encoder fitting or extraction.')
        parent.finish('FINISHED', strict=True)
        write_json(output / 'experiment_state.json', {'status': 'complete', 'parent_run_id': parent.run_id})
        return {'output': str(output), 'parent_run_id': parent.run_id,
            'scores': {r['recipe']['id']: r['oof']['macro_f1'] for r in results}}
    except BaseException as error:
        failure = {'status': 'failed', 'parent_run_id': parent.run_id, 'error_type': type(error).__name__,
            'error': str(error), 'resume_supported': False}
        write_json(output / 'experiment_state.json', failure)
        if child is not None:
            child.write_report(failure)
            child.finish('FAILED', strict=False)
        parent.write_report(failure)
        parent.finish('FAILED', strict=False)
        raise
