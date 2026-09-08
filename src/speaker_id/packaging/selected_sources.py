"""Attested source inputs for a selected release; never shipped in its ZIP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np

from speaker_id.infrastructure.data import confined_path
from speaker_id.models.campp import file_sha256
from speaker_id.inference.selected_policy import F004_SOURCE, ROLES_SHA256
from speaker_id.training.adapted_scoring import (
    verified_export_inventory, validate_adapted_identity, load_adapted_fold_cache)
from speaker_id.training.adaptation_comparison import (
    verify_source_snapshot, verify_final_schedule, _remote_requests)
from speaker_id.training.candidate_fusion import validate_historical_candidate, _candidate_remote_requests
from speaker_id.training.candidate_comparison import verify_candidate_cache, _public_remote_requests, SOURCE
from speaker_id.training.contracts import load_contract
from speaker_id.training.frozen_suite import validated_cache

FAMILIES = {'S006f': ('S006', 'adapted_only', 7), 'S007b': ('S007', 'advanced_only', 3),
            'S008c': ('S008', 'public_advanced', 4), 'S009d': ('S009', 'adapted_advanced', 5)}
EXPORT_KEYS = {'run', 'parent_run_id', 'git_commit', 'children', 'export_manifest_paths', 'export_manifest_sha256'}
FIXED = {'schema_version': 1, 'experiment_code': 'P002', 'readiness_config': 'configs/train/campp_coverage.json',
    'output_root': 'artifacts/releases', 'expected_vast_instance_id': 50079023, 'tracking_required': True,
    'encoder_fold': 0, 'alphas': [0.0, .25, .5, .75, 1.0], 'alpha_tie_order': [0.0, 1.0, .25, .5, .75],
    'unknown_weights': [0.0, .25, .5, .75, 1.0], 'margin_weights': [0.0, .5],
    'threshold_candidates': 201, 'probability_temperature': .05,
    'inference': {'seconds': 180.0, 'maximum_windows': 1}, 'roles_sha256': ROLES_SHA256,
    'expected_counts': {'source': 4529, 'known_references': 2217, 'unknown_references': 2223,
                        'zeros': 89, 'q0_queries': 999, 'frozen_queries': 4440},
    'new_encoder_updates': 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, size=64):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{' + str(size) + '}', value) is not None


def project_file(root, name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
        and not PurePosixPath(name).is_absolute() and PurePosixPath(name).as_posix() == name
        and '..' not in PurePosixPath(name).parts, 'Input must be a canonical relative project path')
    path = confined_path(root, Path(name))
    require(path.is_file(), 'Pinned input is missing: ' + name)
    return path


def validate_config(config):
    require(set(config) == set(FIXED) | {'selection'}, 'Unknown or missing selected-build configuration fields')
    require(all(config[k] == v for k, v in FIXED.items()), 'Final-fit procedure differs from the fixed release policy')
    selection = config['selection']
    require(set(selection) == {'recipe_id', 'family', 'source', 'report_sha256', 'verification'}, 'Incomplete selection proof')
    require(selection['recipe_id'] in FAMILIES, 'Unsupported selected completed recipe')
    code, family, count = FAMILIES[selection['recipe_id']]
    require(selection['family'] == family and digest(selection['report_sha256']), 'Selection family/report binding differs')
    source = selection['source']
    require(set(source) == EXPORT_KEYS and digest(source['parent_run_id'], 32) and digest(source['git_commit'], 40)
        and digest(source['export_manifest_sha256']) and isinstance(source['export_manifest_paths'], list)
        and source['export_manifest_paths'] and len(source['children']) == count - 1
        and all(name.startswith(code) and digest(rid, 32) for name, rid in source['children'].items())
        and len(set(source['children'].values()) | {source['parent_run_id']}) == count
        and selection['recipe_id'] in source['children'], 'Selected export/run identity is incomplete')
    pin = selection['verification']
    require(set(pin) == {'path', 'sha256'} and digest(pin['sha256']), 'Independent completion proof must be byte-pinned')


def _json(directory, name, inventory):
    require(name in inventory, 'Required evidence absent from retained export: ' + name)
    return json.loads((directory / name).read_text(encoding='utf-8'))


def verify_selected_audit(audit, selection, report):
    source, recipe = selection['source'], selection['recipe_id']
    expected = set(source['children'].values()) | {source['parent_run_id']}
    runs = audit.get('runs', [])
    manifest_hash = audit.get('export_manifest_sha256', audit.get('archive_verification', {}).get('manifest_sha256'))
    require(audit.get('status') == 'verified' and audit.get('parent_run_id') == source['parent_run_id']
        and audit.get('run_name') == Path(source['run']).name and audit.get('git_commit') == source['git_commit']
        and manifest_hash == source['export_manifest_sha256'] and len(runs) == len(expected)
        and {row.get('run_id') for row in runs} == expected
        and all(row.get('status') == 'FINISHED' and row.get('git_commit') == source['git_commit'] for row in runs),
        'Selected source requires complete independent immutable-source and owned-run verification')
    score = report['oof']['macro_f1']
    require(np.isfinite(score) and 0 <= score <= 1 and audit['metrics'][recipe]['macro_f1'] == score,
        'Selected development OOF score differs from independently recomputed results')
    return {'verification_status': 'verified', 'precursor_oof_macro_f1_447': score,
            'scope': 'Selected development procedure; not an OOF claim for the final fitted payload'}


def verify_selection(root, selection):
    """Authenticate captured selection bytes without recomputing an old all-src signature."""
    source = selection['source']
    directory, export, inventory = verified_export_inventory(root, source)
    captured = _json(directory, 'resolved_config.json', inventory)
    require(captured == _json(directory, 'tracking/artifacts/resolved_config.json', inventory), 'Captured configuration changed')
    for name in ('experiment_state.json', 'experiment_report.json'):
        record = _json(directory, name, inventory)
        require(record.get('status') == 'complete' and record.get('parent_run_id') == source['parent_run_id'], 'Incomplete selected run')
    parent = _json(directory, 'tracking/run_state.json', inventory)
    require(parent.get('run_id') == source['parent_run_id'] and parent.get('remote_status') == 'FINISHED'
        and parent.get('last_sync_error') is None and parent['tags'].get('mlflow.source.git.commit') == source['git_commit'],
        'Selected parent receipt differs')
    metadata = _json(directory, 'tracking/artifacts/source_manifest.json', inventory)
    snapshot = directory / 'tracking/artifacts/source_snapshot.zip'
    require('tracking/artifacts/source_snapshot.zip' in inventory and metadata.get('schema_version') == 2
        and metadata.get('src_dirty') is False and metadata.get('git_commit') == source['git_commit']
        and metadata.get('archive_sha256') == file_sha256(snapshot), 'Historical source snapshot identity differs')
    entries = {row['path']: row for row in metadata['files']}
    require(len(entries) == len(metadata['files']) == metadata['file_count'], 'Duplicate source inventory')
    with zipfile.ZipFile(snapshot) as archive:
        require(len(archive.infolist()) == len(entries) and set(archive.namelist()) == set(entries), 'Incomplete historical source ZIP')
        for item in archive.infolist():
            path, row = PurePosixPath(item.filename), entries[item.filename]
            require(item.filename.startswith('src/') and path.as_posix() == item.filename and '..' not in path.parts
                and '\\' not in item.filename and ':' not in item.filename and not item.is_dir()
                and not stat.S_ISLNK(item.external_attr >> 16) and item.file_size == row['bytes']
                and hashlib.sha256(archive.read(item)).hexdigest() == row['sha256'], 'Unsafe or changed historical source entry')
    launcher = directory / 'tracking/artifacts/input_configs/launcher.py'
    inputs = _json(directory, 'tracking/artifacts/inputs_manifest.json', inventory)
    require('tracking/artifacts/input_configs/launcher.py' in inventory
        and inputs.get('launcher', {}).get('sha256') == file_sha256(launcher), 'Historical launcher does not match receipt')
    requests = [(source['parent_run_id'], None, [(name, directory / 'tracking/artifacts' / name) for name in
        ('resolved_config.json', 'source_manifest.json', 'source_snapshot.zip', 'input_configs/launcher.py')]
        + [('experiment_report.json', directory / 'experiment_report.json')])]
    for code, rid in source['children'].items():
        child = _json(directory, code + '/tracking/run_state.json', inventory)
        original = _json(directory, code + '/tracking/artifacts/resolved_config.json', inventory)
        recipe = original.get('recipe')
        require(child.get('run_id') == rid and child.get('remote_status') == 'FINISHED' and child.get('last_sync_error') is None
            and child['tags'].get('mlflow.parentRunId') == source['parent_run_id']
            and child['tags'].get('mlflow.source.git.commit') == source['git_commit'] and original.get('suite') == captured['suite']
            and (recipe if isinstance(recipe, str) else recipe.get('id')) == code, 'Selected child/configuration differs')
        files = [('resolved_config.json', directory / code / 'tracking/artifacts/resolved_config.json')]
        files += [(name, directory / 'tracking/artifacts' / name) for name in ('source_manifest.json', 'source_snapshot.zip')]
        files += [(name, directory / code / name) for name in ('experiment_report.json', 'oof_predictions.csv')]
        requests.append((rid, source['parent_run_id'], files))
    recipe_name = selection['recipe_id'] + '/experiment_report.json'
    require(inventory[recipe_name]['sha256'] == selection['report_sha256'], 'Selected report is not the pinned reviewed report')
    report = _json(directory, recipe_name, inventory)
    audit_path = project_file(root, selection['verification']['path'])
    require(file_sha256(audit_path) == selection['verification']['sha256'], 'Independent audit bytes changed')
    audit = verify_selected_audit(json.loads(audit_path.read_text(encoding='utf-8')), selection, report)
    return {'directory': directory, 'captured': captured, 'report': report, 'inventory': inventory,
            'export': export, 'remote_requests': requests, 'proof': {**audit, 'source_snapshot_sha256': file_sha256(snapshot),
                'verification_sha256': selection['verification']['sha256'], 'selection': selection}}


def merge_remote_requests(requests):
    """One readback directory per run, retaining every distinct artifact check."""
    merged = {}
    for run_id, parent, files in requests:
        if run_id not in merged:
            merged[run_id] = (parent, {})
        expected_parent, inventory = merged[run_id]
        require(parent == expected_parent, 'Duplicate run has conflicting parent identity')
        for remote_path, local_path in files:
            if remote_path in inventory:
                require(file_sha256(inventory[remote_path]) == file_sha256(local_path),
                        'Duplicate remote artifact has conflicting expected local bytes')
            else:
                inventory[remote_path] = local_path
    return [(run_id, parent, list(inventory.items())) for run_id, (parent, inventory) in merged.items()]


def load_sources(root, config, *, verify_audio=False):
    """Read historical caches only after identity checks; no forward or optimizer."""
    validate_config(config)
    contract = load_contract(project_file(root, config['readiness_config']), root, verify_audio=verify_audio)
    require(contract['input_hashes']['roles'] == ROLES_SHA256 and len(contract['manifest']) == 4529
        and len(contract['labels']) == 447, 'Original data/role/label contract differs')
    selected = verify_selection(root, config['selection'])
    suite, family = selected['captured']['suite'], config['selection']['family']
    vectors, masks, assets, proof = {}, [], {}, {'selection': selected['proof']}
    captured_configs = {'selection': selected['captured']}
    requests, exports = list(selected['remote_requests']), {'selection': selected['export']}
    if family in ('adapted_only', 'adapted_advanced'):
        source = suite['sources']['F004'] if family == 'adapted_only' else suite['source_adapted']
        require(source['config'] == 'configs/train/campp_finetune_head600.json' and source['completed_steps'] == 1100
            and source['parent_run_id'] == F004_SOURCE['parent_run_id'] and source['git_commit'] == F004_SOURCE['source_git_commit']
            and source['folds']['0']['child_run_id'] == F004_SOURCE['child_run_id']
            and source['folds']['0']['checkpoint_sha256'] == F004_SOURCE['checkpoint_sha256'], 'Release fixes the original F004 fold 0')
        require(family != 'adapted_only' or suite['expanded_arm'] is True, 'S006 source must be the expanded procedure')
        adapted = load_contract(project_file(root, source['config']), root, verify_audio=verify_audio)
        require({k: v for k, v in adapted['input_hashes'].items() if k != 'model_config'}
            == {k: v for k, v in contract['input_hashes'].items() if k != 'model_config'}, 'Adapted data/roles differ from readiness')
        directory, export, inventory = verified_export_inventory(root, source)
        original, fold_proof = validate_adapted_identity(directory, source, adapted, inventory)
        captured_configs['F004'] = original
        snapshot = verify_source_snapshot(directory, source, original, inventory)
        schedule = verify_final_schedule(directory, source, adapted, inventory)
        values, valid, files = load_adapted_fold_cache(directory, adapted, source['signature'], inventory, 0)
        vectors['adapted'], assets['adapted'] = values, {'checkpoint': directory / 'fold_0/last.pt', 'source': source}
        masks.append(valid)
        proof['adapted'] = {'source': source, 'folds': fold_proof, 'snapshot': snapshot, 'schedules': schedule, 'cache_files': files}
        exports['F004'] = export
        requests += _remote_requests({'sources': {'F004': source}, 'controls': {}}, {'F004': directory}, {})
        contract_for_queries = adapted
    else:
        contract_for_queries = contract
    if family != 'adapted_only':
        source = suite['source_advanced'] if family != 'advanced_only' else {
            **config['selection']['source'], 'source_signature': selected['captured']['candidate_identity']['signature']}
        directory, export, inventory = verified_export_inventory(root, source)
        identity, candidate_proof = validate_historical_candidate(directory, source, contract, inventory, root)
        captured_configs['S007'] = _json(directory, 'resolved_config.json', inventory)
        receipt = _json(directory, 'candidate_cache_manifest.json', inventory)
        values, valid = verify_candidate_cache(directory / 'candidate_embedding_cache', identity, contract['manifest'], receipt)
        vectors['advanced'] = values
        masks.append(valid)
        base = directory / 'tracking/artifacts/candidate_model'
        assets['advanced'] = {'config_path': base / 'model_config.json', 'config': identity['model'],
            'weights': base / Path(identity['model']['weights_path']).name,
            'source_record': {'encoder_kind': 'advanced_public_192', 'encoder_updates': 0, 'weights_sha256': identity['weights_sha256'],
                'source_parent_run_id': source['parent_run_id'], 'source_signature': identity['signature'], 'source_git_commit': source['git_commit']}}
        proof['advanced'], exports['S007'] = candidate_proof, export
        requests += _candidate_remote_requests(directory, source)
    if family == 'public_advanced':
        source = suite['source_public']
        require(source == SOURCE, 'Selected public source is not the exact B002 full-utterance cache')
        values, valid, public = validated_cache(root, root / source['source_run'], contract, source['source_parent_run_id'])
        require(_json(selected['directory'], 'source_provenance.json', selected['inventory'])['public_cache'] == public,
            'Public cache differs from the exact cache used by the selected completed procedure')
        vectors['public'] = values
        masks.append(valid)
        assets['public'] = {'config': contract['model'], 'weights': project_file(root, contract['model']['weights_path']),
            'source_record': {'encoder_kind': 'public_voxceleb_512', 'encoder_updates': 0, 'weights_sha256': contract['model']['weights_sha256'],
                'source_parent_run_id': source['source_parent_run_id'], 'source_signature': public['source_signature'],
                'source_git_commit': public['source_git_commit']}}
        proof['public'] = public
        captured_configs['B002'] = json.loads((root / source['source_run'] / 'resolved_config.json').read_text(encoding='utf-8'))
        requests += _public_remote_requests(root, source)
    require(masks and all(np.array_equal(masks[0], mask) for mask in masks)
        and int((~masks[0]).sum()) == 89, 'Selected encoders disagree on original row order/zero semantics')
    return {'contract': contract, 'query_contract': contract_for_queries, 'vectors': vectors, 'valid': masks[0],
        'assets': assets, 'proof': proof, 'exports': exports, 'remote_requests': merge_remote_requests(requests), 'selection': selected,
        'captured_configs': captured_configs}
