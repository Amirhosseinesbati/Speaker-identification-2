"""Populate a clean local postprocessing checkout with immutable input hardlinks.

This does not clone Git, open MLflow, read raw audio or fit any model. The caller
first creates a detached local checkout of the published source revision. Source
inputs are never edited by the scoring runner; new outputs have separate paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = [
    'data/processed/eda_v1',
    'artifacts/training/B002_20260907T150724Z_9b11fe4b',
    'artifacts/training/S002_20260907T153007Z_ec71bae8',
    'artifacts/training/S007_20260907T224739Z_9be25eef',
    'artifacts/training/S008_20260907T233351Z_b35f3c28',
    'artifacts/models/campp', 'artifacts/models/campp_advanced',
    'reports/research/decision_postprocessing_20260908',
]
FILES = [
    'artifacts/infrastructure/S007_verification/server_export_manifest.json',
    'artifacts/infrastructure/S008_verification/server_export_manifest.json',
    'artifacts/infrastructure/S008_verification/verification.json',
    'artifacts/infrastructure/S011_verification/S011_20260908T114324Z_4d842981/verification.json',
    'artifacts/infrastructure/mlflow_state.json',
    'artifacts/infrastructure/S012_preparation/dependency_audit.json',
    'artifacts/infrastructure/S012_preparation/sklearn_install_verification.json',
    'artifacts/infrastructure/S012_preparation/nested_cases_validation.json',
    'artifacts/infrastructure/S012_preparation/tree_validation.json',
    'artifacts/infrastructure/S012_preparation/scoring_validation.json',
    'artifacts/infrastructure/S012_preparation/protocol_design.md',
    'artifacts/infrastructure/S012_preparation/research/manifest.json',
    'artifacts/infrastructure/S012_preparation/research/addendum_manifest.json',
    'artifacts/infrastructure/S011_preparation/local_cuda_readiness.json',
    'artifacts/infrastructure/S011_preparation/local_install_verification.json',
    'artifacts/infrastructure/S011_preparation/mlflow_readonly_connectivity.json',
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(root, *arguments):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    return subprocess.check_output(['git', '-C', str(root), *arguments], env=env,
        text=True, encoding='utf-8', timeout=30).strip()


def populate(destination, commit, suite='S012'):
    destination = Path(destination).resolve()
    if not (re.fullmatch('[a-f0-9]{40}', commit) and destination.is_relative_to(ROOT / 'tmp')
            and destination != ROOT / 'tmp' and (destination / '.git').exists()
            and git(destination, 'rev-parse', 'HEAD') == commit
            and not git(destination, 'status', '--porcelain', '--', 'src', 'scripts', 'configs', 'pyproject.toml', 'uv.lock')):
        raise ValueError('Require a clean isolated checkout at the exact published commit under project tmp/')
    if suite not in ('S012', 'S013'):
        raise ValueError('Unregistered local postprocessing suite')
    directories, input_files = list(DIRECTORIES), list(FILES)
    if suite == 'S013':
        config = json.loads((ROOT / 'configs/postprocessing/campp_s013.json').read_text(encoding='utf-8'))
        prior = config['s012_prerequisite']
        if not (isinstance(prior.get('run_path'), str)
                and re.fullmatch(r'artifacts/training/S012_\d{8}T\d{6}Z_[a-f0-9]{8}', prior['run_path'])
                and prior.get('verification_path') == 'artifacts/infrastructure/S012_verification/'
                    + prior['run_path'].split('/')[-1] + '/verification.json'):
            raise ValueError('S013 requires the pinned completed S012 input paths')
        if (digest(ROOT / prior['verification_path']) != prior['verification_sha256']
                or digest(ROOT / prior['run_path'] / 'experiment_report.json') != prior['report_sha256']):
            raise ValueError('S012 completion proof differs from the S013 pins')
        directories.append(prior['run_path'])
        input_files += [prior['verification_path'],
            'artifacts/infrastructure/S012_preparation/research/metric_next_route_manifest.json',
            'artifacts/infrastructure/S013_preparation/metric_validation.json']
    sources = []
    for relative in directories:
        folder = ROOT / relative
        if not folder.is_dir():
            raise ValueError('Required historical directory is absent: ' + relative)
        sources.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources.extend(ROOT / relative for relative in input_files)
    receipt = []
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        if (not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(ROOT)
                or source.name.startswith('.env') or source.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}):
            raise ValueError('Unsafe or missing local input: ' + relative.as_posix())
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Destination input escapes the isolated checkout')
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = 'copy' if relative.as_posix() == 'artifacts/infrastructure/mlflow_state.json' else 'hardlink'
        expected = digest(source)
        if target.exists():
            if target.is_symlink() or digest(target) != expected:
                raise ValueError('Existing destination input differs: ' + relative.as_posix())
        elif mode == 'copy':
            shutil.copyfile(source, target)
        else:
            # CreateHardLinkW needs explicit long-path syntax for deeply nested
            # historical tracking files even when normal Path reads succeed.
            prefix = '\\\\?\\' if os.name == 'nt' else ''
            os.link(prefix + str(source), prefix + str(target))
        if digest(target) != expected:
            raise ValueError('Isolated local input hash differs: ' + relative.as_posix())
        receipt.append({'path': relative.as_posix(), 'bytes': source.stat().st_size,
                        'sha256': expected, 'transport': mode})
    result = {'status': 'verified_local_inputs_ready_no_experiment_started', 'git_commit': commit,
              'source_root': str(ROOT), 'execution_root': str(destination), 'files': receipt,
              'new_output_directory_shared': False, 'raw_audio_copied': False,
              'tokens_copied': False, 'remote_server_accessed': False}
    output = destination / f'artifacts/infrastructure/{suite}_preparation/isolated_workspace.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'files': len(receipt), 'receipt': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--suite', choices=['S012', 'S013'], default='S012')
    args = parser.parse_args()
    populate(args.destination, args.commit, args.suite)
