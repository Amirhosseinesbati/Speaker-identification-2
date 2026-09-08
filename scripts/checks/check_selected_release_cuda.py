"""Explicit forward-only selected-release QA, not the organizer evaluator image.

Default invocation validates arguments only. --execute verifies existing source
caches and runs representative CUDA inference in a fresh directory outside the
checkout, with empty model caches and Python socket calls disabled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

ROOT = next((path for path in Path(__file__).resolve().parents
             if (path / 'pyproject.toml').is_file() and (path / 'configs').is_dir()), Path(__file__).resolve().parents[2])
TOLERANCE = {'maximum_absolute_embedding_difference': .002, 'minimum_embedding_cosine': .9999,
             'speaker_decisions_must_match': True, 'bit_identical_embeddings_required': False}
CORRUPT = '__qa_corrupt__.wav'

WORKER = r'''
import hashlib, json, os, pathlib, runpy, socket, sys, time
started = time.monotonic()
attempts, backwards = [], []
def blocked(*args, **kwargs):
    attempts.append('network entry point called')
    raise RuntimeError('Network is disabled for selected-release QA')
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
root, inputs, out, result, vectors = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root / 'src' if (root / 'src/speaker_id').is_dir() else root))
import numpy as np
import torch, torchaudio, soundfile, scipy
def no_backward(*args, **kwargs):
    backwards.append('backward requested')
    raise RuntimeError('Backward is forbidden in release QA')
torch.Tensor.backward = no_backward
torch.autograd.backward = no_backward
torch.set_num_threads(4)
assert os.environ.get('VAST_INSTANCE_ID') == '50079023'
assert torch.cuda.is_available() and '3090' in torch.cuda.get_device_name(0)
torch.cuda.reset_peak_memory_stats()
from speaker_id.inference import selected_runtime as runtime
from speaker_id.inference.scoring import score_embeddings
payload = runtime.verify_payload(root)
model_config, policy = payload['models'], payload['policy']
calibration = json.loads((root / 'assets/calibration.json').read_text())
with np.load(root / 'assets/gallery.npz', allow_pickle=False) as saved:
    gallery = {key: saved[key] for key in saved.files}
encoders, device = runtime._load_models(model_config, root, 'cuda')
names, embeddings, valid = [], [], []
for path in sorted(inputs.iterdir()):
    if not path.is_file() or path.name == '__qa_corrupt__.wav':
        continue
    vector, info = runtime._extract(encoders, policy, path, device)
    names.append(path.name); embeddings.append(vector); valid.append(info['nonzero_signal'])
valid = np.asarray(valid, dtype=bool)
prob = score_embeddings(np.asarray(embeddings), valid, gallery, calibration)
assert prob.shape == (len(names), 447)
assert np.isfinite(prob).all() and (prob >= 0).all()
assert np.allclose(prob.sum(axis=1), 1, atol=1e-12)
assert (prob[~valid].argmax(axis=1) == 0).all()
np.savez(vectors, audio_file=np.asarray(names), embedding=np.asarray(embeddings), valid=valid, probabilities=prob)
del encoders
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('existing output must be replaced atomically\n')
sys.argv = [str(root / 'submission.py'), '--data-dir', str(inputs), '--predictions-file-path', str(out)]
try:
    runpy.run_path(str(root / 'submission.py'), run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
protected = [inputs / names[0], root / 'assets/policy.json', root / 'manifest.json']
for target in protected:
    before = target.read_bytes()
    try:
        runtime.run_submission(root, inputs, target, device='cuda')
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe output overwrite was accepted')
    assert target.read_bytes() == before
previous_output = out.read_bytes()
original_loader = runtime._load_models
def model_failure(*args, **kwargs):
    raise RuntimeError('Synthetic fail-closed loader probe')
runtime._load_models = model_failure
try:
    try:
        runtime.run_submission(root, inputs, out, device='cuda')
    except RuntimeError as error:
        assert str(error) == 'Synthetic fail-closed loader probe'
    else:
        raise AssertionError('Injected model failure did not abort')
finally:
    runtime._load_models = original_loader
assert out.read_bytes() == previous_output
runtime.verify_payload(root)
for forbidden in ('mlflow', 'speaker_id.training', 'speaker_id.tracking', 'huggingface_hub', 'modelscope'):
    assert not any(name == forbidden or name.startswith(forbidden + '.') for name in sys.modules), forbidden
assert not attempts and not backwards
result.write_text(json.dumps({'status': 'passed', 'network_attempts': len(attempts),
    'network_block': 'Python socket connect/connect_ex/create_connection/getaddrinfo',
    'device': 'cuda', 'cuda_device_name': torch.cuda.get_device_name(0), 'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(), 'optimizer_steps': 0, 'backward_calls': len(backwards),
    'torch': torch.__version__, 'torchaudio': torchaudio.__version__, 'numpy': np.__version__,
    'scipy': scipy.__version__, 'soundfile': soundfile.__version__, 'python': sys.version,
    'policy_kind': policy['kind'], 'embedding_dim': policy['embedding_dim'],
    'real_audio_files': len(names), 'protected_output_targets': len(protected),
    'existing_output_replaced': True, 'model_failure_preserved_output': True,
    'payload_unchanged': True, 'forbidden_imports_absent': True,
    'elapsed_seconds': time.monotonic()-started}, indent=2))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def no_links(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                    for p in (path, *path.parents)), 'Linked paths are forbidden')
    return path.resolve()


def select_examples(rows, project_root=ROOT):
    """Use the existing P001 selector without changing its representative policy."""
    spec = importlib.util.spec_from_file_location('_p001_offline_cases', project_root / 'scripts/checks/check_release_offline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_examples(rows)


def isolated_environment(base, run_dir):
    allowed = {'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'COMSPEC', 'PATHEXT', 'LD_LIBRARY_PATH'}
    env = {k: v for k, v in base.items() if k.upper() in allowed}
    env.update({'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '0',
        'VAST_INSTANCE_ID': '50079023', 'HF_HOME': str(run_dir / 'h'), 'TORCH_HOME': str(run_dir / 't'), 'XDG_CACHE_HOME': str(run_dir / 'x'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUTF8': '1',
        'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    return env


def extract_verified_archive(archive_path, destination, manifest):
    """Extract the exact built ZIP into a new short directory, without overwrites."""
    require(not destination.exists(), 'QA package extraction must use a fresh directory')
    expected = set(manifest['files']) | {'manifest.json'}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        require(len(infos) == len(expected) and set(archive.namelist()) == expected, 'ZIP file set differs from manifest')
        require(len({name.casefold() for name in expected}) == len(expected), 'Case-colliding ZIP members')
        for info in infos:
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == info.filename
                and '\\' not in info.filename and ':' not in info.filename and not info.is_dir()
                and not any(part.startswith('.') for part in path.parts)
                and not stat.S_ISLNK(info.external_attr >> 16), 'Unsafe ZIP member')
            payload = archive.read(info)
            if info.filename == 'manifest.json':
                require(json.loads(payload) == manifest, 'ZIP manifest differs from build manifest')
            else:
                record = manifest['files'][info.filename]
                require(len(payload) == record['bytes'] and hashlib.sha256(payload).hexdigest() == record['sha256'], 'ZIP member digest differs')
        destination.mkdir(parents=True, exist_ok=False)
        for info in infos:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(archive.read(info))


def selected_vectors(policy, sources):
    """Use captured source vectors; endpoint bytes bypass fusion exactly."""
    from speaker_id.packaging.selected import family_vectors
    from speaker_id.inference.selected_policy import validate_policy
    validate_policy(policy)
    values = family_vectors(sources['vectors'], sources['valid'], sources['family'], policy['advanced_weight'])
    require(values.shape == (len(sources['valid']), policy['embedding_dim']), 'Selected source-cache shape differs')
    return values


def check_parity(observed, reference, valid):
    import numpy as np
    require(observed.shape == reference.shape and observed.ndim == 2 and observed.dtype == reference.dtype == np.float32
            and valid.shape == (len(observed),) and valid.dtype == np.bool_
            and np.isfinite(observed).all() and np.isfinite(reference).all(), 'Invalid cross-device vectors')
    require(not np.any(observed[~valid]) and not np.any(reference[~valid]), 'Original zero-signal embeddings must stay exactly zero')
    evidence = []
    for actual, expected, nonzero in zip(observed, reference, valid):
        difference = float(np.abs(actual - expected).max())
        cosine = float(actual.astype(np.float64) @ expected.astype(np.float64)) if nonzero else None
        require(difference <= TOLERANCE['maximum_absolute_embedding_difference']
            and (cosine is None or cosine >= TOLERANCE['minimum_embedding_cosine']), 'CUDA release extraction differs from its attested CUDA cache')
        evidence.append({'maximum_absolute_difference': difference, 'cosine': cosine})
    return evidence


def read_predictions(path, labels, names):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ['audio_file', 'speaker_id'], 'Competition CSV columns differ')
        rows = list(reader)
    require(len(rows) == len(names) and len({row['audio_file'] for row in rows}) == len(rows)
        and {row['audio_file'] for row in rows} == set(names)
        and all(row['speaker_id'] in labels for row in rows), 'CSV must cover each input exactly once with a valid label')
    return {row['audio_file']: row['speaker_id'] for row in rows}


def execute(build_dir, report_path, *, project_root=ROOT):
    require(os.environ.get('VAST_INSTANCE_ID') == '50079023', 'Fresh CUDA QA requires the authorized instance 50079023')
    import numpy as np
    import torch
    require(torch.cuda.is_available() and '3090' in torch.cuda.get_device_name(0), 'Fresh CUDA QA requires the authorized RTX3090')
    project_root, build_dir, report_path = map(no_links, (project_root, build_dir, report_path))
    sys.path.insert(0, str(project_root / 'src'))
    from speaker_id.inference.selected_runtime import verify_payload, _labels
    from speaker_id.packaging.selected_sources import load_sources
    require(not report_path.exists(), 'QA report must not overwrite existing evidence')
    capture = report_path.parent / (report_path.stem + '_files')
    require(not capture.exists(), 'QA capture directory already exists')
    capture.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'scope': 'Independent selected-release representative offline CUDA QA',
        'optimizer_steps': 0, 'backward_calls': 0, 'leaderboard_validation': 'pending'}
    try:
        package = no_links(build_dir / 'package')
        payload = verify_payload(package)
        build = read_json(build_dir / 'build_report.json')
        captured = read_json(build_dir / 'tracking/artifacts/resolved_config.json')
        config = captured['package']
        require(build['status'] == 'built_pending_offline_qa' and build['policy'] == payload['policy']
            and build['provenance'] == payload['provenance'] and build['release_id'] == payload['manifest']['release_id'], 'Build report differs from actual release')
        selection = config['selection']
        require(all(payload['provenance']['selection'][key] == selection[key] for key in ('recipe_id', 'family', 'report_sha256'))
            and payload['provenance']['selection']['parent_run_id'] == selection['source']['parent_run_id'], 'QA source differs from selected procedure')
        sources = load_sources(project_root, config, verify_audio=False)
        require(read_json(build_dir / 'source_provenance.json') == sources['proof'], 'Build source-cache proof differs from independently revalidated sources')
        for component in payload['models']:
            if component != 'adapted':
                require(payload['provenance']['model_sources'][component] == sources['assets'][component]['source_record'],
                        'Bundled public source identity differs from the attested selected cache')
        sources['family'] = config['selection']['family']
        vectors = selected_vectors(payload['policy'], sources)
        rows, valid = sources['contract']['manifest'], sources['valid']
        require(len(rows) == 4529 and vectors.shape[0] == 4529 and int((~valid).sum()) == 89, 'Original source rows or zeros changed')
        examples = select_examples(rows, project_root)
        require(len(examples) == 7, 'Expected the original seven unique representative examples')
        indices = {row['audio_file']: i for i, row in enumerate(rows)}
        names = sorted(item['row']['audio_file'] for item in examples)
        sample = np.asarray([indices[name] for name in names])
        cached, cache_valid = vectors[sample], valid[sample]
        np.savez(capture / 'source_cache_vectors.npz', audio_file=np.asarray(names), embedding=cached, valid=cache_valid)
        archive = no_links(build_dir / build['archive']['archive_name'])
        require(archive.parent == build_dir and archive.stat().st_size == build['archive']['archive_bytes']
            and sha256(archive) == build['archive']['archive_sha256'], 'Built ZIP digest differs')
        with tempfile.TemporaryDirectory(prefix='p2q_') as temporary:
            run_dir = no_links(temporary)
            require(not run_dir.is_relative_to(project_root), 'QA temporary working directory must be outside the checkout')
            isolated, inputs, cwd = run_dir / 'p', run_dir / 'i', run_dir / 'w'
            extract_verified_archive(archive, isolated, payload['manifest'])
            inputs.mkdir(); cwd.mkdir()
            for item in examples:
                row = item['row']
                source = no_links(project_root / 'data/raw' / row['audio_file'])
                require(source.parent == (project_root / 'data/raw').resolve() and sha256(source) == row['input_sha256'], 'Representative audio SHA differs')
                shutil.copyfile(source, inputs / row['audio_file'])
                require(sha256(inputs / row['audio_file']) == row['input_sha256'], 'Copied representative audio changed')
            (inputs / CORRUPT).write_bytes(b'Intentionally invalid audio for selected-release decoding fallback.\n')
            env = isolated_environment(os.environ, run_dir)
            for key in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
                Path(env[key]).mkdir()
                require(not list(Path(env[key]).iterdir()), 'Model cache is not initially empty')
            output, evidence, extracted = capture / 'predictions.csv', capture / 'worker_report.json', capture / 'vectors.npz'
            completed = subprocess.run([sys.executable, '-I', '-B', '-c', WORKER, str(isolated), str(inputs), str(output), str(evidence), str(extracted)],
                cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8', timeout=1200)
            (capture / 'stdout.log').write_text(completed.stdout, encoding='utf-8')
            (capture / 'stderr.log').write_text(completed.stderr, encoding='utf-8')
            require(completed.returncode == 0, 'Offline subprocess failed; inspect the retained stderr log')
            worker = read_json(evidence)
            require(worker['status'] == 'passed' and worker['real_audio_files'] == 7, 'Offline worker did not complete every case')
            labels = _labels(package / 'assets/labels.json')
            predictions = read_predictions(output, labels, names + [CORRUPT])
            require(predictions[CORRUPT] == 'unknown', 'Corrupt audio must receive unknown')
            with np.load(extracted, allow_pickle=False) as observed:
                require(observed['audio_file'].tolist() == names and np.array_equal(observed['valid'], cache_valid), 'Audio order or zero semantics differ from source cache')
                parity = check_parity(observed['embedding'], cached, cache_valid)
                with np.load(package / 'assets/gallery.npz', allow_pickle=False) as saved:
                    gallery = {key: saved[key] for key in saved.files}
                source_prefix = 'src/' if (package / 'src/speaker_id').is_dir() else ''
                spec = importlib.util.spec_from_file_location('_p002_portable_scoring', package / (source_prefix + 'speaker_id/inference/scoring.py'))
                scoring = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scoring)
                probability = scoring.score_embeddings(cached, cache_valid, gallery, read_json(package / 'assets/calibration.json'))
                require(np.array_equal(probability.argmax(1), observed['probabilities'].argmax(1)), 'CUDA audio and CUDA cache decisions differ')
                require(all(predictions[name] == labels[int(i)] for name, i in zip(names, observed['probabilities'].argmax(1))), 'CLI probability argmax differs')
                drift = float(np.abs(probability - observed['probabilities']).max())
            report.update(worker)
        # Temporary payload/audio/cache copies were confined and removed; evidence remains.
        report.update({'status': 'passed', 'release_id': build['release_id'], 'release_dir': str(package),
            'build_parent_run_id': build['parent_run_id'], 'selection': payload['provenance']['selection'],
            'policy': payload['policy'], 'archive': build['archive'], 'source_rows': len(rows), 'source_zero_rows': 89,
            'build_report_sha256': sha256(build_dir / 'build_report.json'),
            'resolved_build_config_sha256': sha256(build_dir / 'tracking/artifacts/resolved_config.json'),
            'source_provenance_sha256': sha256(build_dir / 'source_provenance.json'),
            'source_model_bindings': payload['provenance']['model_sources'],
            'isolated_outside_checkout': True, 'initial_model_caches_empty': True, 'project_credentials_absent_from_worker_env': True,
            'cuda_vs_source_cache_decisions_exact': True, 'cuda_vs_source_cache_max_probability_difference': drift,
            'cross_device_tolerance': TOLERANCE, 'cli_probability_argmax_parity': True, 'input_coverage_exact': True,
            'corrupt_audio_unknown': True, 'synthetic_corrupt_files': 1,
            'cases': [{'audio_file': item['row']['audio_file'], 'input_sha256': item['row']['input_sha256'], 'cases': item['cases']} for item in examples],
            'embedding_parity': [{'audio_file': name, **item} for name, item in zip(names, parity)],
            'limitations': ['Host Python dependencies; organizer evaluator image is not supplied.',
                'Network blocking covers Python socket APIs, not an OS network namespace.',
                'Fresh CUDA audio forward versus previously attested CUDA caches on the authorized RTX3090.',
                'Representative matched decisions do not guarantee parity for every future input.',
                'No calibration, encoder fitting or quality estimate is performed; this is not a leaderboard result.']})
    except BaseException as error:
        report.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        report['verified_at'] = datetime.now(timezone.utc).isoformat()
        report['capture_directory'] = str(capture)
        report['evidence_files'] = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                                  for path in sorted(capture.iterdir()) if path.is_file()}
        with report_path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--execute', action='store_true', help='Explicitly permit representative forward-only offline QA')
    args = parser.parse_args(argv)
    if not args.execute:
        result = {'status': 'arguments_validated_only', 'forward_calls': 0, 'source_caches_read': False,
                  'build_dir': str(args.build_dir), 'report': str(args.report), 'execute_required': True}
    else:
        result = execute(args.build_dir, args.report, project_root=args.project_root)
    print(json.dumps({key: result[key] for key in ('status', 'build_dir', 'report', 'execute_required') if key in result}, indent=2))
    return result


if __name__ == '__main__':
    main()
