"""Read-only CPU/fresh-CUDA evidence comparison; never loads an encoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def no_links(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in (path, *path.parents)), 'Linked evidence is forbidden')
    return path.resolve()


def read_report(path, device):
    path = no_links(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    require(report['status'] == 'passed' and report['device'] == device
        and report['real_audio_files'] == 7 and report['synthetic_corrupt_files'] == 1
        and report['source_rows'] == 4529 and report['source_zero_rows'] == 89
        and report['optimizer_steps'] == 0 and report['backward_calls'] == 0 and report['network_attempts'] == 0,
        'QA report does not attest the complete forward-only test')
    required = ('isolated_outside_checkout', 'initial_model_caches_empty', 'project_credentials_absent_from_worker_env',
        'cli_probability_argmax_parity', 'input_coverage_exact', 'corrupt_audio_unknown', 'payload_unchanged',
        'forbidden_imports_absent', 'existing_output_replaced', 'model_failure_preserved_output')
    require(all(report.get(key) is True for key in required), 'A required offline/isolation/output check failed')
    key = 'cpu_vs_cuda_cache_decisions_exact' if device == 'cpu' else 'cuda_vs_source_cache_decisions_exact'
    require(report.get(key) is True, 'Source-cache decisions must match')
    if device == 'cuda':
        require('3090' in report.get('cuda_device_name', '') and report.get('peak_cuda_memory_bytes', 0) > 0,
                'Fresh CUDA worker must report the actual RTX3090 and allocated model memory')
    tolerance = {'maximum_absolute_embedding_difference': .002, 'minimum_embedding_cosine': .9999,
        'speaker_decisions_must_match': True, 'bit_identical_embeddings_required': False}
    require(report['cross_device_tolerance'] == tolerance, 'QA relaxed the established numerical/decision guard')
    capture = no_links(path.parent / (path.stem + '_files'))
    require(capture.is_dir(), 'Missing report-adjacent capture directory')
    files = report['evidence_files']
    require({'source_cache_vectors.npz', 'vectors.npz', 'predictions.csv', 'worker_report.json', 'stdout.log', 'stderr.log'} <= set(files), 'Incomplete offline capture')
    for name, record in files.items():
        require(Path(name).name == name and name not in ('.', '..') and ':' not in name and '\\' not in name, 'Unsafe capture filename')
        item = no_links(capture / name)
        require(item.parent == capture and item.is_file() and item.stat().st_size == record['bytes']
            and sha(item) == record['sha256'], 'Captured QA artifact bytes changed')
    worker = json.loads((capture / 'worker_report.json').read_text(encoding='utf-8'))
    require(all(worker.get(key) == report.get(key) for key in ('status', 'device', 'network_attempts',
        'optimizer_steps', 'backward_calls', 'real_audio_files', 'policy_kind', 'embedding_dim')),
        'Parent QA report contradicts its actual isolated worker')
    result = {}
    for name in ('source_cache_vectors', 'vectors'):
        with np.load(capture / (name + '.npz'), allow_pickle=False) as saved:
            result[name] = {key: saved[key].copy() for key in saved.files}
    with (capture / 'predictions.csv').open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ['audio_file', 'speaker_id'], 'CSV columns changed')
        result['predictions'] = list(reader)
    result.update({'report': report, 'report_sha256': sha(path)})
    return result


def compare(cpu, cuda):
    left, right = cpu['report'], cuda['report']
    identity = ('release_id', 'build_parent_run_id', 'selection', 'policy', 'archive', 'source_model_bindings',
        'build_report_sha256', 'resolved_build_config_sha256', 'source_provenance_sha256', 'cases')
    require(all(left[key] == right[key] for key in identity), 'CPU and CUDA reports refer to different selected builds or raw examples')
    cases = left['cases']
    names = sorted(case['audio_file'] for case in cases)
    require(len(cases) == len(set(names)) == 7 and all(isinstance(case['input_sha256'], str)
            and len(case['input_sha256']) == 64 for case in cases), 'Incomplete raw-audio identity')
    reference = cpu['source_cache_vectors']
    require(set(reference) == set(cuda['source_cache_vectors']) == {'audio_file', 'embedding', 'valid'}
        and all(np.array_equal(reference[key], cuda['source_cache_vectors'][key]) for key in reference), 'CPU/CUDA source references differ')
    valid, cached = reference['valid'], reference['embedding']
    require(reference['audio_file'].tolist() == names and valid.dtype == np.bool_ and valid.shape == (7,)
        and int((~valid).sum()) == 1 and cached.dtype == np.float32
        and cached.shape == (7, left['policy']['embedding_dim']) and np.isfinite(cached).all()
        and not np.any(cached[~valid]) and np.allclose(np.linalg.norm(cached[valid], axis=1), 1, rtol=0, atol=1e-5),
        'Source reference row/dimension/zero semantics differ')
    for item in (cpu, cuda):
        actual = item['vectors']
        require(set(actual) == {'audio_file', 'embedding', 'valid', 'probabilities'} and actual['audio_file'].tolist() == names
            and np.array_equal(actual['valid'], valid) and actual['embedding'].shape == cached.shape
            and actual['embedding'].dtype == np.float32 and np.isfinite(actual['embedding']).all()
            and not np.any(actual['embedding'][~valid])
            and np.allclose(np.linalg.norm(actual['embedding'][valid], axis=1), 1, rtol=0, atol=1e-5),
            'Extracted vector shape/order/validity changed')
        probability = actual['probabilities']
        require(probability.shape == (7, 447) and np.isfinite(probability).all() and (probability >= 0).all()
            and np.allclose(probability.sum(1), 1, rtol=0, atol=1e-12) and (probability[~valid].argmax(1) == 0).all(),
            'Invalid 447-way probabilities or zero fallback')
        predictions = item['predictions']
        require(len(predictions) == len({row['audio_file'] for row in predictions}) == 8
            and {row['audio_file'] for row in predictions} == set(names) | {'__qa_corrupt__.wav'}
            and next(row['speaker_id'] for row in predictions if row['audio_file'] == '__qa_corrupt__.wav') == 'unknown', 'CLI input/corrupt coverage differs')
    require(sorted(cpu['predictions'], key=lambda row: row['audio_file']) == sorted(cuda['predictions'], key=lambda row: row['audio_file'])
        and np.array_equal(cpu['vectors']['probabilities'].argmax(1), cuda['vectors']['probabilities'].argmax(1)), 'CPU/CUDA actual speaker decisions differ')
    comparisons = []
    for label, actual, expected in [('cpu_to_cache', cpu['vectors']['embedding'], cached),
                                    ('cuda_to_cache', cuda['vectors']['embedding'], cached),
                                    ('cpu_to_cuda', cpu['vectors']['embedding'], cuda['vectors']['embedding'])]:
        differences = np.abs(actual - expected).max(axis=1)
        dots = (actual[valid].astype(np.float64) * expected[valid].astype(np.float64)).sum(axis=1)
        require(float(differences.max()) <= .002 and float(dots.min()) >= .9999, 'Fresh device/source-cache extraction exceeds the unchanged P001 tolerance')
        comparisons.append({'comparison': label, 'maximum_absolute_difference': float(differences.max()), 'minimum_nonzero_dot_product': float(dots.min())})
    return {'schema_version': 1, 'status': 'passed', 'release_id': left['release_id'], 'archive': left['archive'],
        'selection': left['selection'], 'policy': left['policy'], 'cpu_report_sha256': cpu['report_sha256'],
        'cuda_report_sha256': cuda['report_sha256'], 'same_raw_audio_and_cache_references': True,
        'real_audio_files': 7, 'synthetic_corrupt_files': 1, 'source_rows': 4529, 'source_zero_rows': 89,
        'fresh_cpu_cuda_and_source_cache_decisions_exact': True, 'comparisons': comparisons,
        'cpu_cuda_max_probability_difference': float(np.abs(cpu['vectors']['probabilities'] - cuda['vectors']['probabilities']).max()),
        'cross_device_tolerance': left['cross_device_tolerance'], 'optimizer_steps': 0, 'backward_calls': 0,
        'leaderboard_validation': 'pending',
        'limitations': ['Host CPU and authorized RTX3090 representative QA, not the unavailable organizer image.',
            'Socket blocking is not an OS network namespace.', 'Seven matched audio examples do not guarantee parity on every future input.',
            'This verifier only reads saved evidence; no model forward, fitting, calibration or network call.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-report', type=Path, required=True)
    parser.add_argument('--cuda-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    output = no_links(args.report)
    require(not output.exists(), 'Combined report must not overwrite existing evidence')
    result = compare(read_report(args.cpu_report, 'cpu'), read_report(args.cuda_report, 'cuda'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'report': str(output)}))
    return result


if __name__ == '__main__':
    main()
