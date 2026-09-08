"""CP001: bounded same-server CPU profiling, without recognition or calibration."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
import uuid

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY, IDENTITY_POLICY
from speaker_id.data.splits import truth
from speaker_id.infrastructure.data import confined_path
from speaker_id.models.campp import file_sha256
from speaker_id.tracking.snapshot import write_json

FIXED = {'schema_version': 1, 'experiment_code': 'CP001', 'run_name': 'CP001-frozen-dual-encoder-cpu-feasibility',
    'device': 'cpu', 'instance_id': 50079023, 'hostname': '4593b1f57a8e',
    'workspace': '/workspace/Speaker-identification-2', 'threads': 4, 'max_elapsed_seconds': 600,
    'source_suite': 'configs/train/campp_gain.json', 'inference': {'seconds': 180.0, 'maximum_windows': 1},
    'gain_policy': GAIN_POLICY, 'minimum_memory_headroom_gib': 8, 'minimum_disk_free_gib': 10,
    'selection': 'fixed first/middle/last, first zero, shortest/longest nonzero, lowest nonzero RMS; no labels',
    'output_root': 'artifacts/infrastructure/cpu_pilots', 'encoder_updates': 0,
    'mlflow_payload': 'configs_source_hashes_reports_and_scalar_metrics_no_embeddings'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pilot_config(config):
    require(json.dumps(config, sort_keys=True, allow_nan=False) == json.dumps(FIXED, sort_keys=True),
            'CP001 requires the exact fixed CPU pilot contract')


def pilot_indices(manifest):
    """No label, role, prediction or error information participates in selection."""
    nonzero = [i for i, row in enumerate(manifest) if truth(row['has_nonzero_signal'])]
    zeros = [i for i, row in enumerate(manifest) if not truth(row['has_nonzero_signal'])]
    require(nonzero and zeros, 'The pilot requires both signal and zero representatives')
    require(all(math.isfinite(float(manifest[i]['duration_seconds'])) and float(manifest[i]['duration_seconds']) > 0
                and math.isfinite(float(manifest[i]['mono_rms_dbfs'])) for i in nonzero), 'Invalid pilot metadata')
    return sorted({0, len(manifest)//2, len(manifest)-1, zeros[0],
        min(nonzero, key=lambda i: float(manifest[i]['duration_seconds'])),
        max(nonzero, key=lambda i: float(manifest[i]['duration_seconds'])),
        min(nonzero, key=lambda i: float(manifest[i]['mono_rms_dbfs']))})


def validate_capacity(capacity, config):
    require(all(math.isfinite(float(capacity[k])) for k in ('quota_cores', 'affinity_cpus', 'memory_headroom_bytes', 'disk_free_bytes')),
            'Missing finite CPU resource limits')
    require(min(capacity['quota_cores'], capacity['affinity_cpus']) >= config['threads'], 'CPU quota is below four threads')
    require(capacity['memory_headroom_bytes'] >= config['minimum_memory_headroom_gib']*2**30,
            'Insufficient container memory headroom')
    require(capacity['disk_free_bytes'] >= config['minimum_disk_free_gib']*2**30, 'Insufficient workspace disk space')


def server_cpu_gate(root, config):
    validate_pilot_config(config)
    root = Path(root)
    require(platform.system() == 'Linux' and root == root.resolve() and str(root) == config['workspace']
        and socket.gethostname() == config['hostname'] and os.environ.get('VAST_INSTANCE_ID') == str(config['instance_id']),
        'CPU pilot is authorized only on the existing verified Vast server')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True, timeout=15).strip()
    commit = git('rev-parse', 'HEAD')
    require(git('remote','get-url','origin') == 'https://github.com/Amirhosseinesbati/Speaker-identification-2.git'
        and git('branch','--show-current') == 'develop' and not git('status','--porcelain=v1','--untracked-files=all'),
        'CPU pilot requires a clean committed Git workspace')
    marker_path = confined_path(root, 'artifacts/infrastructure/instance.json')
    require(not marker_path.is_symlink(), 'Instance marker cannot be a symlink')
    marker = json.loads(marker_path.read_text())
    require(marker.get('status') == 'verified' and marker.get('verified_via') == 'vast_api_and_ssh'
        and marker.get('instance_id') == config['instance_id'] and marker.get('hostname') == config['hostname']
        and marker.get('remote_workspace') == str(root) and marker.get('git_commit') == commit,
        'Operational instance identity must match the current source')
    cg = Path('/sys/fs/cgroup')
    quota, period = (cg/'cpu.max').read_text().split()
    affinity = len(os.sched_getaffinity(0))
    maximum = (cg/'memory.max').read_text().strip()
    require(maximum.isdigit(), 'Explicit container RAM limit is required')
    memory_limit, memory_current = int(maximum), int((cg/'memory.current').read_text())
    available = next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    result = {'git_commit': commit, 'instance_id': config['instance_id'], 'hostname': config['hostname'],
        'quota_cores': affinity if quota == 'max' else int(quota)/int(period), 'cpu_max': [quota, period],
        'affinity_cpus': affinity, 'memory_limit_bytes': memory_limit, 'memory_current_bytes': memory_current,
        'memory_headroom_bytes': min(memory_limit-memory_current, available), 'disk_free_bytes': shutil.disk_usage(root).free,
        'cuda_queried': False, 'gpu_readiness_overridden': False, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    validate_capacity(result, config)
    return result


def check_deadline(started, maximum, *, now=None):
    if (time.monotonic() if now is None else now)-started >= maximum:
        raise TimeoutError('CP001 profiling budget exhausted; no full experiment may start automatically')


def effective_seconds(row):
    return min(180.0, max(1.0, float(row['duration_seconds']))) if truth(row['has_nonzero_signal']) else 0.0


def estimate_runtime(manifest, records, capacity):
    """A metadata-extrema sample is a profiling estimate, not measured parallel throughput."""
    total = sum(effective_seconds(row) for row in manifest)
    zeros = sum(not truth(row['has_nonzero_signal']) for row in manifest)
    estimates = {}
    for frontend in ('identity', 'gain'):
        rows = [row for row in records if row['frontend'] == frontend]
        nonzero = [row for row in rows if row['effective_audio_seconds'] > 0]
        require(nonzero and all(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] > 0 for row in rows),
                'Positive finite timings are required for an estimate')
        rate = sum(row['elapsed_seconds'] for row in nonzero)/sum(row['effective_audio_seconds'] for row in nonzero)
        slowest = max(row['elapsed_seconds']/row['effective_audio_seconds'] for row in nonzero)
        zero_cost = max((row['elapsed_seconds'] for row in rows if not row['valid']), default=0)
        estimates[frontend] = {'observed_seconds_per_effective_audio_second': rate,
            'sequential_estimate_seconds': total*rate+zeros*zero_cost,
            'conservative_sequential_seconds': 2*(total*max(rate, slowest)+zeros*zero_cost)}
    sequential = sum(v['conservative_sequential_seconds'] for v in estimates.values())
    return {'effective_nonzero_audio_seconds_per_frontend': total, 'zero_files': zeros, 'frontends': estimates,
        'conservative_two_frontend_sequential_seconds': sequential,
        'hypothetical_three_worker_seconds': sequential*1.5/3 if min(capacity['quota_cores'],capacity['affinity_cpus']) >= 12 else None,
        'three_worker_threads_total': 12, 'parallel_scaling_measured': False,
        'parallel_memory_capacity_validated': False, 'automatic_full_run_authorized': False,
        'limitations': 'Nonrandom extrema sample, decode overhead and shared CPU contention affect estimates; the three-worker estimate assumes 1.5x contention and needs its own memory/concurrency validation.'}


def best_effort_hashes(items, hasher):
    """An after-failure audit cannot replace the original failure with a hashing error."""
    hashes, errors = {}, {}
    for name,value in items.items():
        try:
            hashes[name] = hasher(value)
        except Exception as error:
            errors[name] = type(error).__name__
    return hashes, errors


def execute_pilot(root, config_path, pilot, contract, source_config):
    """One tracked parent, two frozen CPU encoders, at most seven metadata-selected files."""
    from speaker_id.packaging.selected_sources import load_sources, project_file
    from speaker_id.training.candidate_comparison import encoder_state_sha256
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    task_started = time.monotonic()
    capacity = server_cpu_gate(root, pilot)
    binding = ExperimentBinding(**json.loads((root/'artifacts/infrastructure/mlflow_state.json').read_text())['binding'])
    require(binding.experiment_id == '1', 'CP001 must use the owned experiment 1')
    output = confined_path(root, Path(pilot['output_root'])/('CP001_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]))
    output.mkdir(parents=True, exist_ok=False)
    inputs = {'pilot_config': config_path, 'suite_config': root/pilot['source_suite'],
        'source_release_config': root/'configs/package/campp_selected.json', 'launcher': root/'scripts/checks/probe_gain_cpu.py'}
    inputs.update({key: root/contract['config'][key] for key in ('manifest','folds','roles','label_map','model_config')})
    resolved = {'pilot': pilot, 'source_release_config': source_config, 'capacity': capacity,
        'data_contract': {key: contract[key] for key in ('config','model','input_hashes','code_hashes','signature')},
        'metadata_indices': pilot_indices(contract['manifest']), 'no_recognition_metrics_or_calibration': True}
    write_json(output/'resolved_config.json', resolved)
    tracker = DurableMLflowRun.prepare(project_root=root, spool_dir=output/'tracking', binding=binding,
        run_name=pilot['run_name'], config=resolved, input_paths=inputs, run_kind='cpu_feasibility_pilot', training_started=False)
    records, encoders, before, weights_before, paths = [], {}, {}, {}, {}
    started = None
    progress_tracking_seconds = 0.0
    try:
        for name in ('pilot_config','suite_config','source_release_config','launcher'):
            tracker.add_artifact(inputs[name], 'input_configs/'+name+inputs[name].suffix)
        tracker.flush(strict=True); tracker.verify_artifacts(); tracker.verify_remote_metadata()
        sources = load_sources(root, source_config, verify_audio=True)
        require(sources['contract']['signature'] == contract['signature'] and set(sources['assets']) == {'public','advanced'},
                'CPU pilot requires the exact source-bound frozen public and advanced encoders')
        paths = {name: project_file(root, asset['config']['weights_path']) for name, asset in sources['assets'].items()}
        weights_before = {name: file_sha256(path) for name,path in paths.items()}
        require(all(weights_before[name] == sources['assets'][name]['source_record']['weights_sha256'] for name in paths),
                'Frozen checkpoint differs from the verified source')
        provenance = {'historical_sources': sources['proof'], 'raw_audio_full_sha_verified': True,
            'raw_files_verified': len(contract['manifest']), 'model_sources': {n:a['source_record'] for n,a in sources['assets'].items()},
            'weight_file_sha256_before': weights_before, 'model_configs':{n:a['config'] for n,a in sources['assets'].items()}, 'device': 'cpu', 'encoder_updates': 0}
        write_json(output/'source_provenance.json', provenance); tracker.add_artifact(output/'source_provenance.json')
        tracker.flush(strict=True); tracker.verify_artifacts(); tracker.verify_remote_metadata()
        import resource
        import torch
        from speaker_id.models.campp import load_campp
        from speaker_id.candidates.campp_advanced import load_advanced
        from speaker_id.candidates.gain_frontend import extract_gain_pair
        torch.set_num_threads(pilot['threads']); torch.set_num_interop_threads(1)
        encoders = {'public': load_campp(sources['assets']['public']['config'], root, 'cpu'),
                    'advanced': load_advanced(sources['assets']['advanced']['config'], root, 'cpu')}
        for encoder in encoders.values():
            encoder.requires_grad_(False).eval()
            require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in encoder.parameters()), 'Models must remain FP32 CPU')
        before = {name: encoder_state_sha256(model) for name,model in encoders.items()}
        started = time.monotonic()
        for index in resolved['metadata_indices']:
            row = contract['manifest'][index]
            raw = project_file(root, contract['config']['data_dir']+'/'+row['audio_file'])
            for frontend, policy in (('identity', IDENTITY_POLICY), ('gain', pilot['gain_policy'])):
                check_deadline(started, pilot['max_elapsed_seconds']); tick = time.monotonic()
                vectors, info = extract_gain_pair(encoders['public'], encoders['advanced'], raw,
                    device='cpu', policy=policy, **pilot['inference'])
                elapsed = time.monotonic()-tick
                valid = bool(info['nonzero_signal'])
                require(valid == bool(sources['valid'][index]) and file_sha256(raw) == row['input_sha256'], 'Input/eligibility changed')
                comparisons = {}
                for name,dim in (('public',512),('advanced',192)):
                    vector = vectors[name]
                    require(vector.shape == (dim,) and vector.dtype == np.float32 and np.isfinite(vector).all()
                        and (np.isclose(np.linalg.norm(vector),1,atol=1e-5) if valid else not np.any(vector)), 'Invalid CPU embedding')
                    if frontend == 'identity':
                        cached = sources['vectors'][name][index].astype(np.float64); current = vector.astype(np.float64)
                        comparisons[name] = {'max_abs':float(np.max(np.abs(current-cached))),
                            'l2':float(np.linalg.norm(current-cached)), 'cosine':float(np.dot(current,cached)/
                            (np.linalg.norm(current)*np.linalg.norm(cached))) if valid else None}
                path = output/f'{index:04d}_{frontend}.npz'
                with path.open('xb') as handle:
                    np.savez_compressed(handle, **vectors, valid=valid, audio_file=row['audio_file'], audio_sha256=row['input_sha256'])
                records.append({'index':index,'audio_file':row['audio_file'],'audio_sha256':row['input_sha256'],
                    'frontend':frontend,'valid':valid,'effective_audio_seconds':effective_seconds(row),
                    'elapsed_seconds':elapsed,'frontend_diagnostics':info,'cpu_vs_gpu_identity_diagnostic':comparisons,
                    'vectors':{n:{'dimension':int(v.size),'dtype':str(v.dtype),'finite':True,'norm':float(np.linalg.norm(v))} for n,v in vectors.items()},
                    'cache_file':path.name,'cache_sha256':file_sha256(path),'bytes':path.stat().st_size})
                write_json(output/'progress.json', {'status':'profiling','records':records})
                tracker.log_metrics({f'{frontend}/pair_seconds':elapsed,'completed_pairs':len(records)}, step=len(records),sync=False)
                progress_path = output/'progress'/f'{len(records):02d}.json'
                write_json(progress_path, {'status':'profiling','completed_pairs':len(records),'records':records})
                tracker.add_artifact(progress_path, f'progress/{len(records):02d}.json')
                tracking_tick = time.monotonic()
                try:
                    tracker.flush(strict=True)
                finally:
                    progress_tracking_seconds += time.monotonic()-tracking_tick
                print(json.dumps({'stage':'cpu_pair','completed':len(records),'audio_file':row['audio_file'],'frontend':frontend,'seconds':elapsed}),flush=True)
                check_deadline(started, pilot['max_elapsed_seconds'])
        after = {name: encoder_state_sha256(model) for name,model in encoders.items()}
        weights_after = {name:file_sha256(path) for name,path in paths.items()}
        require(before == after and weights_before == weights_after, 'Frozen model or weight file changed')
        report = {'status':'complete','parent_run_id':tracker.run_id,'git_commit':capacity['git_commit'],
            'device':'cpu','threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'torch_version':torch.__version__,'elapsed_profile_seconds':time.monotonic()-started,
            'setup_seconds':started-task_started,'wall_seconds_before_final_tracking':time.monotonic()-task_started,
            'progress_tracking_seconds':progress_tracking_seconds,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'capacity':capacity,
            'raw_audio_full_sha_verified':True,'raw_files_verified':len(contract['manifest']),
            'encoder_state_sha256_before':before,'encoder_state_sha256_after':after,
            'weight_file_sha256_before':weights_before,'weight_file_sha256_after':weights_after,
            'records':records,'estimates':estimate_runtime(contract['manifest'],records,capacity),
            'encoder_updates':0,'recognition_scoring_or_calibration':False,'gpu_health_claim':False,
            'embedding_artifacts_uploaded':False,
            'deadline_scope':'600-second soft profiling budget checked between cases, including live progress tracking; setup and final tracking excluded; an in-flight native call may overrun'}
        write_json(output/'pilot_report.json', report); tracker.add_artifact(output/'pilot_report.json')
        tracker.write_report(report); tracker.flush(strict=True)
        roundtrip = tracker.verify_artifacts(); tracker.verify_remote_metadata()
        tracker.finish(strict=True); metadata = tracker.verify_remote_metadata()
        write_json(output/'verification.json', {'status':'passed','artifact_roundtrip':roundtrip,'metadata':metadata,
            'total_wall_seconds':time.monotonic()-task_started})
        return {'output_directory':str(output),'parent_run_id':tracker.run_id, 'report':report}
    except Exception as error:
        after_states, state_errors = best_effort_hashes(encoders,encoder_state_sha256) if before else ({},{})
        after_weights, weight_errors = best_effort_hashes(paths,file_sha256) if weights_before else ({},{})
        failure = {'status':'failed','parent_run_id':tracker.run_id,'error':tracker.redactor.text(str(error)),
            'error_type':type(error).__name__,'completed_pairs':len(records),'records':records,
            'elapsed_profile_seconds':0 if started is None else time.monotonic()-started,
            'setup_seconds':time.monotonic()-task_started if started is None else started-task_started,
            'total_wall_seconds':time.monotonic()-task_started,'encoder_updates':0,'no_automatic_retry':True,
            'progress_tracking_seconds':progress_tracking_seconds,
            'encoder_state_sha256_before':before,
            'encoder_state_sha256_after':after_states,'state_hash_errors':state_errors,
            'weight_file_sha256_before':weights_before,
            'weight_file_sha256_after':after_weights,'weight_hash_errors':weight_errors}
        write_json(output/'failure.json', failure); tracker.add_artifact(output/'failure.json'); tracker.write_report(failure)
        tracker.finish('FAILED',strict=True)
        raise
