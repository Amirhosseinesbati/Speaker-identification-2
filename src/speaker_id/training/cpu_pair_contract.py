"""C001's explicit CPU protocol; the original S010 CUDA contract is unchanged."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY, IDENTITY_POLICY
from speaker_id.infrastructure.cpu_feasibility import FIXED as PILOT_CONFIG, server_cpu_gate
from speaker_id.models.campp import file_sha256
from speaker_id.training.gain_suite import (
    ALPHAS, TIE_ORDER, DECISION, SELECTION, SOURCE_CONFIG, SOURCE_CONFIG_SHA,
    load_gain_inputs, require,
)
from speaker_id.training.fusion_suite import project_path

CP001 = {'directory': 'artifacts/infrastructure/cpu_pilots/CP001_20260908T131952Z_ba9bdeee',
    'report_sha256': 'edbcd09d3c03c7edd0d8cdf0a2ade9d2799f6860d21e03e7f8c1a7fd5116515d',
    'parent_run_id': '5edec7fb3ebf4de1be47d61fba456db5',
    'git_commit': '212a9aa23575c6e59993e7c8cef5b01be2f5904e'}
EXECUTION = {'device': 'cpu', 'threads': 4, 'interop_threads': 1, 'worker_count': 1,
    'maximum_extraction_seconds': 28800, 'progress_every_pairs': 50}
FIXED = {'schema_version': 1, 'experiment_code': 'C001',
    'run_name': 'C001-campp-matched-cpu-fixed-rms-boost',
    'readiness_config': 'configs/train/campp_coverage.json', 'output_root': 'artifacts/training/cpu_gain',
    'source_release_config': SOURCE_CONFIG, 'source_release_config_sha256': SOURCE_CONFIG_SHA,
    'gain_policy': GAIN_POLICY, 'identity_policy': IDENTITY_POLICY,
    'alphas': list(ALPHAS), 'alpha_tie_order': list(TIE_ORDER),
    'unknown_weights': [0.0, .25, .5, .75, 1.0], 'margin_weights': [0.0, .5],
    'threshold_candidates': 201, 'probability_temperature': .05,
    'selection_policy': SELECTION, 'decision_rule': DECISION,
    'primary_contrast': 'C001d minus C001b; historical GPU-cache control is never a selectable frontend',
    'recipes': ['C001a_historical_control', 'C001b_cpu_identity', 'C001c_cpu_gain', 'C001d_inner_frontend_choice'],
    'execution': EXECUTION, 'cp001': CP001,
    'mlflow_payload': 'configs_source_hashes_reports_and_scalar_metrics_no_embeddings'}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def validate_cpu_gain_config(suite):
    require(canonical(suite) == canonical(FIXED), 'C001 requires its exact matched CPU protocol, pilot and budget')


def load_cpu_gain_inputs(root, suite):
    """Metadata only; no model, scoring, new MLflow run or inference."""
    validate_cpu_gain_config(suite)
    original = json.loads(project_path(root, 'configs/train/campp_gain.json', 'configs/train').read_text())
    return load_gain_inputs(root, original)


def verify_pilot_report(report, state, suite):
    expected = suite['cp001']
    require(report['status'] == 'complete' and report['parent_run_id'] == expected['parent_run_id']
        and report['git_commit'] == expected['git_commit']
        and report['device'] == 'cpu' and report['threads'] == 4 and report['interop_threads'] == 1
        and report['capacity']['instance_id'] == 50079023
        and report['encoder_updates'] == 0 and report['embedding_artifacts_uploaded'] is False
        and report['raw_audio_full_sha_verified'] is True and report['raw_files_verified'] == 4529
        and report['recognition_scoring_or_calibration'] is False and report['gpu_health_claim'] is False,
        'CP001 completed CPU evidence is required, without an encoder-training or GPU-health claim')
    expected_pairs = [(i, name) for i in (0,35,583,2060,2264,3632,4528) for name in ('identity','gain')]
    require([(r['index'],r['frontend']) for r in report['records']] == expected_pairs,
        'The completed pilot must cover every fixed metadata representative')
    for prefix in ('encoder_state','weight_file'):
        before, after = (report[prefix+'_sha256_'+when] for when in ('before','after'))
        require(before == after and set(before) == {'public','advanced'}
            and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in before.values()),
            'CP001 models must remain frozen with complete actual hash evidence')
    require(state['run_id'] == expected['parent_run_id'] and state['remote_status'] == 'FINISHED'
        and state['pending_status'] is None and state['last_sync_error'] is None
        and state['binding']['experiment_id'] == '1'
        and state['tags']['mlflow.source.git.commit'] == expected['git_commit']
        and not any(name.lower().endswith('.npz') for name in state['uploaded_artifacts']),
        'CP001 tracking evidence must be complete and exclude embedding uploads')


def prepare_cpu_execution(root, suite):
    """Same-host CPU gate and fresh historical pilot readback before actual models."""
    validate_cpu_gain_config(suite)
    root = Path(root)
    capacity = server_cpu_gate(root, PILOT_CONFIG)
    directory = project_path(root, suite['cp001']['directory'], 'artifacts/infrastructure/cpu_pilots')
    report_path = directory/'pilot_report.json'
    require(file_sha256(report_path) == suite['cp001']['report_sha256'], 'Completed CP001 report changed')
    report = json.loads(report_path.read_text())
    state = json.loads((directory/'tracking/run_state.json').read_text())
    verify_pilot_report(report,state,suite)
    verification = json.loads((directory/'verification.json').read_text())
    require(verification['status'] == verification['artifact_roundtrip']['status'] == verification['metadata']['status'] == 'passed'
        and verification['metadata']['remote_run_status'] == 'FINISHED', 'CP001 final verification must pass')
    from speaker_id.tracking import ExperimentBinding
    from speaker_id.tracking.mlflow import make_client, _check_remote_binding
    binding = ExperimentBinding(**state['binding']); binding.validate()
    current_binding = json.loads((root/'artifacts/infrastructure/mlflow_state.json').read_text())['binding']
    require(current_binding == state['binding'], 'CPU comparison must retain the owned experiment binding')
    client = make_client(binding.tracking_endpoint); _check_remote_binding(client,binding)
    actual = client.get_run(suite['cp001']['parent_run_id'])
    require(actual.info.status == 'FINISHED' and actual.info.experiment_id == '1'
        and all(actual.data.tags.get(k) == v for k,v in state['tags'].items()), 'Fresh CP001 metadata does not match')
    with tempfile.TemporaryDirectory(prefix='cp001_readback_',dir=root/'artifacts/infrastructure') as temporary:
        fetched = Path(client.download_artifacts(suite['cp001']['parent_run_id'],'pilot_report.json',temporary))
        require(file_sha256(fetched) == suite['cp001']['report_sha256'], 'Fresh CP001 MLflow report SHA differs')
    require(min(capacity['quota_cores'],capacity['affinity_cpus']) >= 4, 'Single-worker CPU capacity no longer available')
    backend = capture_cpu_backend()
    return {'capacity':capacity,'backend':backend,'cp001_evidence':{
        **suite['cp001'],'fresh_report_sha256_verified':True,'fresh_remote_status':'FINISHED',
        'measured_profile_seconds':report['elapsed_profile_seconds'],'measured_peak_rss_kib':report['peak_rss_kib'],
        'weighted_extraction_estimate_seconds':sum(r['sequential_estimate_seconds'] for r in report['estimates']['frontends'].values()),
        'conservative_extrapolation_seconds':report['estimates']['conservative_two_frontend_sequential_seconds'],
        'execution_choice':'One measured four-thread worker; no parallel scaling assumption; soft eight-hour extraction budget',
        'pilot_vectors_reused':False,'estimate_is_not_a_runtime_guarantee':True}}


def static_torch_build(torch_root, extension_path, wheel_metadata):
    """Hash installed CPU implementation files without a runtime CUDA probe."""
    torch_root,extension_path=Path(torch_root).resolve(),Path(extension_path).resolve()
    require(extension_path.is_relative_to(torch_root), 'Torch extension must belong to the imported package')
    paths=[torch_root/'version.py',extension_path,torch_root/'lib/libtorch_cpu.so']
    require(len(set(paths))==3 and all(p.is_file() and not p.is_symlink() for p in paths),
        'Complete installed CPU Torch build files are required')
    require(isinstance(wheel_metadata,str) and wheel_metadata.strip(), 'Torch wheel metadata is missing')
    return {'method':'installed_cpu_files_and_wheel_metadata_sha256_v1',
        'files':{p.relative_to(torch_root).as_posix():file_sha256(p) for p in paths},
        'wheel_metadata_sha256':hashlib.sha256(wheel_metadata.encode()).hexdigest()}


def capture_cpu_backend():
    import torch
    import soundfile
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        require(os.environ.get(key) == '4', 'Explicit four-thread numerical environment required: '+key)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','soundfile','torch','torchaudio')}
    build=static_torch_build(Path(torch.__file__).parent,torch._C.__file__,
        importlib.metadata.distribution('torch').read_text('WHEEL'))
    result={'schema_version':1,'device':'cpu','tensor_dtype':'float32','worker_count':1,
        'torch_intraop_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'python_version':platform.python_version(),'versions':versions,'libsndfile_version':soundfile.__libsndfile_version__,
        'torch_build_manifest':build,'torch_build_sha256':hashlib.sha256(canonical(build)).hexdigest(),
        'numpy_build_sha256':hashlib.sha256(numpy_config.getvalue().encode()).hexdigest(),
        'cpu_capability':torch.backends.cpu.get_cpu_capability(), 'mkldnn_enabled':bool(torch.backends.mkldnn.enabled),
        'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'thread_environment':{key:os.environ[key] for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'encoder_updates':0,'cuda_queried':False}
    validate_backend(result)
    return result


def validate_backend(backend):
    fields={'schema_version','device','tensor_dtype','worker_count','torch_intraop_threads','torch_interop_threads',
        'python_version','versions','libsndfile_version','torch_build_manifest','torch_build_sha256','numpy_build_sha256','cpu_capability',
        'mkldnn_enabled','deterministic_algorithms_enabled','thread_environment','encoder_updates','cuda_queried'}
    require(set(backend)==fields and backend['schema_version']==1 and backend['device']=='cpu'
        and backend['tensor_dtype']=='float32' and backend['worker_count']==1
        and backend['torch_intraop_threads']==4 and backend['torch_interop_threads']==1
        and backend['encoder_updates']==0 and backend['cuda_queried'] is False
        and set(backend['versions'])=={'numpy','scipy','soundfile','torch','torchaudio'}
        and backend['thread_environment']=={key:'4' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'C001 backend must describe the measured single CPU worker')
    require(all(type(backend[k]) is int for k in ('schema_version','worker_count','torch_intraop_threads','torch_interop_threads','encoder_updates'))
        and all(type(backend[k]) is bool for k in ('mkldnn_enabled','deterministic_algorithms_enabled')),
        'Backend flags and counts require exact types')
    require(all(isinstance(backend[k],str) and backend[k] for k in ('python_version','libsndfile_version','cpu_capability'))
        and all(isinstance(v,str) and v for v in backend['versions'].values()), 'Incomplete observed CPU versions')
    for key in ('torch_build_sha256','numpy_build_sha256'):
        require(isinstance(backend[key],str) and len(backend[key])==64 and all(c in '0123456789abcdef' for c in backend[key]),
            'Invalid CPU build digest')
    build=backend['torch_build_manifest']
    require(type(build) is dict and set(build)=={'method','files','wheel_metadata_sha256'}
        and build['method']=='installed_cpu_files_and_wheel_metadata_sha256_v1'
        and type(build['files']) is dict and len(build['files'])==3
        and {'version.py','lib/libtorch_cpu.so'}.issubset(build['files'])
        and all(isinstance(k,str) and not Path(k).is_absolute() and '..' not in Path(k).parts
                and '\\' not in k and ':' not in k for k in build['files'])
        and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
                for v in [*build['files'].values(),build['wheel_metadata_sha256']])
        and hashlib.sha256(canonical(build)).hexdigest()==backend['torch_build_sha256'],
        'Installed CPU build manifest must match its signed digest')


def build_cpu_identity(root,suite,contract,sources,frontend,backend):
    validate_cpu_gain_config(suite); validate_backend(backend)
    require(frontend in ('identity','gain'), 'Only two matched CPU frontends are allowed')
    body={'schema_version':1,'frontend_policy':IDENTITY_POLICY if frontend=='identity' else GAIN_POLICY,
        'inference':contract['config']['inference'],'data_input_hashes':contract['input_hashes'],
        'labels':contract['labels'],'embedding_dims':{'public':512,'advanced':192},
        'model_sources':{name:asset['source_record'] for name,asset in sources['assets'].items()},
        'model_configs':{name:asset['config'] for name,asset in sources['assets'].items()},
        'code_hashes':{**contract['code_hashes'],'scripts/score_gain_cpu.py':file_sha256(Path(root)/'scripts/score_gain_cpu.py')},
        'backend':backend,'experiment_code':'C001','base_feature_implementation_unchanged':True,'encoder_updates':0}
    return {**body,'signature':hashlib.sha256(canonical(body)).hexdigest()}
