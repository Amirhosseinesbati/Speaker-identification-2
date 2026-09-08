"""One frozen CPU worker producing fresh identity/gain caches; no recognition scoring."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from speaker_id.audio.gain import IDENTITY_POLICY
from speaker_id.data.splits import truth
from speaker_id.infrastructure.data import confined_path
from speaker_id.models.campp import file_sha256
from speaker_id.tracking.snapshot import write_json
from speaker_id.training.gain_suite import verify_gain_cache

NAMES = ('public', 'advanced')
FRONTENDS = ('identity', 'gain')
DIMS = {'public':512, 'advanced':192}


def require(value, message):
    if not value:
        raise ValueError(message)


def check_budget(started, maximum, *, now=None):
    if (time.monotonic() if now is None else now)-started >= maximum:
        raise TimeoutError('Paired CPU extraction soft budget exceeded; partial files preserved; no implicit resume')


def publish_npz(path, vectors, valid, row, signature):
    """Exclusive partial + fsync + atomic no-replace link; collision never overwrites."""
    temporary = path.with_suffix('.partial')
    require(not path.exists() and not temporary.exists(), 'Fresh CPU cache collision')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **vectors, valid=bool(valid), signature=signature,
            audio_file=row['audio_file'], audio_sha256=row['input_sha256'])
        handle.flush(); os.fsync(handle.fileno())
    os.link(temporary,path)
    temporary.unlink()
    if os.name == 'posix':
        fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {'audio_file':row['audio_file'],'audio_sha256':row['input_sha256'],
        'cache_file':path.name,'bytes':path.stat().st_size,'cache_sha256':file_sha256(path),'valid':bool(valid)}


def validate_pair(vectors, info, valid, policy):
    require(set(vectors) == set(DIMS) and type(info['nonzero_signal']) is bool
        and info['nonzero_signal'] == valid and info['gain']['policy'] == policy['name'], 'Frontend validity/policy changed')
    for name,dimension in DIMS.items():
        vector = vectors[name]
        require(isinstance(vector,np.ndarray) and vector.shape == (dimension,) and vector.dtype == np.float32
            and np.isfinite(vector).all() and (np.isclose(np.linalg.norm(vector),1,atol=1e-5) if valid
                else vector.tobytes() == np.zeros(dimension,dtype=np.float32).tobytes()), 'Malformed CPU vector')
    require(type(info['gain']['applied']) is bool and np.isfinite(info['gain']['gain'])
        and info['gain']['gain'] >= 1 and info['gain']['applied'] == (info['gain']['gain'] != 1), 'Invalid gain application flag')
    require((valid and policy['name'] != 'identity_v1') or info['gain']['gain'] == 1,
            'Identity and zero-signal policies cannot change waveform gain')


def require_noop_equal(identity, gain, gain_info):
    if not gain_info['gain']['applied']:
        require(all(identity[name].dtype == gain[name].dtype and identity[name].tobytes() == gain[name].tobytes()
                    for name in NAMES), 'No-op gain changed the exact CPU embedding bytes')


def diagnostic(cpu, gpu, valid):
    result = {}
    for name in NAMES:
        current, historical = cpu[name].astype(np.float64),gpu[name].astype(np.float64)
        require(current.shape == historical.shape and np.isfinite(historical).all(), 'Historical diagnostic shape/value changed')
        result[name] = {'max_abs':float(np.max(np.abs(current-historical))),
            'l2':float(np.linalg.norm(current-historical)), 'cosine':float(np.dot(current,historical)/
            (np.linalg.norm(current)*np.linalg.norm(historical))) if valid else None}
    return result


def summarize_diagnostics(rows):
    return {name:{'files':len(rows),'nonzero_files':sum(row[name]['cosine'] is not None for row in rows),
        'maximum_absolute_difference':max(row[name]['max_abs'] for row in rows),
        'maximum_l2_difference':max(row[name]['l2'] for row in rows),
        'minimum_nonzero_cosine':min((row[name]['cosine'] for row in rows if row[name]['cosine'] is not None),default=None),
        'diagnostic_only_no_acceptance_threshold':True} for name in NAMES}


def best_effort_hashes(items, hasher):
    hashes, errors = {}, {}
    for name,value in items.items():
        try:
            hashes[name] = hasher(value)
        except Exception as error:
            errors[name] = type(error).__name__
    return hashes, errors


def extract_cpu_pair_caches(root,suite,contract,sources,output,control,backend,*,progress_callback):
    """Callback payloads contain only identities, scalar diagnostics and file receipts."""
    from speaker_id.training.cpu_pair_contract import build_cpu_identity
    from speaker_id.training.candidate_comparison import encoder_state_sha256
    started = time.monotonic()
    root,output = Path(root).resolve(),confined_path(Path(root),output)
    settings = suite['execution']
    require(settings == {'device':'cpu','threads':4,'interop_threads':1,'worker_count':1,
        'maximum_extraction_seconds':28800,'progress_every_pairs':50}, 'Unexpected CPU worker execution policy')
    require(all(control.get(key) for key in ('exact_prediction_reproduction','exact_pooled_metrics',
        'exact_inner_alpha_curves','exact_probability_and_support_arrays')), 'Exact historical control must precede extraction')
    require(set(sources['assets']) == set(NAMES) and output.is_dir(), 'Missing dual-encoder assets/output')
    manifest = contract['manifest']
    require(len(manifest) > 0 and sources['valid'].shape == (len(manifest),)
        and all(sources['vectors'][name].shape == (len(manifest),DIMS[name]) for name in NAMES), 'Source row/dimension mismatch')
    names = [row['audio_file'] for row in manifest]
    require(len(set(names)) == len(names) and len({Path(name).stem for name in names}) == len(names)
        and all(Path(name).name == name and '/' not in name and '\\' not in name and ':' not in name for name in names),
        'Audio/cache filenames must remain unique and flat')
    caches = {name:confined_path(root,output/(name+'_embedding_cache')) for name in FRONTENDS}
    pair_directory = confined_path(root,output/'pair_receipts')
    for path in (*caches.values(),pair_directory):
        require(not path.exists(), 'Paired caches are fresh-only; existing/CP001 caches cannot be reused')
    for name in ('identity_cache_identity.json','gain_cache_identity.json','identity_cache_manifest.json',
            'gain_cache_manifest.json','paired_execution_report.json','paired_extraction_progress.json','paired_extraction_failure.json'):
        require(not (output/name).exists(), 'Fresh worker metadata already exists; implicit resume is forbidden')
    identities,records,encoders,paths = {},{name:[] for name in FRONTENDS},{},{}
    before,weights_before,drift = {},{},[]
    try:
        identities = {name:build_cpu_identity(root,suite,contract,sources,name,backend) for name in FRONTENDS}
        require(identities['identity']['signature'] != identities['gain']['signature'], 'Frontends require independent signatures')
        for name in FRONTENDS:
            write_json(output/(name+'_cache_identity.json'),identities[name]); caches[name].mkdir(exist_ok=False)
        pair_directory.mkdir(exist_ok=False)
        progress_callback('identities',{'identities':identities})
        paths = {name:confined_path(root,asset['config']['weights_path']) for name,asset in sources['assets'].items()}
        weights_before = {name:file_sha256(path) for name,path in paths.items()}
        require(all(weights_before[name] == sources['assets'][name]['source_record']['weights_sha256']
            == identities['identity']['model_sources'][name]['weights_sha256']
            == identities['gain']['model_sources'][name]['weights_sha256'] for name in NAMES), 'Frozen source weights changed')
        import torch
        from speaker_id.models.campp import load_campp
        from speaker_id.candidates.campp_advanced import load_advanced
        from speaker_id.candidates.gain_frontend import extract_gain_pair
        require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, 'Parent must configure CPU threads exactly once')
        encoders['public'] = load_campp(sources['assets']['public']['config'],root,'cpu')
        encoders['advanced'] = load_advanced(sources['assets']['advanced']['config'],root,'cpu')
        for encoder in encoders.values():
            encoder.requires_grad_(False).eval()
            require(not any(module.training for module in encoder.modules()), 'All modules must remain in eval mode')
            require(all(tensor.device.type == 'cpu' and (not tensor.is_floating_point() or tensor.dtype == torch.float32)
                for tensor in (*encoder.parameters(),*encoder.buffers())), 'Models/buffers must remain on CPU in FP32')
        before = {name:encoder_state_sha256(model) for name,model in encoders.items()}
        noops = 0
        for index,row in enumerate(manifest):
            check_budget(started,settings['maximum_extraction_seconds'])
            raw = confined_path(root,Path(contract['config']['data_dir'])/row['audio_file'])
            expected_valid = truth(row['has_nonzero_signal'])
            require(expected_valid == bool(sources['valid'][index]), 'Original zero signal mask changed')
            values,infos,pair_records = {},{},{}
            for frontend,policy in (('identity',IDENTITY_POLICY),('gain',suite['gain_policy'])):
                require(raw.is_file() and file_sha256(raw) == row['input_sha256'], 'Raw input changed before extraction')
                tick = time.monotonic()
                values[frontend],infos[frontend] = extract_gain_pair(encoders['public'],encoders['advanced'],raw,
                    device='cpu',policy=policy,**contract['config']['inference'])
                duration = time.monotonic()-tick
                validate_pair(values[frontend],infos[frontend],expected_valid,policy)
                require(file_sha256(raw) == row['input_sha256'], 'Raw input changed during extraction')
                if frontend == 'gain':
                    require_noop_equal(values['identity'],values['gain'],infos['gain'])
                    noops += int(not infos['gain']['gain']['applied'])
                record = publish_npz(caches[frontend]/(Path(row['audio_file']).stem+'.npz'),values[frontend],
                    expected_valid,row,identities[frontend]['signature'])
                record.update(frontend_diagnostics=infos[frontend],elapsed_seconds=duration)
                records[frontend].append(record); pair_records[frontend] = record
            numerical = diagnostic(values['identity'],{name:sources['vectors'][name][index] for name in NAMES},expected_valid)
            drift.append(numerical)
            pair_path = pair_directory/f'{index:05d}.json'
            require(not pair_path.exists(), 'Pair receipt collision')
            write_json(pair_path,{'index':index,'records':pair_records,'cpu_vs_gpu_identity':numerical})
            progress = {'status':'extracting','completed_pairs':index+1,'total':len(manifest),
                'elapsed_seconds':time.monotonic()-started,'signatures':{n:i['signature'] for n,i in identities.items()},
                'last_pair_receipt':pair_path.relative_to(output).as_posix(),'last_pair_receipt_sha256':file_sha256(pair_path),
                'no_op_gain_pairs':noops,'encoder_updates':0}
            write_json(output/'paired_extraction_progress.json',progress)
            if (index+1)%settings['progress_every_pairs'] == 0 or index+1 == len(manifest):
                progress_callback('progress',progress)
                print(json.dumps({'stage':'cpu_paired_extraction','completed_pairs':index+1,'elapsed_seconds':progress['elapsed_seconds']}),flush=True)
            check_budget(started,settings['maximum_extraction_seconds'])
        after = {name:encoder_state_sha256(model) for name,model in encoders.items()}
        weights_after = {name:file_sha256(path) for name,path in paths.items()}
        require(before == after and weights_before == weights_after, 'Frozen tensors or weight files changed')
        receipts = {frontend:{'schema_version':1,'identity':identities[frontend],'file_count':len(manifest),
            'files':records[frontend],'encoder_state_sha256_before':before,'encoder_state_sha256_after':after,
            'weight_file_sha256_before':weights_before,'weight_file_sha256_after':weights_after,
            'elapsed_seconds':time.monotonic()-started,'encoder_updates':0} for frontend in FRONTENDS}
        vectors,masks = {},{}
        for frontend in FRONTENDS:
            write_json(output/(frontend+'_cache_manifest.json'),receipts[frontend])
            vectors[frontend],masks[frontend] = verify_gain_cache(caches[frontend],identities[frontend],manifest,receipts[frontend])
            require(np.array_equal(masks[frontend],sources['valid']), 'Completed cache changed original validity')
        report = {'status':'complete','completed_pairs':len(manifest),'elapsed_seconds':time.monotonic()-started,
            'backend':backend,'no_op_gain_pairs':noops,'no_op_bitwise_parity_verified':True,
            'encoder_state_sha256_before':before,'encoder_state_sha256_after':after,
            'weight_file_sha256_before':weights_before,'weight_file_sha256_after':weights_after,
            'cpu_vs_gpu_identity':summarize_diagnostics(drift),'encoder_updates':0,'raw_audio_sha_before_and_after_each_frontend':True,
            'cache_manifest_sha256':{n:file_sha256(output/(n+'_cache_manifest.json')) for n in FRONTENDS},
            'deadline_scope':'8-hour soft budget from worker entry through completed pairs, including periodic progress callbacks; final cache verification/completion callback excluded',
            'gpu_health_claim':False,'recognition_scoring_or_calibration':False}
        write_json(output/'paired_execution_report.json',report)
        progress_callback('complete',{'identities':identities,'receipts':receipts,'execution_report':report})
        return {'vectors':vectors,'valid':masks['identity'],'identities':identities,'receipts':receipts,'execution_report':report}
    except Exception as error:
        state_after,state_errors = best_effort_hashes(encoders,encoder_state_sha256)
        weights_after,weight_errors = best_effort_hashes(paths,file_sha256)
        failure = {'status':'failed','error_type':type(error).__name__,'completed_frontend_files':{n:len(v) for n,v in records.items()},
            'elapsed_seconds':time.monotonic()-started,'encoder_updates':0,'no_implicit_retry_or_resume':True,
            'identities':identities,'encoder_state_sha256_before':before,'encoder_state_sha256_after':state_after,
            'weight_file_sha256_before':weights_before,'weight_file_sha256_after':weights_after,
            'state_hash_errors':state_errors,'weight_hash_errors':weight_errors,'files_preserved':True}
        write_json(output/'paired_extraction_failure.json',failure)
        try:
            progress_callback('failure',failure)
        except Exception as callback_error:
            failure['failure_callback_error_type'] = type(callback_error).__name__
            write_json(output/'paired_extraction_failure.json',failure)
        raise
