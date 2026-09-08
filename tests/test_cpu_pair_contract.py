"""CPU protocol and provenance rejection tests; no model or audio execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from speaker_id.training import cpu_pair_contract as cpu


def backend():
    return {'schema_version':1,'device':'cpu','tensor_dtype':'float32','worker_count':1,
        'torch_intraop_threads':4,'torch_interop_threads':1,'python_version':'3.12.3',
        'versions':{k:'1.0' for k in ('numpy','scipy','soundfile','torch','torchaudio')},
        'libsndfile_version':'1.2.0','torch_build_sha256':'a'*64,'numpy_build_sha256':'b'*64,
        'cpu_capability':'AVX2','mkldnn_enabled':True,'deterministic_algorithms_enabled':False,
        'thread_environment':{k:'4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'encoder_updates':0,'cuda_queried':False}


def pilot():
    report={'status':'complete','parent_run_id':cpu.CP001['parent_run_id'],'git_commit':cpu.CP001['git_commit'],
        'device':'cpu','threads':4,'interop_threads':1,'capacity':{'instance_id':50079023},
        'encoder_updates':0,'embedding_artifacts_uploaded':False,'raw_audio_full_sha_verified':True,
        'raw_files_verified':4529,'recognition_scoring_or_calibration':False,'gpu_health_claim':False,
        'records':[{'index':i,'frontend':f} for i in (0,35,583,2060,2264,3632,4528) for f in ('identity','gain')]}
    for prefix in ('encoder_state','weight_file'):
        for when in ('before','after'):
            report[prefix+'_sha256_'+when]={'public':'a'*64,'advanced':'b'*64}
    state={'run_id':cpu.CP001['parent_run_id'],'remote_status':'FINISHED','pending_status':None,
        'last_sync_error':None,'binding':{'experiment_id':'1'},
        'tags':{'mlflow.source.git.commit':cpu.CP001['git_commit']},'uploaded_artifacts':{'report.json':'c'*64}}
    return report,state


class CPUContractTests(unittest.TestCase):
    def test_fixed_config_matches_file_and_disallows_backend_search_or_extra_fields(self):
        root=Path(__file__).resolve().parents[1]
        cpu.validate_cpu_gain_config(json.loads((root/'configs/train/campp_gain_cpu.json').read_text()))
        for key,value in [('schema_version',True),('experiment_code','S010'),('recipes',[]),
                ('mlflow_payload','include_embeddings'),('extra',1)]:
            changed=deepcopy(cpu.FIXED); changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cpu.validate_cpu_gain_config(changed)
        for field,value in [('threads',12),('worker_count',3),('maximum_extraction_seconds',999999)]:
            changed=deepcopy(cpu.FIXED);changed['execution'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):cpu.validate_cpu_gain_config(changed)

    def test_backend_rejects_hidden_precision_threads_or_missing_provenance(self):
        cpu.validate_backend(backend())
        for key,value in [('device','cuda'),('tensor_dtype','float16'),('worker_count',True),
                ('torch_intraop_threads',8),('versions',{}),('torch_build_sha256','x'*64),
                ('thread_environment',{}),('cuda_queried',True),('encoder_updates',1)]:
            changed=backend();changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cpu.validate_backend(changed)

    def test_no_local_execution_or_network_when_host_gate_fails(self):
        with (patch.object(cpu,'server_cpu_gate',side_effect=ValueError('wrong host')),
              patch.object(cpu,'capture_cpu_backend') as capture):
            with self.assertRaises(ValueError):cpu.prepare_cpu_execution(Path.cwd(),deepcopy(cpu.FIXED))
            capture.assert_not_called()

    def test_pilot_requires_exact_completed_prefix_and_frozen_models(self):
        report,state=pilot();cpu.verify_pilot_report(report,state,cpu.FIXED)
        for mutate in (lambda r:r.update(status='failed'),lambda r:r['records'].pop(),
                lambda r:r['records'].reverse(),lambda r:r.update(embedding_artifacts_uploaded=True),
                lambda r:r['encoder_state_sha256_after'].update(public='c'*64),
                lambda r:r.update(raw_files_verified=7)):
            changed=deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):cpu.verify_pilot_report(changed,state,cpu.FIXED)

    def test_pilot_tracking_rejects_unfinished_wrong_commit_and_embedding_upload(self):
        report,state=pilot()
        for mutate in (lambda s:s.update(remote_status='RUNNING'),lambda s:s.update(last_sync_error='offline'),
                lambda s:s['tags'].update({'mlflow.source.git.commit':'a'*40}),
                lambda s:s['binding'].update(experiment_id='0'),
                lambda s:s['uploaded_artifacts'].update({'0000_identity.npz':'a'*64})):
            changed=deepcopy(state);mutate(changed)
            with self.assertRaises(ValueError):cpu.verify_pilot_report(report,changed,cpu.FIXED)

    def test_feature_identity_binds_frontend_backend_sources_and_new_launcher(self):
        contract={'config':{'inference':{'seconds':180.0,'maximum_windows':1}},'input_hashes':{'manifest':'x'},
            'labels':['unknown','s1'],'code_hashes':{'src/module.py':'a'*64}}
        sources={'assets':{name:{'source_record':{'weights_sha256':char*64},'config':{'weights_path':name}}
            for name,char in [('public','b'),('advanced','c')]}}
        with patch.object(cpu,'file_sha256',return_value='d'*64):
            first=cpu.build_cpu_identity(Path.cwd(),cpu.FIXED,contract,sources,'identity',backend())
            gain=cpu.build_cpu_identity(Path.cwd(),cpu.FIXED,contract,sources,'gain',backend())
            changed=backend();changed['cpu_capability']='AVX512'
            different=cpu.build_cpu_identity(Path.cwd(),cpu.FIXED,contract,sources,'identity',changed)
        self.assertNotEqual(first['signature'],gain['signature'])
        self.assertNotEqual(first['signature'],different['signature'])
        self.assertIn('scripts/score_gain_cpu.py',first['code_hashes'])
        self.assertNotIn('scripts/score_gain.py',first['code_hashes'])
        body={k:v for k,v in first.items() if k!='signature'}
        self.assertEqual(first['signature'],hashlib.sha256(cpu.canonical(body)).hexdigest())


if __name__=='__main__':unittest.main()
