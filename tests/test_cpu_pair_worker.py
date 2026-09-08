"""Paired worker checks with tiny files and synthetic NumPy vectors, never real models."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY
from speaker_id.training import cpu_pair_worker as worker


class FakeEncoder:
    training = False
    def requires_grad_(self,value):
        return self
    def eval(self):
        return self
    def modules(self):
        return [self]
    def parameters(self):
        return [SimpleNamespace(device=SimpleNamespace(type='cpu'),dtype='float32',is_floating_point=lambda:True)]
    def buffers(self):
        return []


def vectors(valid,position=0):
    result={name:np.zeros(dimension,dtype=np.float32) for name,dimension in worker.DIMS.items()}
    if valid:
        for value in result.values(): value[position]=1
    return result


def fixture(root):
    (root/'data/raw').mkdir(parents=True)
    (root/'artifacts/models').mkdir(parents=True)
    output=root/'artifacts/training/C001_test';output.mkdir(parents=True)
    rows=[]
    for i,valid in enumerate((True,False,True)):
        name=f'{i}.wav';path=root/'data/raw'/name;path.write_bytes(f'synthetic-{i}'.encode())
        rows.append({'audio_file':name,'input_sha256':worker.file_sha256(path),'has_nonzero_signal':valid,'speaker_id':object()})
    assets={}
    for name in worker.NAMES:
        path=root/'artifacts/models'/f'{name}.bin';path.write_bytes(name.encode())
        assets[name]={'config':{'weights_path':path.relative_to(root).as_posix()},
            'source_record':{'weights_sha256':worker.file_sha256(path)}}
    sources={'assets':assets,'valid':np.asarray([True,False,True]),
        'vectors':{name:np.stack([vectors(valid)[name] for valid in (True,False,True)]) for name in worker.NAMES}}
    suite={'execution':{'device':'cpu','threads':4,'interop_threads':1,'worker_count':1,
        'maximum_extraction_seconds':28800,'progress_every_pairs':50},'gain_policy':GAIN_POLICY}
    contract={'manifest':rows,'config':{'data_dir':'data/raw','inference':{'seconds':180.0,'maximum_windows':1}}}
    control={k:True for k in ('exact_prediction_reproduction','exact_pooled_metrics','exact_inner_alpha_curves','exact_probability_and_support_arrays')}
    return suite,contract,sources,output,control


def signed_identity(root,suite,contract,sources,frontend,backend):
    body={'schema_version':1,'frontend':frontend,'backend':backend,'embedding_dims':worker.DIMS,
        'model_sources':{name:value['source_record'] for name,value in sources['assets'].items()}}
    return {**body,'signature':hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()}


def fake_extract(public,advanced,path,*,device,policy,**kwargs):
    valid=Path(path).stem != '1'
    return vectors(valid),{'nonzero_signal':valid,'gain':{'policy':policy['name'],'applied':False,'gain':1.0}}


def mocks(stack,extract=fake_extract,*,intra=4,inter=1):
    torch=ModuleType('torch');torch.float32='float32'
    torch.get_num_threads=lambda:intra;torch.get_num_interop_threads=lambda:inter
    contract=ModuleType('speaker_id.training.cpu_pair_contract');contract.build_cpu_identity=signed_identity
    stack.enter_context(patch.dict('sys.modules',{'torch':torch,'speaker_id.training.cpu_pair_contract':contract}))
    public=stack.enter_context(patch('speaker_id.models.campp.load_campp',return_value=FakeEncoder()))
    advanced=stack.enter_context(patch('speaker_id.candidates.campp_advanced.load_advanced',return_value=FakeEncoder()))
    stack.enter_context(patch('speaker_id.candidates.gain_frontend.extract_gain_pair',side_effect=extract))
    stack.enter_context(patch('speaker_id.training.candidate_comparison.encoder_state_sha256',return_value='a'*64))
    return public,advanced


class CPUWorkerTests(unittest.TestCase):
    def test_full_synthetic_pair_contract_and_callbacks_have_no_vectors(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
            public,advanced=mocks(stack);events=[]
            def callback(stage,payload):
                json.dumps(payload,allow_nan=False)
                events.append(stage)
                if stage=='identities': public.assert_not_called();advanced.assert_not_called()
            result=worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{'device':'cpu'},progress_callback=callback)
            self.assertEqual(events,['identities','progress','complete'])
            public.assert_called_once();advanced.assert_called_once()
            for frontend in worker.FRONTENDS:
                self.assertEqual(result['receipts'][frontend]['file_count'],3)
                self.assertEqual(result['vectors'][frontend]['public'].shape,(3,512))
                self.assertEqual(len(list((output/(frontend+'_embedding_cache')).glob('*.npz'))),3)
                self.assertFalse(list((output/(frontend+'_embedding_cache')).glob('*.partial')))
            self.assertEqual(result['execution_report']['no_op_gain_pairs'],3)
            self.assertEqual(len(list((output/'pair_receipts').glob('*.json'))),3)
            self.assertFalse(list(output.rglob('*.zip')))

    def test_changed_audio_fails_before_model_forward_and_preserves_hashes(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
            def extract(*args,**kwargs): raise AssertionError('forward must never occur')
            mocks(stack,extract);(root/'data/raw/0.wav').write_bytes(b'changed')
            events=[]
            with self.assertRaisesRegex(ValueError,'before extraction'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda stage,p:events.append(stage))
            failure=json.loads((output/'paired_extraction_failure.json').read_text())
            self.assertEqual(failure['encoder_state_sha256_before'],failure['encoder_state_sha256_after'])
            self.assertEqual(failure['completed_frontend_files'],{'identity':0,'gain':0})
            self.assertEqual(events,['identities','failure'])

    def test_noop_drift_fails_preserving_identity_partial_pair(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
            def drift(*args,**kwargs):
                v,info=fake_extract(*args,**kwargs)
                if kwargs['policy']['name']==GAIN_POLICY['name']:v=vectors(True,1)
                return v,info
            mocks(stack,drift)
            with self.assertRaisesRegex(ValueError,'No-op gain'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda *_:None)
            failure=json.loads((output/'paired_extraction_failure.json').read_text())
            self.assertEqual(failure['completed_frontend_files'],{'identity':1,'gain':0})
            self.assertTrue((output/'identity_embedding_cache/0.npz').is_file())
            self.assertFalse((output/'gain_embedding_cache/0.npz').exists())

    def test_existing_cache_or_metadata_never_resumed_or_overwritten(self):
        for name in ('identity_embedding_cache','identity_cache_identity.json'):
            with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
                path=output/name
                path.mkdir() if name.endswith('cache') else path.write_text('retained')
                public,_=mocks(stack)
                with self.assertRaisesRegex(ValueError,'fresh|Fresh'):
                    worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda *_:None)
                public.assert_not_called()
                if path.is_file():self.assertEqual(path.read_text(),'retained')

    def test_wrong_threads_and_control_stop_before_models(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
            public,_=mocks(stack,inter=4)
            with self.assertRaisesRegex(ValueError,'threads'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda *_:None)
            public.assert_not_called()

    def test_failed_callback_and_after_hash_preserve_original_exception(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root);mocks(stack)
            def callback(stage,payload):raise RuntimeError('tracking stopped')
            with self.assertRaisesRegex(RuntimeError,'tracking stopped'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=callback)
            failure=json.loads((output/'paired_extraction_failure.json').read_text())
            self.assertEqual(failure['failure_callback_error_type'],'RuntimeError')
            hashes,errors=worker.best_effort_hashes({'missing':None},lambda p:(_ for _ in ()).throw(FileNotFoundError()))
            self.assertEqual(errors,{'missing':'FileNotFoundError'})

    def test_soft_budget_boundary_and_gain_zero_semantics(self):
        worker.check_budget(100,28800,now=28899)
        with self.assertRaises(TimeoutError):worker.check_budget(100,28800,now=28900)
        v=vectors(False);info={'nonzero_signal':False,'gain':{'policy':'identity_v1','applied':False,'gain':1.0}}
        worker.validate_pair(v,info,False,{'name':'identity_v1'})
        v['public'][0]=np.float32(-0.0)
        with self.assertRaises(ValueError):worker.validate_pair(v,info,False,{'name':'identity_v1'})

    def test_budget_failure_preserves_completed_pair_without_implicit_resume(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root);mocks(stack)
            stack.enter_context(patch.object(worker,'check_budget',side_effect=[None,TimeoutError('budget')]))
            with self.assertRaisesRegex(TimeoutError,'budget'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda *_:None)
            failure=json.loads((output/'paired_extraction_failure.json').read_text())
            self.assertEqual(failure['completed_frontend_files'],{'identity':1,'gain':1})
            self.assertTrue((output/'pair_receipts/00000.json').is_file())
            self.assertTrue(failure['no_implicit_retry_or_resume'])

    def test_weight_drift_after_forward_is_detected_before_complete_receipts(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();suite,contract,sources,output,control=fixture(root)
            def mutate(*args,**kwargs):
                if Path(args[2]).stem=='2' and kwargs['policy']['name']==GAIN_POLICY['name']:
                    (root/'artifacts/models/public.bin').write_bytes(b'changed-weights')
                return fake_extract(*args,**kwargs)
            mocks(stack,mutate)
            with self.assertRaisesRegex(ValueError,'Frozen tensors or weight files'):
                worker.extract_cpu_pair_caches(root,suite,contract,sources,output,control,{},progress_callback=lambda *_:None)
            failure=json.loads((output/'paired_extraction_failure.json').read_text())
            self.assertNotEqual(failure['weight_file_sha256_before'],failure['weight_file_sha256_after'])
            self.assertFalse((output/'gain_cache_manifest.json').exists())

    def test_atomic_publish_collision_never_overwrites_completed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'x.npz';path.write_bytes(b'original')
            row={'audio_file':'x.wav','input_sha256':'c'*64}
            with self.assertRaises(ValueError):worker.publish_npz(path,vectors(True),True,row,'b'*64)
            self.assertEqual(path.read_bytes(),b'original')


if __name__=='__main__':unittest.main()
