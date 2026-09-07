from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.training.adapted_scoring import (assert_fixed_role_groups, load_adapted_fold_cache,
    validate_adapted_identity, validate_adapted_suite, verified_export_inventory)

ROOT = Path(__file__).resolve().parents[1]


class AdaptedScoringTests(unittest.TestCase):
    def fixture(self, root):
        directory = root / 'artifacts/training/F003_fixture'
        directory.mkdir(parents=True)
        def write(relative, value):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf-8')
        roles = []
        for outer in (0, 1):
            for i, role in enumerate(('fit_enrollment', 'query', 'outer')):
                roles.append({'audio_file': f'{i}.wav', 'outer_fold': outer, 'group_id': f'g{i}',
                    'role': role, 'encoder_fit_allowed': i == 0, 'enrollment_allowed': i == 0,
                    'calibration_query': i == 1, 'outer_evaluation_included': i == 2})
        contract = {'config': {'mode':'fine_tune', 'fold_ids':[0,1]}, 'model':{'weights':'public'},
                    'input_hashes': {'roles':'rolehash'}, 'code_hashes': {'src/speaker_id/training/fit.py':'fitcode'},
                    'manifest': [{'audio_file':f'{i}.wav','input_sha256':str(i)*64,'has_nonzero_signal':i != 2}
                                 for i in range(3)], 'roles':roles}
        original = {'experiment':contract['config'], 'model':contract['model'], 'input_hashes':contract['input_hashes'],
                    'code_hashes':contract['code_hashes'], 'resume':False}
        signature = hashlib.sha256(json.dumps({'config':original['experiment'], 'model':original['model'],
            'input_hashes':original['input_hashes'], 'code_hashes':original['code_hashes']},sort_keys=True).encode()).hexdigest()
        original['signature'] = signature
        source = {'run':'artifacts/training/F003_fixture','parent_run_id':'a'*32,'signature':signature,
                  'git_commit':'c'*40, 'completed_steps':600, 'folds':{},
                  'export_manifest_paths':['artifacts/exports/fixture.json']}
        write('resolved_config.json',original)
        write('experiment_state.json', {'status':'complete','parent_run_id':source['parent_run_id'],
                                       'signature':signature,'attempt':'attempt'})
        write('experiment_report.json', {'status':'complete','signature':signature,
                                         'folds':[{'outer_fold':i,'fit':{'completed_steps':600}} for i in (0,1)]})
        write('tracking/attempt/run_state.json',{'run_id':source['parent_run_id'],'remote_status':'FINISHED'})
        write('tracking/attempt/artifacts/source_manifest.json',{'git_commit':source['git_commit']})
        for outer in (0,1):
            child_id = str(outer+1)*32
            write(f'fold_{outer}/tracking/attempt/run_state.json',{'run_id':child_id,'remote_status':'FINISHED',
                'tags':{'mlflow.parentRunId':source['parent_run_id'],'mlflow.source.git.commit':source['git_commit']}})
            write(f'fold_{outer}/tracking/attempt/artifacts/resolved_config.json',{**original,'outer_fold':outer})
            checkpoint = directory / f'fold_{outer}/last.pt'
            checkpoint.write_bytes(f'checkpoint-for-fold-{outer}'.encode())
            source['folds'][str(outer)] = {'child_run_id':child_id,'checkpoint_sha256':file_sha256(checkpoint)}
            cache = directory / f'fold_{outer}/embedding_cache'
            cache.mkdir()
            for i,row in enumerate(contract['manifest']):
                vector = np.zeros(512,dtype=np.float32)
                if i != 2:
                    vector[outer*3+i] = 1
                np.savez_compressed(cache / f'{i}.npz',embedding=vector,valid=i != 2,
                                    signature=signature,audio_sha256=row['input_sha256'])
        inventory = {path.relative_to(directory).as_posix(): {'path':path.relative_to(directory).as_posix(),
                     'bytes':path.stat().st_size,'sha256':file_sha256(path)} for path in directory.rglob('*') if path.is_file()}
        manifest = {'schema_version':1,'run_name':directory.name,'parent_run_id':source['parent_run_id'],
                    'selected_file_count':len(inventory),'files':list(inventory.values())}
        manifest_path = root / source['export_manifest_paths'][0]
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
        source['export_manifest_sha256'] = file_sha256(manifest_path)
        return directory, source, contract, inventory, manifest_path

    def test_swapped_fold_cache_rejected_even_with_identical_npz_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            directory,source,contract,inventory,_=self.fixture(root)
            first,valid,_=load_adapted_fold_cache(directory,contract,source['signature'],inventory,0)
            second,_,_=load_adapted_fold_cache(directory,contract,source['signature'],inventory,1)
            self.assertFalse(np.array_equal(first,second))
            self.assertEqual(valid.tolist(),[True,True,False])
            self.assertFalse(first[2].any())
            shutil.copyfile(directory/'fold_1/embedding_cache/0.npz',directory/'fold_0/embedding_cache/0.npz')
            with self.assertRaisesRegex(ValueError,'fold binding/hash'):
                load_adapted_fold_cache(directory,contract,source['signature'],inventory,0)

    def test_complete_export_and_fold_identity_accept_original_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            directory,source,contract,inventory,_=self.fixture(root)
            _,_,checked=verified_export_inventory(root,source)
            _,folds=validate_adapted_identity(directory,source,contract,checked)
            self.assertEqual(folds['0']['completed_steps'],600)
            changed=deepcopy(source)
            changed['folds']['0']['checkpoint_sha256']=source['folds']['1']['checkpoint_sha256']
            with self.assertRaisesRegex(ValueError,'fold/child/checkpoint'):
                validate_adapted_identity(directory,changed,contract,inventory)
            changed=deepcopy(contract)
            changed['code_hashes']['src/speaker_id/training/fit.py']='changed-training-logic'
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                validate_adapted_identity(directory,source,changed,inventory)
            changed=deepcopy(contract)
            del changed['code_hashes']['src/speaker_id/training/fit.py']
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                validate_adapted_identity(directory,source,changed,inventory)
            shutil.copyfile(directory/'fold_1/last.pt',directory/'fold_0/last.pt')
            with self.assertRaisesRegex(ValueError,'hash or path mismatch'):
                verified_export_inventory(root,source)

    def test_manifest_bytes_duplicates_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            _,source,_,_,path=self.fixture(root)
            original=json.loads(path.read_text())
            path.write_text(json.dumps(original)+' ')
            with self.assertRaisesRegex(ValueError,'manifest bytes'):
                verified_export_inventory(root,source)
            for altered in ([original['files'][0],original['files'][0]],
                            [{**original['files'][0],'path':'../escape'}]):
                updated={**original,'files':altered,'selected_file_count':len(altered)}
                path.write_text(json.dumps(updated))
                source['export_manifest_sha256']=file_sha256(path)
                with self.assertRaisesRegex(ValueError,'Unsafe, duplicate'):
                    verified_export_inventory(root,source)

    def test_group_leakage_rejected_despite_distinct_file_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,_,contract,_,_=self.fixture(Path(temporary))
            self.assertTrue(assert_fixed_role_groups(contract,0)['fit_query_outer_content_groups_disjoint'])
            for target in (1,2):
                changed=deepcopy(contract)
                changed['roles'][target]['group_id']=changed['roles'][0]['group_id']
                with self.assertRaisesRegex(ValueError,'content-group leakage'):
                    assert_fixed_role_groups(changed,0)

    def test_fixed_grid_and_both_controls_cannot_be_bypassed(self):
        suite=json.loads((ROOT/'configs/train/campp_adapted_scoring.json').read_text())
        validate_adapted_suite(suite)
        for key,value in [('calibration_protocol','leave_content_group_out'),('recipes',suite['recipes'][1:]),
                          ('threshold_candidates',501),('readiness_config','configs/train/campp_finetune_fp32.json')]:
            changed=deepcopy(suite)
            changed[key]=value
            with self.assertRaises(ValueError):
                validate_adapted_suite(changed)
        changed=deepcopy(suite)
        changed['sources']['adapted']['folds']['0']['checkpoint_sha256']=changed['sources']['adapted']['folds']['1']['checkpoint_sha256']
        with self.assertRaisesRegex(ValueError,'must be distinct'):
            validate_adapted_suite(changed)


if __name__ == '__main__':
    unittest.main()
