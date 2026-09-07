from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training.adapted_scoring import RECIPES as S004_RECIPES
from speaker_id.training.expanded_gallery import RECIPES, validate_expanded_suite, verified_s004_controls
from speaker_id.training.heldout_references import heldout_reference_scores
from speaker_id.training.reference_scoring import calibrate_gate, reference_probabilities

ROOT = Path(__file__).resolve().parents[1]


class OuterLabelForbidden(dict):
    def __getitem__(self,key):
        if key=='speaker_id':
            raise AssertionError('Outer labels cannot enter reference scores')
        return super().__getitem__(key)


class HeldoutReferenceTests(unittest.TestCase):
    def fixture(self):
        targets=['A','B','unknown','A','A','B','unknown','unknown','A','unknown']
        groups=['fit_a','fit_b','fit_u','query_a','query_a','query_b','query_u','query_u','outer_a','outer_zero']
        values=np.asarray([[1,0,0],[0,1,0],[0,0,1],[.8,.6,0],[.8,.6,0],[.6,.8,0],
                           [1,0,0],[1,0,0],[.9,.1,0],[0,0,0]],dtype=np.float32)
        values[:-1]/=np.linalg.norm(values[:-1],axis=1,keepdims=True)
        valid=np.asarray([True]*9+[False])
        manifest=[{'audio_file':f'{i}.wav','speaker_id':label} for i,label in enumerate(targets)]
        manifest[8]=OuterLabelForbidden(manifest[8]); manifest[9]=OuterLabelForbidden(manifest[9])
        folds=[{'audio_file':f'{i}.wav','fold':int(i<8),'group_id':group,'train_eligible':i<9}
               for i,group in enumerate(groups)]
        roles=[{'audio_file':f'{i}.wav','outer_fold':0,'group_id':group,'encoder_fit_allowed':i<3,
                'enrollment_allowed':i<2,'calibration_query':3<=i<8,'outer_evaluation_included':i>=8}
               for i,group in enumerate(groups)]
        return {'config':{'fold_ids':[0,1]},'manifest':manifest,'folds':folds,'roles':roles,'labels':['unknown','A','B']},values,valid

    def test_entire_known_and_unknown_query_groups_are_masked_before_maximum(self):
        contract,vectors,valid=self.fixture()
        scores=heldout_reference_scores(contract,vectors,valid,0)
        np.testing.assert_array_equal(scores['calibration_indices'],[3,4,5,6,7])
        np.testing.assert_array_equal(scores['outer_indices'],[8,9])
        # Each duplicated unknown query has a self-like reference in its group.
        # Both must leave; only the original orthogonal unknown reference remains.
        np.testing.assert_array_equal(scores['inner_unknown_similarity'][-2:],[0.,0.])
        self.assertEqual(scores['reference_counts']['removed_query_group_files'].tolist(),[2,2,1,2,2])
        groups=[row['group_id'] for row in contract['folds']]
        for position,q in enumerate(scores['calibration_indices']):
            remaining=[i for i in range(8) if groups[i]!=groups[q]]
            expected=[max(float(vectors[q]@vectors[i]) for i in remaining if contract['manifest'][i]['speaker_id']==label)
                      for label in contract['labels'][1:]]
            np.testing.assert_allclose(scores['inner_known_scores'][position],expected,atol=1e-7)
        np.testing.assert_array_equal(scores['reference_counts']['known_query_own_class_inner_files'][:3],[1,1,1])
        np.testing.assert_array_equal(scores['reference_counts']['known_query_own_class_outer_files'][:3],[3,3,2])

    def test_outer_labels_and_features_cannot_change_inner_calibration(self):
        contract,vectors,valid=self.fixture()
        before=heldout_reference_scores(contract,vectors,valid,0)
        vectors[8]=[0,0,1]
        after=heldout_reference_scores(contract,vectors,valid,0)
        for key in ('calibration_indices','inner_known_scores','inner_unknown_similarity'):
            np.testing.assert_array_equal(before[key],after[key])
        targets=np.asarray([1,1,2,0,0])
        def fit(value):
            return calibrate_gate(value['inner_known_scores'],targets,value['inner_unknown_similarity'],[0,.25,.5,.75,1],[0,.5],201,3)[0]
        self.assertEqual(fit(before),fit(after))
        probabilities=reference_probabilities(after['outer_known_scores'],after['outer_unknown_similarity'],fit(after),after['outer_valid'])
        np.testing.assert_array_equal(probabilities[1],[1.,0.,0.])
        self.assertEqual(probabilities.shape,(2,3))

    def test_fitted_query_ineligible_roles_and_group_conflicts_fail_closed(self):
        contract,vectors,valid=self.fixture()
        for mutation in ('fit_query','fit_group','ineligible','outer_group'):
            changed=deepcopy(contract)
            if mutation=='fit_query': changed['roles'][0]['calibration_query']=True
            if mutation=='fit_group':
                for collection in ('roles','folds'):
                    for i in (3,4): changed[collection][i]['group_id']='fit_a'
            if mutation=='ineligible': changed['folds'][3]['train_eligible']=False
            if mutation=='outer_group':
                changed['roles'][8]['group_id']='query_a'; changed['folds'][8]['group_id']='query_a'
            with self.assertRaises(ValueError): heldout_reference_scores(changed,vectors,valid,0)

    def test_empty_class_after_whole_group_exclusion_is_rejected(self):
        contract,vectors,valid=self.fixture()
        contract['folds'][0]['train_eligible']=False
        contract['roles'][0]['encoder_fit_allowed']=False
        contract['roles'][0]['enrollment_allowed']=False
        with self.assertRaisesRegex(ValueError,'empty class or unknown'):
            heldout_reference_scores(contract,vectors,valid,0)
        contract,vectors,valid=self.fixture()
        contract['labels']=['unknown','B','A']
        with self.assertRaisesRegex(ValueError,'label columns'):
            heldout_reference_scores(contract,vectors,valid,0)

    def suite(self):
        original=json.loads((ROOT/'configs/train/campp_adapted_scoring.json').read_text())
        suite={**original,'experiment_code':'S005','recipes':list(RECIPES),
               'calibration_protocol':'original_heldout_queries_expanded_gallery',
               'control':{'run':'artifacts/training/S004_fixture','parent_run_id':'a'*32,
                    'recipes':{'public':{'id':'S004c','child_run_id':'b'*32},'adapted':{'id':'S004d','child_run_id':'c'*32}}}}
        return original,suite

    def test_completed_source_control_identity_and_grid_are_mandatory(self):
        original,suite=self.suite()
        validate_expanded_suite(suite)
        for key,value in [('control',None),('recipes',list(RECIPES)[1:]),('threshold_candidates',101)]:
            changed={**suite,key:value}
            with self.assertRaises(ValueError): validate_expanded_suite(changed)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/suite['control']['run']
            def write(path,value):
                target=directory/path;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(value),encoding='utf-8')
            identity={'status':'complete','parent_run_id':suite['control']['parent_run_id']}
            proof={'public':{'source_signature':'public'},'adapted':{'checkpoint_sha256':'adapted'}}
            write('experiment_state.json',identity)
            write('experiment_report.json',{**identity,'source_control_checks':{name:{'exact_prediction_reproduction':True} for name in ('public','adapted')}})
            write('tracking/run_state.json',{'run_id':suite['control']['parent_run_id'],'remote_status':'FINISHED'})
            write('tracking/artifacts/resolved_config.json',{'suite':original})
            write('source_provenance.json',proof)
            for name,entry in suite['control']['recipes'].items():
                recipe=next(row for row in S004_RECIPES if row['id']==entry['id'])
                write(entry['id']+'/tracking/run_state.json',{'run_id':entry['child_run_id'],'remote_status':'FINISHED',
                    'tags':{'mlflow.parentRunId':suite['control']['parent_run_id']}})
                write(entry['id']+'/tracking/artifacts/resolved_config.json',{'suite':original,'recipe':recipe})
                write(entry['id']+'/experiment_report.json',{'recipe':recipe,'folds':[{'outer_fold':0},{'outer_fold':1}]})
            directories,checked=verified_s004_controls(root,suite,proof)
            self.assertEqual(set(directories),{'public','adapted'})
            self.assertEqual(checked['parent_run_id'],'a'*32)
            with self.assertRaisesRegex(ValueError,'source cache/checkpoint'):
                verified_s004_controls(root,suite,{'public':{'source_signature':'swapped'},'adapted':proof['adapted']})
            write('S004d/tracking/run_state.json',{'run_id':'c'*32,'remote_status':'RUNNING'})
            with self.assertRaisesRegex(ValueError,'control child'):
                verified_s004_controls(root,suite,proof)


if __name__=='__main__': unittest.main()
