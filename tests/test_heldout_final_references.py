"""Synthetic leakage and support checks; no real embeddings or fitting."""
from copy import deepcopy
import unittest
import numpy as np
from speaker_id.training.heldout_final_references import heldout_final_scores
from speaker_id.inference.scoring import reference_scores


def fixture():
    rows = [
        ('aq', 'a', 'aq', 1, 'query', [1, 0, 0]),
        ('aq_copy', 'a', 'aq', 1, 'query', [1, 0, 0]),
        ('af', 'a', 'af', 1, 'fit', [.8, .6, 0]),
        ('ao', 'a', 'ao', 0, 'outer', [.6, .8, 0]),
        ('bq', 'b', 'bq', 1, 'query', [0, 1, 0]),
        ('bf', 'b', 'bf', 1, 'fit', [0, .8, .6]),
        ('bo', 'b', 'bo', 0, 'outer', [0, .6, .8]),
        ('uq', 'unknown', 'uq', 1, 'query', [0, 0, 1]),
        ('uf', 'unknown', 'uf', 1, 'fit', [.6, 0, .8]),
        ('uo', 'unknown', 'uo', 0, 'outer', [.8, 0, .6]),
        ('zero', 'unknown', 'z', 0, 'outer', [0, 0, 0]),
    ]
    manifest, folds, roles = [], [], []
    for name, label, group, fold, role, vector in rows:
        manifest.append({'audio_file': name, 'speaker_id': label})
        folds.append({'audio_file': name, 'group_id': group, 'fold': fold, 'train_eligible': name != 'zero'})
        roles.append({'audio_file': name, 'outer_fold': 0, 'group_id': group,
            'encoder_fit_allowed': role == 'fit', 'enrollment_allowed': role == 'fit',
            'calibration_query': role == 'query', 'outer_evaluation_included': role == 'outer'})
    return {'manifest': manifest, 'folds': folds, 'roles': roles, 'labels': ['unknown', 'a', 'b']}, np.asarray([row[-1] for row in rows], dtype=np.float32), np.asarray([True]*10+[False])


class HeldoutFinalTests(unittest.TestCase):
    def test_standalone_cosine_parity_with_perturbed_unit_sources(self):
        contract, vectors, valid = fixture()
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((len(vectors), 192)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors *= np.float32(1.000005)
        vectors[-1] = 0
        vectors[1] = vectors[0]
        vectors[2] = vectors[0]  # A different synthetic group with an identical vector.
        result = heldout_final_scores(contract, vectors, valid)
        groups = np.asarray([row['group_id'] for row in contract['folds']])
        refs = result['reference_indices']
        targets = np.asarray([contract['labels'].index(contract['manifest'][i]['speaker_id']) for i in refs])
        for row, query in enumerate(result['calibration_indices']):
            keep_known = groups[refs[targets > 0]] != groups[query]
            keep_unknown = groups[refs[targets == 0]] != groups[query]
            gallery = {key: value[keep_known if key != 'unknown_embeddings' else keep_unknown]
                       for key, value in result['gallery'].items()}
            known, unknown = reference_scores(vectors[query:query+1], gallery, classes=2)
            np.testing.assert_allclose(result['inner_known_scores'][row], known[0], atol=1e-7, rtol=0)
            np.testing.assert_allclose(result['inner_unknown_similarity'][row], unknown[0], atol=1e-7, rtol=0)
        self.assertTrue(np.all(np.abs(result['inner_known_scores']) <= 1))
        self.assertTrue(np.all(np.abs(result['inner_unknown_similarity']) <= 1))

    def test_group_exclusion_and_global_enrollment(self):
        contract, vectors, valid = fixture()
        original = vectors.copy()
        result = heldout_final_scores(contract, vectors, valid)
        self.assertEqual(result['calibration_indices'].tolist(), [0, 1, 4, 7])
        self.assertEqual(result['reference_indices'].tolist(), list(range(10)))
        self.assertEqual(result['gallery']['known_embeddings'].shape, (7, 3))
        self.assertEqual(result['gallery']['unknown_embeddings'].shape, (3, 3))
        self.assertEqual(result['reference_counts']['removed_query_group_files'].tolist(), [2, 2, 1, 1])
        self.assertEqual(result['reference_counts']['inner_known_files_per_class'][0].tolist(), [2, 3])
        # A's two same-content references would produce 1.0 if only one copy left.
        self.assertAlmostEqual(float(result['inner_known_scores'][0, 0]), .8, places=6)
        self.assertAlmostEqual(float(result['inner_unknown_similarity'][-1]), .8, places=6)
        np.testing.assert_array_equal(vectors, original)
        self.assertIn('not OOF', result['provenance']['metric_scope'])

    def test_fit_query_overlap_fails(self):
        contract, vectors, valid = fixture()
        contract['roles'][0]['encoder_fit_allowed'] = True
        with self.assertRaisesRegex(ValueError, 'roles'):
            heldout_final_scores(contract, vectors, valid)

    def test_group_role_split_fails(self):
        contract, vectors, valid = fixture()
        contract['roles'][1].update(calibration_query=False, encoder_fit_allowed=True, enrollment_allowed=True)
        with self.assertRaisesRegex(ValueError, 'content group'):
            heldout_final_scores(contract, vectors, valid)

    def test_index_rule_is_fixed(self):
        contract, vectors, valid = fixture()
        with self.assertRaisesRegex(ValueError, 'fixes fold 0'):
            heldout_final_scores(contract, vectors, valid, encoder_fold=1)

    def test_zero_query_and_dirty_invalid_cache_fail(self):
        contract, vectors, valid = fixture()
        bad = deepcopy(contract)
        bad['roles'][-1]['calibration_query'] = True
        with self.assertRaises(ValueError):
            heldout_final_scores(bad, vectors, valid)
        vectors[-1, 0] = 1
        with self.assertRaises(ValueError):
            heldout_final_scores(contract, vectors, valid)

    def test_group_cannot_cross_outer_roles(self):
        contract, vectors, valid = fixture()
        contract['folds'][3]['group_id'] = 'aq'
        contract['roles'][3]['group_id'] = 'aq'
        with self.assertRaisesRegex(ValueError, 'content group'):
            heldout_final_scores(contract, vectors, valid)


if __name__ == '__main__':
    unittest.main()
