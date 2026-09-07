import unittest
import numpy as np
from speaker_id.eda.geometry import nearest_excluding_groups, mutual_components


class GeometryTests(unittest.TestCase):
    def test_exact_duplicate_group_excluded_from_nearest(self):
        scores = np.array([[1., 1., .2], [1., 1., .3], [.2, .3, 1.]])
        index, values = nearest_excluding_groups(scores, ["a", "a", "b"])
        np.testing.assert_array_equal(index, [2, 2, 1])
        np.testing.assert_allclose(values, [.2, .3, .3])

    def test_mutual_neighbors_not_transitive_person_count(self):
        scores = np.array([[1., .9, .2], [.9, 1., .8], [.2, .8, 1.]])
        self.assertEqual(mutual_components(scores, [.7])[0]["mutual_nearest_pairs"], 1)
