"""Focused synthetic checks for the S033 reference-resampling veto core."""
from dataclasses import replace
import unittest

import numpy as np

from speaker_id.research.s033_reference_resampling import (
    apply_stability_veto,
    frozen_winner_stability,
    make_balanced_resampled_galleries,
)


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class ReferenceResamplingTests(unittest.TestCase):
    def reference_fixture(self):
        # Class 1 contains a reference that supports the false accept and one
        # that does not.  Both support the stable class-1 query on dimension 2.
        references = unit([
            [1.0, 0.0, 0.8], [0.0, 1.0, 0.8],
            [0.7, 0.7, 0.0], [-0.7, 0.7, 0.0],
        ])
        labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
        groups = np.asarray(["a0", "a1", "b0", "b1"], dtype=object)
        galleries = make_balanced_resampled_galleries(
            np.arange(len(references), dtype=np.int64), labels,
            n_resamples=40, per_class=1, seed=77,
        )
        return references, labels, groups, galleries

    def test_resampling_is_seed_deterministic_and_balanced_by_class(self):
        references, labels, _, _ = self.reference_fixture()
        first = make_balanced_resampled_galleries(
            np.arange(len(references)), labels, n_resamples=5, per_class=1, seed=31
        )
        second = make_balanced_resampled_galleries(
            np.arange(len(references)), labels, n_resamples=5, per_class=1, seed=31
        )
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.class_labels, [1, 2])
        for position, label in enumerate(first.class_labels):
            self.assertTrue(np.isin(first.indices[:, position, :], np.flatnonzero(labels == label)).all())

    def test_unstable_false_accept_is_vetoed_while_stable_known_is_retained(self):
        references, _, reference_groups, galleries = self.reference_fixture()
        queries = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        stability = frozen_winner_stability(
            queries, np.asarray([True, True]), np.asarray(["q_false", "q_known"], dtype=object),
            references, np.ones(len(references), dtype=bool), reference_groups, galleries,
            np.asarray([1, 1], dtype=np.int64),
        )
        self.assertGreater(stability[0], 0.0)
        self.assertLess(stability[0], 0.9)
        self.assertEqual(stability[1], 1.0)
        np.testing.assert_array_equal(
            apply_stability_veto(np.asarray([1, 1]), stability, np.asarray([True, True]), minimum_stability=.9),
            [0, 1],
        )

    def test_veto_never_reranks_and_invalid_rows_become_unknown(self):
        baseline = np.asarray([2, 1, 0, 1], dtype=np.int64)
        result = apply_stability_veto(
            baseline, np.asarray([.1, 1., 0., 1.], dtype=np.float64),
            np.asarray([True, True, True, False]), minimum_stability=.5,
        )
        # The first row was speaker 2; veto can only send it to unknown, never
        # promote the runner-up speaker 1.  The invalid final row also fails closed.
        np.testing.assert_array_equal(result, [0, 1, 0, 0])

    def test_invalid_embeddings_and_empty_group_exclusion_are_rejected(self):
        references, labels, reference_groups, galleries = self.reference_fixture()
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            frozen_winner_stability(
                np.asarray([[0., 0., 0.]], dtype=np.float32), np.asarray([True]), np.asarray(["q"], dtype=object),
                references, np.ones(4, dtype=bool), reference_groups, galleries, np.asarray([1]),
            )
        # A complete class is in the same content group as the query.  The
        # scorer must reject rather than quietly leave a self-reference behind.
        grouped = np.asarray(["same", "same", "b0", "b1"], dtype=object)
        full_class_gallery = make_balanced_resampled_galleries(
            np.arange(4), labels, n_resamples=1, per_class=2, seed=4
        )
        with self.assertRaisesRegex(ValueError, "group exclusion leaves an empty class"):
            frozen_winner_stability(
                np.asarray([[1., 0., 0.]], dtype=np.float32), np.asarray([True]), np.asarray(["same"], dtype=object),
                references, np.ones(4, dtype=bool), grouped, full_class_gallery, np.asarray([1]),
            )
        bad = replace(galleries, source_indices=np.asarray([0, 0, 2, 3], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "malformed balanced"):
            frozen_winner_stability(
                np.asarray([[1., 0., 0.]], dtype=np.float32), np.asarray([True]), np.asarray(["q"], dtype=object),
                references, np.ones(4, dtype=bool), reference_groups, bad, np.asarray([1]),
            )


if __name__ == "__main__":
    unittest.main()
