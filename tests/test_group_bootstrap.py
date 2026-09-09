import hashlib
import json
import unittest

import numpy as np

from speaker_id.evaluation.group_bootstrap import (
    paired_whole_group_cluster_bootstrap,
)


def _macro_f1(truth, prediction, class_count):
    support = np.bincount(truth, minlength=class_count)
    predicted = np.bincount(prediction, minlength=class_count)
    correct = np.bincount(
        truth[truth == prediction], minlength=class_count)
    denominator = support + predicted
    return float(np.divide(
        2.0 * correct, denominator,
        out=np.zeros(class_count, dtype=np.float64),
        where=denominator > 0,
    ).mean())


class WholeContentGroupBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64)
        self.before = np.asarray([0, 1, 1, 1, 2, 0], dtype=np.int64)
        self.after = np.asarray([1, 1, 0, 1, 2, 2], dtype=np.int64)
        self.groups = np.asarray(
            ["mixed", "mixed", "single-0", "single-1", "pair-2", "pair-2"])

    def _run(self, **overrides):
        arguments = {
            "class_count": 3,
            "seed": 1729,
            "replicates": 100,
            "quantiles": (0.05, 0.95),
        }
        arguments.update(overrides)
        return paired_whole_group_cluster_bootstrap(
            self.truth, self.before, self.after, self.groups, **arguments)

    def test_mixed_label_group_is_resampled_as_one_indivisible_cluster(self):
        result = self._run()

        group_rows = {
            "mixed": np.asarray([0, 1]),
            "pair-2": np.asarray([4, 5]),
            "single-0": np.asarray([2]),
            "single-1": np.asarray([3]),
        }
        ordered = sorted(group_rows)
        rng = np.random.Generator(np.random.PCG64(1729))
        expected = []
        for _ in range(100):
            draws = rng.integers(0, len(ordered), size=len(ordered),
                                 dtype=np.int64)
            # Each draw contributes every row of that group.  In particular,
            # the two different true labels in "mixed" always share a draw.
            indices = np.concatenate([
                group_rows[ordered[int(draw)]] for draw in draws])
            expected.append(
                _macro_f1(self.truth[indices], self.after[indices], 3)
                - _macro_f1(self.truth[indices], self.before[indices], 3))

        np.testing.assert_array_equal(
            result["deltas"], np.asarray(expected, dtype=np.float64))
        receipt = result["receipt"]
        self.assertFalse(receipt["label_stratified"])
        self.assertEqual(receipt["sampling"]["groups_drawn_per_replicate"], 4)
        self.assertEqual(receipt["counts"]["mixed_label_groups"], 1)
        self.assertEqual(receipt["counts"]["rows_in_mixed_label_groups"], 2)
        self.assertTrue(receipt["mixed_groups"][
            "preserved_as_indivisible_clusters"])

    def test_repeated_call_and_semantically_identical_row_order_are_deterministic(self):
        first = self._run()
        second = self._run()
        np.testing.assert_array_equal(first["deltas"], second["deltas"])
        self.assertEqual(first["receipt"], second["receipt"])

        order = np.asarray([5, 2, 1, 4, 0, 3])
        reordered = paired_whole_group_cluster_bootstrap(
            self.truth[order], self.before[order], self.after[order],
            self.groups[order], class_count=3, seed=1729, replicates=100,
            quantiles=(0.05, 0.95),
        )
        np.testing.assert_array_equal(first["deltas"], reordered["deltas"])
        self.assertEqual(
            first["receipt"]["hashes"]["semantic_input_sha256"],
            reordered["receipt"]["hashes"]["semantic_input_sha256"],
        )
        self.assertEqual(
            first["receipt"]["receipt_sha256"],
            reordered["receipt"]["receipt_sha256"],
        )

    def test_identical_predictions_have_exact_zero_delta_and_verifiable_hashes(self):
        result = paired_whole_group_cluster_bootstrap(
            self.truth, self.before, self.before.copy(), self.groups,
            class_count=3, seed=3, replicates=100,
            quantiles=[0.025, 0.975],
        )
        np.testing.assert_array_equal(result["deltas"], np.zeros(100))
        self.assertFalse(result["deltas"].flags.writeable)
        receipt = result["receipt"]
        self.assertEqual(
            receipt["confidence_interval"]["lower_delta_macro_f1"], 0.0)
        self.assertEqual(
            receipt["confidence_interval"]["upper_delta_macro_f1"], 0.0)
        delta_hash = hashlib.sha256(
            np.ascontiguousarray(result["deltas"], dtype="<f8").tobytes()
        ).hexdigest()
        self.assertEqual(
            receipt["hashes"]["deltas_float64_le_sha256"], delta_hash)
        expected_receipt_hash = hashlib.sha256(json.dumps(
            {key: value for key, value in receipt.items()
             if key != "receipt_sha256"},
            ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(receipt["receipt_sha256"], expected_receipt_hash)

    def test_malformed_inputs_and_configuration_are_rejected(self):
        valid = (self.truth, self.before, self.after, self.groups)
        malformed_calls = [
            (list(self.truth), *valid[1:]),
            (self.truth.reshape(2, 3), *valid[1:]),
            (self.truth.astype(np.float64), *valid[1:]),
            (np.asarray([-1, 1, 0, 1, 2, 2]), *valid[1:]),
            (np.asarray([3, 1, 0, 1, 2, 2]), *valid[1:]),
            (self.truth, self.before[:-1], self.after, self.groups),
            (self.truth, self.before, self.after, self.groups[:-1]),
            (self.truth, self.before, self.after,
             np.asarray(["mixed", "mixed", "", "x", "y", "z"])),
            (self.truth, self.before, self.after,
             np.asarray(["mixed", "mixed", " x", "x", "y", "z"])),
            (self.truth, self.before, self.after,
             np.asarray(["mixed", "mixed", 1, "x", "y", "z"], dtype=object)),
        ]
        for arguments in malformed_calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    paired_whole_group_cluster_bootstrap(
                        *arguments, class_count=3, seed=1, replicates=100)

        bad_configurations = [
            {"class_count": True}, {"class_count": 0},
            {"seed": True}, {"seed": -1},
            {"replicates": True}, {"replicates": 99},
            {"quantiles": (0.95, 0.05)},
            {"quantiles": (0.0, 0.95)},
            {"quantiles": (0.05, 1.0)},
            {"quantiles": (float("nan"), 0.95)},
            {"quantiles": (0.05,)},
            {"quantiles": "0.05,0.95"},
            {"quantiles": (False, 0.95)},
        ]
        for override in bad_configurations:
            settings = {"class_count": 3, "seed": 1, "replicates": 100,
                        "quantiles": (0.05, 0.95)}
            settings.update(override)
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    paired_whole_group_cluster_bootstrap(*valid, **settings)


if __name__ == "__main__":
    unittest.main()
