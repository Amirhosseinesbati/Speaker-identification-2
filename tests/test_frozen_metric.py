"""Independent small-matrix checks for group-balanced frozen metrics."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.postprocessing.frozen_metric import METRIC_SPECS, fit_transform, transform


def fixture():
    values = np.array([
        [1.0, 0.0], [0.8, 0.6], [0.0, 1.0],   # A has two groups, one with two distinct rows
        [-1.0, 0.0], [-0.6, -0.8], [0.6, -0.8],  # B has three groups
        [0.6, 0.8],                            # C singleton contributes mean, not covariance
        [500.0, -400.0], [-200.0, 900.0],       # unrelated unknowns
    ], dtype=np.float32)
    labels = ["A"] * 3 + ["B"] * 3 + ["C"] + ["unknown"] * 2
    groups = ["a0", "a0", "a1", "b0", "b1", "b2", "c0", "u0", "u1"]
    return values, labels, groups


def scalar_scatter(values, labels, groups):
    """Scalar grouping/covariance oracle, independent of NumPy covariance/eigh."""
    gathered = {}
    for value, label, group in zip(values, labels, groups):
        if label == "unknown" or not any(value):
            continue
        gathered.setdefault((label, group), set()).add(tuple(float(x) for x in value))
    by_class = {}
    for (label, _), vectors in sorted(gathered.items()):
        normalized = []
        for vector in sorted(vectors):
            length = math.sqrt(sum(x * x for x in vector))
            normalized.append([x / length for x in vector])
        center = [sum(v[k] for v in normalized) / len(normalized) for k in range(2)]
        length = math.sqrt(sum(x * x for x in center))
        by_class.setdefault(label, []).append([x / length for x in center])
    means = {}
    covariance = [[0.0, 0.0], [0.0, 0.0]]
    contributors = 0
    for label, vectors in sorted(by_class.items()):
        center = [sum(v[k] for v in vectors) / len(vectors) for k in range(2)]
        means[label] = center
        if len(vectors) >= 2:
            contributors += 1
            for i in range(2):
                for j in range(2):
                    covariance[i][j] += sum((v[i] - center[i]) * (v[j] - center[j])
                                            for v in vectors) / (len(vectors) - 1)
    mean = [sum(m[k] for m in means.values()) / len(means) for k in range(2)]
    covariance = [[v / contributors for v in row] for row in covariance]
    return np.array(mean), np.array(covariance)


def analytic_symmetric_power_2d(matrix, power):
    """Closed-form spectral matrix function; no eigenvector routine."""
    a, b, c = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[1, 1])
    gap = math.sqrt((a - c) ** 2 + 4 * b * b)
    low, high = ((a + c - gap) / 2, (a + c + gap) / 2)
    if gap == 0:
        return low ** (-power) * np.eye(2)
    return (low ** (-power) * np.eye(2)
            + (high ** (-power) - low ** (-power)) / gap * (matrix - low * np.eye(2)))


class FrozenMetricTests(unittest.TestCase):
    def test_fixed_recipe_registry(self):
        self.assertEqual(len(METRIC_SPECS), 6)
        self.assertEqual([x["kind"] for x in METRIC_SPECS[:2]], ["identity", "centering"])
        self.assertEqual({(x["shrinkage"], x["power"]) for x in METRIC_SPECS[2:]},
                         {(0.5, 0.25), (0.5, 0.5), (0.9, 0.25), (0.9, 0.5)})

    def test_class_balanced_scatter_matches_scalar_closed_form_oracle(self):
        values, labels, groups = fixture()
        expected_mean, W = scalar_scatter(values, labels, groups)
        nu = float(np.trace(W) / 2)
        for spec in METRIC_SPECS[2:]:
            with self.subTest(spec=spec["id"]):
                payload = fit_transform(values, labels, groups, spec)
                regularized = (1 - spec["shrinkage"]) * W + spec["shrinkage"] * nu * np.eye(2)
                expected = analytic_symmetric_power_2d(regularized, spec["power"])
                np.testing.assert_allclose(payload["mean"], expected_mean, rtol=0, atol=2e-16)
                np.testing.assert_allclose(payload["matrix"], expected, rtol=0, atol=3e-14)
                meta = payload["metadata"]
                self.assertEqual(meta["within_contributing_labels"], ["A", "B"])
                self.assertEqual(meta["known_labels"], ["A", "B", "C"])
                self.assertEqual(meta["diagnostics"]["residual_degrees_of_freedom"], 3)
                self.assertAlmostEqual(meta["diagnostics"]["within_trace"], float(np.trace(W)), places=14)
                self.assertEqual(meta["known_group_count"], 6)
                json.dumps(meta, allow_nan=False)

    def test_rotation_covariance_and_transformed_geometry(self):
        values, labels, groups = fixture()
        rotation = np.array([[0, -1], [1, 0]], dtype=np.float64)
        rotated = (values.astype(np.float64) @ rotation.T).astype(np.float32)
        for spec in METRIC_SPECS[1:]:
            with self.subTest(spec=spec["id"]):
                original = fit_transform(values, labels, groups, spec)
                changed = fit_transform(rotated, labels, groups, spec)
                np.testing.assert_allclose(changed["mean"], rotation @ original["mean"], atol=2e-16, rtol=0)
                np.testing.assert_allclose(changed["matrix"],
                                           rotation @ original["matrix"] @ rotation.T, atol=3e-14, rtol=0)
                np.testing.assert_allclose(transform(changed, rotated),
                                           transform(original, values) @ rotation.T, atol=2e-7, rtol=0)

    def test_duplicate_file_and_input_order_cannot_reweight_groups(self):
        values, labels, groups = fixture()
        # Duplicate only one of a0's two distinct vectors; a plain file mean
        # would change its direction, whereas a deduplicated group mean cannot.
        expanded = np.concatenate([values, values[[0, 0, 3, 3, 3]]])
        expanded_labels = labels + [labels[i] for i in (0, 0, 3, 3, 3)]
        expanded_groups = groups + [groups[i] for i in (0, 0, 3, 3, 3)]
        order = np.random.default_rng(310).permutation(len(expanded))
        for spec in METRIC_SPECS:
            original = fit_transform(values, labels, groups, spec)
            repeated = fit_transform(expanded[order], [expanded_labels[i] for i in order],
                                     [expanded_groups[i] for i in order], spec)
            with self.subTest(spec=spec["id"]):
                np.testing.assert_array_equal(original["mean"], repeated["mean"])
                np.testing.assert_array_equal(original["matrix"], repeated["matrix"])
                self.assertEqual(original["metadata"]["canonical_fit_groups_sha256"],
                                 repeated["metadata"]["canonical_fit_groups_sha256"])
                self.assertNotEqual(original["metadata"]["input_rows"], repeated["metadata"]["input_rows"])

    def test_unknown_and_zero_known_rows_do_not_enter_supervised_statistics(self):
        values, labels, groups = fixture()
        for spec in METRIC_SPECS:
            base = fit_transform(values[:7], labels[:7], groups[:7], spec)
            augmented = fit_transform(np.r_[values, np.zeros((1, 2), dtype=np.float32)],
                                      labels + ["D"], groups + ["d_zero"], spec)
            altered_unknown = values.copy()
            altered_unknown[-2:] *= np.float32(-1e10)
            changed = fit_transform(altered_unknown, labels, groups, spec)
            with self.subTest(spec=spec["id"]):
                for key in ("mean", "matrix"):
                    np.testing.assert_array_equal(base[key], augmented[key])
                    np.testing.assert_array_equal(base[key], changed[key])
                self.assertEqual(augmented["metadata"]["unknown_rows_ignored"], 2)
                self.assertEqual(augmented["metadata"]["known_zero_rows_ignored"], 1)
                self.assertNotIn("D", augmented["metadata"]["known_labels"])
                self.assertEqual(base["metadata"]["canonical_fit_groups_sha256"],
                                 augmented["metadata"]["canonical_fit_groups_sha256"])

    def test_string_and_integer_unknown_conventions_match(self):
        values, labels, groups = fixture()
        mapping = {"unknown": 0, "A": 1, "B": 2, "C": 3}
        integer_labels = np.array([mapping[x] for x in labels], dtype=np.int64)
        string = fit_transform(values, labels, groups, METRIC_SPECS[2])
        integer = fit_transform(values, integer_labels, groups, METRIC_SPECS[2])
        np.testing.assert_array_equal(string["mean"], integer["mean"])
        np.testing.assert_array_equal(string["matrix"], integer["matrix"])
        self.assertEqual(integer["metadata"]["unknown_label"], 0)
        self.assertEqual(integer["metadata"]["within_contributing_labels"], [1, 2])

    def test_transform_normalizes_float32_and_preserves_exact_zero(self):
        values, labels, groups = fixture()
        queries = np.array([[3, 4], [0, 0], [-8, 1]], dtype=np.float32)
        for spec in METRIC_SPECS:
            payload = fit_transform(values, labels, groups, spec)
            before = queries.copy()
            result = transform(payload, queries)
            with self.subTest(spec=spec["id"]):
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result[1], np.zeros(2, dtype=np.float32))
                np.testing.assert_allclose(np.linalg.norm(result[[0, 2]].astype(np.float64), axis=1),
                                           [1, 1], rtol=0, atol=6e-8)
                np.testing.assert_array_equal(queries, before)
                self.assertEqual(transform(payload, np.empty((0, 2), dtype=np.float32)).shape, (0, 2))
        identity = fit_transform(values, labels, groups, METRIC_SPECS[0])
        np.testing.assert_array_equal(transform(identity, queries)[0], np.array([.6, .8], dtype=np.float32))

    def test_rank_deficient_scatter_has_finite_positive_shrunk_metric(self):
        values = np.zeros((6, 12), dtype=np.float32)
        values[:, 0] = 1
        for pair in range(3):
            values[2 * pair, pair + 1] = np.float32(2 ** -20)
            values[2 * pair + 1, pair + 1] = np.float32(-(2 ** -20))
        labels = ["A", "A", "B", "B", "C", "C"]
        groups = [f"g{i}" for i in range(6)]
        for spec in METRIC_SPECS[2:]:
            payload = fit_transform(values, labels, groups, spec)
            with self.subTest(spec=spec["id"]):
                diagnostics = payload["metadata"]["diagnostics"]
                self.assertLess(diagnostics["within_numeric_rank"], 12)
                self.assertEqual(diagnostics["residual_degrees_of_freedom"], 3)
                self.assertGreater(diagnostics["regularized_min_eigenvalue"], 0)
                self.assertTrue(np.isfinite(payload["matrix"]).all())
                self.assertTrue(np.all(np.linalg.eigvalsh(payload["matrix"]) > 0))
                self.assertTrue(np.isfinite(transform(payload, values)).all())

    def test_reject_degenerate_fit_and_nonzero_centered_query(self):
        same = np.array([[1, 0], [1, 0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "variance"):
            fit_transform(same, ["A", "A"], ["g0", "g1"], METRIC_SPECS[2])
        with self.assertRaisesRegex(ValueError, "two independent"):
            fit_transform(same, ["A", "B"], ["g0", "g1"], METRIC_SPECS[2])
        opposite = np.array([[1, 0], [-1, 0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "cancels"):
            fit_transform(opposite, ["A", "A"], ["g0", "g0"], METRIC_SPECS[1])
        center = fit_transform(same, ["A", "A"], ["g0", "g1"], METRIC_SPECS[1])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            transform(center, same)
        # Exact invalid zeros bypass centering; they do not become -mean.
        np.testing.assert_array_equal(transform(center, np.zeros((2, 2), dtype=np.float32)),
                                      np.zeros((2, 2), dtype=np.float32))

    def test_reject_conflicting_groups_bad_shapes_types_and_corrupt_payload(self):
        values, labels, groups = fixture()
        wrong_groups = groups.copy()
        wrong_groups[3] = "a0"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            fit_transform(values, labels, wrong_groups, METRIC_SPECS[2])
        for bad_values in (values.astype(np.float64), values[:, 0],
                           np.full_like(values, np.inf)):
            with self.assertRaises(ValueError):
                fit_transform(bad_values, labels, groups, METRIC_SPECS[2])
        with self.assertRaises(ValueError):
            fit_transform(values, labels[:-1], groups, METRIC_SPECS[2])
        with self.assertRaises(ValueError):
            fit_transform(values, [True] * len(values), groups, METRIC_SPECS[2])
        payload = fit_transform(values, labels, groups, METRIC_SPECS[2])
        corrupt = deepcopy(payload)
        corrupt["matrix"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "receipt"):
            transform(corrupt, values)
        corrupt = deepcopy(payload)
        corrupt["metadata"]["spec"]["power"] = 0.75
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            transform(corrupt, values)
        with self.assertRaises(ValueError):
            transform(payload, np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
