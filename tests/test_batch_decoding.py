"""Portable and label-free batch-decoding tests for S014."""
from dataclasses import asdict, replace
import inspect
import unittest

import numpy as np

from speaker_id.inference.batch_decoding import (
    DistributionAlignmentConfig,
    PoweredRatioAlignmentConfig,
    QueryExpansionConfig,
    build_dual_view_reciprocal_graph,
    distribution_alignment,
    powered_ratio_alignment,
    soft_distribution_alignment,
    uniform_query_expansion,
    validate_alignment_config,
    validate_expansion_config,
    validate_powered_ratio_config,
)


CLASSES = 447


def probability_fixture(rows=96, seed=1401):
    rng = np.random.default_rng(seed)
    logits = rng.normal(-3.0, 1.25, size=(rows, CLASSES))
    logits[:, 0] += rng.normal(3.0, 1.0, size=rows)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    values[0, 10:] = 0
    values[0, :10] /= values[0, :10].sum()
    return values


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class ConfigValidationTests(unittest.TestCase):
    def test_exact_dict_schema_and_types(self):
        alignment = DistributionAlignmentConfig()
        powered = PoweredRatioAlignmentConfig()
        expansion = QueryExpansionConfig()
        self.assertEqual(validate_alignment_config(asdict(alignment)), alignment)
        self.assertEqual(validate_powered_ratio_config(asdict(powered)), powered)
        self.assertEqual(validate_expansion_config(asdict(expansion)), expansion)
        cases = [
            (alignment, validate_alignment_config, "schema_version", True),
            (alignment, validate_alignment_config, "official_unknown_prior", 1),
            (alignment, validate_alignment_config, "strength", -0.1),
            (alignment, validate_alignment_config, "temperature", 0.0),
            (alignment, validate_alignment_config, "epsilon", float("nan")),
            (alignment, validate_alignment_config, "max_iterations", True),
            (alignment, validate_alignment_config, "tolerance", 0.0),
            (powered, validate_powered_ratio_config, "schema_version", True),
            (powered, validate_powered_ratio_config, "unknown_strength", -0.1),
            (powered, validate_powered_ratio_config, "known_strength", 1.1),
            (powered, validate_powered_ratio_config, "epsilon", 0.0),
            (expansion, validate_expansion_config, "k", True),
            (expansion, validate_expansion_config, "fused_cosine_floor", 1.1),
            (expansion, validate_expansion_config, "beta", -0.1),
            (expansion, validate_expansion_config, "block_size", 0),
        ]
        for config, validator, field, bad in cases:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator(replace(config, **{field: bad}))
        missing = asdict(alignment)
        missing.pop("epsilon")
        with self.assertRaises(ValueError):
            validate_alignment_config(missing)
        extra = asdict(expansion)
        extra["labels"] = []
        with self.assertRaises(ValueError):
            validate_expansion_config(extra)

    def test_public_apis_cannot_receive_labels_or_groups(self):
        for function in (distribution_alignment, powered_ratio_alignment,
                         soft_distribution_alignment,
                         build_dual_view_reciprocal_graph,
                         uniform_query_expansion):
            parameters = set(inspect.signature(function).parameters)
            self.assertTrue(parameters.isdisjoint(
                {"truth", "labels", "speaker", "groups"}))


class PoweredRatioAlignmentTests(unittest.TestCase):
    def test_preregistered_formula_and_report_are_exact(self):
        values = probability_fixture(rows=37, seed=314)
        valid = np.ones(len(values), dtype=bool)
        valid[[2, 21]] = False
        # The release contract makes invalid rows deterministic unknown.
        values[~valid] = 0
        values[~valid, 0] = 1
        result = distribution_alignment(values, valid)
        observed = values[valid].mean(axis=0)
        observed_known = observed[1:] / observed[1:].sum()
        expected_ratios = np.r_[0.5 / observed[0],
                                (0.5 / 446) / observed[1:]]
        expected_factors = np.sqrt(expected_ratios)
        expected_factors[0] = 1.0
        weighted = values[valid] * expected_factors
        expected = weighted / weighted.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.observed_prior, observed,
                                   rtol=0, atol=2e-16)
        np.testing.assert_allclose(result.observed_known_conditional,
                                   observed_known, rtol=0, atol=2e-16)
        np.testing.assert_allclose(result.prior_ratios, expected_ratios,
                                   rtol=0, atol=2e-14)
        np.testing.assert_allclose(result.factors, expected_factors,
                                   rtol=0, atol=2e-14)
        np.testing.assert_allclose(result.probabilities[valid], expected,
                                   rtol=0, atol=2e-15)
        np.testing.assert_array_equal(result.probabilities[~valid, 0], 1)
        np.testing.assert_array_equal(result.probabilities[~valid, 1:], 0)
        np.testing.assert_allclose(result.adjusted_prior,
            result.probabilities[valid].mean(axis=0), rtol=0, atol=2e-16)
        self.assertEqual(result.unknown_factor, 1.0)
        self.assertTrue(
            result.unknown_probability_may_change_via_row_normalization)
        self.assertTrue(np.any(result.probabilities[valid, 0]
                               != values[valid, 0]))

    def test_deterministic_permutation_equivariant_and_finite_with_zeros(self):
        values = probability_fixture(rows=61, seed=71)
        valid = np.ones(len(values), dtype=bool)
        valid[::13] = False
        values[~valid] = 0
        values[~valid, 0] = 1
        first = powered_ratio_alignment(values, valid)
        second = powered_ratio_alignment(values.copy(), valid.copy())
        np.testing.assert_array_equal(first.probabilities,
                                      second.probabilities)
        np.testing.assert_array_equal(first.factors, second.factors)
        self.assertTrue(np.isfinite(first.prior_ratios).all())
        self.assertTrue(np.isfinite(first.probabilities).all())
        np.testing.assert_allclose(first.probabilities.sum(axis=1), 1,
                                   rtol=0, atol=2e-15)
        permutation = np.random.default_rng(882).permutation(len(values))
        inverse = np.argsort(permutation)
        shuffled = powered_ratio_alignment(values[permutation],
                                            valid[permutation])
        np.testing.assert_allclose(shuffled.probabilities[inverse],
            first.probabilities, rtol=0, atol=2e-15)
        np.testing.assert_allclose(shuffled.factors, first.factors,
                                   rtol=0, atol=2e-14)

    def test_both_zero_strengths_are_exact_bypass_for_valid_rows(self):
        values = probability_fixture(rows=11).astype(np.float32)
        valid = np.ones(11, dtype=bool)
        config = replace(PoweredRatioAlignmentConfig(),
                         unknown_strength=0.0, known_strength=0.0)
        result = powered_ratio_alignment(values, valid, asdict(config))
        np.testing.assert_array_equal(result.probabilities, values)
        self.assertEqual(result.probabilities.dtype, values.dtype)
        np.testing.assert_array_equal(result.factors, np.ones(CLASSES))
        self.assertEqual(result.unknown_factor, 1.0)

    def test_invalid_rows_are_forced_to_exact_unknown_and_excluded(self):
        values = probability_fixture(rows=7)
        valid = np.array([True, True, False, True, False, True, True])
        reference = powered_ratio_alignment(values[valid],
                                            np.ones(valid.sum(), dtype=bool))
        result = powered_ratio_alignment(values, valid)
        np.testing.assert_array_equal(result.probabilities[~valid, 0], 1)
        np.testing.assert_array_equal(result.probabilities[~valid, 1:], 0)
        np.testing.assert_allclose(result.observed_prior,
                                   reference.observed_prior, rtol=0, atol=0)
        np.testing.assert_allclose(result.factors, reference.factors,
                                   rtol=0, atol=0)


class DistributionAlignmentTests(unittest.TestCase):
    def test_finite_simplex_target_residual_and_invalid_exclusion(self):
        values = probability_fixture()
        valid = np.ones(len(values), dtype=bool)
        valid[[3, 41]] = False
        values[~valid] = 0
        values[~valid, 0] = 1
        original_invalid = values[~valid].copy()
        result = soft_distribution_alignment(values, valid)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 200)
        self.assertLessEqual(result.residual, 1e-8)
        self.assertEqual(result.valid_rows, int(valid.sum()))
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertTrue(np.isfinite(result.duals).all())
        self.assertTrue(np.isfinite(result.residual_trace).all())
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1,
                                   rtol=0, atol=2e-15)
        np.testing.assert_array_equal(result.probabilities[~valid], original_invalid)
        np.testing.assert_allclose(result.achieved_prior,
            result.probabilities[valid].mean(axis=0), rtol=0, atol=2e-16)
        np.testing.assert_allclose(result.achieved_prior, result.target_prior,
                                   rtol=0, atol=1e-8)
        expected = 0.5 * result.observed_prior + 0.5 * result.official_prior
        np.testing.assert_allclose(result.target_prior, expected,
                                   rtol=0, atol=2e-16)
        self.assertAlmostEqual(result.official_prior[0], 0.5)
        np.testing.assert_array_equal(result.official_prior[1:],
                                      np.full(446, 1 / 892))
        self.assertEqual(result.duals[0], 0.0)

    def test_deterministic_and_permutation_equivariant(self):
        values = probability_fixture(rows=73, seed=19)
        valid = np.ones(len(values), dtype=bool)
        valid[::17] = False
        values[~valid] = 0
        values[~valid, 0] = 1
        first = soft_distribution_alignment(values, valid)
        second = soft_distribution_alignment(values.copy(), valid.copy())
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        np.testing.assert_array_equal(first.duals, second.duals)
        np.testing.assert_array_equal(first.residual_trace, second.residual_trace)
        permutation = np.random.default_rng(99).permutation(len(values))
        inverse = np.argsort(permutation)
        shuffled = soft_distribution_alignment(values[permutation],
                                                valid[permutation])
        np.testing.assert_allclose(shuffled.probabilities[inverse],
            first.probabilities, rtol=0, atol=2e-15)
        np.testing.assert_allclose(shuffled.duals, first.duals,
                                   rtol=0, atol=2e-14)

    def test_zero_strength_is_exact_bypass(self):
        values = probability_fixture(rows=9).astype(np.float32)
        valid = np.array([True, False, True, True, False,
                          True, True, True, True])
        config = replace(DistributionAlignmentConfig(), strength=0.0)
        result = soft_distribution_alignment(values, valid, asdict(config))
        np.testing.assert_array_equal(result.probabilities, values)
        self.assertEqual(result.probabilities.dtype, values.dtype)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.residual, 0.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.residual_trace.shape, (0,))
        np.testing.assert_array_equal(result.duals, np.zeros(CLASSES))

    def test_all_invalid_is_a_finite_fixed_batch(self):
        values = probability_fixture(rows=4)
        valid = np.zeros(4, dtype=bool)
        result = soft_distribution_alignment(values, valid)
        np.testing.assert_array_equal(result.probabilities, values)
        self.assertEqual(result.valid_rows, 0)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.observed_prior).all())
        np.testing.assert_array_equal(result.observed_prior,
                                      result.official_prior)

    def test_rejects_wrong_prior_width_and_bad_probability_inputs(self):
        valid = np.ones(3, dtype=bool)
        with self.assertRaises(ValueError):
            soft_distribution_alignment(np.full((3, 4), 0.25), valid)
        bad_cases = [
            (np.ones((3, CLASSES)), valid),
            (np.full((3, CLASSES), 1 / CLASSES),
             np.ones(2, dtype=bool)),
        ]
        nonfinite = np.full((3, CLASSES), 1 / CLASSES)
        nonfinite[0, 1] = np.nan
        bad_cases.append((nonfinite, valid))
        for values, mask in bad_cases:
            with self.subTest(shape=np.asarray(values).shape), self.assertRaises(ValueError):
                soft_distribution_alignment(values, mask)


class ReciprocalGraphTests(unittest.TestCase):
    def test_self_and_nonreciprocal_neighbors_are_excluded(self):
        angles = np.array([0.00, 0.04, 0.15, 1.20])
        view = np.column_stack((np.cos(angles), np.sin(angles)))
        valid = np.ones(4, dtype=bool)
        config = replace(QueryExpansionConfig(), k=1,
                         fused_cosine_floor=-1.0, block_size=2)
        graph = build_dual_view_reciprocal_graph(view, view, valid, config)
        edge_set = {tuple(edge) for edge in graph.edges.tolist()}
        self.assertIn((0, 1), edge_set)
        self.assertNotIn((1, 2), edge_set)  # 2 -> 1, but 1 -> 0.
        self.assertTrue(all(left < right for left, right in edge_set))
        self.assertFalse(any(left == right for left, right in edge_set))

    def test_dual_view_intersection_and_cosine_floor(self):
        a = unit_rows([[1, 0], [.99, .1], [0, 1]])
        b = unit_rows([[1, 0], [0, 1], [.99, .1]])
        valid = np.ones(3, dtype=bool)
        config = replace(QueryExpansionConfig(), k=1,
                         fused_cosine_floor=-1.0)
        graph = build_dual_view_reciprocal_graph(a, b, valid, config)
        self.assertEqual(len(graph.edges), 0)
        pair = unit_rows([[1, 0], [0, 1]])
        floor = build_dual_view_reciprocal_graph(
            pair, pair, np.ones(2, dtype=bool),
            replace(QueryExpansionConfig(), k=1, fused_cosine_floor=0.75))
        self.assertEqual(len(floor.edges), 0)

    def test_invalid_rows_are_excluded_even_when_nonfinite(self):
        view_a = np.array([[1., 0.], [np.nan, np.nan], [.99, .02]])
        view_b = np.array([[0., 1.], [np.inf, 0.], [.01, .99]])
        valid = np.array([True, False, True])
        graph = build_dual_view_reciprocal_graph(
            view_a, view_b, valid,
            replace(QueryExpansionConfig(), k=1,
                    fused_cosine_floor=-1.0, block_size=1))
        np.testing.assert_array_equal(graph.edges, [[0, 2]])
        self.assertEqual(graph.degrees[1], 0)

    def test_graph_is_permutation_equivariant_and_block_invariant(self):
        rng = np.random.default_rng(819)
        a = unit_rows(rng.normal(size=(31, 8)))
        b = unit_rows(rng.normal(size=(31, 5)))
        valid = np.ones(31, dtype=bool)
        valid[[4, 19]] = False
        config = replace(QueryExpansionConfig(), k=3,
                         fused_cosine_floor=-1.0, block_size=4)
        baseline = build_dual_view_reciprocal_graph(a, b, valid, config)
        one_block = build_dual_view_reciprocal_graph(
            a, b, valid, replace(config, block_size=100))
        self.assertEqual({tuple(x) for x in baseline.edges.tolist()},
                         {tuple(x) for x in one_block.edges.tolist()})
        permutation = rng.permutation(len(a))
        shuffled = build_dual_view_reciprocal_graph(
            a[permutation], b[permutation], valid[permutation], config)
        mapped = {tuple(sorted((int(permutation[i]), int(permutation[j]))))
                  for i, j in shuffled.edges}
        expected = {tuple(x) for x in baseline.edges.tolist()}
        self.assertEqual(mapped, expected)


class UniformExpansionTests(unittest.TestCase):
    def test_one_pass_uniform_update_and_fixed_rows(self):
        view = unit_rows([[1, 0], [.98, .2], [.96, -.25], [-1, 0]])
        valid = np.array([True, True, True, False])
        config = replace(QueryExpansionConfig(), k=3,
                         fused_cosine_floor=0.75, beta=0.25, block_size=2)
        graph = build_dual_view_reciprocal_graph(view, view, valid, config)
        result = uniform_query_expansion(view, view.copy(), valid,
                                         graph, config)
        np.testing.assert_array_equal(result.degrees, [2, 2, 2, 0])
        mean = view[[1, 2]].mean(axis=0)
        expected = .75 * view[0] + .25 * mean
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result.view_a[0], expected,
                                   rtol=0, atol=2e-15)
        np.testing.assert_array_equal(result.view_a[3], view[3])
        np.testing.assert_array_equal(
            result.expanded_rows, [True, True, True, False])
        np.testing.assert_allclose(np.linalg.norm(result.view_a[:3], axis=1),
                                   1, rtol=0, atol=2e-15)

    def test_zero_beta_is_exact_bypass(self):
        view_a = unit_rows([[1, 0], [.98, .2], [0, 1]]).astype(np.float32)
        view_b = unit_rows([[0, 1], [.2, .98], [1, 0]]).astype(np.float32)
        valid = np.ones(3, dtype=bool)
        graph_config = replace(QueryExpansionConfig(), k=2,
                               fused_cosine_floor=-1.0)
        graph = build_dual_view_reciprocal_graph(
            view_a, view_b, valid, graph_config)
        bypass = replace(graph_config, beta=0.0)
        result = uniform_query_expansion(
            view_a, view_b, valid, graph, asdict(bypass))
        np.testing.assert_array_equal(result.view_a, view_a)
        np.testing.assert_array_equal(result.view_b, view_b)
        self.assertEqual(result.view_a.dtype, view_a.dtype)
        self.assertEqual(result.view_b.dtype, view_b.dtype)
        self.assertFalse(result.expanded_rows.any())


if __name__ == "__main__":
    unittest.main()
