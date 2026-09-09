"""Focused arithmetic and contract tests for the S017 QMF scorer."""
from __future__ import annotations

import json
import unittest

import numpy as np

from speaker_id.postprocessing.qmf_scoring import (
    SCORE_ONLY_FEATURE_ORDER,
    SCORE_ONLY,
    SCORE_QUALITY,
    SCORE_QUALITY_FEATURE_ORDER,
    fit_qmf_logistic,
    group_equal_binary_balanced_weights,
    is_known_targets,
    predict_qmf_logit,
    qmf_decisions,
    qmf_feature_view,
    qmf_probabilities,
    select_qmf_policy,
    select_qmf_threshold,
)


class QmfScoringTests(unittest.TestCase):
    def test_direct_is_known_target_and_group_equal_binary_balance(self):
        truth = np.asarray([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
        # Known-identity mistakes stay positive: the target does not ask whether
        # the fixed known winner is the correct known identity.
        known_guess = np.asarray([1, 2, 1, 2, 2, 1, 1], dtype=np.int64)
        target = is_known_targets(truth, classes=3)
        np.testing.assert_array_equal(target, [0, 0, 0, 0, 1, 1, 1])
        self.assertNotEqual(known_guess[-1], truth[-1])
        self.assertEqual(target[-1], 1)
        groups = np.asarray(["u0", "u0", "u0", "u1", "k0", "k0", "k1"])
        weights = group_equal_binary_balanced_weights(target, groups)
        self.assertAlmostEqual(weights[target == 0].sum(), len(target) / 2)
        self.assertAlmostEqual(weights[target == 1].sum(), len(target) / 2)
        for left, right in (("u0", "u1"), ("k0", "k1")):
            self.assertAlmostEqual(weights[groups == left].sum(), weights[groups == right].sum())

    def test_group_cannot_mix_binary_targets(self):
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            group_equal_binary_balanced_weights(np.asarray([0, 1]), np.asarray(["same", "same"]))

    def test_fit_is_deterministic_json_safe_and_uses_scale_floor(self):
        primary = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        features = np.column_stack((primary, *[np.full(len(primary), 7.0)
                                               for _ in SCORE_ONLY_FEATURE_ORDER[1:]]))
        target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        groups = np.asarray([f"g{i}" for i in range(len(target))])
        first = fit_qmf_logistic(features, target, groups, SCORE_ONLY_FEATURE_ORDER)
        second = fit_qmf_logistic(features, target, groups, SCORE_ONLY_FEATURE_ORDER)
        self.assertEqual(first, second)
        self.assertEqual(first["target"], "is_known")
        self.assertEqual(first["feature_order"], list(SCORE_ONLY_FEATURE_ORDER))
        self.assertEqual(first["feature_set"], SCORE_ONLY)
        self.assertEqual(first["scale"][1], first["scale_floor"])
        json.dumps(first, allow_nan=False, sort_keys=True)
        logits = predict_qmf_logit(features, first, feature_order=SCORE_ONLY_FEATURE_ORDER)
        self.assertTrue(np.all(np.diff(logits) > 0))

    def test_feature_order_and_nonfinite_values_fail_closed(self):
        x = np.zeros((2, len(SCORE_ONLY_FEATURE_ORDER)), dtype=np.float64)
        x[:, 0] = [-1.0, 1.0]
        model = fit_qmf_logistic(x, np.asarray([0.0, 1.0]), np.asarray(["u", "k"]),
                                 SCORE_ONLY_FEATURE_ORDER)
        with self.assertRaisesRegex(ValueError, "order changed"):
            predict_qmf_logit(x, model, feature_order=list(reversed(SCORE_ONLY_FEATURE_ORDER)))
        with self.assertRaises(TypeError):
            predict_qmf_logit(x, model)  # type: ignore[call-arg]
        bad = x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_qmf_logit(bad, model, feature_order=SCORE_ONLY_FEATURE_ORDER)

    def test_qmf_feature_view_pins_the_nine_and_eleven_canonical_columns(self):
        from speaker_id.postprocessing.decision_scoring import FEATURE_NAMES

        full = np.tile(np.arange(len(FEATURE_NAMES), dtype=np.float64), (3, 1))
        scores, score_order = qmf_feature_view(full, FEATURE_NAMES, SCORE_ONLY)
        quality, quality_order = qmf_feature_view(full, FEATURE_NAMES, SCORE_QUALITY)
        expected_indices = [1, 3, 8, 9, 11, 12, 21, 22, 23]
        np.testing.assert_array_equal(scores[0], expected_indices)
        np.testing.assert_array_equal(quality[0], expected_indices + [26, 27])
        self.assertEqual(score_order, list(SCORE_ONLY_FEATURE_ORDER))
        self.assertEqual(quality_order, list(SCORE_QUALITY_FEATURE_ORDER))
        with self.assertRaisesRegex(ValueError, "schema changed"):
            qmf_feature_view(full, list(reversed(FEATURE_NAMES)), SCORE_ONLY)
        with self.assertRaisesRegex(ValueError, "preregistered"):
            fit_qmf_logistic(full[:, :2], np.asarray([0, 1, 1]),
                             np.asarray(["u", "k0", "k1"]), ["arbitrary", "columns"])

    def test_qmf_only_accepts_or_rejects_fixed_winner_and_invalid_is_unknown(self):
        scores = np.asarray([[0.1, 0.8, 0.2], [0.9, 0.1, 0.2], [0.2, 0.3, 0.7],
                             [0.7, 0.2, 0.1]], dtype=np.float32)
        logits = np.asarray([1.0, -1.0, 0.0, 5.0])
        valid = np.asarray([True, True, True, False])
        expected = np.asarray([2, 0, 0, 0])
        np.testing.assert_array_equal(qmf_decisions(scores, logits, 0.0, valid), expected)
        probabilities = qmf_probabilities(scores, logits, 0.0, valid)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), expected)
        np.testing.assert_array_equal(probabilities[-1], [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12))

    def test_positive_sub_ulp_margin_still_accepts_and_exact_tie_rejects(self):
        scores = np.asarray([[0.8, 0.2], [0.8, 0.2]])
        logits = np.asarray([np.nextafter(0.0, 1.0), 0.0])
        valid = np.asarray([True, True])
        expected = np.asarray([1, 0])
        np.testing.assert_array_equal(qmf_decisions(scores, logits, 0.0, valid), expected)
        np.testing.assert_array_equal(qmf_probabilities(scores, logits, 0.0, valid).argmax(1), expected)

    def test_threshold_uses_fixed_quantiles_and_prefers_fewer_baseline_changes(self):
        logits = np.asarray([-2.0, -1.0, 1.0, 2.0])
        truth = np.asarray([0, 0, 1, 2], dtype=np.int64)
        guess = np.asarray([1, 2, 1, 2], dtype=np.int64)
        valid = np.ones(4, dtype=bool)
        baseline = np.asarray([0, 0, 1, 2], dtype=np.int64)
        selected, curve = select_qmf_threshold(logits, truth, guess, valid, baseline, classes=3)
        self.assertEqual(selected["meta_macro_f1_447"], 1.0)
        self.assertEqual(selected["changed_from_baseline"], 0)
        self.assertGreaterEqual(len(curve), 4)
        self.assertTrue(all(set(row) == {"threshold", "meta_macro_f1_447", "changed_from_baseline"}
                            for row in curve))

    def test_policy_exact_score_ties_prefer_baseline_then_score_only(self):
        truth = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        guess = np.asarray([1, 1, 2, 2, 1, 2], dtype=np.int64)
        valid = np.ones(6, dtype=bool)
        assignments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        perfect = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        logits = np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
        tied = select_qmf_policy(truth, guess, valid, perfect, assignments, [
            {"id": "quality", "feature_set": SCORE_QUALITY, "known_logits": logits},
            {"id": "scores", "feature_set": SCORE_ONLY, "known_logits": logits},
        ], classes=3)
        self.assertEqual(tied["selected"]["id"], "baseline")
        self.assertTrue(tied["baseline_fallback"])

        imperfect = np.asarray([1, 0, 2, 2, 0, 2], dtype=np.int64)
        improved = select_qmf_policy(truth, guess, valid, imperfect, assignments, [
            {"id": "quality", "feature_set": SCORE_QUALITY, "known_logits": logits},
            {"id": "scores", "feature_set": SCORE_ONLY, "known_logits": logits},
        ], classes=3)
        self.assertEqual(improved["selected"]["id"], "scores")
        self.assertFalse(improved["baseline_fallback"])
        self.assertEqual(improved["tie_order"], ["baseline", SCORE_ONLY, SCORE_QUALITY])
        self.assertEqual(len(improved["selected"]["meta_fold_macro_f1_447"]), 3)

    def test_policy_falls_back_when_a_candidate_fails_one_meta_fold(self):
        per_fold = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 10)
        truth = np.tile(per_fold, 3)
        guess = np.where(truth == 0, 1, truth)
        valid = np.ones(len(truth), dtype=bool)
        assignments = np.repeat(np.arange(3, dtype=np.int64), len(per_fold))
        baseline = truth.copy()
        # Six baseline mistakes in folds zero/one are repaired, while one new
        # mistake is introduced in fold two.  The candidate wins pooled but
        # must still fail the explicit per-fold robustness gate.
        baseline[[0, 1, 2, 30, 31, 32]] = [1, 0, 0, 1, 0, 0]
        logits = np.where(truth == 0, -1.0, 1.0)
        logits[60] = 1.0
        result = select_qmf_policy(truth, guess, valid, baseline, assignments, [
            {"id": "unstable", "feature_set": SCORE_ONLY, "known_logits": logits},
        ], classes=3)
        self.assertEqual(result["selected"]["id"], "baseline")
        self.assertTrue(result["baseline_fallback"])
        self.assertFalse(result["candidates"][1]["eligible_for_selection"])
        self.assertGreater(result["candidates"][1]["meta_gain"], 0.001)
        self.assertLess(result["candidates"][1]["meta_fold_delta"][2], -0.002)

    def test_model_schema_rejects_boolean_version_and_feature_set_drift(self):
        x = np.zeros((2, len(SCORE_ONLY_FEATURE_ORDER)), dtype=np.float64)
        x[:, 0] = [-1.0, 1.0]
        model = fit_qmf_logistic(x, np.asarray([0.0, 1.0]), np.asarray(["u", "k"]),
                                 SCORE_ONLY_FEATURE_ORDER)
        for key, value in (("schema_version", True), ("feature_set", SCORE_QUALITY)):
            changed = dict(model)
            changed[key] = value
            with self.assertRaises(ValueError):
                predict_qmf_logit(x, changed, feature_order=SCORE_ONLY_FEATURE_ORDER)

    def test_policy_requires_exact_three_fold_coverage(self):
        with self.assertRaisesRegex(ValueError, "three-fold"):
            select_qmf_policy(
                np.asarray([0, 1, 0, 1]), np.ones(4, dtype=np.int64),
                np.ones(4, dtype=bool), np.asarray([0, 1, 0, 1]),
                np.asarray([0, 0, 1, 1]), [], classes=3,
            )

    def test_malformed_threshold_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly 201"):
            select_qmf_threshold(
                np.asarray([-1.0, 1.0]), np.asarray([0, 1]), np.asarray([1, 1]),
                np.ones(2, dtype=bool), np.asarray([0, 1]), quantiles=101, classes=3,
            )


if __name__ == "__main__":
    unittest.main()
