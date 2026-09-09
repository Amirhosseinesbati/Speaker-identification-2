"""Synthetic audits of reference-isolated metric fitting and frozen selection."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.postprocessing import metric_scoring as metric
from speaker_id.postprocessing.nested_cases import make_meta_plan


class LabelDenied(dict):
    def __getitem__(self, key):
        if key == "speaker_id":
            raise AssertionError("Original outer label was read")
        return super().__getitem__(key)


def fixture():
    rng = np.random.default_rng(1329)
    labels = ["unknown", "a", "b", "c"]
    manifest, folds, vectors = [], [], []
    for label, count in (("a", 5), ("b", 5), ("c", 1), ("unknown", 9)):
        for group in range(count):
            name = f"file_{len(manifest)}.wav"
            manifest.append({"audio_file": name, "speaker_id": label})
            folds.append({"audio_file": name, "speaker_id": label, "fold": "1", "train_eligible": "true",
                "group_id": f"{label}_{group}"})
            value = rng.normal(size=8).astype(np.float32) * .2
            if label != "unknown":
                value[labels.index(label)] += 1
            vectors.append(value)
    train_count = len(manifest)
    for label in labels:
        name = f"file_{len(manifest)}.wav"
        manifest.append(LabelDenied({"audio_file": name, "speaker_id": label}))
        folds.append(LabelDenied({"audio_file": name, "speaker_id": label, "fold": "0", "train_eligible": "true",
            "group_id": "outer_" + label}))
        vectors.append(rng.normal(size=8).astype(np.float32))
    values = np.asarray(vectors, dtype=np.float32)
    valid = np.ones(len(values), dtype=bool)
    plan = make_meta_plan(valid, manifest, folds, labels, 0)
    targets = np.full(len(values), -1, dtype=np.int64)
    for i in plan["reference_global_indices"]:
        targets[i] = labels.index(manifest[int(i)]["speaker_id"])
    prepared = {"classes": 3, "inner_truth": targets[plan["query_global_indices"]],
        "scores_by_alpha": {0.: {"calibration_indices": plan["query_global_indices"],
            "outer_indices": np.arange(train_count, len(values)), "known_labels": labels[1:],
            "provenance": {"reference_indices": plan["reference_global_indices"].tolist()}}}}
    probabilities = np.zeros((len(labels), len(labels)))
    probabilities[:, 0] = 1
    baseline = {"policy": {"id": "baseline", "advanced_weight": .5},
        "calibration": {"threshold": .2, "unknown_weight": .25, "margin_weight": 0., "inner_macro_f1_447": .96},
        "probabilities": probabilities}
    return values, targets, valid, manifest, folds, labels, prepared, baseline, plan


class MetricScoringTests(unittest.TestCase):
    def test_heldout_values_and_targets_cannot_change_fitted_geometry(self):
        values, targets, _, _, _, labels, _, _, plan = fixture()
        entry = plan["cases"][0]
        refs, query = entry["reference_global_indices"], entry["validation_global_indices"]
        spec = metric.METRIC_SPECS[2]
        first = metric._fit_score_case(values, targets, plan["groups"], refs, query, spec, len(labels) - 1)
        changed = values.copy()
        changed[query] *= -3
        changed_targets = targets.copy()
        changed_targets[query] = 100000
        second = metric._fit_score_case(changed, changed_targets, plan["groups"], refs, query, spec, len(labels) - 1)
        np.testing.assert_array_equal(first["payload"]["mean"], second["payload"]["mean"])
        np.testing.assert_array_equal(first["payload"]["matrix"], second["payload"]["matrix"])
        self.assertEqual(first["payload"]["metadata"], second["payload"]["metadata"])
        np.testing.assert_array_equal(first["transformed_references"], second["transformed_references"])
        self.assertFalse(np.array_equal(first["known_scores"], second["known_scores"]))

    def test_unknown_reference_vectors_never_enter_mean_or_covariance(self):
        values, targets, _, _, _, labels, _, _, plan = fixture()
        entry = plan["cases"][0]
        refs, query = entry["reference_global_indices"], entry["validation_global_indices"]
        first = metric._fit_score_case(values, targets, plan["groups"], refs, query, metric.METRIC_SPECS[-1], len(labels) - 1)
        changed = values.copy()
        changed[refs[targets[refs] == 0]] = np.arange(8, dtype=np.float32) + .7
        second = metric._fit_score_case(changed, targets, plan["groups"], refs, query, metric.METRIC_SPECS[-1], len(labels) - 1)
        np.testing.assert_array_equal(first["payload"]["mean"], second["payload"]["mean"])
        np.testing.assert_array_equal(first["payload"]["matrix"], second["payload"]["matrix"])
        self.assertEqual(first["payload"]["metadata"], second["payload"]["metadata"])
        np.testing.assert_array_equal(first["known_scores"], second["known_scores"])
        self.assertFalse(np.array_equal(first["unknown_similarity"], second["unknown_similarity"]))

    def test_group_overlap_or_missing_background_is_rejected(self):
        values, targets, _, _, _, labels, _, _, plan = fixture()
        entry = plan["cases"][0]
        refs, query = entry["reference_global_indices"], entry["validation_global_indices"]
        with self.assertRaisesRegex(ValueError, "heldout content group"):
            metric._fit_score_case(values, targets, plan["groups"], np.r_[refs, query[:1]], query,
                metric.METRIC_SPECS[0], len(labels) - 1)
        with self.assertRaisesRegex(ValueError, "background"):
            metric._fit_score_case(values, targets, plan["groups"], refs[targets[refs] > 0], query,
                metric.METRIC_SPECS[0], len(labels) - 1)

    def test_complete_nested_search_never_reads_outer_labels_and_freezes_before_outer_fits(self):
        values, _, valid, manifest, folds, labels, prepared, baseline, plan = fixture()
        with patch.object(metric, "fit_transform", wraps=metric.fit_transform) as fitted:
            result = metric._prepare_from_values(prepared, baseline, values, valid, manifest, folds, labels, 0)
        self.assertEqual(fitted.call_count, 18)
        self.assertFalse(result["provenance"]["outer_scores_computed"])
        self.assertEqual(len(result["candidate_summary"]), 6)
        self.assertEqual(result["provenance"]["advanced_weight"], .5)
        np.testing.assert_array_equal(result["source"]["query_indices"], plan["query_global_indices"])
        singleton = next(i for i, row in enumerate(manifest[:20]) if row["speaker_id"] == "c")
        self.assertNotIn(singleton, result["source"]["query_indices"])
        for cases in result["cases"].values():
            for case in cases:
                self.assertIn(singleton, case["reference_global_indices"])
                self.assertFalse(set(case["reference_group_ids"]) & set(case["query_group_ids"]))
        with patch.object(metric, "fit_transform", wraps=metric.fit_transform) as fitted:
            output = metric.evaluate_metric_outer(result, baseline)
        self.assertEqual(fitted.call_count, 6)
        self.assertFalse(output["original_outer_labels_read"])
        self.assertEqual(len(output["results"]), 6)
        for item in output["results"].values():
            self.assertEqual(item["probabilities"].shape, (4, 4))
            np.testing.assert_allclose(item["probabilities"].sum(axis=1), 1, rtol=0, atol=1e-15)
        result["selection"]["overall_metric_id"] = "centering_only"
        with self.assertRaisesRegex(ValueError, "selection changed"):
            metric.evaluate_metric_outer(result, baseline)

    def test_full_gate_curves_include_all_folds_and_selected_predictions(self):
        known = np.asarray([[.9, .2], [.2, .8], [.6, .5], [.4, .6], [.95, .2], [.3, .85]], dtype=np.float32)
        unknown = np.asarray([.2, .2, .9, .9, .2, .3], dtype=np.float32)
        truth, folds = np.asarray([1, 2, 0, 0, 1, 2]), np.asarray([0, 1, 2, 0, 1, 2])
        selected, curves, predictions, margin = metric._calibrate_meta(known, unknown, truth, folds, 3)
        self.assertEqual({(r["unknown_weight"], r["margin_weight"]) for r in curves},
            {(u, m) for u in metric.UNKNOWN_WEIGHTS for m in metric.MARGIN_WEIGHTS})
        for row in curves:
            self.assertEqual(len(row["meta_fold_macro_f1_447"]), 3)
        self.assertEqual(metric.macro_f1_indices(truth, predictions, 3), selected["meta_macro_f1_447"])
        np.testing.assert_array_equal(predictions, np.where(margin > 0, known.argmax(axis=1) + 1, 0))

    def test_promotion_uses_identity_control_and_requires_every_fold(self):
        rows = [{"id": spec["id"], "calibration": {"meta_macro_f1_447": .9,
            "meta_fold_macro_f1_447": [.9, .9, .9]}} for spec in metric.METRIC_SPECS]
        selection = metric.select_metric_policy(deepcopy(rows))
        self.assertTrue(selection["baseline_retained"])
        changed = deepcopy(rows)
        changed[1]["calibration"] = {"meta_macro_f1_447": .92, "meta_fold_macro_f1_447": [.91, .897, .95]}
        self.assertTrue(metric.select_metric_policy(changed)["baseline_retained"])
        changed = deepcopy(rows)
        changed[2]["calibration"] = {"meta_macro_f1_447": .905, "meta_fold_macro_f1_447": [.9, .905, .91]}
        selection = metric.select_metric_policy(changed)
        self.assertEqual(selection["overall_metric_id"], metric.METRIC_SPECS[2]["id"])
        self.assertFalse(selection["identity_control_is_historical_baseline"])

    def test_outer_fallback_reuses_exact_historical_probabilities(self):
        values, _, valid, manifest, folds, labels, prepared, baseline, _ = fixture()
        with patch.object(metric, "select_metric_policy", return_value={"overall_metric_id": None, "baseline_retained": True}):
            result = metric._prepare_from_values(prepared, baseline, values, valid, manifest, folds, labels, 0)
        output = metric.evaluate_metric_outer(result, baseline)
        self.assertIs(output["overall"]["probabilities"], baseline["probabilities"])
        self.assertEqual(output["overall"]["policy"], baseline["policy"])
        self.assertTrue(output["historical_fallback_probability_bytes_preserved"])


if __name__ == "__main__":
    unittest.main()
