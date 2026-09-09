"""Synthetic sklearn/NumPy parity and corrupt-asset tests for S012 trees."""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.inference.decision_trees import feature_matrix, predict_export, validate_export
from speaker_id.postprocessing.tree_models import MODEL_SPECS, SEED, fit_export

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def tiny_payload():
    return {
        "schema_version": 1, "id": "synthetic", "family": "decision_tree",
        "n_features": 2, "classes": [0, 1], "input_dtype": "float32",
        "aggregation": "single", "base_margin": 0.0, "learning_rate": 1.0,
        "feature_importances": [1.0, 0.0], "metadata": {},
        "trees": [{"children_left": [1, -1, -1], "children_right": [2, -1, -1],
                   "feature": [0, -2, -2], "threshold": [0.15, -2.0, -2.0],
                   "value": [0.5, 0.2, 0.8]}],
    }


def native_model(spec):
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    constructors = {"decision_tree": DecisionTreeClassifier,
                    "random_forest": RandomForestClassifier,
                    "extra_trees": ExtraTreesClassifier,
                    "gradient_boosting": GradientBoostingClassifier}
    params = dict(spec["params"], random_state=SEED)
    if spec["family"] == "gradient_boosting":
        params.update(loss="log_loss", subsample=1.0, n_iter_no_change=None)
    else:
        params["class_weight"] = None
    return constructors[spec["family"]](**params)


class PortableTreeTests(unittest.TestCase):
    def test_single_mean_and_stable_sigmoid(self):
        payload = tiny_payload()
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(predict_export(payload, X), [0.2, 0.8])
        for family in ("random_forest", "extra_trees"):
            forest = deepcopy(payload)
            forest.update(family=family, aggregation="mean")
            other = deepcopy(payload["trees"][0])
            other["value"] = [0.5, 0.4, 0.6]
            forest["trees"].append(other)
            np.testing.assert_allclose(predict_export(forest, X), [0.3, 0.7],
                                       rtol=0, atol=1e-16)
        boost = deepcopy(payload)
        boost.update(family="gradient_boosting", aggregation="additive_logit",
                     base_margin=0.25, learning_rate=0.5)
        boost["trees"][0]["value"] = [0.0, -2000.0, 2000.0]
        np.testing.assert_array_equal(predict_export(boost, X), [0.0, 1.0])

    def test_float32_comparison_is_observable_at_boundary(self):
        payload = tiny_payload()
        threshold = payload["trees"][0]["threshold"][0]
        probe = np.array([[threshold, 0.0]], dtype=np.float64)
        self.assertTrue(probe[0, 0] <= threshold)
        # Convert the scalar back to Python float: NumPy 2 scalar promotion
        # would otherwise cast the Python threshold down to float32 too.
        self.assertFalse(float(np.float32(probe[0, 0])) <= threshold)
        np.testing.assert_array_equal(predict_export(payload, probe), [0.8])
        empty = predict_export(payload, np.empty((0, 2)))
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.float64)

    def test_reject_malformed_payloads(self):
        cases = {}
        for name, key, value in [
            ("schema", "schema_version", True),
            ("classes", "classes", [1, 0]),
            ("boolean_classes", "classes", [False, True]),
            ("width", "n_features", True),
            ("dtype", "input_dtype", "float64"),
            ("family", "family", "unknown"),
            ("aggregation", "aggregation", "mean"),
            ("base", "base_margin", 1),
            ("rate", "learning_rate", float("inf")),
            ("huge_rate", "learning_rate", 1 << 4096),
            ("no_trees", "trees", []),
            ("importance_sum", "feature_importances", [0.1, 0.1]),
            ("nonjson", "metadata", {"value": object()}),
            ("nan_metadata", "metadata", {"value": float("nan")}),
        ]:
            malformed = tiny_payload()
            malformed[key] = value
            cases[name] = malformed
        extra_key = tiny_payload()
        extra_key["unexpected"] = True
        cases["extra_key"] = extra_key
        for name, field, index, value in [
            ("out_of_range", "children_left", 0, 3),
            ("fractional_index", "children_left", 0, 1.0),
            ("huge_index", "children_left", 0, 1 << 100),
            ("same_children", "children_right", 0, 1),
            ("half_leaf", "children_left", 0, -1),
            ("self_cycle", "children_left", 0, 0),
            ("bad_feature", "feature", 0, 2),
            ("bad_leaf_feature", "feature", 1, 0),
            ("bad_leaf_threshold", "threshold", 1, 0),
            ("infinite_threshold", "threshold", 0, float("inf")),
            ("nonfinite_output", "value", 1, float("nan")),
            ("negative_probability", "value", 1, -0.01),
            ("too_large_probability", "value", 1, 1.01),
        ]:
            malformed = tiny_payload()
            malformed["trees"][0][field][index] = value
            cases[name] = malformed
        unaligned = tiny_payload()
        unaligned["trees"][0]["value"].pop()
        cases["unaligned"] = unaligned
        disconnected = tiny_payload()
        disconnected["trees"] = [{
            "children_left": [1, -1, -1, 4, 3, -1, -1],
            "children_right": [2, -1, -1, 5, 6, -1, -1],
            "feature": [0, -2, -2, 0, 0, -2, -2],
            "threshold": [0.1, -2.0, -2.0, 0.1, 0.1, -2.0, -2.0],
            "value": [0.5] * 7,
        }]
        cases["disconnected_cycle_with_valid_parent_counts"] = disconnected
        for name, malformed in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                predict_export(malformed, [[0.0, 0.0]])

    def test_reject_bad_features_before_and_after_cast(self):
        for X in ([1, 2], [[0, 1, 2]], [[float("nan"), 0]],
                  [[float("inf"), 0]], [[1e100, 0]], [["0", "1"]],
                  [[complex(0, 1), 0]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                predict_export(tiny_payload(), X)
        for X in (np.empty((4, 0)), np.array(3)):
            with self.assertRaises(ValueError):
                feature_matrix(X)

    def test_no_sklearn_import_needed_for_inference(self):
        import builtins
        real_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise AssertionError("Portable inference imported sklearn")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            output = predict_export(json.loads(json.dumps(tiny_payload())), [[0, 0]])
        np.testing.assert_array_equal(output, [0.2])

    def test_fitter_rejects_target_or_spec_without_library(self):
        X = np.zeros((4, 2))
        for target in ([0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 0],
                       [0, 1, float("nan"), 0], [0, 1], [[0], [1], [0], [1]]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                fit_export(MODEL_SPECS[0], X, target)
        spec = deepcopy(MODEL_SPECS[0])
        spec["params"]["max_depth"] = 99
        with self.assertRaises(ValueError):
            fit_export(spec, X, [0, 1, 0, 1])


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn 1.8.0 readiness installation is required")
class SklearnExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(818)
        cls.X = rng.normal(size=(400, 28))
        signal = cls.X[:, 0] + 0.5 * cls.X[:, 1] * cls.X[:, 2] + 0.2 * rng.normal(size=400)
        cls.y = np.zeros(400, dtype=np.int64)
        cls.y[np.argsort(signal)[200:]] = 1
        cls.check = rng.normal(size=(91, 28))
        cls.exports = {}
        cls.native = {}
        for spec in MODEL_SPECS:
            cls.exports[spec["id"]] = fit_export(spec, cls.X, cls.y, check_features=cls.check)
            native = native_model(spec)
            native.fit(cls.X, cls.y, sample_weight=np.ones(400))
            cls.native[spec["id"]] = native

    def test_all_six_real_estimators_match_after_json_roundtrip(self):
        for spec in MODEL_SPECS:
            with self.subTest(model=spec["id"]):
                payload = self.exports[spec["id"]]
                native = self.native[spec["id"]]
                serialized = json.loads(json.dumps(payload, allow_nan=False))
                for X in (self.X, self.check):
                    portable = predict_export(serialized, X)
                    np.testing.assert_allclose(portable, native.predict_proba(X)[:, 1],
                                               atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(portable > 0.5,
                                                  native.predict_proba(X)[:, 1] > 0.5)
                    self.assertEqual(portable.dtype, np.float64)
                self.assertEqual(payload["metadata"]["class_counts"], [200, 200])
                self.assertEqual(payload["metadata"]["weighted_class_mass"], [200, 200])
                self.assertEqual(payload["metadata"]["training_rows"], 400)
                self.assertEqual(payload["n_features"], 28)
                self.assertEqual(payload["metadata"]["sklearn_version"], "1.8.0")
                self.assertTrue(payload["metadata"]["export_parity"]["check_features"]["performed"])
                self.assertEqual(payload["metadata"]["export_parity"]["check_features"]["rows"], 91)
                validate_export(serialized)

    def test_tree_threshold_neighbors_match_native_float32_semantics(self):
        # Probes around every split exercise both float64 rounding and adjacent
        # float32 values; the separate one-split fixture below ensures routing
        # actually reaches a nonrepresentable threshold.
        for spec in MODEL_SPECS:
            payload = self.exports[spec["id"]]
            probes = []
            for tree in payload["trees"]:
                for feature, threshold in zip(tree["feature"], tree["threshold"]):
                    if feature < 0:
                        continue
                    f32 = np.float32(threshold)
                    neighbors = [np.nextafter(f32, np.float32(-np.inf)), f32,
                                 np.nextafter(f32, np.float32(np.inf)),
                                 np.nextafter(threshold, -np.inf), threshold,
                                 np.nextafter(threshold, np.inf)]
                    for neighbor in neighbors:
                        row = np.zeros(28)
                        row[feature] = neighbor
                        probes.append(row)
            with self.subTest(model=spec["id"]):
                if probes:
                    X = np.asarray(probes)
                    np.testing.assert_allclose(predict_export(payload, X),
                                               self.native[spec["id"]].predict_proba(X)[:, 1],
                                               rtol=0, atol=1e-12)
        X = np.zeros((400, 28))
        X[:200, 0], X[200:, 0] = 0.1, 0.2
        y = np.repeat([0, 1], 200)
        payload = fit_export(MODEL_SPECS[0], X, y)
        native = native_model(MODEL_SPECS[0]).fit(X, y, sample_weight=np.ones(400))
        threshold = payload["trees"][0]["threshold"][0]
        probe = np.zeros((3, 28))
        probe[:, 0] = [np.nextafter(threshold, -np.inf), threshold,
                       np.nextafter(threshold, np.inf)]
        expected = native.predict_proba(probe)[:, 1]
        np.testing.assert_array_equal(predict_export(payload, probe), expected)
        naive_without_float32 = (probe[:, 0] > threshold).astype(float)
        self.assertTrue(np.any(naive_without_float32 != expected))

    def test_unbalanced_target_weights_have_equal_binary_mass(self):
        X, y = self.X.copy(), np.repeat([0, 1], [320, 80])
        payload = fit_export(MODEL_SPECS[0], X, y)
        np.testing.assert_array_equal(X, self.X)
        self.assertEqual(payload["metadata"]["class_counts"], [320, 80])
        np.testing.assert_allclose(payload["metadata"]["class_sample_weights"], [0.625, 2.5],
                                   atol=0, rtol=0)
        self.assertEqual(payload["metadata"]["weighted_class_mass"], [200, 200])
        self.assertIsNone(payload["metadata"]["fit_params"]["class_weight"])
        self.assertFalse(payload["metadata"]["export_parity"]["check_features"]["performed"])

    def test_check_features_do_not_change_the_fitted_export(self):
        spec = MODEL_SPECS[0]
        without = fit_export(spec, self.X, self.y)
        with_check = self.exports[spec["id"]]
        for key in without:
            if key != "metadata":
                self.assertEqual(without[key], with_check[key])
        for key in without["metadata"]:
            if key != "export_parity":
                self.assertEqual(without["metadata"][key], with_check["metadata"][key])


if __name__ == "__main__":
    unittest.main()
