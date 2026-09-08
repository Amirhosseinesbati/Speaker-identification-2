"""S012 source guards, score-scope separation, safe archives and portability gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.postprocessing import decision_suite as suite


class Tracker:
    def __init__(self):
        self.paths = []

    def add_artifact(self, path, relative=None):
        self.paths.append((path, relative))

    def flush(self, **kwargs):
        return True


class DecisionSuiteTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/postprocessing/campp_s012.json").read_text(encoding="utf-8"))

    def test_frozen_grid_and_baseline_pins_reject_protocol_changes(self):
        suite.validate_config(self.config)
        edits = [
            lambda c: c.update(s011_verification_sha256="0" * 64),
            lambda c: c.update(source_package_config_sha256="0" * 64),
            lambda c: c.update(loss_weighting="group_balanced"),
            lambda c: c["promotion"].update(minimum_pooled_gain=0),
            lambda c: c["model_specs"][0]["params"].update(max_depth=20),
            lambda c: c["feature_names"].reverse(),
            lambda c: c["modes"].append({"mode": "full", "band": .1}),
            lambda c: c["baseline_numerical_policy"].update(probability_atol=1e-4),
        ]
        for edit in edits:
            changed = deepcopy(self.config)
            edit(changed)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                suite.validate_config(changed)

    def test_baseline_fallback_meta_score_never_uses_historical_fit_score(self):
        baseline = {"calibration": {"threshold": .21}, "inner_macro_f1_447": .96755}
        selection = {"baseline_meta_macro_f1_447": .91, "baseline_meta_fold_macro_f1_447": [.90, .91, .92]}
        for result in (baseline, {**baseline, "model": None, "meta_selection": {"baseline_retained": True}}):
            observed = suite._selection_fields(result, baseline, selection)
            self.assertEqual(observed["selection_meta_macro_f1_447"], .91)
            self.assertEqual(observed["historical_full_pool_baseline_calibration_macro_f1_447"], .96755)
            self.assertTrue(observed["baseline_retained"])
        result = {"model": {"placeholder": True}, "meta_selection": {"meta_macro_f1_447": .915,
            "meta_fold_macro_f1_447": [.913, .914, .918]}}
        observed = suite._selection_fields(result, baseline, selection)
        self.assertEqual(observed["selection_meta_macro_f1_447"], .915)
        self.assertFalse(observed["baseline_retained"])

    def test_archive_hashes_actual_portable_json_bytes_and_rejects_unsafe_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {"trees": [{"value": [.1, .2]}], "classes": [0, 1]}
            receipt = suite._zip_json(directory / "models.zip", {"final/model/scores_only.json": payload})
            duplicate = suite._zip_json(directory / "duplicate.zip", {"final/model/scores_only.json": payload})
            self.assertEqual(receipt["sha256"], duplicate["sha256"])
            with zipfile.ZipFile(directory / "models.zip") as archive:
                data = archive.read("final/model/scores_only.json")
                self.assertEqual(json.loads(data), payload)
                self.assertEqual(hashlib.sha256(data).hexdigest(), receipt["members"]["final/model/scores_only.json"]["sha256"])
            with self.assertRaises(ValueError):
                suite._zip_json(directory / "escape.zip", {"../model.json": payload})
            with self.assertRaises(ValueError):
                suite._zip_json(directory / "nan.zip", {"model.json": {"x": float("nan")}})
            with self.assertRaises(TypeError):
                suite._zip_json(directory / "object.zip", {"model.json": {"x": object()}})

    def test_array_artifacts_forbid_pickle_and_nonfinite_features(self):
        for invalid in (np.asarray([object()]), np.asarray([np.nan]), np.asarray([np.inf])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                suite._array_fields({"features": invalid})
        good = {"groups": np.asarray(["group-a", "group-b"]), "truth": np.asarray([0, 1]),
            "features": np.ones((2, 28)), "notes": "kept separately"}
        self.assertEqual(set(suite._array_fields(good)), {"groups", "truth", "features"})

    def test_cpu_feature_threshold_sensitivity_blocks_promotion_without_rewriting_oof(self):
        from speaker_id.postprocessing.decision_scoring import decision_probabilities
        known = np.asarray([[.8, .4], [.7, .5]], dtype=np.float32)
        cuda_features = np.zeros((2, 28), dtype=np.float64)
        cpu_features = cuda_features.copy()
        cpu_features[0, 0] = 1e-7
        common = {"feature_names": self.config["feature_names"], "indices": np.asarray([5, 9]),
            "known_scores": known, "margin": np.asarray([.03, -.03]), "valid": np.ones(2, dtype=bool)}
        cpu = {**common, "features": cpu_features}
        cuda = {**common, "features": cuda_features}
        original = decision_probabilities(known, np.asarray([.3, -.3]), common["valid"])
        search = {"outer_features": cuda, "results": {"decision_tree": {
            "model": {"unused_by_mock": True}, "policy": {"feature_set": "scores_only", "mode": "full", "band": None, "threshold": .5},
            "scores": {"outer_tree_confidence": np.asarray([.8, .2])}, "probabilities": original},
            "overall": {"model": None}}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("speaker_id.postprocessing.decision_scoring.decision_features", return_value=cpu) as builder, \
                patch("speaker_id.inference.decision_trees.predict_export", return_value=np.asarray([.2, .2])):
            report = suite._deployment_parity(Path(temporary), {}, {}, search, Tracker(), 0)
            builder.assert_called_once_with({}, {}, "outer", device="cpu")
            self.assertTrue(report["families"]["decision_tree"]["promotion_blocked"])
            self.assertEqual(report["families"]["decision_tree"]["decision_disagreements"], 1)
            self.assertEqual(report["families"]["decision_tree"]["disagreement_indices"], [5])
            self.assertFalse(report["families"]["overall"]["promotion_blocked"])
            self.assertTrue(np.array_equal(original, search["results"]["decision_tree"]["probabilities"]))
            with np.load(Path(temporary) / "deployment_cpu_arrays.npz", allow_pickle=False) as arrays:
                self.assertTrue(np.array_equal(arrays["cpu_indices"], [5, 9]))
                self.assertEqual(arrays["decision_tree_cpu_probabilities"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
