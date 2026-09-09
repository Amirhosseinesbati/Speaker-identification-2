"""Focused protocol, sealing and promotion tests for S017."""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.postprocessing import qmf_suite as suite


class QmfSuiteTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "configs/postprocessing/campp_s017_qmf.json").read_text(encoding="utf-8")
        )

    def test_config_pins_source_features_nested_gate_promotion_and_retention(self):
        suite.validate_config(self.config)
        edits = [
            lambda value: value["source"]["artifacts"]["identity_cache_manifest"].update(sha256="0" * 64),
            lambda value: value["feature_sets"]["scores_only"].reverse(),
            lambda value: value["logistic"].update(l2_penalty=.01),
            lambda value: value["nested_validation"].update(meta_assignment_salt="new-split"),
            lambda value: value["nested_validation"].update(minimum_pooled_meta_gain=0.0),
            lambda value: value["promotion"].update(minimum_pooled_macro_f1_delta=0.0),
            lambda value: value["retention"].update(local_transfer="always"),
        ]
        for edit in edits:
            changed = deepcopy(self.config)
            edit(changed)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                suite.validate_config(changed)

    def test_validate_is_read_only_when_server_cache_is_absent(self):
        observed = suite.validate(ROOT, Path("configs/postprocessing/campp_s017_qmf.json"))
        self.assertEqual(observed["status"], "validated_no_experiment_started")
        self.assertFalse(observed["source_present_on_this_host"])
        self.assertFalse(observed["source_cache_transfer_required"])
        self.assertEqual(observed["local_transfer_policy"], "promotion_only")

    def test_success_path_has_no_normal_mlflow_writes_after_parent_finish(self):
        source = inspect.getsource(suite.execute)
        marker = "parent_verification = _finish_verified(parent)"
        self.assertEqual(source.count(marker), 1)
        tail = source.split(marker, 1)[1].split("\n    except BaseException", 1)[0]
        for forbidden in (
            "parent.add_artifact(",
            "parent.write_report(",
            "parent.flush(",
            "parent.log_metrics(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, tail)
        self.assertIn('tracking_terminal_verification.json', tail)

    def test_outer_label_masking_preserves_only_training_truth(self):
        contract = {
            "manifest": [
                {"audio_file": "a.wav", "speaker_id": "known-a"},
                {"audio_file": "b.wav", "speaker_id": "unknown"},
            ],
            "folds": [
                {"audio_file": "a.wav", "fold": "0"},
                {"audio_file": "b.wav", "fold": "1"},
            ],
        }
        fold0 = suite._masked_manifest(contract, 0)
        self.assertEqual(fold0[0]["speaker_id"], "__outer_truth_withheld__")
        self.assertEqual(fold0[1]["speaker_id"], "unknown")
        self.assertEqual(contract["manifest"][0]["speaker_id"], "known-a")

    @staticmethod
    def _probabilities(predictions):
        values = np.zeros((len(predictions), 447), dtype=np.float64)
        values[np.arange(len(predictions)), predictions] = 1.0
        return values

    def test_both_prediction_and_model_bytes_are_bound_by_pretruth_seal(self):
        results = {
            name: {"probabilities": self._probabilities(np.asarray([0, 3])),
                   "policy": {"id": name}, "meta_selection": {"id": name}}
            for name in suite.RECIPES
        }
        search = {
            "results": results,
            "outer_features": {"indices": np.asarray([4, 9])},
            "final_models": {"scores_only": {"schema": 1}},
            "selection": {"selected": {"id": "baseline", "predictions": np.asarray([0, 3]), "curve": []},
                          "candidates": [{"id": "baseline", "predictions": np.asarray([0, 3]), "curve": []}],
                          "baseline_fallback": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            fold = suite._seal_fold(Path(temporary), 0, {}, {"exact": True}, search,
                                    {"all_exact_predictions": True}, {"source": "fixed"})
            indices, predictions = suite._read_sealed_predictions(fold, "selector")
            np.testing.assert_array_equal(indices, [4, 9])
            np.testing.assert_array_equal(predictions, [0, 3])
            (fold["directory"] / "qmf_models.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seal changed"):
                suite._read_sealed_predictions(fold, "selector")

    def test_promotion_requires_every_condition(self):
        baseline = {"macro_f1": .9565, "accuracy": .959,
                    "errors": {"unknown_to_known": 60, "known_to_other_known": 12}}
        selector = {"macro_f1": .9581, "accuracy": .960,
                    "errors": {"unknown_to_known": 62, "known_to_other_known": 12}}
        folds = {
            "baseline": {"0": {"macro_f1": .95}, "1": {"macro_f1": .96}},
            "selector": {"0": {"macro_f1": .951}, "1": {"macro_f1": .961}},
        }
        passed = suite._promotion(self.config, baseline, selector, {"lower": -.0005}, folds, True)
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["local_transfer_allowed"])
        failed = suite._promotion(self.config, baseline, selector, {"lower": -.0011}, folds, True)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["local_transfer_allowed"])
        self.assertEqual(failed["retained_recipe"], "baseline")

    def test_bootstrap_is_deterministic_and_group_stratified(self):
        labels = ["unknown", "known"] + [f"unused-{i}" for i in range(445)]
        manifest = [
            {"audio_file": "u0.wav", "speaker_id": "unknown"},
            {"audio_file": "u1.wav", "speaker_id": "unknown"},
            {"audio_file": "k0.wav", "speaker_id": "known"},
            {"audio_file": "k1.wav", "speaker_id": "known"},
        ]
        contract = {"labels": labels, "manifest": manifest,
                    "folds": [{"audio_file": row["audio_file"], "group_id": row["audio_file"]}
                              for row in manifest]}
        before = [{"audio_file": row["audio_file"], "speaker_id": "unknown"} for row in manifest]
        after = [{"audio_file": row["audio_file"], "speaker_id": row["speaker_id"]} for row in manifest]
        config = deepcopy(self.config)
        config["bootstrap"]["replicates"] = 20
        first = suite._bootstrap(contract, before, after, config)
        second = suite._bootstrap(contract, before, after, config)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 20)
        self.assertEqual(first["true_class_strata"], 2)


if __name__ == "__main__":
    unittest.main()
