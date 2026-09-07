"""Contracts and leakage tests; no test starts optimization or full extraction."""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from speaker_id.models.campp import crop_waveform, read_mono
from speaker_id.training.contracts import load_contract
from speaker_id.training.scoring import build_prototypes, fit_threshold, score_probabilities

ROOT = Path(__file__).resolve().parents[1]


class CAMPPContractTests(unittest.TestCase):
    def test_content_decode_resampling_and_antiphase_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misnamed.mp3"
            x = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32) * .2
            sf.write(path, np.column_stack((x, -x)), 48000, format="WAV", subtype="PCM_16")
            output = read_mono(path)
            self.assertEqual(output.shape, (16000,))
            self.assertGreater(np.sqrt(np.mean(output ** 2)), .1)

    def test_short_crop_is_padded_and_long_crop_uses_position(self):
        self.assertEqual(crop_waveform(np.ones(1, dtype=np.float32), 3).shape, (16000,))
        signal = np.arange(64000, dtype=np.float32)
        self.assertEqual(crop_waveform(signal, 1, position=1)[0], 48000)
        with self.assertRaises(ValueError):
            crop_waveform(signal, 1, position=2)

    def test_nonfinite_audio_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.wav"
            sf.write(path, np.asarray([0., np.nan], dtype=np.float32), 16000, subtype="FLOAT")
            with self.assertRaises(ValueError):
                read_mono(path)

    def test_threshold_and_447_argmax_agree_including_ties_and_zero(self):
        scores = np.full((4, 446), -.8)
        scores[:, 10] = [.7, .3, .5, .9]
        probabilities = score_probabilities(scores, .5, valid=np.asarray([True, True, True, False]))
        np.testing.assert_array_equal(probabilities.argmax(1), [11, 0, 0, 0])
        np.testing.assert_allclose(probabilities.sum(1), 1)
        self.assertEqual(probabilities.shape, (4, 447))

    def test_calibration_uses_only_explicit_query_scores(self):
        scores = np.full((4, 446), -.8)
        scores[0, 0], scores[1, 1] = .9, .8
        scores[2, 2], scores[3, 3] = .2, .1
        threshold, curve = fit_threshold(scores, np.asarray([1, 2, 0, 0]), 11)
        predictions = score_probabilities(scores, threshold).argmax(1)
        np.testing.assert_array_equal(predictions, [1, 2, 0, 0])
        self.assertTrue(all("inner_macro_f1_447" in row for row in curve))

    def test_prototype_requires_every_known_class(self):
        vectors = np.eye(446, dtype=np.float32)
        prototype = build_prototypes(vectors, np.arange(1, 447))
        np.testing.assert_allclose(prototype, vectors)
        with self.assertRaises(ValueError):
            build_prototypes(vectors[:-1], np.arange(1, 446))

    def _fixture(self, directory):
        root = Path(directory)
        (root / "src").mkdir()
        (root / "scripts").mkdir()
        (root / "scripts/train.py").write_text("# fixture\n")
        config = json.loads((ROOT / "configs/train/campp_baseline.json").read_text())
        config["expected_source_files"] = 894
        config["model_config"], config["manifest"], config["folds"], config["roles"], config["label_map"] = "model.json", "manifest.csv", "folds.csv", "roles.csv", "labels.json"
        (root / "model.json").write_text((ROOT / "configs/model/campp.json").read_text())
        labels = ["unknown"] + [f"speaker{i}" for i in range(446)]
        (root / "labels.json").write_text(json.dumps({"labels": labels}))
        manifest, folds, roles = [], [], []
        for fold in range(2):
            for index, label in enumerate(labels):
                name = f"{fold}_{index}.wav"
                manifest.append({"audio_file": name, "speaker_id": label, "input_sha256": "unused"})
                folds.append({"audio_file": name, "speaker_id": label, "fold": fold, "group_id": name, "train_eligible": True})
        for outer in range(2):
            for row in folds:
                evaluating = row["fold"] == outer
                roles.append({"outer_fold": outer, "audio_file": row["audio_file"], "speaker_id": row["speaker_id"],
                              "group_id": row["group_id"], "role": "outer_validation" if evaluating else "known_enrollment",
                              "encoder_fit_allowed": not evaluating, "enrollment_allowed": not evaluating and row["speaker_id"] != "unknown",
                              "calibration_query": False, "outer_evaluation_included": evaluating})
        for name, rows in (("manifest.csv", manifest), ("folds.csv", folds), ("roles.csv", roles)):
            with (root / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (root / "config.json").write_text(json.dumps(config))
        return root, roles

    def test_outer_leakage_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root, rows = self._fixture(directory)
            load_contract(root / "config.json", root)
            rows[1]["encoder_fit_allowed"] = True
            with (root / "roles.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Outer validation leakage"):
                load_contract(root / "config.json", root)

    def test_code_change_invalidates_resume_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._fixture(directory)
            first = load_contract(root / "config.json", root)["signature"]
            (root / "src/new.py").write_text("# changed implementation\n")
            second = load_contract(root / "config.json", root)["signature"]
            self.assertNotEqual(first, second)

    def test_disabling_tracking_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._fixture(directory)
            config = json.loads((root / "config.json").read_text())
            config["tracking_required"] = False
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "mandatory"):
                load_contract(root / "config.json", root)

    def test_unimplemented_frontend_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._fixture(directory)
            model = json.loads((root / "model.json").read_text())
            model["frontend"]["vad"] = True
            (root / "model.json").write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, "frontend"):
                load_contract(root / "config.json", root)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is required by the training environment")
    def test_fold_evaluation_preserves_all_rows_and_probability_argmax(self):
        from speaker_id.training.runner import evaluate_fold
        labels = ["unknown"] + [f"speaker{i}" for i in range(446)]
        config = json.loads((ROOT / "configs/train/campp_baseline.json").read_text())
        manifest, roles, vectors = [], [], []
        unit = np.eye(512, dtype=np.float32)
        def append(name, label, vector, role, nonzero=True):
            manifest.append({"audio_file": name, "speaker_id": label, "duration_seconds": 10,
                             "mono_rms_dbfs": -20 if nonzero else -240, "has_nonzero_signal": nonzero})
            roles.append({"audio_file": name, "speaker_id": label, "outer_fold": 0,
                          "enrollment_allowed": role == "enrollment", "calibration_query": role == "query",
                          "outer_evaluation_included": role == "outer"})
            vectors.append(vector)
        for index, label in enumerate(labels[1:]):
            append(f"enroll{index}", label, unit[index], "enrollment")
            append(f"query{index}", label, unit[index], "query")
        append("query_unknown", "unknown", unit[-1], "query")
        append("outer_known", labels[1], unit[0], "outer")
        append("outer_unknown", "unknown", unit[-1], "outer")
        append("outer_zero", labels[2], np.zeros(512), "outer", False)
        class Tracker:
            def log_metrics(self, *args, **kwargs):
                pass
            def add_artifact(self, path, *args, **kwargs):
                if not path.is_file():
                    raise AssertionError("Artifact was not created")
        with tempfile.TemporaryDirectory() as directory:
            predictions, report = evaluate_fold({"labels": labels, "manifest": manifest, "config": config},
                                                roles, np.asarray(vectors),
                                                np.asarray([bool(r["has_nonzero_signal"]) for r in manifest]),
                                                Path(directory), Tracker())
            self.assertEqual(len(predictions), 3)
            self.assertEqual(report["outer"]["row_count"], 3)
            self.assertEqual(predictions[-1]["speaker_id"], "unknown")
            with np.load(Path(directory) / "outer_probabilities.npz", allow_pickle=False) as stored:
                self.assertEqual([labels[int(i)] for i in stored["probabilities"].argmax(1)], [r["speaker_id"] for r in predictions])
            self.assertTrue((Path(directory) / "file_diagnostics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
