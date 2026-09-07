"""Portable scoring parity, payload integrity, and CLI contracts without Torch."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from speaker_id.inference.runtime import IntegrityError, run_submission, verify_payload
from speaker_id.inference.scoring import reference_scores, score_embeddings, validate_calibration, validate_gallery
from speaker_id.training.reference_scoring import known_scores, reference_probabilities

ROOT = Path(__file__).resolve().parents[1]


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class InferenceScoringTests(unittest.TestCase):
    def test_parity_with_training_max_reference_and_gate(self):
        rng = np.random.default_rng(41)
        gallery = {"known_embeddings": unit(rng.normal(size=(9, 16))),
                   "known_targets": np.repeat(np.arange(1, 4, dtype=np.int64), 3),
                   "unknown_embeddings": unit(rng.normal(size=(5, 16)))}
        queries = unit(rng.normal(size=(7, 16)))
        normalized = unit(queries)
        expected_known = np.clip(known_scores(normalized, gallery["known_embeddings"], gallery["known_targets"], "max_reference", classes=3), -1, 1)
        expected_unknown = np.clip((normalized @ gallery["unknown_embeddings"].T).max(axis=1), -1, 1)
        actual_known, actual_unknown = reference_scores(queries, gallery, classes=3)
        np.testing.assert_array_equal(actual_known, expected_known)
        np.testing.assert_array_equal(actual_unknown, expected_unknown)
        calibration = {"unknown_weight": .75, "margin_weight": .5, "threshold": .18, "temperature": .05}
        valid = np.asarray([True, True, False, True, True, True, True])
        expected = reference_probabilities(expected_known, expected_unknown, calibration, valid)
        actual = score_embeddings(queries, valid, gallery, calibration, classes=3)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=1), 1, rtol=0, atol=1e-14)

    def test_unknown_ties_known_ties_and_invalid_queries(self):
        gallery = {"known_embeddings": np.eye(3, dtype=np.float32),
                   "known_targets": np.arange(1, 4, dtype=np.int64),
                   "unknown_embeddings": unit([[1, 1, 1]])}
        calibration = {"unknown_weight": 0., "margin_weight": 0., "threshold": 1.}
        predictions = score_embeddings(np.asarray([[1, 0, 0], [0, 0, 0], [np.nan, 0, 0]], dtype=np.float32), np.ones(3, dtype=bool), gallery, calibration, classes=3)
        self.assertEqual(predictions.argmax(axis=1).tolist(), [0, 0, 0])
        np.testing.assert_array_equal(predictions[1:, 0], [1, 1])
        calibration["threshold"] = -1.
        tied_known = score_embeddings(np.asarray([[1, 1, 0]], dtype=np.float32), np.asarray([True]), gallery, calibration, classes=3)
        self.assertEqual(int(tied_known.argmax(axis=1)[0]), 1)

    def test_gallery_rejects_missing_class_bad_norm_and_wrong_target_dtype(self):
        base = {"known_embeddings": np.eye(3, dtype=np.float32), "known_targets": np.arange(1, 4, dtype=np.int64),
                "unknown_embeddings": unit([[1, 1, 1]])}
        for replacement in ({"known_targets": np.asarray([1, 1, 3], dtype=np.int64)},
                            {"known_targets": np.arange(1, 4, dtype=np.int32)},
                            {"known_embeddings": np.eye(3, dtype=np.float32) * 2},
                            {"unknown_embeddings": np.asarray([[np.nan, 0, 0]], dtype=np.float32)}):
            with self.subTest(replacement=list(replacement)), self.assertRaises(ValueError):
                validate_gallery({**base, **replacement}, classes=3, embedding_dim=3)

    def test_release_calibration_rejects_unsupported_windows_and_nonfinite(self):
        base = {"unknown_weight": 1., "margin_weight": .5, "threshold": .1, "temperature": .05,
                "inference": {"seconds": 180., "maximum_windows": 1}}
        validate_calibration(base, require_inference=True)
        for replacement in ({"threshold": float("nan")}, {"temperature": 0},
                            {"inference": {"seconds": 6., "maximum_windows": 3}},
                            {"inference": {"seconds": 180., "maximum_windows": True}}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_calibration({**base, **replacement}, require_inference=True)


class PortableRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.package, self.audio = self.base / "package", self.base / "audio"
        self.package.mkdir()
        self.audio.mkdir()
        for name in ("submission.py", "speaker_id/inference/runtime.py", "speaker_id/inference/scoring.py", "speaker_id/models/campp.py"):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic payload file.\n", encoding="utf-8")
        assets = self.package / "assets"
        assets.mkdir()
        (assets / "campplus_voxceleb.bin").write_bytes(b"synthetic verified fixture")
        config = json.loads((ROOT / "configs/model/campp.json").read_text())
        config["weights_path"] = "assets/campplus_voxceleb.bin"
        (assets / "model_config.json").write_text(json.dumps(config))
        self.labels = ["unknown"] + [str(uuid.UUID(int=index)) for index in range(1, 447)]
        (assets / "labels.json").write_text(json.dumps({"labels": self.labels, "unknown_index": 0}))
        gallery = {"known_embeddings": np.eye(512, dtype=np.float32)[:446],
                   "known_targets": np.arange(1, 447, dtype=np.int64),
                   "unknown_embeddings": np.eye(512, dtype=np.float32)[511:]}
        np.savez_compressed(assets / "gallery.npz", **gallery)
        (assets / "calibration.json").write_text(json.dumps({"unknown_weight": 1., "margin_weight": .5,
            "threshold": .2, "temperature": .05, "inference": {"seconds": 180., "maximum_windows": 1}}))
        self.manifest = {"schema_version": 1, "release_id": "P001-fixture", "leaderboard_validation": "pending",
                         "files": {path.relative_to(self.package).as_posix(): {"bytes": path.stat().st_size,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for path in self.package.rglob("*") if path.is_file()}}
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.package / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def fake_extract(self, encoder, path, *, device):
        if path.name.startswith("bad"):
            raise OSError("Synthetic decoder failure")
        vector = np.zeros(512, dtype=np.float32)
        if not path.name.startswith("zero"):
            vector[0] = 1
        return vector, {"nonzero_signal": bool(vector.any())}

    def test_manifest_detects_tampering_unlisted_and_escaping_paths(self):
        verify_payload(self.package)
        changed = self.package / "submission.py"
        changed.write_text("changed\n")
        with self.assertRaises(IntegrityError):
            verify_payload(self.package)
        changed.write_text("# Synthetic payload file.\n")
        extra = self.package / "unexpected.py"
        extra.write_text("pass\n")
        with self.assertRaises(IntegrityError):
            verify_payload(self.package)
        extra.unlink()
        self.manifest["files"]["../outside.py"] = {"bytes": 0, "sha256": "0" * 64}
        self.write_manifest()
        with self.assertRaises(IntegrityError):
            verify_payload(self.package)

    def test_directory_symlinks_are_rejected(self):
        extra = self.package / "external"
        try:
            extra.symlink_to(self.audio, target_is_directory=True)
        except OSError:
            self.skipTest("Creating a directory symlink is unavailable on this host")
        with self.assertRaises(IntegrityError):
            verify_payload(self.package)

    def test_junction_ancestor_is_rejected_without_requiring_admin_privileges(self):
        assets = (self.package / "assets").resolve()
        with patch.object(type(assets), "is_junction", lambda path: path == assets):
            with self.assertRaisesRegex(IntegrityError, "junction"):
                verify_payload(self.package)

    def test_nested_output_sorted_names_zero_and_decode_failure(self):
        for name in ("zero.mp3", "known.mp3", "bad.wav"):
            (self.audio / name).write_bytes(b"fixture")
        (self.audio / "labels.csv").write_text("ignored metadata")
        output = self.base / "new" / "nested" / "predictions.csv"
        with patch("speaker_id.inference.runtime._load_encoder", return_value=(object(), "cpu")), \
             patch("speaker_id.inference.runtime._extract", side_effect=self.fake_extract), contextlib.redirect_stderr(io.StringIO()):
            result = run_submission(self.package, self.audio, output)
        self.assertEqual(result["files"], 3)
        with output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["audio_file", "speaker_id"])
            rows = list(reader)
        self.assertEqual(rows, [{"audio_file": "bad.wav", "speaker_id": "unknown"},
                               {"audio_file": "known.mp3", "speaker_id": self.labels[1]},
                               {"audio_file": "zero.mp3", "speaker_id": "unknown"}])
        verify_payload(self.package)

    def test_model_failure_is_fatal_and_inputs_cannot_be_overwritten(self):
        audio = self.audio / "known.mp3"
        audio.write_bytes(b"fixture")
        output = self.base / "predictions.csv"
        with patch("speaker_id.inference.runtime._load_encoder", return_value=(object(), "cpu")), \
             patch("speaker_id.inference.runtime._extract", side_effect=RuntimeError("Synthetic GPU/model error")):
            with self.assertRaises(RuntimeError):
                run_submission(self.package, self.audio, output)
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            run_submission(self.package, self.audio, audio)
        self.assertEqual(audio.read_bytes(), b"fixture")

    def test_actual_cli_flags_and_basename_only_output_are_repeatable(self):
        (self.audio / "known.mp3").write_bytes(b"fixture")
        spec = importlib.util.spec_from_file_location("portable_submission_test", ROOT / "submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = self.package
        previous = Path.cwd()
        try:
            os.chdir(self.package)
            with patch("speaker_id.inference.runtime._load_encoder", return_value=(object(), "cpu")), \
                 patch("speaker_id.inference.runtime._extract", side_effect=self.fake_extract), contextlib.redirect_stdout(io.StringIO()):
                for _ in range(2):
                    result = module.main(["--data-dir", str(self.audio), "--predictions-file-path", "predictions.csv"])
                    self.assertEqual(result["files"], 1)
            self.assertTrue((self.package / "predictions.csv").is_file())
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
