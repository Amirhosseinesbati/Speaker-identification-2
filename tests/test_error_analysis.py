"""Saved-output diagnostic contracts; tests never run a model or select thresholds."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.evaluation.error_analysis import analyze_saved_probabilities


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.labels = ["unknown", "speaker-a", "speaker-b"] + [f"unused-{i}" for i in range(444)]
        self.specs = [
            ("f0a", 0, "speaker-a", True, [.6, .3, .1]),
            ("f0b", 0, "speaker-b", True, [.1, .6, .3]),
            ("f0zero", 0, "speaker-a", False, [1., 0., 0.]),
            ("f0unknown", 0, "unknown", True, [.2, .6, .2]),
            ("f1a", 1, "speaker-a", True, [.1, .8, .1]),
            ("f1b", 1, "speaker-b", True, [.1, .1, .8]),
            ("f1unknown", 1, "unknown", True, [.8, .1, .1]),
        ]
        self.manifest = self.root / "manifest.csv"
        self.roles = self.root / "roles.csv"
        self.label_map = self.root / "label_map.json"
        self.label_map.write_text(json.dumps({"labels": self.labels}))
        manifest = [{"audio_file": name, "speaker_id": label, "has_nonzero_signal": nonzero,
                     "duration_seconds": 40 if nonzero else .1, "mono_rms_dbfs": -20 if nonzero else -240}
                    for name, fold, label, nonzero, values in self.specs]
        roles = []
        for outer in (0, 1):
            for name, fold, label, nonzero, values in self.specs:
                enrolling = outer != fold and label != "unknown" and nonzero
                roles.append({"audio_file": name, "speaker_id": label, "outer_fold": outer,
                              "group_id": name, "outer_evaluation_included": outer == fold,
                              "enrollment_allowed": enrolling, "encoder_fit_allowed": enrolling,
                              "calibration_query": False})
            selected = [row for row in self.specs if row[1] == outer]
            probabilities = np.zeros((len(selected), 447), dtype=np.float64)
            probabilities[:, :3] = [row[4] for row in selected]
            directory = self.root / f"fold_{outer}"
            directory.mkdir()
            np.savez(directory / "outer_probabilities.npz", probabilities=probabilities,
                     audio_files=np.asarray([row[0] for row in selected]), labels=np.asarray(self.labels))
        for path, rows in ((self.manifest, manifest), (self.roles, roles)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.refresh_hashes()

    def tearDown(self):
        self.temporary.cleanup()

    def refresh_hashes(self):
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in (("manifest", self.manifest), ("roles", self.roles), ("label_map", self.label_map))}
        (self.root / "resolved_config.json").write_text(json.dumps({"input_hashes": hashes}))

    def analyze(self):
        return analyze_saved_probabilities(self.root, self.manifest, self.roles, self.label_map)

    def test_ranks_ignore_unknown_threshold_but_not_zero_signal(self):
        report, rows, classes = self.analyze()
        self.assertEqual(report["nonzero_known"]["files"], 4)
        self.assertEqual(report["nonzero_known"]["top1_accuracy"], .75)
        self.assertEqual(report["nonzero_known"]["top2_accuracy"], 1.)
        self.assertEqual(report["nonzero_known"]["true_top1_but_rejected"], 1)
        self.assertEqual(report["known_zero_signal_files_excluded_from_ranking"], 1)
        zero = next(row for row in rows if row["audio_file"] == "f0zero")
        self.assertIsNone(zero["known_rank"])
        self.assertIsNone(zero["top_known_speaker_id"])
        oracle = report["ground_truth_knownness_oracle_with_current_known_argmax"]
        self.assertEqual(oracle["errors"], {"known_to_unknown": 1, "unknown_to_known": 0, "known_to_other_known": 1})
        self.assertEqual(oracle["row_count"], 7)
        self.assertEqual(oracle["class_count"], 447)
        self.assertEqual(len(classes), 4)

    def test_input_hash_change_is_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "original input hash"):
            self.analyze()

    def test_outer_role_leakage_is_rejected(self):
        rows = list(csv.DictReader(self.roles.open(newline="")))
        rows[0]["enrollment_allowed"] = "True"
        with self.roles.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "leaked"):
            self.analyze()

    def test_saved_label_order_change_is_rejected(self):
        path = self.root / "fold_0/outer_probabilities.npz"
        with np.load(path, allow_pickle=False) as archive:
            contents = {name: archive[name].copy() for name in archive.files}
        contents["labels"][[1, 2]] = contents["labels"][[2, 1]]
        np.savez(path, **contents)
        with self.assertRaisesRegex(ValueError, "label order"):
            self.analyze()

    def test_missing_outer_probability_row_is_rejected(self):
        path = self.root / "fold_0/outer_probabilities.npz"
        with np.load(path, allow_pickle=False) as archive:
            contents = {name: archive[name].copy() for name in archive.files}
        contents["probabilities"] = contents["probabilities"][:-1]
        contents["audio_files"] = contents["audio_files"][:-1]
        np.savez(path, **contents)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
