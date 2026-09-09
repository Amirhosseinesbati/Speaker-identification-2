"""Contracts for the post-hoc U->K oracle audit; no model or audio required."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_u2k_oracle", ROOT / "scripts" / "analysis" / "analyze_u2k_oracle.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class U2KOracleAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.labels = ["unknown", "A", "B"] + [f"unused-{index}" for index in range(444)]
        self.reference = self.root / "reference.csv"
        self.predictions = self.root / "oof_predictions.csv"
        self.folds = self.root / "folds.csv"
        self.label_map = self.root / "labels.json"
        self.label_map.write_text(json.dumps({"labels": self.labels}), encoding="utf-8")
        self.reference_rows = [
            {"audio_file": "u-short.wav", "speaker_id": "unknown", "duration_seconds": "0.5", "has_nonzero_signal": "True"},
            {"audio_file": "u-long.wav", "speaker_id": "unknown", "duration_seconds": "31", "has_nonzero_signal": "True"},
            {"audio_file": "u-correct.wav", "speaker_id": "unknown", "duration_seconds": "4", "has_nonzero_signal": "False"},
            {"audio_file": "a-correct.wav", "speaker_id": "A", "duration_seconds": "8", "has_nonzero_signal": "True"},
            {"audio_file": "a-rejected.wav", "speaker_id": "A", "duration_seconds": "8", "has_nonzero_signal": "True"},
            {"audio_file": "a-wrong-known.wav", "speaker_id": "A", "duration_seconds": "8", "has_nonzero_signal": "True"},
            {"audio_file": "b-correct.wav", "speaker_id": "B", "duration_seconds": "8", "has_nonzero_signal": "True"},
        ]
        self.prediction_rows = [
            {"audio_file": "u-short.wav", "speaker_id": "A"},
            {"audio_file": "u-long.wav", "speaker_id": "B"},
            {"audio_file": "u-correct.wav", "speaker_id": "unknown"},
            {"audio_file": "a-correct.wav", "speaker_id": "A"},
            {"audio_file": "a-rejected.wav", "speaker_id": "unknown"},
            {"audio_file": "a-wrong-known.wav", "speaker_id": "B"},
            {"audio_file": "b-correct.wav", "speaker_id": "B"},
        ]
        fold_rows = [
            {"audio_file": "u-short.wav", "speaker_id": "unknown", "fold": "0", "group_id": "gu-short"},
            {"audio_file": "u-long.wav", "speaker_id": "unknown", "fold": "1", "group_id": "gu-long"},
            {"audio_file": "u-correct.wav", "speaker_id": "unknown", "fold": "0", "group_id": "gu-correct"},
            {"audio_file": "a-correct.wav", "speaker_id": "A", "fold": "0", "group_id": "ga"},
            {"audio_file": "a-rejected.wav", "speaker_id": "A", "fold": "1", "group_id": "ga"},
            {"audio_file": "a-wrong-known.wav", "speaker_id": "A", "fold": "1", "group_id": "ga2"},
            {"audio_file": "b-correct.wav", "speaker_id": "B", "fold": "0", "group_id": "gb"},
        ]
        self._write(self.reference, self.reference_rows)
        self._write(self.predictions, self.prediction_rows)
        self._write(self.folds, fold_rows)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write(path: Path, rows: list[dict]):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _analyze(self):
        return module.analyze_u2k_oracle(self.reference, self.predictions, self.label_map, self.folds)

    def test_changes_only_true_unknown_predicted_known_rows(self):
        report, rows, by_label = self._analyze()
        self.assertEqual(report["baseline"]["errors"], {
            "known_to_unknown": 1, "unknown_to_known": 2, "known_to_other_known": 1,
        })
        self.assertEqual(report["u2k_oracle"]["errors"], {
            "known_to_unknown": 1, "unknown_to_known": 0, "known_to_other_known": 1,
        })
        self.assertEqual(report["delta"]["corrected_rows"], 2)
        self.assertAlmostEqual(report["delta"]["accuracy"], 2 / 7)
        self.assertGreater(report["delta"]["macro_f1"], 0)
        changed = [row for row in rows if row["is_u2k"]]
        self.assertEqual({row["audio_file"] for row in changed}, {"u-short.wav", "u-long.wav"})
        self.assertTrue(all(row["oracle_prediction"] == "unknown" for row in changed))
        self.assertTrue(all(
            row["oracle_prediction"] == row["baseline_prediction"]
            for row in rows if not row["is_u2k"]
        ))
        self.assertEqual({row["predicted_speaker_id"]: row["u2k_rows_removed_as_false_positives"] for row in by_label},
                         {"A": 1, "B": 1})
        self.assertEqual(report["true_unknown_population_slices"]["duration_band"]["under_1s"]["baseline_unknown_to_known"], 1)
        self.assertEqual(report["true_unknown_population_slices"]["outer_fold"]["1"]["baseline_unknown_to_known"], 1)
        self.assertEqual(report["content_group_summary"]["u2k_affected_content_groups"], 2)

    def test_prediction_coverage_mismatch_is_rejected(self):
        self._write(self.predictions, self.prediction_rows[:-1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            self._analyze()

    def test_fold_label_mismatch_is_rejected(self):
        with self.folds.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["speaker_id"] = "A"
        self._write(self.folds, rows)
        with self.assertRaisesRegex(ValueError, "differs"):
            self._analyze()

    def test_writer_refuses_existing_directory_and_never_marks_oracle_eligible(self):
        report, rows, by_label = self._analyze()
        self.assertFalse(report["eligible_for_model_selection"])
        self.assertFalse(report["eligible_for_submission"])
        output = self.root / "output"
        module.write_analysis(output, report, rows, by_label)
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output / "u2k_rows.csv").is_file())
        self.assertTrue((output / "u2k_by_predicted_label.csv").is_file())
        with self.assertRaises(FileExistsError):
            module.write_analysis(output, report, rows, by_label)


if __name__ == "__main__":
    unittest.main()
