"""Contracts for read-only paired diagnosis of completed F005 versus C002b."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.evaluation.f005_paired_analysis import analyze_f005_vs_c002b
from speaker_id.evaluation.metrics import score_predictions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class F005PairedAnalysisTests(unittest.TestCase):
    """Synthetic receipts mirror F005's actual on-disk evaluation layout."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.f005_dir = self.root / "F005"
        self.c002b_dir = self.root / "C002b"
        self.manifest_path = self.root / "manifest.csv"
        self.folds_path = self.root / "folds.csv"
        self.roles_path = self.root / "roles.csv"
        self.label_map_path = self.root / "label_map.json"
        self.labels = ["unknown", "speaker-a", "speaker-b"] + [
            f"unused-{index}" for index in range(444)
        ]
        self.specs = [
            # name, outer fold, truth, duration, C002b final prediction, F005 selected prediction
            ("f0_a.wav", 0, "speaker-a", 2.5, "speaker-a", "unknown"),
            ("f0_b.wav", 0, "speaker-b", 4.0, "unknown", "speaker-b"),
            ("f0_u.wav", 0, "unknown", 7.0, "unknown", "speaker-a"),
            ("f1_a.wav", 1, "speaker-a", 9.0, "speaker-b", "speaker-a"),
            ("f1_b.wav", 1, "speaker-b", 31.0, "speaker-b", "speaker-b"),
            ("f1_u.wav", 1, "unknown", 40.0, "speaker-a", "speaker-a"),
        ]
        self._write_inputs()
        self._write_c002b()
        self._write_f005()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_inputs(self) -> None:
        manifest, folds, roles = [], [], []
        for index, (name, fold, speaker, duration, _c002b, _f005) in enumerate(self.specs):
            manifest.append({
                "audio_file": name,
                "speaker_id": speaker,
                "duration_seconds": duration,
                "has_nonzero_signal": True,
                "mono_rms_dbfs": -20.0,
            })
            folds.append({
                "audio_file": name,
                "speaker_id": speaker,
                "group_id": f"group-{index}",
                "fold": fold,
                "train_eligible": True,
            })
            for outer in (0, 1):
                is_outer = outer == fold
                is_known = speaker != "unknown"
                roles.append({
                    "audio_file": name,
                    "speaker_id": speaker,
                    "outer_fold": outer,
                    "group_id": f"group-{index}",
                    "role": "outer" if is_outer else ("fit_enrollment" if is_known else "query"),
                    "outer_evaluation_included": is_outer,
                    "encoder_fit_allowed": not is_outer and is_known,
                    "enrollment_allowed": not is_outer and is_known,
                    "calibration_query": not is_outer and not is_known,
                })
        _write_csv(self.manifest_path, manifest)
        _write_csv(self.folds_path, folds)
        _write_csv(self.roles_path, roles)
        _write_json(self.label_map_path, {"labels": self.labels, "unknown_index": 0})

    def _write_c002b(self) -> None:
        predictions = []
        label_index = {label: index for index, label in enumerate(self.labels)}
        for outer in (0, 1):
            selected = [row for row in self.specs if row[1] == outer]
            probabilities = np.zeros((len(selected), len(self.labels)), dtype=np.float64)
            for index, (name, _fold, _truth, _duration, c002b, _f005) in enumerate(selected):
                probabilities[index, label_index[c002b]] = 1.0
                predictions.append({"audio_file": name, "speaker_id": c002b})
            directory = self.c002b_dir / f"fold_{outer}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / "outer_probabilities.npz",
                probabilities=probabilities,
                audio_files=np.asarray([row[0] for row in selected]),
                labels=np.asarray(self.labels),
            )
        _write_csv(self.c002b_dir / "oof_predictions.csv", predictions)

    def _write_f005(self) -> None:
        manifest_by_name = {
            row["audio_file"]: row
            for row in csv.DictReader(self.manifest_path.open(encoding="utf-8", newline=""))
        }
        folds_by_name = {
            row["audio_file"]: row
            for row in csv.DictReader(self.folds_path.open(encoding="utf-8", newline=""))
        }
        historical, selected = [], []
        for outer in (0, 1):
            items = [row for row in self.specs if row[1] == outer]
            reference = [
                {
                    "audio_file": name,
                    "speaker_id": manifest_by_name[name]["speaker_id"],
                    "group_id": folds_by_name[name]["group_id"],
                    "duration_seconds": manifest_by_name[name]["duration_seconds"],
                    "has_nonzero_signal": manifest_by_name[name]["has_nonzero_signal"],
                }
                for name, _fold, _speaker, _duration, _c002b, _f005 in items
            ]
            c002b_predictions = [
                {"audio_file": name, "speaker_id": c002b}
                for name, _fold, _speaker, _duration, c002b, _f005 in items
            ]
            f005_predictions = [
                {"audio_file": name, "speaker_id": f005}
                for name, _fold, _speaker, _duration, _c002b, f005 in items
            ]
            historical.extend(c002b_predictions)
            selected.extend(f005_predictions)
            comparator_predictions = {
                "frozen_same_protocol": c002b_predictions,
                "fresh_control": f005_predictions,
                "selected_arm": f005_predictions,
            }
            comparators = {
                name: {
                    "policy": {},
                    "predictions": predictions,
                    # The stored known-top1 is deliberately distinct from final
                    # rejection predictions for the relevant diagnostic rows.
                    "known_top1_predictions": [
                        {
                            "audio_file": item["audio_file"],
                            "speaker_id": (
                                item["speaker_id"]
                                if item["speaker_id"] != "unknown"
                                else "speaker-a"
                            ),
                        }
                        for item in predictions
                    ],
                    "metrics": score_predictions(reference, predictions, self.labels),
                    "duration_slices": {},
                }
                for name, predictions in comparator_predictions.items()
            }
            body = {
                "schema_version": "f005-one-shot-outer-evaluation-v1",
                "experiment_signature": "f005-fixture-signature",
                "outer_fold": outer,
                "selected_arm": "control",
                "policy_seal_sha256": "a" * 64,
                "policy_file_sha256": "b" * 64,
                "outer_reference": reference,
                "comparators": comparators,
                "one_shot_outer_evaluation": True,
                "outer_truth_first_access_stage": "after_all_three_policy_seals_reloaded",
            }
            _write_json(
                self.f005_dir / "evaluation" / f"fold_{outer}.json",
                {**body, "evaluation_sha256": _canonical_sha(body)},
            )
        source_hashes = {
            "C002b/oof_predictions.csv": _sha256(self.c002b_dir / "oof_predictions.csv"),
            **{
                f"C002b/fold_{outer}/outer_probabilities.npz": _sha256(
                    self.c002b_dir / f"fold_{outer}" / "outer_probabilities.npz"
                )
                for outer in (0, 1)
            },
        }
        _write_json(self.f005_dir / "source_verification.json", {"c002b_artifacts": source_hashes})
        input_hashes = {
            name: _sha256(path)
            for name, path in (
                ("manifest", self.manifest_path),
                ("folds", self.folds_path),
                ("roles", self.roles_path),
                ("label_map", self.label_map_path),
            )
        }
        _write_json(self.f005_dir / "resolved_config.json", {
            "experiment_signature": "f005-fixture-signature",
            "identity": {"readiness_input_hashes": input_hashes},
        })
        _write_json(self.f005_dir / "experiment_report.json", {
            "status": "complete",
            "experiment_signature": "f005-fixture-signature",
            "result": {
                "schema_version": "f005-oof-promotion-decision-v1",
                "experiment_signature": "f005-fixture-signature",
                "selected_arms_by_outer": {"0": "control", "1": "control"},
                "selection_kind": "control",
                "metrics": {
                    "historical_c002b": score_predictions(
                        self._reference_rows(), historical, self.labels,
                    ),
                    "fresh_control": score_predictions(self._reference_rows(), selected, self.labels),
                    "selected_arm": score_predictions(self._reference_rows(), selected, self.labels),
                },
                "decision": "retain_c002b",
            },
        })

    def _reference_rows(self) -> list[dict]:
        return [
            {"audio_file": name, "speaker_id": speaker}
            for name, _fold, speaker, _duration, _c002b, _f005 in self.specs
        ]

    def _evaluation(self, outer: int) -> dict:
        path = self.f005_dir / "evaluation" / f"fold_{outer}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _replace_evaluation(self, outer: int, value: dict) -> None:
        body = {key: item for key, item in value.items() if key != "evaluation_sha256"}
        _write_json(
            self.f005_dir / "evaluation" / f"fold_{outer}.json",
            {**body, "evaluation_sha256": _canonical_sha(body)},
        )

    def _refresh_c002b_source_receipts(self) -> None:
        path = self.f005_dir / "source_verification.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["c002b_artifacts"] = {
            "C002b/oof_predictions.csv": _sha256(self.c002b_dir / "oof_predictions.csv"),
            **{
                f"C002b/fold_{outer}/outer_probabilities.npz": _sha256(
                    self.c002b_dir / f"fold_{outer}" / "outer_probabilities.npz"
                )
                for outer in (0, 1)
            },
        }
        _write_json(path, receipt)

    def _replace_c002b_npz(self, outer: int, **changes) -> None:
        path = self.c002b_dir / f"fold_{outer}" / "outer_probabilities.npz"
        with np.load(path, allow_pickle=False) as saved:
            contents = {name: saved[name].copy() for name in saved.files}
        contents.update(changes)
        np.savez_compressed(path, **contents)
        self._refresh_c002b_source_receipts()

    def _refresh_f005_selected_report_metric(self) -> None:
        """Keep the aggregate receipt internally consistent for a negative test."""
        path = self.f005_dir / "experiment_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        selected = [
            row
            for outer in (0, 1)
            for row in self._evaluation(outer)["comparators"]["selected_arm"]["predictions"]
        ]
        report["result"]["metrics"]["selected_arm"] = score_predictions(
            self._reference_rows(), selected, self.labels,
        )
        _write_json(path, report)

    def analyze(self):
        return analyze_f005_vs_c002b(
            self.f005_dir,
            self.c002b_dir,
            self.manifest_path,
            self.folds_path,
            self.roles_path,
            self.label_map_path,
        )

    def test_complete_paired_analysis_reports_transitions_and_integrity(self):
        report, paired_rows, class_rows = self.analyze()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["analysis_kind"], "f005_vs_c002b_paired_saved_prediction_analysis")
        self.assertEqual(report["row_count"], len(self.specs))
        self.assertEqual(report["class_count"], len(self.labels))
        self.assertEqual(report["paired_transitions"]["c002b_correct_to_f005_wrong"], 2)
        self.assertEqual(report["paired_transitions"]["c002b_wrong_to_f005_correct"], 2)
        self.assertEqual(report["paired_transitions"]["both_correct"], 1)
        self.assertEqual(report["paired_transitions"]["both_wrong"], 1)
        self.assertTrue(report["integrity"]["full_oof_coverage"])
        self.assertTrue(report["integrity"]["f005_selected_matches_fresh_control"])
        self.assertTrue(report["integrity"]["c002b_csv_matches_probability_argmax"])
        self.assertEqual({row["audio_file"] for row in paired_rows}, {row[0] for row in self.specs})
        self.assertEqual(len(paired_rows), len(self.specs))
        self.assertEqual(len(class_rows), len(self.labels))
        self.assertEqual({row["speaker_id"] for row in class_rows}, set(self.labels))
        deteriorated = next(row for row in paired_rows if row["audio_file"] == "f0_a.wav")
        self.assertEqual(deteriorated["c002b_prediction"], "speaker-a")
        self.assertEqual(deteriorated["f005_selected_prediction"], "unknown")
        self.assertEqual(deteriorated["c002b_error_mode"], "correct")
        self.assertEqual(deteriorated["f005_error_mode"], "known_to_unknown")

    def test_rejects_missing_f005_outer_coverage(self):
        evaluation = self._evaluation(0)
        evaluation["outer_reference"] = evaluation["outer_reference"][:-1]
        for comparator in evaluation["comparators"].values():
            comparator["predictions"] = comparator["predictions"][:-1]
            comparator["known_top1_predictions"] = comparator["known_top1_predictions"][:-1]
        self._replace_evaluation(0, evaluation)

        with self.assertRaises(ValueError):
            self.analyze()

    def test_rejects_duplicate_f005_prediction_identity(self):
        evaluation = self._evaluation(0)
        for comparator in evaluation["comparators"].values():
            comparator["predictions"][1]["audio_file"] = comparator["predictions"][0]["audio_file"]
            comparator["known_top1_predictions"][1]["audio_file"] = comparator["known_top1_predictions"][0]["audio_file"]
        self._replace_evaluation(0, evaluation)

        with self.assertRaises(ValueError):
            self.analyze()

    def test_rejects_selected_control_when_predictions_differ_from_fresh_control(self):
        evaluation = self._evaluation(0)
        selected = evaluation["comparators"]["selected_arm"]
        selected["predictions"][0]["speaker_id"] = "speaker-b"
        selected["known_top1_predictions"][0]["speaker_id"] = "speaker-b"
        self._replace_evaluation(0, evaluation)
        self._refresh_f005_selected_report_metric()

        with self.assertRaises(ValueError):
            self.analyze()

    def test_rejects_c002b_csv_and_probability_argmax_disagreement(self):
        path = self.c002b_dir / "fold_0" / "outer_probabilities.npz"
        with np.load(path, allow_pickle=False) as saved:
            probabilities = saved["probabilities"].copy()
        probabilities[0] = 0.0
        probabilities[0, self.labels.index("speaker-b")] = 1.0
        self._replace_c002b_npz(0, probabilities=probabilities)

        with self.assertRaises(ValueError):
            self.analyze()

    def test_rejects_c002b_probability_label_order_mismatch(self):
        path = self.c002b_dir / "fold_0" / "outer_probabilities.npz"
        with np.load(path, allow_pickle=False) as saved:
            probabilities = saved["probabilities"].copy()
        # Keep semantic NPZ argmax labels equal to the C002b CSV.  Rejection
        # must therefore come from the saved-order mismatch itself, rather than
        # a downstream CSV-vs-NPZ prediction disagreement.
        probabilities[:, [1, 2]] = probabilities[:, [2, 1]]
        labels = np.asarray(self.labels)
        labels[[1, 2]] = labels[[2, 1]]
        self._replace_c002b_npz(0, probabilities=probabilities, labels=labels)

        with self.assertRaises(ValueError):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
