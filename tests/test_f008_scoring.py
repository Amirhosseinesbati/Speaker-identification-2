import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training import f008_scoring as scoring


def _labels():
    return ["unknown"] + [f"k{index}" for index in range(1, 447)]


def _contract():
    manifest, folds = [], []
    for fold in (0, 1):
        for kind, label in (
            ("reference", "k1"), ("reference", "k2"),
            ("query", "k1"), ("query", "k2"), ("query", "unknown"),
        ):
            name = f"fold{fold}_{kind}_{label}.wav"
            manifest.append({
                "audio_file": name,
                "speaker_id": label,
                "duration_seconds": 4.0,
                "has_nonzero_signal": True,
            })
            folds.append({
                "audio_file": name,
                "group_id": f"group_{fold}_{kind}_{label}",
                "fold": fold,
                "train_eligible": True,
            })
    roles = []
    for outer in (0, 1):
        for row, fold in zip(manifest, folds, strict=True):
            is_outer = fold["fold"] == outer
            is_reference = "_reference_" in row["audio_file"]
            roles.append({
                "audio_file": row["audio_file"],
                "speaker_id": row["speaker_id"],
                "group_id": fold["group_id"],
                "outer_fold": outer,
                "encoder_fit_allowed": not is_outer and is_reference,
                "enrollment_allowed": False,
                "calibration_query": not is_outer and not is_reference,
                "outer_evaluation_included": is_outer,
            })
    return {
        "signature": "a" * 64,
        "config": {"fold_ids": [0, 1]},
        "labels": _labels(),
        "manifest": manifest,
        "folds": folds,
        "roles": roles,
    }


def _spec():
    return {
        "arm_ids": ["control_f005", "energy_005", "uniform_005"],
        "control_arm_id": "control_f005",
        "arm_tie_order": ["control_f005", "energy_005", "uniform_005"],
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
        "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
        "margin_weights": [0.0, 0.5],
        # Two quantiles keep this synthetic test fast.  F008's actual config
        # supplies 201 through scoring_spec_from_f008_config.
        "threshold_candidates": 2,
        "probability_temperature": 0.05,
        "maximum_known_preservation_decline_vs_control": 0.001,
        "class_count": 447,
    }


def _embeddings(row_count):
    public = np.zeros((row_count, 512), dtype=np.float32)
    advanced = np.zeros((row_count, 192), dtype=np.float32)
    public[:, 0] = 1.0
    advanced[:, 0] = 1.0
    return public, advanced, np.ones(row_count, dtype=np.bool_)


def _source_receipt():
    return {
        "schema_version": "f007-f005-source-receipt-v1",
        "source_run_directory": "/synthetic/f005",
        "experiment_state": {
            "path": "experiment_state.json",
            "sha256": "b" * 64,
            "status": "complete",
            "experiment_signature": "c" * 64,
        },
        "fold_ids": [0, 1],
        "folds": [
            {
                "outer_fold": outer,
                "full_scoring_control": {
                    "identity": {"path": f"fold_{outer}/identity.json", "sha256": "d" * 64},
                    "receipt": {"path": f"fold_{outer}/receipt.json", "sha256": "e" * 64},
                },
            }
            for outer in (0, 1)
        ],
    }


def _fake_heldout(contract, embeddings, valid, outer):
    # The scorer must receive guarded outer rows.  Calibration labels remain
    # readable, so the fake can construct scores without leaking outer truth.
    positions = {row["audio_file"]: index for index, row in enumerate(contract["manifest"])}
    outer_index = positions[next(
        row["audio_file"] for row in contract["roles"]
        if row["outer_fold"] == outer and row["outer_evaluation_included"]
    )]
    with unittest.TestCase().assertRaises(RuntimeError):
        _ = contract["manifest"][outer_index]["speaker_id"]

    roles = [row for row in contract["roles"] if row["outer_fold"] == outer]
    calibration = np.asarray([
        positions[row["audio_file"]] for row in roles if row["calibration_query"]
    ], dtype=np.int64)
    outer_indices = np.asarray([
        positions[row["audio_file"]] for row in roles if row["outer_evaluation_included"]
    ], dtype=np.int64)
    inner = np.full((len(calibration), 446), -0.5, dtype=np.float32)
    unknown = np.empty(len(calibration), dtype=np.float32)
    for index, row_index in enumerate(calibration):
        label = contract["manifest"][int(row_index)]["speaker_id"]
        if label == "k1":
            inner[index, 0], unknown[index] = 0.9, 0.1
        elif label == "k2":
            inner[index, 1], unknown[index] = 0.9, 0.1
        else:
            inner[index, 0], unknown[index] = 0.1, 0.9
    outer_scores = np.full((len(outer_indices), 446), -0.5, dtype=np.float32)
    outer_scores[:, 0] = 0.9
    return {
        "calibration_indices": calibration,
        "outer_indices": outer_indices,
        "known_labels": _labels()[1:],
        "inner_known_scores": inner,
        "outer_known_scores": outer_scores,
        "inner_unknown_similarity": unknown,
        "outer_unknown_similarity": np.full(len(outer_indices), 0.1, dtype=np.float32),
        "outer_valid": np.asarray(valid[outer_indices], dtype=np.bool_),
        "reference_counts": {},
        "provenance": {
            "protocol": "original_heldout_queries_expanded_gallery_v1",
            "outer_fold": outer,
        },
    }


def _minimal_policy(full_f1, known_f1):
    return {
        "inner_metrics": {
            "inner_macro_f1_full_label_map": full_f1,
            "known_query_macro_f1_over_observed_labels": known_f1,
        },
    }


class F008ScoringTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()
        self.spec = _spec()
        self.public, self.advanced, self.valid = _embeddings(len(self.contract["manifest"]))
        self.source = _source_receipt()

    def _binding(self, outer):
        return scoring.bind_authenticated_f005_control_embeddings(
            self.source, outer, embeddings=self.advanced, valid=self.valid,
        )

    def _prepare(self, root, outer):
        return scoring.prepare_and_seal_pretruth(
            self.contract, outer,
            public_embeddings=self.public,
            frozen_advanced_embeddings=self.advanced,
            f005_control_embeddings=self.advanced,
            f005_source_receipt=self.source,
            f005_control_binding=self._binding(outer),
            f008_advanced_embeddings_by_arm={
                arm: self.advanced.copy() for arm in self.spec["arm_ids"]
            },
            valid=self.valid,
            scoring_spec=self.spec,
            policy_seal_path=root / f"policy-{outer}.json",
            heldout_scorer=_fake_heldout,
        )

    def test_pretruth_seal_is_immutable_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare(root, 0)
            self.assertEqual(prepared["selected_arm"], "control_f005")
            self.assertEqual(set(prepared["score_bundles"]), set(scoring.COMPARATORS))
            self.assertFalse(prepared["outer_truth_read"])
            reloaded = scoring.reload_pretruth_seal(
                root / "policy-0.json", self.contract, 0, scoring_spec=self.spec,
                f005_source_receipt=self.source, f005_control_binding=self._binding(0),
            )
            self.assertEqual(reloaded["seal"], prepared["policy_reload"]["seal"])
            with self.assertRaises(FileExistsError):
                self._prepare(root, 0)

            altered = json.loads((root / "policy-0.json").read_text(encoding="utf-8"))
            altered["selected_arm"] = "energy_005"
            (root / "policy-0.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                scoring.reload_pretruth_seal(
                    root / "policy-0.json", self.contract, 0, scoring_spec=self.spec,
                )

    def test_arm_selection_rejects_high_f1_arm_that_breaks_known_preservation(self):
        policies = {
            "control_f005": _minimal_policy(0.70, 0.90),
            # Stronger full F1 is not enough when known F1 falls below .899.
            "energy_005": _minimal_policy(0.95, 0.85),
            "uniform_005": _minimal_policy(0.80, 0.90),
        }
        selected = scoring.select_arm_from_policies(policies, self.spec)
        self.assertEqual(selected["selected_arm"], "uniform_005")
        self.assertEqual(selected["rejected_by_known_preservation"], ["energy_005"])

        policies["energy_005"] = _minimal_policy(0.80, 0.90)
        tied = scoring.select_arm_from_policies(policies, self.spec)
        self.assertEqual(tied["selected_arm"], "energy_005")

    def test_all_fold_reloads_precede_outer_truth_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = {outer: self._prepare(root, outer) for outer in (0, 1)}
            bindings = {outer: self._binding(outer) for outer in (0, 1)}
            all_reloads = scoring.reload_all_pretruth_seals(
                self.contract, scoring_spec=self.spec,
                policy_paths_by_outer={outer: root / f"policy-{outer}.json" for outer in (0, 1)},
                f005_source_receipt=self.source,
                f005_control_bindings_by_outer=bindings,
            )

            class TruthTrap(dict):
                reads = 0

                def get(self, key, default=None):
                    if key == "speaker_id":
                        type(self).reads += 1
                        raise AssertionError("outer truth was read before all seals")
                    return super().get(key, default)

            incomplete = {**all_reloads, "policy_reloads": {0: all_reloads["policy_reloads"][0]}}
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    self.contract, prepared[0], incomplete,
                    [TruthTrap({"audio_file": "x", "speaker_id": "k1", "group_id": "g",
                                "duration_seconds": 1.0, "has_nonzero_signal": True})],
                    self.contract["labels"], scoring_spec=self.spec,
                    evaluation_path=root / "must-not-exist.json",
                )
            self.assertEqual(TruthTrap.reads, 0)

            outer_rows = [
                {**row, "group_id": self.contract["folds"][index]["group_id"]}
                for index, row in enumerate(self.contract["manifest"])
                if self.contract["folds"][index]["fold"] == 0
            ]
            receipt = scoring.evaluate_outer_once(
                self.contract, prepared[0], all_reloads, outer_rows,
                self.contract["labels"], scoring_spec=self.spec,
                evaluation_path=root / "outer-0.json",
            )
            self.assertTrue(receipt["one_shot_outer_evaluation"])
            self.assertEqual(set(receipt["metrics"]), set(scoring.COMPARATORS))
            with self.assertRaises(FileExistsError):
                scoring.evaluate_outer_once(
                    self.contract, prepared[0], all_reloads, outer_rows,
                    self.contract["labels"], scoring_spec=self.spec,
                    evaluation_path=root / "outer-0.json",
                )

    def test_source_control_binding_rejects_array_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = self.advanced.copy()
            changed[:, 0], changed[:, 1] = 0.0, 1.0
            with self.assertRaises(ValueError):
                scoring.prepare_and_seal_pretruth(
                    self.contract, 0,
                    public_embeddings=self.public,
                    frozen_advanced_embeddings=self.advanced,
                    f005_control_embeddings=changed,
                    f005_source_receipt=self.source,
                    f005_control_binding=self._binding(0),
                    f008_advanced_embeddings_by_arm={
                        arm: changed.copy() for arm in self.spec["arm_ids"]
                    },
                    valid=self.valid, scoring_spec=self.spec,
                    policy_seal_path=root / "drift.json", heldout_scorer=_fake_heldout,
                )

    def test_outer_duration_normalizes_csv_and_numpy_scalars_without_weakening_checks(self):
        self.assertEqual(scoring.normalize_outer_duration("4.25"), 4.25)
        self.assertEqual(
            scoring.normalize_outer_duration(np.float32(4.25)),
            float(np.float32(4.25)),
        )
        self.assertIsInstance(scoring.normalize_outer_duration(np.int64(4)), float)
        for value in (True, np.bool_(True), "nan", np.float32(np.inf), "-0.1", "not-a-number"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    scoring.normalize_outer_duration(value)

    def test_outer_receipt_canonicalizes_numpy_duration_scalar(self):
        original = self.contract
        self.contract = _contract()
        for row in self.contract["manifest"]:
            row["duration_seconds"] = np.float32(4.25)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared = {outer: self._prepare(root, outer) for outer in (0, 1)}
                all_reloads = scoring.reload_all_pretruth_seals(
                    self.contract, scoring_spec=self.spec,
                    policy_paths_by_outer={outer: root / f"policy-{outer}.json" for outer in (0, 1)},
                    f005_source_receipt=self.source,
                    f005_control_bindings_by_outer={outer: self._binding(outer) for outer in (0, 1)},
                )
                outer_rows = [
                    {**row, "group_id": self.contract["folds"][index]["group_id"]}
                    for index, row in enumerate(self.contract["manifest"])
                    if self.contract["folds"][index]["fold"] == 0
                ]
                receipt = scoring.evaluate_outer_once(
                    self.contract, prepared[0], all_reloads, outer_rows,
                    self.contract["labels"], scoring_spec=self.spec,
                    evaluation_path=root / "outer-numpy-duration.json",
                )
                self.assertTrue(all(
                    type(row["duration_seconds"]) is float
                    for row in receipt["outer_reference"]
                ))
        finally:
            self.contract = original

    def test_module_is_pure_and_does_not_depend_on_outer_crossfit(self):
        source = inspect.getsource(scoring)
        self.assertNotIn("import torch", source)
        self.assertNotIn("crossfit" + "_scores", source)
        self.assertIn("all disk-reloaded pretruth seals", inspect.getdoc(scoring.evaluate_outer_once))


if __name__ == "__main__":
    unittest.main()
