import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import f007_scoring as scoring


def _config():
    return {
        "experiment_code": "F007",
        "fold_ids": [0, 1],
        "arms": [
            {"id": "control_f005"}, {"id": "l2sp_001"}, {"id": "l2sp_01"},
        ],
        "selection": {
            "known_query_scope": "original_group_disjoint_calibration_query_rows_only",
            "metric": "known_query_macro_f1_over_observed_labels_then_top1_accuracy",
            "arm_tie_order": list(scoring.ARM_IDS),
            "unknown_calibration_hidden_until_arm_sealed": True,
            "outer_labels_forbidden_until_all_policies_sealed": True,
            "refit_after_selection": False,
        },
        "scoring": {
            "protocol": "c002b_family_disjoint_roles_v1",
            "reference_method": "max_reference",
            "fusion": "same_reference_sqrt_weighted_encoder_concatenation",
            "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
            "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "margin_weights": [0.0, 0.5],
            "threshold_candidates": 201,
            "probability_temperature": 0.05,
            "historical_c002b_fixed_policy": "diagnostic_only_never_selectable",
        },
    }


def _contract():
    rows, folds = [], []
    for fold in (0, 1):
        for kind, label in (("ref", "k1"), ("ref", "k2"),
                            ("query", "k1"), ("query", "k2"),
                            ("query", "unknown")):
            name = f"fold{fold}_{kind}_{label}.wav"
            group = f"g_{fold}_{kind}_{label}"
            rows.append({
                "audio_file": name, "speaker_id": label,
                "duration_seconds": 4.0, "has_nonzero_signal": True,
            })
            folds.append({
                "audio_file": name, "group_id": group,
                "fold": fold, "train_eligible": True,
            })
    roles = []
    for outer in (0, 1):
        for row, split in zip(rows, folds, strict=True):
            kind = "ref" if "_ref_" in row["audio_file"] else "query"
            evaluation = split["fold"] == outer
            roles.append({
                "audio_file": row["audio_file"], "speaker_id": row["speaker_id"],
                "group_id": split["group_id"], "outer_fold": outer,
                "encoder_fit_allowed": (not evaluation and kind == "ref"),
                "enrollment_allowed": False,
                "calibration_query": (not evaluation and kind == "query"),
                "outer_evaluation_included": evaluation,
            })
    return {
        "signature": "a" * 64, "config": _config(),
        "labels": ["unknown", "k1", "k2"],
        "manifest": rows, "folds": folds, "roles": roles,
    }


def _embeddings(row_count):
    public = np.zeros((row_count, 512), dtype=np.float32)
    advanced = np.zeros((row_count, 192), dtype=np.float32)
    public[:, 0], advanced[:, 0] = 1.0, 1.0
    return public, advanced, np.ones(row_count, dtype=np.bool_)


def _fake_heldout(contract, embeddings, valid, outer):
    # The fake proves that policy construction receives the guarded outer rows.
    outer_index = next(
        index for index, fold in enumerate(contract["folds"])
        if int(fold["fold"]) == outer
    )
    with unittest.TestCase().assertRaises(RuntimeError):
        _ = contract["manifest"][outer_index]["speaker_id"]
    positions = {row["audio_file"]: index for index, row in enumerate(contract["manifest"])}
    query = np.asarray([
        positions[row["audio_file"]] for row in contract["roles"]
        if int(row["outer_fold"]) == outer and row["calibration_query"]
    ], dtype=np.int64)
    outer_indices = np.asarray([
        index for index, fold in enumerate(contract["folds"])
        if int(fold["fold"]) == outer
    ], dtype=np.int64)
    inner = np.zeros((len(query), 2), dtype=np.float32)
    for position, index in enumerate(query):
        label = contract["manifest"][int(index)]["speaker_id"]
        inner[position] = {"k1": (0.9, 0.1), "k2": (0.1, 0.9), "unknown": (0.2, 0.1)}[label]
    outer_known = np.tile(np.asarray([[0.9, 0.1]], dtype=np.float32), (len(outer_indices), 1))
    return {
        "calibration_indices": query, "outer_indices": outer_indices,
        "known_labels": ["k1", "k2"],
        "inner_known_scores": inner, "outer_known_scores": outer_known,
        "inner_unknown_similarity": np.asarray([0.1, 0.1, 0.8], dtype=np.float32),
        "outer_unknown_similarity": np.full(len(outer_indices), 0.1, dtype=np.float32),
        "outer_valid": np.asarray(valid[outer_indices], dtype=np.bool_),
        "reference_counts": {},
        "provenance": {
            "protocol": "original_heldout_queries_expanded_gallery_v1", "outer_fold": outer,
        },
    }


class F007ScoringTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()
        self.public, self.advanced, self.valid = _embeddings(len(self.contract["manifest"]))

    def _selection(self, root, outer):
        return scoring.select_and_seal_known_arm(
            self.contract, outer,
            embeddings_by_arm={arm: self.advanced.copy() for arm in scoring.ARM_IDS},
            valid=self.valid, selection_seal_path=root / f"selection-{outer}.json",
        )

    def test_known_only_group_disjoint_selection_seals_control_on_tie(self):
        evidence = scoring.known_selection_scores(
            self.contract, self.advanced, self.valid, 0,
        )
        query_labels = [
            self.contract["manifest"][int(index)]["speaker_id"]
            for index in evidence["known_calibration_indices"]
        ]
        self.assertEqual(query_labels, ["k1", "k2"])
        self.assertTrue(np.all(evidence["reference_support"] >= 1))
        self.assertFalse(evidence["provenance"]["unknown_similarity_computed"])
        self.assertFalse(evidence["provenance"]["unknown_reference_cohort_materialized"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reload = self._selection(root, 0)
            self.assertEqual(reload["seal"]["selected_arm"], "control_f005")
            self.assertEqual(
                reload["seal"]["scientific_conclusion"],
                "control_retained_on_known_calibration",
            )
            self.assertEqual(
                scoring.reload_selection_seal(root / "selection-0.json", self.contract, 0)["seal"],
                reload["seal"],
            )

    def test_policy_waits_for_selection_and_outer_waits_for_both_fold_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forged = {"kind": "f007_selection_disk_reload", "disk_reloaded": False}
            with patch.object(scoring, "heldout_reference_scores") as heldout:
                with self.assertRaises(ValueError):
                    scoring.prepare_and_seal_policy(
                        self.contract, 0, public_embeddings=self.public,
                        frozen_advanced_embeddings=self.advanced,
                        reused_control_embeddings=self.advanced,
                        selected_arm_embeddings=self.advanced, valid=self.valid,
                        selection_reload=forged, policy_seal_path=root / "forged.json",
                    )
                heldout.assert_not_called()

            selections = {outer: self._selection(root, outer) for outer in (0, 1)}
            pretruth = {}
            policy_paths = {}
            for outer in (0, 1):
                path = root / f"policy-{outer}.json"
                pretruth[outer] = scoring.prepare_and_seal_policy(
                    self.contract, outer, public_embeddings=self.public,
                    frozen_advanced_embeddings=self.advanced,
                    reused_control_embeddings=self.advanced,
                    selected_arm_embeddings=self.advanced, valid=self.valid,
                    selection_reload=selections[outer], policy_seal_path=path,
                    heldout_scorer=_fake_heldout,
                )
                policy_paths[outer] = path
            all_reloads = scoring.reload_all_policy_seals(
                self.contract, policy_paths_by_outer=policy_paths,
                selection_reloads_by_outer=selections,
            )

            class TruthTrap(dict):
                reads = 0

                def get(self, key, default=None):
                    if key == "speaker_id":
                        type(self).reads += 1
                        raise AssertionError("outer truth was read too early")
                    return super().get(key, default)

                def __getitem__(self, key):
                    if key == "speaker_id":
                        type(self).reads += 1
                        raise AssertionError("outer truth was read too early")
                    return super().__getitem__(key)

            trap = TruthTrap({
                "audio_file": "x", "speaker_id": "k1", "group_id": "g",
                "duration_seconds": 1.0, "has_nonzero_signal": True,
            })
            incomplete = {**all_reloads, "policy_reloads": {0: all_reloads["policy_reloads"][0]}}
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    self.contract, pretruth[0], incomplete, [trap], self.contract["labels"],
                    evaluation_path=root / "forbidden.json",
                )
            self.assertEqual(TruthTrap.reads, 0)

            outer_rows = [
                {**row, "group_id": self.contract["folds"][index]["group_id"]}
                for index, row in enumerate(self.contract["manifest"])
                if self.contract["folds"][index]["fold"] == 0
            ]
            fake_metrics = {
                "row_count": len(outer_rows), "class_count": 3, "macro_f1": 0.5,
                "accuracy": 0.5,
                "errors": {"known_to_unknown": 0, "unknown_to_known": 0, "known_to_other_known": 0},
            }
            with patch.object(scoring, "score_predictions", return_value=fake_metrics):
                result = scoring.evaluate_outer_once(
                    self.contract, pretruth[0], all_reloads, outer_rows, self.contract["labels"],
                    evaluation_path=root / "outer-0.json",
                )
            self.assertEqual(set(result["comparators"]), set(scoring.COMPARATORS))
            with self.assertRaises(FileExistsError):
                with patch.object(scoring, "score_predictions", return_value=fake_metrics):
                    scoring.evaluate_outer_once(
                        self.contract, pretruth[0], all_reloads, outer_rows, self.contract["labels"],
                        evaluation_path=root / "outer-0.json",
                    )

    def test_module_stays_role_safe_and_does_not_use_crossfit_scorer(self):
        source = inspect.getsource(scoring)
        self.assertNotIn("crossfit" + "_scores", source)
        self.assertNotIn("f005_scoring", source)
        self.assertIn("both disk-reloaded policy seals", inspect.getdoc(scoring.evaluate_outer_once))


if __name__ == "__main__":
    unittest.main()
