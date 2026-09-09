import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.evaluation.metrics import score_predictions
from speaker_id.training import f005_scoring as scoring


def _config(class_count=3):
    return {
        "fold_ids": [0, 1],
        "arms": [{"id": arm} for arm in scoring.ARM_IDS],
        "views": {"long_seconds": 8.0},
        "scoring": {
            "protocol": "heldout_reference_scores_original_disjoint_roles",
            "reference_method": "max_reference",
            "alphas": list(scoring.ALPHAS),
            "alpha_tie_order": list(scoring.TIE_ORDER),
            "unknown_weights": list(scoring.UNKNOWN_WEIGHTS),
            "margin_weights": list(scoring.MARGIN_WEIGHTS),
            "threshold_candidates": 201,
            "probability_temperature": 0.05,
            "no_outer_tuning": True,
            "no_exact_c002b_reproduction_claim": True,
        },
        "bootstrap": {
            "kind": "paired_whole_content_group_unstratified",
            "mixed_label_groups_preserved": True,
            "true_class_purity_assumed": False,
            "seed": 11,
            "replicates": 100,
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
        },
        "promotion": {
            "goal_target_oof_macro_f1": 0.965,
            "incumbent_promotion_is_distinct_from_goal_completion": True,
            "minimum_treatment_delta_vs_fresh_control": 0.003,
            "minimum_selected_delta_vs_c002b": 0.003,
            "minimum_control_delta_vs_c002b_when_selected": 0.003,
            "minimum_accuracy_delta_vs_c002b": 0.0,
            "minimum_short_known_top1_delta_vs_c002b": 0.0,
            "minimum_each_fold_delta_vs_control": -0.001,
            "minimum_each_fold_delta_vs_c002b": -0.001,
            "maximum_unknown_to_known_increase_vs_c002b": 2,
            "maximum_known_to_other_known_increase_vs_c002b": 0,
            "minimum_group_bootstrap_lower_bound_vs_control": -0.0005,
            "minimum_group_bootstrap_lower_bound_vs_c002b": -0.0005,
            "require_exact_cpu_cuda_prediction_parity": True,
            "all_conditions_required": True,
            "otherwise": "retain_c002b",
        },
        "evaluation_classes": class_count,
    }


def _selection_seal(signature="experiment", outer=0, arm="treatment_mse01"):
    metrics = {
        name: {
            "macro_f1_observed_known_labels": 0.6 if name == arm else 0.5,
            "top1_accuracy": 0.6 if name == arm else 0.5,
        }
        for name in scoring.ARM_IDS
    }
    body = {
        "schema_version": 1,
        "experiment_signature": signature,
        "outer_fold": outer,
        "selected_arm": arm,
        "scientific_conclusion": (
            "consistency_not_supported_on_known_calibration"
            if arm == "control" else "consistency_candidate_selected"
        ),
        "arm_metrics": metrics,
        "selection_order": [
            "macro_f1_observed_known_labels", "top1_accuracy", "fixed_arm_tie_order"
        ],
        "known_calibration_indices_sha256": "a" * 64,
        "known_query_rows": 1,
        "observed_known_labels": 1,
        "absent_known_labels": [],
        "unknown_calibration_indices_materialized": False,
        "unknown_reference_cohort_materialized": False,
        "unknown_similarity_computed": False,
        "outer_rows_or_labels_read": False,
        "refit_after_selection": False,
    }
    return {**body, "seal_sha256": scoring._sha(body)}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _pretruth_contract():
    manifest = [
        {"audio_file": "known.wav", "speaker_id": "k1", "duration_seconds": "9.0", "has_nonzero_signal": "true"},
        {"audio_file": "unknown.wav", "speaker_id": "unknown", "duration_seconds": "4.0", "has_nonzero_signal": "true"},
        {"audio_file": "outer.wav", "speaker_id": "k2", "duration_seconds": "2.0", "has_nonzero_signal": "true"},
        {"audio_file": "reference.wav", "speaker_id": "k2", "duration_seconds": "9.0", "has_nonzero_signal": "true"},
    ]
    roles = []
    for outer in (0, 1):
        for index, row in enumerate(manifest):
            roles.append({
                **row,
                "outer_fold": outer,
                "outer_evaluation_included": index == 2,
            })
    return {
        "signature": "experiment",
        "config": _config(),
        "labels": ["unknown"] + [f"k{i}" for i in range(1, 447)],
        "manifest": manifest,
        "roles": roles,
        "folds": [
            {"audio_file": row["audio_file"], "group_id": f"g{index}",
             "fold": 0 if index == 2 else 1}
            for index, row in enumerate(manifest)
        ],
    }


def _fake_heldout(contract, embeddings, valid, outer):
    # The scorer receives a raising sentinel in place of every outer label.
    with unittest.TestCase().assertRaises(RuntimeError):
        _ = contract["manifest"][2]["speaker_id"]
    inner_known = np.full((2, 446), -0.2, dtype=np.float32)
    outer_known = np.full((1, 446), -0.2, dtype=np.float32)
    inner_known[0, 0], inner_known[1, 0] = 0.9, 0.1
    outer_known[0, 1] = 0.9
    return {
        "calibration_indices": np.asarray([0, 1], dtype=np.int64),
        "outer_indices": np.asarray([2], dtype=np.int64),
        "known_labels": [f"k{i}" for i in range(1, 447)],
        "inner_known_scores": inner_known,
        "outer_known_scores": outer_known,
        "inner_unknown_similarity": np.asarray([0.1, 0.8], dtype=np.float32),
        "outer_unknown_similarity": np.asarray([0.1], dtype=np.float32),
        "outer_valid": np.asarray([True]),
        "provenance": {
            "protocol": "original_heldout_queries_expanded_gallery_v1",
            "outer_fold": outer,
        },
    }


class F005PretruthScoringTests(unittest.TestCase):
    def setUp(self):
        self.contract = _pretruth_contract()

    def _embeddings(self):
        public = np.zeros((4, 512), dtype=np.float32)
        frozen = np.zeros((4, 192), dtype=np.float32)
        public[:, 0] = 1
        frozen[:, 0] = 1
        arms = {arm: frozen.copy() for arm in ("control", "treatment_mse01")}
        return public, frozen, arms, np.ones(4, dtype=np.bool_)

    def test_full_heldout_scoring_cannot_start_before_valid_disk_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arm.json"
            malformed = _selection_seal()
            malformed["unknown_similarity_computed"] = True
            body = {key: value for key, value in malformed.items() if key != "seal_sha256"}
            malformed["seal_sha256"] = scoring._sha(body)
            _write_json(path, malformed)
            with self.assertRaises(ValueError):
                scoring.reload_arm_selection_seal(path, self.contract, 0)

            public, frozen, arms, valid = self._embeddings()
            forged = {
                "kind": "f005_arm_selection_disk_reload",
                "disk_reloaded": True,
                "path": str(path),
            }
            with patch.object(scoring, "heldout_reference_scores") as scorer:
                with self.assertRaises((ValueError, KeyError)):
                    scoring.prepare_and_seal_inner_policies(
                        self.contract, 0, public_embeddings=public,
                        frozen_advanced_embeddings=frozen,
                        advanced_embeddings_by_arm=arms, valid=valid,
                        arm_selection_reload=forged,
                        policy_seal_path=root / "policy.json",
                    )
                scorer.assert_not_called()

    def test_arm_winner_is_recomputed_instead_of_trusting_sealed_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            seal = _selection_seal()
            seal["selected_arm"] = "treatment_mse05"
            body = {key: value for key, value in seal.items() if key != "seal_sha256"}
            seal["seal_sha256"] = scoring._sha(body)
            _write_json(path, seal)
            with self.assertRaises(ValueError):
                scoring.reload_arm_selection_seal(path, self.contract, 0)

    def test_three_policies_are_sealed_before_outer_truth_and_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_path = root / "arm.json"
            _write_json(arm_path, _selection_seal())
            arm = scoring.reload_arm_selection_seal(arm_path, self.contract, 0)
            public, frozen, arms, valid = self._embeddings()
            policy_path = root / "policies.json"
            with patch.object(scoring, "heldout_reference_scores", side_effect=_fake_heldout) as scorer:
                prepared = scoring.prepare_and_seal_inner_policies(
                    self.contract, 0, public_embeddings=public,
                    frozen_advanced_embeddings=frozen,
                    advanced_embeddings_by_arm=arms, valid=valid,
                    arm_selection_reload=arm, policy_seal_path=policy_path,
                )
            self.assertEqual(scorer.call_count, 3 * len(scoring.ALPHAS))
            self.assertEqual(set(prepared["score_bundles"]), set(scoring.COMPARATORS))
            seal = prepared["policy_reload"]["seal"]
            self.assertEqual(set(seal["policies"]), set(scoring.COMPARATORS))
            self.assertTrue(seal["unknown_scoring_started_after_arm_seal_reload"])
            self.assertFalse(seal["outer_truth_read"])
            self.assertEqual(set(arms), {"control", "treatment_mse01"})

            with patch.object(scoring, "heldout_reference_scores", side_effect=_fake_heldout) as recovered_scorer:
                recovered = scoring.rebuild_pretruth_from_policy_seal(
                    self.contract, 0, public_embeddings=public,
                    frozen_advanced_embeddings=frozen,
                    advanced_embeddings_by_arm=arms, valid=valid,
                    arm_selection_reload=arm,
                    policy_reload=prepared["policy_reload"],
                )
            self.assertEqual(recovered_scorer.call_count, 3)
            self.assertEqual(set(recovered), set(prepared))
            for key in set(prepared) - {"score_bundles"}:
                self.assertEqual(recovered[key], prepared[key])
            for comparator in scoring.COMPARATORS:
                self.assertEqual(
                    scoring._score_evidence(recovered["score_bundles"][comparator]),
                    scoring._score_evidence(prepared["score_bundles"][comparator]),
                )

            truth = [{
                "audio_file": "outer.wav", "speaker_id": "k2", "group_id": "g2",
                "duration_seconds": 2.0, "has_nonzero_signal": True,
            }]
            evaluation_path = root / "outer.json"
            result = scoring.evaluate_outer_once(
                prepared, prepared["policy_reload"], truth,
                self.contract["labels"], evaluation_path=evaluation_path,
            )
            self.assertEqual(result["comparators"]["selected_arm"]["metrics"]["accuracy"], 1.0)
            with self.assertRaises(FileExistsError):
                scoring.evaluate_outer_once(
                    prepared, prepared["policy_reload"], truth,
                    self.contract["labels"], evaluation_path=evaluation_path,
                )

    def test_rejected_arm_full_embeddings_are_forbidden_after_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_path = root / "arm.json"
            _write_json(arm_path, _selection_seal())
            arm = scoring.reload_arm_selection_seal(arm_path, self.contract, 0)
            public, frozen, arms, valid = self._embeddings()
            arms["treatment_mse0"] = frozen.copy()
            with patch.object(scoring, "heldout_reference_scores") as scorer:
                with self.assertRaises(ValueError):
                    scoring.prepare_and_seal_inner_policies(
                        self.contract, 0, public_embeddings=public,
                        frozen_advanced_embeddings=frozen,
                        advanced_embeddings_by_arm=arms, valid=valid,
                        arm_selection_reload=arm,
                        policy_seal_path=root / "policy.json",
                    )
                scorer.assert_not_called()

    def test_policy_tamper_and_score_alignment_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_path = root / "arm.json"
            _write_json(arm_path, _selection_seal())
            arm = scoring.reload_arm_selection_seal(arm_path, self.contract, 0)
            public, frozen, arms, valid = self._embeddings()
            with patch.object(scoring, "heldout_reference_scores", side_effect=_fake_heldout):
                prepared = scoring.prepare_and_seal_inner_policies(
                    self.contract, 0, public_embeddings=public,
                    frozen_advanced_embeddings=frozen,
                    advanced_embeddings_by_arm=arms, valid=valid,
                    arm_selection_reload=arm, policy_seal_path=root / "policy.json",
                )
            prepared["score_bundles"]["selected_arm"]["outer_known_scores"][0, 0] += 0.01
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    prepared, prepared["policy_reload"], [], self.contract["labels"],
                    evaluation_path=root / "outer.json",
                )

            # A caller cannot keep using a reload token after the disk seal changes.
            policy_path = root / "policy.json"
            altered = json.loads(policy_path.read_text(encoding="utf-8"))
            altered["policies"]["selected_arm"]["calibration"]["threshold"] += 0.01
            altered_body = {
                key: value for key, value in altered.items() if key != "seal_sha256"
            }
            altered["seal_sha256"] = scoring._sha(altered_body)
            _write_json(root / "altered.json", altered)
            policy_path.unlink()
            (root / "altered.json").replace(policy_path)
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    prepared, prepared["policy_reload"], [], self.contract["labels"],
                    evaluation_path=root / "outer-after-seal-drift.json",
                )

    def test_label_order_and_public_metadata_fail_before_truth_access(self):
        class TruthTrap(dict):
            reads = 0

            def get(self, key, default=None):
                if key == "speaker_id":
                    type(self).reads += 1
                    raise AssertionError("truth was accessed")
                return super().get(key, default)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_path = root / "arm.json"
            _write_json(arm_path, _selection_seal())
            arm = scoring.reload_arm_selection_seal(arm_path, self.contract, 0)
            public, frozen, arms, valid = self._embeddings()
            with patch.object(scoring, "heldout_reference_scores", side_effect=_fake_heldout):
                prepared = scoring.prepare_and_seal_inner_policies(
                    self.contract, 0, public_embeddings=public,
                    frozen_advanced_embeddings=frozen,
                    advanced_embeddings_by_arm=arms, valid=valid,
                    arm_selection_reload=arm, policy_seal_path=root / "policy.json",
                )
            trap = TruthTrap({
                "audio_file": "outer.wav", "speaker_id": "k2", "group_id": "g2",
                "duration_seconds": 3.0, "has_nonzero_signal": True,
            })
            wrong_labels = self.contract["labels"].copy()
            wrong_labels[1], wrong_labels[2] = wrong_labels[2], wrong_labels[1]
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    prepared, prepared["policy_reload"], [trap], wrong_labels,
                    evaluation_path=root / "wrong-labels.json",
                )
            self.assertEqual(TruthTrap.reads, 0)
            with self.assertRaises(ValueError):
                scoring.evaluate_outer_once(
                    prepared, prepared["policy_reload"], [trap], self.contract["labels"],
                    evaluation_path=root / "wrong-metadata.json",
                )
            self.assertEqual(TruthTrap.reads, 0)

    def test_module_has_no_full_crossfit_dependency_or_outer_truth_pretruth_argument(self):
        source = inspect.getsource(scoring)
        forbidden = "crossfit" + "_scores"
        self.assertNotIn(forbidden, source)
        signature = inspect.signature(scoring.prepare_and_seal_inner_policies)
        self.assertNotIn("outer_truth", signature.parameters)


def _prediction_rows(reference, guesses):
    return [
        {"audio_file": row["audio_file"], "speaker_id": guess}
        for row, guess in zip(reference, guesses, strict=True)
    ]


def _fold_receipt(signature, outer, arm, reference, selected, control, frozen):
    labels = ["unknown"] + [f"k{i}" for i in range(1, 447)]
    comparators = {}
    for name, guesses in (
        ("selected_arm", selected), ("fresh_control", control),
        ("frozen_same_protocol", frozen),
    ):
        rows = _prediction_rows(reference, guesses)
        comparators[name] = {
            "policy": {}, "predictions": rows,
            "known_top1_predictions": rows,
            "metrics": score_predictions(reference, rows, labels),
            "duration_slices": {},
        }
    body = {
        "schema_version": scoring.EVALUATION_SCHEMA,
        "experiment_signature": signature,
        "outer_fold": outer,
        "selected_arm": arm,
        "policy_seal_sha256": "a" * 64,
        "policy_file_sha256": "b" * 64,
        "outer_reference": reference,
        "comparators": comparators,
        "one_shot_outer_evaluation": True,
        "outer_truth_first_access_stage": "after_all_three_policy_seals_reloaded",
    }
    return {**body, "evaluation_sha256": scoring._sha(body)}


class F005PromotionTests(unittest.TestCase):
    def _case(self, arm="treatment_mse01", candidate_equals_control=False):
        labels = ["unknown"] + [f"k{i}" for i in range(1, 447)]
        reference, folds = [], []
        truth_labels = ("unknown", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8")
        for outer in (0, 1):
            for index, speaker in enumerate(truth_labels):
                name = f"f{outer}-{index}.wav"
                reference.append({
                    "audio_file": name, "speaker_id": speaker,
                    "group_id": f"g{outer}-{index}",
                    "duration_seconds": 2.0 if index else 10.0,
                    "has_nonzero_signal": True,
                })
                folds.append({"audio_file": name, "group_id": f"g{outer}-{index}", "fold": outer})
        contract = {
            "signature": "aggregate", "labels": labels, "manifest": reference,
            "folds": folds, "config": _config(),
        }
        results = []
        historical, historical_top1 = [], []
        for outer in (0, 1):
            rows_per_fold = len(truth_labels)
            rows = reference[outer * rows_per_fold:(outer + 1) * rows_per_fold]
            truth_values = [row["speaker_id"] for row in rows]
            control = truth_values.copy()
            if arm != "control":
                control[1] = "unknown"
                control[2] = "unknown"
            selected = control.copy() if candidate_equals_control else truth_values
            frozen = control.copy()
            results.append(_fold_receipt(
                "aggregate", outer, arm, rows, selected, control, frozen,
            ))
            old = truth_values.copy()
            old[1] = "k2"
            old[2] = "k3"
            historical.extend(_prediction_rows(rows, old))
            historical_top1.extend(_prediction_rows(rows, old))
        return contract, results, historical, historical_top1

    def test_treatment_must_beat_control_and_c002b(self):
        contract, results, old, old_top1 = self._case()
        report = scoring.aggregate_oof_and_decide(
            contract, results, historical_c002b_predictions=old,
            historical_c002b_known_top1_predictions=old_top1,
            exact_cpu_cuda_prediction_parity=True,
        )
        self.assertEqual(report["selection_kind"], "treatment")
        self.assertTrue(report["promote_new_incumbent"])
        self.assertFalse(report["goal_reached"])
        self.assertIn("development_metric_goal_reached", report)
        self.assertEqual(report["goal_outstanding"], [
            "clean_reproduction", "offline_package_qa"
        ])
        self.assertEqual(report["decision"], "promote_selected_f005")

        contract, results, old, old_top1 = self._case(candidate_equals_control=True)
        rejected = scoring.aggregate_oof_and_decide(
            contract, results, historical_c002b_predictions=old,
            historical_c002b_known_top1_predictions=old_top1,
            exact_cpu_cuda_prediction_parity=True,
        )
        self.assertFalse(rejected["conditions"]["conditional_pooled_macro_f1_gain"])
        self.assertFalse(rejected["promote_new_incumbent"])

    def test_selected_control_is_a_candidate_and_uses_c002b_gain_gate(self):
        contract, results, old, old_top1 = self._case(
            arm="control", candidate_equals_control=True,
        )
        report = scoring.aggregate_oof_and_decide(
            contract, results, historical_c002b_predictions=old,
            historical_c002b_known_top1_predictions=old_top1,
            exact_cpu_cuda_prediction_parity=True,
        )
        self.assertEqual(report["selection_kind"], "control")
        self.assertTrue(report["conditions"]["conditional_pooled_macro_f1_gain"])
        self.assertTrue(report["promote_new_incumbent"])

    def test_alignment_and_parity_are_hard_guards(self):
        contract, results, old, old_top1 = self._case()
        missing = old[:-1]
        with self.assertRaises(ValueError):
            scoring.aggregate_oof_and_decide(
                contract, results, historical_c002b_predictions=missing,
                historical_c002b_known_top1_predictions=old_top1,
                exact_cpu_cuda_prediction_parity=True,
            )
        report = scoring.aggregate_oof_and_decide(
            contract, results, historical_c002b_predictions=old,
            historical_c002b_known_top1_predictions=old_top1,
            exact_cpu_cuda_prediction_parity=False,
        )
        self.assertFalse(report["conditions"]["exact_cpu_cuda_prediction_parity"])
        self.assertFalse(report["promote_new_incumbent"])

    def test_aggregate_rejects_public_metadata_drift_even_with_rehashed_receipt(self):
        contract, results, old, old_top1 = self._case()
        result = json.loads(json.dumps(results[0]))
        result["outer_reference"][0]["duration_seconds"] = 99.0
        body = {key: value for key, value in result.items() if key != "evaluation_sha256"}
        result["evaluation_sha256"] = scoring._sha(body)
        with self.assertRaises(ValueError):
            scoring.aggregate_oof_and_decide(
                contract, [result, results[1]], historical_c002b_predictions=old,
                historical_c002b_known_top1_predictions=old_top1,
                exact_cpu_cuda_prediction_parity=True,
            )


if __name__ == "__main__":
    unittest.main()
