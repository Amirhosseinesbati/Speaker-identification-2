"""Synthetic safety tests for the separate F008 E0 outer stage."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training import f008_e0_evaluation as e0
from speaker_id.training import f008_scoring as scoring


def _labels() -> list[str]:
    return ["unknown"] + [f"k{index}" for index in range(1, 447)]


def _contract() -> dict[str, object]:
    manifest, folds = [], []
    for fold in (0, 1):
        for kind, label in (
            ("reference", "k1"), ("reference", "k2"),
            ("query", "k1"), ("query", "k2"), ("query", "unknown"),
        ):
            name = f"fold{fold}_{kind}_{label}.wav"
            manifest.append({
                "audio_file": name, "speaker_id": label,
                "duration_seconds": 4.0, "has_nonzero_signal": True,
            })
            folds.append({
                "audio_file": name, "group_id": f"group_{fold}_{kind}_{label}",
                "fold": fold, "train_eligible": True,
            })
    roles = []
    for outer in (0, 1):
        for row, fold in zip(manifest, folds, strict=True):
            is_outer = fold["fold"] == outer
            is_reference = "_reference_" in row["audio_file"]
            roles.append({
                "audio_file": row["audio_file"], "speaker_id": row["speaker_id"],
                "group_id": fold["group_id"], "outer_fold": outer,
                "encoder_fit_allowed": not is_outer and is_reference,
                "enrollment_allowed": False,
                "calibration_query": not is_outer and not is_reference,
                "outer_evaluation_included": is_outer,
            })
    return {
        "signature": "a" * 64,
        "config": {"fold_ids": [0, 1]},
        "labels": _labels(), "manifest": manifest, "folds": folds, "roles": roles,
    }


def _spec() -> dict[str, object]:
    return {
        "arm_ids": list(e0.E0_ACTIVE_ARMS),
        "control_arm_id": "control_f005",
        "arm_tie_order": list(e0.E0_ACTIVE_ARMS),
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_tie_order": [0.0, 1.0, 0.25, 0.5, 0.75],
        "unknown_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
        "margin_weights": [0.0, 0.5],
        "threshold_candidates": 2,
        "probability_temperature": 0.05,
        "maximum_known_preservation_decline_vs_control": 0.001,
        "class_count": 447,
    }


def _config() -> dict[str, object]:
    spec = _spec()
    return {
        "fold_ids": [0, 1], "evaluation_classes": 447,
        "arms": [
            {"id": "control_f005", "kind": "control", "lambda": 0.0},
            {"id": "energy_005", "kind": "energy", "lambda": 0.05},
        ],
        "selection": {
            "tie_order": list(e0.E0_ACTIVE_ARMS),
            "maximum_known_preservation_decline_vs_control": spec[
                "maximum_known_preservation_decline_vs_control"
            ],
        },
        "scoring": {
            key: spec[key] for key in (
                "alphas", "alpha_tie_order", "unknown_weights", "margin_weights",
                "threshold_candidates", "probability_temperature",
            )
        },
    }


def _embeddings(row_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    public = np.zeros((row_count, 512), dtype=np.float32)
    advanced = np.zeros((row_count, 192), dtype=np.float32)
    public[:, 0] = 1.0
    advanced[:, 0] = 1.0
    return public, advanced, np.ones(row_count, dtype=np.bool_)


def _source_receipt() -> dict[str, object]:
    return {
        "schema_version": "f007-f005-source-receipt-v1",
        "source_run_directory": "/synthetic/f005",
        "experiment_state": {
            "path": "experiment_state.json", "sha256": "b" * 64,
            "status": "complete", "experiment_signature": "c" * 64,
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
    positions = {row["audio_file"]: index for index, row in enumerate(contract["manifest"])}
    outer_index = positions[next(
        row["audio_file"] for row in contract["roles"]
        if row["outer_fold"] == outer and row["outer_evaluation_included"]
    )]
    # The recompute path must still guard outer labels before a policy exists.
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
    for position, row_index in enumerate(calibration):
        label = contract["manifest"][int(row_index)]["speaker_id"]
        if label == "k1":
            inner[position, 0], unknown[position] = 0.9, 0.1
        elif label == "k2":
            inner[position, 1], unknown[position] = 0.9, 0.1
        else:
            inner[position, 0], unknown[position] = 0.1, 0.9
    outer_scores = np.full((len(outer_indices), 446), -0.5, dtype=np.float32)
    outer_scores[:, 0] = 0.9
    return {
        "calibration_indices": calibration, "outer_indices": outer_indices,
        "known_labels": _labels()[1:], "inner_known_scores": inner,
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


class F008E0OuterEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = _contract()
        self.spec = _spec()
        self.public, self.advanced, self.valid = _embeddings(len(self.contract["manifest"]))
        self.source = _source_receipt()

    def _binding(self, outer: int) -> dict[str, object]:
        return scoring.bind_authenticated_f005_control_embeddings(
            self.source, outer, embeddings=self.advanced, valid=self.valid,
        )

    def _prepared(self, root: Path, outer: int) -> e0.E0PreparedFold:
        policy = root / f"policy-{outer}.json"
        scoring.prepare_and_seal_pretruth(
            self.contract, outer, public_embeddings=self.public,
            frozen_advanced_embeddings=self.advanced,
            f005_control_embeddings=self.advanced,
            f005_source_receipt=self.source, f005_control_binding=self._binding(outer),
            f008_advanced_embeddings_by_arm={
                "control_f005": self.advanced.copy(),
                "energy_005": self.advanced.copy(),
            },
            valid=self.valid, scoring_spec=self.spec, policy_seal_path=policy,
            heldout_scorer=_fake_heldout,
        )
        before = policy.read_bytes()
        rebuilt = scoring.rebuild_and_validate_pretruth_bundle(
            self.contract, outer, public_embeddings=self.public,
            frozen_advanced_embeddings=self.advanced,
            f005_control_embeddings=self.advanced,
            f005_source_receipt=self.source, f005_control_binding=self._binding(outer),
            f008_advanced_embeddings_by_arm={
                "control_f005": self.advanced.copy(),
                "energy_005": self.advanced.copy(),
            },
            valid=self.valid, scoring_spec=self.spec, policy_seal_path=policy,
            heldout_scorer=_fake_heldout,
        )
        self.assertEqual(policy.read_bytes(), before)
        self.assertEqual(rebuilt["selected_arm"], "control_f005")
        return e0.E0PreparedFold(
            outer_fold=outer, pretruth=rebuilt, policy_path=policy,
            f005_control_binding=self._binding(outer),
            checkpoint_receipt_sha256="f" * 64,
            cache_identity_signature="0" * 64,
            cache_receipt_sha256="1" * 64,
        )

    def test_named_arms_are_evaluated_when_control_was_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = {outer: self._prepared(root, outer) for outer in (0, 1)}
            output = root / "outer"; output.mkdir()
            report, receipts = e0.evaluate_rebuilt_e0_outer(
                contract=self.contract, config=_config(),
                source_receipt=self.source, prepared=prepared, output_directory=output,
            )
            self.assertEqual(set(receipts), {0, 1})
            for receipt in receipts.values():
                self.assertEqual(receipt["selected_arm"], "control_f005")
                self.assertEqual(set(receipt["predictions"]), set(e0.E0_ACTIVE_ARMS))
                self.assertEqual(receipt["active_arm_ids"], list(e0.E0_ACTIVE_ARMS))
            self.assertEqual(report["active_arm_ids"], list(e0.E0_ACTIVE_ARMS))
            self.assertEqual(len(report["class_f1_deltas_energy_minus_control"]), 447)
            self.assertEqual(report["deltas"]["macro_f1_energy_minus_control"], 0.0)
            self.assertFalse(report["selection_or_promotion_allowed"])
            self.assertFalse(report["local_model_transfer"])
            self.assertTrue((output / "outer_evaluation_report.json").is_file())

    def test_materialized_outer_rows_normalize_csv_duration_metadata(self) -> None:
        original = self.contract
        self.contract = _contract()
        for row in self.contract["manifest"]:
            row["duration_seconds"] = "4.25"
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared = {outer: self._prepared(root, outer) for outer in (0, 1)}
                output = root / "outer"
                output.mkdir()
                report, receipts = e0.evaluate_rebuilt_e0_outer(
                    contract=self.contract, config=_config(), source_receipt=self.source,
                    prepared=prepared, output_directory=output,
                )
                self.assertEqual(set(receipts), {0, 1})
                self.assertEqual(report["metrics"]["control_f005"]["row_count"], 10)
                self.assertTrue(all(
                    type(row["duration_seconds"]) is float
                    for receipt in receipts.values() for row in receipt["outer_reference"]
                ))
        finally:
            self.contract = original

    def test_every_seal_must_reload_before_truth_provider_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = {outer: self._prepared(root, outer) for outer in (0, 1)}
            output = root / "outer"; output.mkdir()
            calls = []

            def provider(*_args):
                calls.append(True)
                return []

            incomplete = {0: prepared[0]}
            with self.assertRaises(ValueError):
                e0.evaluate_rebuilt_e0_outer(
                    contract=self.contract, config=_config(),
                    source_receipt=self.source, prepared=incomplete, output_directory=output,
                    outer_truth_provider=provider,
                )
            self.assertEqual(calls, [])

    def test_completed_screen_requires_exact_e0_boundary(self) -> None:
        screen = {
            "status": "complete", "f008_config_signature": "a" * 64,
            "f005_contract_signature": "b" * 64,
            "screen": {
                "id": "E0", "active_arms": list(e0.E0_ACTIVE_ARMS),
                "deferred_uniform": True, "outer_evaluation_called": False,
                "outer_labels_used_for_selection": False,
                "selection_or_promotion_allowed": False,
            },
            "outer_evaluation_called": False,
            "promotion_decision": "forbidden_for_E0_calibration_only_screen",
            "folds": [
                {
                    "outer_fold": outer, "checkpoint_receipt_sha256": "c" * 64,
                    "cache_receipt_sha256": "d" * 64,
                    "pretruth_seal_sha256": "e" * 64,
                }
                for outer in (0, 1)
            ],
        }
        checked = e0.validate_completed_e0_screen(
            screen, f008_signature="a" * 64, f005_signature="b" * 64, fold_ids=[0, 1],
        )
        self.assertEqual(set(checked["folds_by_outer"]), {0, 1})
        changed = copy.deepcopy(screen)
        changed["screen"]["outer_evaluation_called"] = True
        with self.assertRaises(ValueError):
            e0.validate_completed_e0_screen(
                changed, f008_signature="a" * 64, f005_signature="b" * 64, fold_ids=[0, 1],
            )


if __name__ == "__main__":
    unittest.main()
