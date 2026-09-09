"""Targeted fail-closed tests for F007 protocol and outer-truth gating."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from speaker_id.training import f007_contract as contract
from speaker_id.training.f007_source import F007_F005_SOURCE_RECEIPT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def _sha(character: str) -> str:
    return character * 64


def source_receipt(config: dict) -> dict:
    folds = []
    for outer, character in ((0, "a"), (1, "b")):
        shared = _sha(character)
        folds.append({
            "outer_fold": outer,
            "shared_head": {
                "checkpoint": {
                    "path": f"training/fold_{outer}/shared_head/shared_head.pt",
                    "sha256": shared,
                },
                "unit_report": {
                    "path": f"training/fold_{outer}/shared_head/unit_report.json",
                    "sha256": _sha("c" if outer == 0 else "d"),
                },
            },
            "control_tail": {
                "checkpoint": {
                    "path": f"training/fold_{outer}/tails/control/last.pt",
                    "sha256": _sha("e" if outer == 0 else "f"),
                },
                "unit_report": {
                    "path": f"training/fold_{outer}/tails/control/unit_report.json",
                    "sha256": _sha("1" if outer == 0 else "2"),
                },
                "shared_head_checkpoint_sha256": shared,
            },
            "full_scoring_control": {
                "identity": {
                    "path": f"full_scoring/fold_{outer}/control/full_scoring_cache_identity.json",
                    "sha256": _sha("3" if outer == 0 else "4"),
                    "signature": _sha("5" if outer == 0 else "6"),
                },
                "receipt": {
                    "path": f"full_scoring/fold_{outer}/control/full_scoring_cache_receipt.json",
                    "sha256": _sha("7" if outer == 0 else "8"),
                    "receipt_sha256": _sha("9" if outer == 0 else "0"),
                    "file_count": 4529,
                },
            },
        })
    return {
        "schema_version": F007_F005_SOURCE_RECEIPT_SCHEMA,
        "source_run_directory": config["source_f005"]["run_dir"],
        "experiment_state": {
            "path": "experiment_state.json", "sha256": _sha("d"),
            "status": "complete", "experiment_signature": _sha("e"),
        },
        "fold_ids": [0, 1], "folds": folds,
    }


def arm_metrics(*, l2sp_wins: bool = True) -> dict:
    return {
        "control_f005": {
            "macro_f1_observed_known_labels": 0.95,
            "top1_accuracy": 0.95,
        },
        "l2sp_001": {
            "macro_f1_observed_known_labels": 0.96 if l2sp_wins else 0.95,
            "top1_accuracy": 0.96 if l2sp_wins else 0.95,
        },
        "l2sp_01": {
            "macro_f1_observed_known_labels": 0.94,
            "top1_accuracy": 0.94,
        },
    }


def known_receipts() -> dict:
    return {
        arm: {"sha256": _sha(character), "rows": 443}
        for arm, character in zip(contract.ARM_IDS, ("a", "b", "c"), strict=True)
    }


def policies(selected: str) -> dict:
    return {
        "frozen_same_protocol": {
            "model_source": "c002b_frozen", "alpha": 0.5,
            "unknown_weight": 0.75, "margin_weight": 0.5, "threshold": 0.25,
            "calibration_receipt_sha256": _sha("1"),
        },
        "reused_control": {
            "model_source": "control_f005", "alpha": 0.5,
            "unknown_weight": 0.75, "margin_weight": 0.5, "threshold": 0.25,
            "calibration_receipt_sha256": _sha("2"),
        },
        "selected_arm": {
            "model_source": selected, "alpha": 0.5,
            "unknown_weight": 0.75, "margin_weight": 0.5, "threshold": 0.25,
            "calibration_receipt_sha256": _sha("3"),
        },
    }


class F007ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "f007"
        self.output.mkdir()
        self.config = json.loads(
            (ROOT / "configs/train/campp_f007_l2sp.json").read_text(encoding="utf-8")
        )
        self.contract = contract.build_f007_contract(self.config, source_receipt(self.config))

    def tearDown(self):
        self.temporary.cleanup()

    def test_preregistered_config_and_source_receipt_are_bound_immutably(self):
        contract.validate_f007_config(self.config)
        contract.validate_f007_contract(self.contract)
        self.assertEqual(self.contract["arm_ids"], list(contract.ARM_IDS))
        self.assertEqual(self.contract["source_f005_receipt"]["fold_ids"], [0, 1])
        changed = deepcopy(self.config)
        changed["arms"][1]["lambda"] = 0.02
        with self.assertRaisesRegex(ValueError, "exact preregistered"):
            contract.validate_f007_config(changed)
        tampered = deepcopy(self.contract)
        tampered["source_f005_receipt"]["folds"][0]["control_tail"]["shared_head_checkpoint_sha256"] = _sha("0")
        with self.assertRaises(ValueError):
            contract.validate_f007_contract(tampered)

    def test_arm_seal_recomputes_winner_and_cannot_be_replaced(self):
        path = self.output / "selection/fold_0/arm_selection_seal.json"
        reload = contract.write_arm_selection_seal(
            path, self.contract, 0, arm_metrics(), known_receipts(),
        )
        self.assertEqual(reload["seal"]["selected_arm"], "l2sp_001")
        self.assertFalse(reload["seal"]["unknown_similarity_computed"])
        with self.assertRaises(FileExistsError):
            contract.write_arm_selection_seal(
                path, self.contract, 0, arm_metrics(l2sp_wins=False), known_receipts(),
            )
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["selected_arm"] = "control_f005"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recomputed known-query winner"):
            contract.reload_arm_selection_seal(path, self.contract, 0)

    def test_policy_seal_binds_reloaded_arm_and_grid(self):
        arm_path = self.output / "selection/fold_0/arm_selection_seal.json"
        arm = contract.write_arm_selection_seal(
            arm_path, self.contract, 0, arm_metrics(), known_receipts(),
        )
        policy_path = self.output / "full_scoring/fold_0/policy_seal.json"
        policy = contract.write_policy_seal(
            policy_path, self.contract, 0, arm, policies("l2sp_001"),
        )
        self.assertEqual(policy["seal"]["selected_arm"], "l2sp_001")
        self.assertEqual(policy["seal"]["policies"]["reused_control"]["model_source"], "control_f005")
        bad = policies("l2sp_001")
        bad["selected_arm"]["alpha"] = 0.33
        with self.assertRaisesRegex(ValueError, "scoring grid"):
            contract.policy_seal(self.contract, 0, arm, bad)
        stale = deepcopy(arm)
        stale["seal_sha256"] = _sha("f")
        with self.assertRaisesRegex(ValueError, "reload changed"):
            contract.policy_seal(self.contract, 0, stale, policies("l2sp_001"))

    def test_outer_truth_cannot_unlock_until_both_disk_reloaded_policy_seals(self):
        state = contract.fresh_f007_state(self.contract, self.output)
        with self.assertRaisesRegex(ValueError, "all fold policies"):
            contract.unlock_outer_truth(state, self.contract, self.output)
        arm_reloads = {}
        for outer in (0, 1):
            arm_path = self.output / f"selection/fold_{outer}/arm_selection_seal.json"
            arm = contract.write_arm_selection_seal(
                arm_path, self.contract, outer, arm_metrics(), known_receipts(),
            )
            contract.register_arm_selection_seal(state, self.contract, self.output, arm)
            arm_reloads[outer] = arm
        policy_path = self.output / "full_scoring/fold_0/policy_seal.json"
        policy = contract.write_policy_seal(
            policy_path, self.contract, 0, arm_reloads[0], policies("l2sp_001"),
        )
        contract.register_policy_seal(state, self.contract, self.output, policy)
        with self.assertRaisesRegex(ValueError, "all fold policies"):
            contract.unlock_outer_truth(state, self.contract, self.output)
        policy_path = self.output / "full_scoring/fold_1/policy_seal.json"
        policy = contract.write_policy_seal(
            policy_path, self.contract, 1, arm_reloads[1], policies("l2sp_001"),
        )
        contract.register_policy_seal(state, self.contract, self.output, policy)
        permit = contract.unlock_outer_truth(state, self.contract, self.output)
        contract.require_outer_truth_permit(permit, state, self.contract, self.output, 0)
        self.assertTrue(state["outer_truth_materialized"])
        self.assertEqual(state["phase"], "one_shot_outer_evaluation")
        with self.assertRaisesRegex(ValueError, "already unlocked"):
            contract.unlock_outer_truth(state, self.contract, self.output)

    def test_state_rejects_an_outer_truth_claim_without_policy_seals(self):
        state = contract.fresh_f007_state(self.contract, self.output)
        state["outer_truth_materialized"] = True
        state["outer_truth_access"] = {
            "schema_version": 1, "policy_digest": _sha("a"), "fold_ids": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "before every fold policy"):
            contract.validate_f007_state(state, self.contract, self.output, verify_seals=False)


if __name__ == "__main__":
    unittest.main()
