"""F005 pairing, known-only selection, sealing and resume-contract tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import f005_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def small_planning_contract():
    config = json.loads((ROOT / "configs/train/campp_f005_consistency.json").read_text())
    config["fold_ids"] = [0]
    labels = ["unknown"] + [f"speaker{i:03d}" for i in range(446)]
    roles = [{"outer_fold": 0, "audio_file": f"{label}.wav", "speaker_id": label,
              "group_id": f"g-{label}", "encoder_fit_allowed": True}
             for label in labels[1:]]
    return {"config": config, "labels": labels, "roles": roles, "signature": "a" * 64}


def scoring_fixture():
    config = json.loads((ROOT / "configs/train/campp_f005_consistency.json").read_text())
    labels = ["unknown"] + [f"speaker{i:03d}" for i in range(446)]
    manifest, folds, roles = [], [], []
    def add(name, label, group, role):
        manifest.append({"audio_file": name, "speaker_id": label})
        folds.append({"audio_file": name, "speaker_id": label, "group_id": group,
                      "fold": 1 if role != "outer" else 0, "train_eligible": role != "outer"})
        roles.append({"outer_fold": 0, "audio_file": name, "speaker_id": label, "group_id": group,
                      "encoder_fit_allowed": role == "fit", "enrollment_allowed": role == "enroll",
                      "calibration_query": role == "query", "outer_evaluation_included": role == "outer"})
    for index, label in enumerate(labels[1:]):
        add(f"fit-{index}.wav", label, f"fit-g-{index}", "fit")
        add(f"second-{index}.wav", label, f"query-g-{index}", "query" if index < 443 else "enroll")
    add("unknown-query.wav", "unknown", "unknown-query-group", "query")
    add("outer.wav", labels[1], "outer-group", "outer")
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(len(manifest), 192)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = np.ones(len(manifest), dtype=np.bool_)
    contract = {"config": config, "labels": labels, "manifest": manifest, "folds": folds,
                "roles": roles, "signature": "b" * 64}
    return contract, vectors, valid


class F005RunnerTests(unittest.TestCase):
    def test_all_arms_share_step_plan_and_head_then_fork_layout(self):
        contract = small_planning_contract()
        first = runner.training_step_plan(contract, 0, 600)
        self.assertEqual(first, runner.training_step_plan(contract, 0, 600))
        self.assertEqual(len(first), 32)
        pairing = runner.pairing_identity(contract, 0)
        identities = [runner.arm_identity(contract, 0, arm, shared_head_checkpoint_sha256="1" * 64,
                                          tail_plan_sha256="2" * 64)
                      for arm in runner.ARM_IDS]
        self.assertEqual({item["pairing_signature"] for item in identities}, {pairing["signature"]})
        self.assertEqual(len({item["signature"] for item in identities}), 4)
        plan = runner.execution_plan(contract)
        self.assertEqual(plan["unit_count"], 5)
        self.assertEqual(plan["shared_head_units"], 1)
        self.assertEqual(plan["tail_arm_units"], 4)

    def test_shared_checkpoint_is_only_valid_fork_and_tail_resume_cannot_cross_arm(self):
        contract = small_planning_contract(); fit = contract["config"]["fit"]
        head_identity = runner.shared_head_identity(contract, 0, plan_sha256="3" * 64)
        head_meta = runner.shared_head_checkpoint_metadata(head_identity, fit)
        payload = {"metadata": head_meta, "encoder": object(), "head": object(), "optimizer": object(),
                   "torch_rng": object(), "cuda_rng": object()}
        self.assertTrue(runner.validate_shared_head_payload(payload, head_identity, fit)["byte_identical_fork_source"])
        partial = deepcopy(payload)
        partial["metadata"] = runner.shared_head_checkpoint_metadata(head_identity, fit, 500)
        self.assertFalse(runner.validate_shared_head_payload(partial, head_identity, fit)["byte_identical_fork_source"])
        control = runner.arm_identity(contract, 0, "control", shared_head_checkpoint_sha256="4" * 64,
                                      tail_plan_sha256="5" * 64)
        treatment = runner.arm_identity(contract, 0, "treatment_mse0", shared_head_checkpoint_sha256="4" * 64,
                                        tail_plan_sha256="5" * 64)
        tail = {**payload, "metadata": runner.checkpoint_metadata(control, fit, 700)}
        runner.validate_resume_payload(tail, control, fit)
        with self.assertRaises(ValueError): runner.validate_resume_payload(tail, treatment, fit)

    def test_known_preselection_never_materializes_unknown_and_rejects_cross_role_group(self):
        contract, vectors, valid = scoring_fixture()
        result = runner.known_selection_scores(vectors, valid, contract, 0)
        self.assertEqual(len(result["known_calibration_indices"]), 443)
        self.assertEqual(result["known_scores"].shape, (443, 446))
        self.assertFalse(result["provenance"]["unknown_query_indices_materialized"])
        self.assertFalse(result["provenance"]["unknown_reference_cohort_materialized"])
        self.assertFalse(result["provenance"]["unknown_similarity_computed"])
        self.assertNotIn("inner_unknown_similarity", result)
        changed = deepcopy(contract)
        changed["folds"][1]["group_id"] = changed["folds"][0]["group_id"]
        changed["roles"][1]["group_id"] = changed["roles"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "crossing heldout roles"):
            runner.known_selection_scores(vectors, valid, changed, 0)

    def test_control_can_win_and_seal_is_per_outer_known_only(self):
        contract, vectors, valid = scoring_fixture()
        base = runner.known_selection_scores(vectors, valid, contract, 0)
        indices = base["known_calibration_indices"]
        truth = [contract["labels"].index(contract["manifest"][int(i)]["speaker_id"]) - 1 for i in indices]
        arms = {}
        for position, arm in enumerate(runner.ARM_IDS):
            item = deepcopy(base); item["known_scores"] = np.full_like(base["known_scores"], -1)
            for row, target in enumerate(truth):
                item["known_scores"][row, target if arm == "control" else (target + position) % 446] = 1
            arms[arm] = item
        seal = runner.select_arm(contract, 0, arms)
        self.assertEqual(seal["selected_arm"], "control")
        self.assertEqual(seal["scientific_conclusion"], "consistency_not_supported_on_known_calibration")
        self.assertFalse(seal["unknown_similarity_computed"])
        with tempfile.TemporaryDirectory() as directory:
            receipt = runner.write_and_reload_seal(Path(directory) / "selection.json", seal)
            self.assertEqual(receipt["file_sha256"], receipt["expected_bytes_sha256"])

    def test_seal_detects_writer_byte_drift(self):
        seal_body = {"schema_version": 1, "outer_fold": 0}
        seal = {**seal_body, "seal_sha256": runner.hashlib.sha256(runner.canonical(seal_body)).hexdigest()}
        def corrupt(path, payload): Path(path).write_text(json.dumps(payload), encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "write_json", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "round-trip"):
                runner.write_and_reload_seal(Path(directory) / "seal.json", seal)


if __name__ == "__main__":
    unittest.main()
