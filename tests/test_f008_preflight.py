"""Pure receipt and deterministic-crop tests for the F008 source preflight."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speaker_id.training.f008_preflight import (
    F008_PREFLIGHT_SCHEMA,
    build_preflight_receipt,
    source_crop_seed,
    validate_f005_control_selection,
    verify_preflight_receipt,
)
from speaker_id.training.f008_protocol import role_pools


def _role(name: str, speaker: str, group: str, role: str) -> dict:
    flags = {
        "known_enrollment": (True, True, False, False),
        "unknown_development": (True, False, False, False),
        "known_calibration_query": (False, False, True, False),
        "unknown_calibration_query": (False, False, True, False),
        "outer_validation": (False, False, False, True),
    }[role]
    return {
        "outer_fold": 0, "audio_file": name, "speaker_id": speaker, "group_id": group,
        "role": role, "encoder_fit_allowed": flags[0], "enrollment_allowed": flags[1],
        "calibration_query": flags[2], "outer_evaluation_included": flags[3],
        "source_train_eligible": True,
    }


def _pools() -> dict:
    return role_pools([
        _role("known-a.wav", "a", "known-a", "known_enrollment"),
        _role("known-b.wav", "b", "known-b", "known_enrollment"),
        _role("unknown-a.wav", "unknown", "unknown-a", "unknown_development"),
        _role("unknown-b.wav", "unknown", "unknown-b", "unknown_development"),
        _role("known-cal.wav", "a", "known-cal", "known_calibration_query"),
        _role("unknown-cal.wav", "unknown", "unknown-cal", "unknown_calibration_query"),
        _role("outer.wav", "a", "outer", "outer_validation"),
    ], 0)


class F008PreflightReceiptTests(unittest.TestCase):
    def test_crop_seed_is_stable_and_role_stream_bound(self) -> None:
        pool = _pools()
        first = source_crop_seed(pool["signature"], outer_fold=0, stream="known", audio_file="known-a.wav")
        self.assertEqual(
            first,
            source_crop_seed(pool["signature"], outer_fold=0, stream="known", audio_file="known-a.wav"),
        )
        self.assertNotEqual(
            first,
            source_crop_seed(pool["signature"], outer_fold=0, stream="unknown", audio_file="known-a.wav"),
        )
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)

    def test_receipt_seals_source_margin_without_per_file_energies(self) -> None:
        pool = _pools()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "shared_head.pt"
            checkpoint.write_bytes(b"authenticated-shared-head-test-bytes")
            receipt = build_preflight_receipt(
                f008_signature="a" * 64,
                f005_signature="b" * 64,
                f005_source_receipt={"schema_version": "source", "folds": []},
                outer_fold=0,
                role_pool=pool,
                shared_head_checkpoint=checkpoint,
                energy_margin_config={
                    "known_maximum_quantile": 0.95,
                    "declared_minimum_energy_gap": 0.02,
                },
                known_energies=[-0.8, -0.7, -0.6],
                unknown_energies=[-0.5, -0.4],
                source_view_algorithm="test_paired_view_v1",
            )
        self.assertEqual(receipt["schema_version"], F008_PREFLIGHT_SCHEMA)
        self.assertEqual(receipt["known_energy_views"], 3)
        self.assertEqual(receipt["unknown_energy_views"], 2)
        self.assertEqual(verify_preflight_receipt(receipt), receipt)
        rendered = str(receipt)
        self.assertNotIn("unknown-a.wav", rendered)
        self.assertNotIn("known_energies", rendered)
        self.assertNotIn("unknown_energies", rendered)
        self.assertGreater(
            receipt["energy_margin_plan"]["minimum_unknown_energy"],
            receipt["energy_margin_plan"]["maximum_known_energy"],
        )

    def test_receipt_rejects_source_or_margin_tampering(self) -> None:
        pool = _pools()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "shared_head.pt"
            checkpoint.write_bytes(b"authenticated-shared-head-test-bytes")
            receipt = build_preflight_receipt(
                f008_signature="a" * 64, f005_signature="b" * 64,
                f005_source_receipt={"schema_version": "source", "folds": []}, outer_fold=0,
                role_pool=pool, shared_head_checkpoint=checkpoint,
                energy_margin_config={"known_maximum_quantile": 0.95,
                                      "declared_minimum_energy_gap": 0.02},
                known_energies=[-0.8, -0.7], unknown_energies=[-0.4, -0.3],
                source_view_algorithm="test_paired_view_v1",
            )
        changed = deepcopy(receipt)
        changed["role_pool_signature"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_preflight_receipt(changed)
        changed_margin = deepcopy(receipt)
        changed_margin["energy_margin_plan"]["maximum_known_energy"] += 0.01
        with self.assertRaisesRegex(ValueError, "identity changed|digest changed|boundaries are inconsistent"):
            verify_preflight_receipt(changed_margin)

    def test_f005_control_selection_requires_pinned_state_and_seal_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for fold, seal_hash in (("0", "a" * 64), ("1", "b" * 64)):
                path = root / "selection" / f"fold_{fold}" / "arm_selection_seal.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"selected_arm": "control", "seal_sha256": seal_hash}),
                                encoding="utf-8")
                expected[fold] = {
                    "selected_arm": "control", "seal_sha256": seal_hash,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            state = {
                "status": "complete",
                "arm_seals": {
                    fold: {**entry, "disk_reloaded": True} for fold, entry in expected.items()
                },
            }
            (root / "experiment_state.json").write_text(json.dumps(state), encoding="utf-8")
            source = {
                "arm_selection_seals": expected,
                "required_selected_arm_by_outer_fold": {"0": "control", "1": "control"},
            }
            observed = validate_f005_control_selection(root, source)
            self.assertEqual(observed, expected)
            altered = deepcopy(state)
            altered["arm_seals"]["1"]["selected_arm"] = "treatment"
            (root / "experiment_state.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected arm differs"):
                validate_f005_control_selection(root, source)


if __name__ == "__main__":
    unittest.main()
