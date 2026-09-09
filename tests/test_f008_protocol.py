"""Focused metadata-only tests for the F008 role and sampling protocol."""
from __future__ import annotations

from copy import deepcopy
import unittest

from speaker_id.training import f008_protocol as protocol


def _role(
    name: str,
    speaker: object,
    group: str,
    role: str,
    *,
    outer: int | str = 0,
) -> dict:
    flags = {
        "known_enrollment": (True, True, False, False),
        "unknown_development": (True, False, False, False),
        "known_calibration_query": (False, False, True, False),
        "unknown_calibration_query": (False, False, True, False),
        "outer_validation": (False, False, False, True),
        "training_excluded": (False, False, False, False),
    }[role]
    return {
        "outer_fold": outer,
        "audio_file": name,
        "speaker_id": speaker,
        "group_id": group,
        "role": role,
        "encoder_fit_allowed": flags[0],
        "enrollment_allowed": flags[1],
        "calibration_query": flags[2],
        "outer_evaluation_included": flags[3],
        "source_train_eligible": role != "training_excluded",
    }


def _roles() -> list[dict]:
    return [
        _role("known-a-1.wav", "A", "known-a", "known_enrollment", outer="0"),
        _role("known-a-2.wav", "A", "known-a", "known_enrollment"),
        _role("known-b.wav", "B", "known-b", "known_enrollment"),
        _role("unknown-0-a.wav", "unknown", "unknown-0", "unknown_development"),
        _role("unknown-0-b.wav", "unknown", "unknown-0", "unknown_development"),
        _role("unknown-1.wav", "unknown", "unknown-1", "unknown_development"),
        _role("unknown-2.wav", "unknown", "unknown-2", "unknown_development"),
        _role("known-query.wav", "A", "known-query", "known_calibration_query"),
        _role("unknown-query.wav", "unknown", "unknown-query", "unknown_calibration_query"),
        _role("outer.wav", "A", "outer", "outer_validation"),
        _role("excluded.wav", "unknown", "excluded", "training_excluded"),
        _role("other-fold.wav", "A", "other-fold", "outer_validation", outer=1),
    ]


class F008RoleProtocolTests(unittest.TestCase):
    def test_role_pools_are_exact_order_independent_and_do_not_mutate_input(self) -> None:
        roles = _roles()
        before = deepcopy(roles)
        pools = protocol.role_pools(roles, 0)
        reversed_pools = protocol.role_pools(list(reversed(roles)), 0)

        self.assertEqual(pools, reversed_pools)
        self.assertEqual(roles, before)
        self.assertEqual(
            [row["audio_file"] for row in pools["known_rows"]],
            ["known-a-1.wav", "known-a-2.wav", "known-b.wav"],
        )
        self.assertEqual(
            [row["audio_file"] for row in pools["unknown_rows"]],
            ["unknown-0-a.wav", "unknown-0-b.wav", "unknown-1.wav", "unknown-2.wav"],
        )
        self.assertEqual(pools["unknown_group_ids"], ["unknown-0", "unknown-1", "unknown-2"])
        self.assertNotIn("unknown-query.wav", {row["audio_file"] for row in pools["unknown_rows"]})
        self.assertEqual(len(pools["signature"]), 64)

    def test_pool_creation_does_not_read_outer_speaker_identity(self) -> None:
        class HiddenOuterTruth:
            def __str__(self) -> str:
                raise AssertionError("outer truth was materialized")

            def __eq__(self, _other: object) -> bool:
                raise AssertionError("outer truth was materialized")

        roles = _roles()
        next(row for row in roles if row["role"] == "outer_validation")["speaker_id"] = HiddenOuterTruth()
        pools = protocol.role_pools(roles, 0)
        self.assertEqual(pools["outer_group_ids"], ["outer"])

    def test_unknown_fit_groups_cannot_overlap_query_or_outer_roles(self) -> None:
        for role in ("unknown_calibration_query", "outer_validation"):
            with self.subTest(role=role):
                roles = _roles()
                next(row for row in roles if row["role"] == role)["group_id"] = "unknown-0"
                with self.assertRaisesRegex(ValueError, "unknown encoder-fit groups overlap"):
                    protocol.role_pools(roles, 0)

    def test_malformed_roles_and_empty_populations_fail_closed(self) -> None:
        cases = []
        duplicate = _roles()
        duplicate[1]["audio_file"] = duplicate[0]["audio_file"]
        cases.append(duplicate)
        bad_flag = _roles()
        bad_flag[3]["calibration_query"] = True
        cases.append(bad_flag)
        wrong_unknown = _roles()
        wrong_unknown[3]["speaker_id"] = "A"
        cases.append(wrong_unknown)
        conflicting_group = _roles()
        conflicting_group[2]["group_id"] = "known-a"
        cases.append(conflicting_group)
        no_unknown_fit = [row for row in _roles() if row["role"] != "unknown_development"]
        cases.append(no_unknown_fit)
        for roles in cases:
            with self.subTest(case=len(roles)):
                with self.assertRaises(ValueError):
                    protocol.role_pools(roles, 0)

    def test_unknown_plan_is_counter_derived_group_balanced_and_target_free(self) -> None:
        pools = protocol.role_pools(_roles(), 0)
        first = protocol.unknown_exposure_plan(pools, 17, seed=20260909, samples_per_step=3)
        again = protocol.unknown_exposure_plan(pools, 17, seed=20260909, samples_per_step=3)
        protocol.validate_unknown_exposure_plan(first, pools)

        self.assertEqual(first, again)
        self.assertEqual({row["slot"] for row in first["rows"]}, {0, 1, 2})
        self.assertEqual(len({row["group_id"] for row in first["rows"]}), 3)
        source_members = {
            group: {row["audio_file"] for row in pools["unknown_rows"] if row["group_id"] == group}
            for group in pools["unknown_group_ids"]
        }
        for row in first["rows"]:
            self.assertEqual(set(row), {"slot", "audio_file", "group_id", "crop_seed", "stream"})
            self.assertIn(row["audio_file"], source_members[row["group_id"]])
            self.assertEqual(row["stream"], "unknown_oe")
            self.assertIs(type(row["crop_seed"]), int)
            self.assertGreaterEqual(row["crop_seed"], 0)
            self.assertLess(row["crop_seed"], 1 << 63)

        self.assertNotEqual(
            first["signature"],
            protocol.unknown_exposure_plan(pools, 18, seed=20260909, samples_per_step=3)["signature"],
        )
        self.assertNotEqual(
            first["signature"],
            protocol.unknown_exposure_plan(pools, 17, seed=20260910, samples_per_step=3)["signature"],
        )

    def test_plan_rejects_replacement_bad_counters_and_tampering(self) -> None:
        pools = protocol.role_pools(_roles(), 0)
        with self.assertRaisesRegex(ValueError, "forbids content-group replacement"):
            protocol.unknown_exposure_plan(pools, 0, seed=1, samples_per_step=4)
        for arguments in (
            (False, 1, 1),
            (0, False, 1),
            (0, 1, 0),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    protocol.unknown_exposure_plan(
                        pools, arguments[0], seed=arguments[1], samples_per_step=arguments[2],
                    )

        plan = protocol.unknown_exposure_plan(pools, 0, seed=1, samples_per_step=2)
        changed = deepcopy(plan)
        changed["rows"][0]["crop_seed"] += 1
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.validate_unknown_exposure_plan(changed, pools)
        changed_pool = deepcopy(pools)
        changed_pool["unknown_rows"][0]["audio_file"] = "mutated.wav"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.unknown_exposure_plan(changed_pool, 0, seed=1, samples_per_step=2)

    def test_plan_range_hash_is_stable_and_order_sensitive(self) -> None:
        pools = protocol.role_pools(_roles(), 0)
        first = protocol.unknown_plan_range_sha256(
            pools, 0, 4, seed=7, samples_per_step=2,
        )
        self.assertEqual(
            first,
            protocol.unknown_plan_range_sha256(pools, 0, 4, seed=7, samples_per_step=2),
        )
        self.assertNotEqual(
            first,
            protocol.unknown_plan_range_sha256(pools, 1, 4, seed=7, samples_per_step=2),
        )
        self.assertNotEqual(
            first,
            protocol.unknown_plan_range_sha256(pools, 0, 4, seed=8, samples_per_step=2),
        )
        with self.assertRaises(ValueError):
            protocol.unknown_plan_range_sha256(pools, 4, 4, seed=7, samples_per_step=2)


if __name__ == "__main__":
    unittest.main()
