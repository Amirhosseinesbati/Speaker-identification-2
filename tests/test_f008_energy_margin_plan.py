"""Focused, dependency-free checks for F008's sealed energy margin planner."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest

from speaker_id.training.f008_energy_margin_plan import (
    F008_ENERGY_MARGIN_PLAN_SCHEMA,
    F008_ENERGY_ORIENTATION,
    F008_ENERGY_QUANTILE_METHOD,
    build_energy_margin_plan,
    verify_energy_margin_plan,
)


CONFIG = {
    "known_maximum_quantile": 0.95,
    "declared_minimum_energy_gap": 0.02,
}


class F008EnergyMarginPlanTests(unittest.TestCase):
    def test_linear_known_quantile_sets_both_boundaries_and_orientation(self) -> None:
        plan = build_energy_margin_plan(
            known_energies=[4.0, 0.0, 2.0, 1.0],
            unknown_energies=[5.0, 7.0, 6.0],
            energy_margin_config=CONFIG,
        )

        # position = (4 - 1) * .95 = 2.85, so linear interpolation gives 3.7.
        self.assertTrue(math.isclose(plan["maximum_known_energy"], 3.7, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(plan["minimum_unknown_energy"], 3.72, abs_tol=1.0e-12))
        self.assertEqual(plan["schema_version"], F008_ENERGY_MARGIN_PLAN_SCHEMA)
        self.assertEqual(plan["energy_orientation"], F008_ENERGY_ORIENTATION)
        self.assertEqual(plan["quantile_method"], F008_ENERGY_QUANTILE_METHOD)
        self.assertEqual(plan["unknown_energy_role"], "diagnostics_only_does_not_set_margins")
        self.assertEqual(plan["known_diagnostics"]["count"], 4)
        self.assertEqual(plan["unknown_diagnostics"]["count"], 3)
        self.assertEqual(verify_energy_margin_plan(plan), plan)
        self.assertEqual(json.loads(json.dumps(plan, allow_nan=False)), plan)

    def test_order_and_unknown_distribution_do_not_move_margins(self) -> None:
        first = build_energy_margin_plan(
            known_energies=[1.0, -3.0, 4.0, 2.0],
            unknown_energies=[10.0, 11.0],
            energy_margin_config=CONFIG,
        )
        reordered = build_energy_margin_plan(
            known_energies=[2.0, 4.0, -3.0, 1.0],
            unknown_energies=[11.0, 10.0],
            energy_margin_config=CONFIG,
        )
        changed_unknowns = build_energy_margin_plan(
            known_energies=[1.0, -3.0, 4.0, 2.0],
            unknown_energies=[-100.0, 100.0],
            energy_margin_config=CONFIG,
        )

        self.assertEqual(first, reordered)
        for key in ("maximum_known_energy", "minimum_unknown_energy"):
            self.assertEqual(first[key], changed_unknowns[key])
        self.assertNotEqual(first["unknown_diagnostics"], changed_unknowns["unknown_diagnostics"])

    def test_singleton_known_vector_is_well_defined(self) -> None:
        plan = build_energy_margin_plan(
            known_energies=[-0.4],
            unknown_energies=[0.2],
            energy_margin_config=CONFIG,
        )
        self.assertEqual(plan["maximum_known_energy"], -0.4)
        self.assertEqual(plan["minimum_unknown_energy"], -0.38)
        self.assertEqual(plan["known_diagnostics"]["boundary_satisfied_fraction"], 1.0)

    def test_digest_rejects_boundary_or_diagnostic_tampering(self) -> None:
        plan = build_energy_margin_plan(
            known_energies=[-1.0, 0.0, 1.0],
            unknown_energies=[2.0, 3.0],
            energy_margin_config=CONFIG,
        )
        altered_boundary = deepcopy(plan)
        altered_boundary["minimum_unknown_energy"] += 0.01
        with self.assertRaisesRegex(ValueError, "boundaries are inconsistent"):
            verify_energy_margin_plan(altered_boundary)

        altered_diagnostic = deepcopy(plan)
        altered_diagnostic["unknown_diagnostics"]["mean"] += 0.1
        with self.assertRaisesRegex(ValueError, "digest changed"):
            verify_energy_margin_plan(altered_diagnostic)

    def test_rejects_bad_config_vectors_and_unrepresentable_gap(self) -> None:
        invalid_configurations = [
            {},
            {"known_maximum_quantile": 0.0, "declared_minimum_energy_gap": 0.02},
            {"known_maximum_quantile": 1.0, "declared_minimum_energy_gap": 0.02},
            {"known_maximum_quantile": True, "declared_minimum_energy_gap": 0.02},
            {"known_maximum_quantile": 0.95, "declared_minimum_energy_gap": 0.0},
            {"known_maximum_quantile": 0.95, "declared_minimum_energy_gap": float("inf")},
        ]
        for config in invalid_configurations:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    build_energy_margin_plan([0.0], [1.0], config)

        invalid_vectors = [
            ([], [1.0]),
            ([0.0], []),
            ([float("nan")], [1.0]),
            ([0.0], [float("inf")]),
            ("0.0", [1.0]),
            ([0.0], {"not": "a vector"}),
            ([True], [1.0]),
        ]
        for known, unknown in invalid_vectors:
            with self.subTest(known=repr(known), unknown=repr(unknown)):
                with self.assertRaises((TypeError, ValueError)):
                    build_energy_margin_plan(known, unknown, CONFIG)

        with self.assertRaises(ValueError):
            build_energy_margin_plan(
                [1.0e308], [0.0],
                {"known_maximum_quantile": 0.95, "declared_minimum_energy_gap": 1.0e308},
            )

    def test_plan_schema_is_fail_closed(self) -> None:
        plan = build_energy_margin_plan([0.0, 1.0], [2.0], CONFIG)
        missing = dict(plan)
        del missing["unknown_energy_role"]
        with self.assertRaisesRegex(ValueError, "schema changed"):
            verify_energy_margin_plan(missing)

        wrong_orientation = dict(plan)
        wrong_orientation["energy_orientation"] = "reversed"
        wrong_orientation["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_energy_margin_plan(wrong_orientation)


if __name__ == "__main__":
    unittest.main()
