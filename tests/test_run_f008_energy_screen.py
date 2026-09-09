"""Pure contract checks for the focused F008 E0 launcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from speaker_id.training.f008_config import config_signature, load_f008_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/run_f008_energy_screen.py"
SPEC = importlib.util.spec_from_file_location("f008_energy_screen_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class F008EnergyScreenLauncherTests(unittest.TestCase):
    def test_e0_reduces_only_the_active_scoring_arm_inventory(self) -> None:
        config = load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")
        before = config_signature(config)
        spec = launcher.reduced_energy_screen_spec(config)
        self.assertEqual(spec["arm_ids"], ["control_f005", "energy_005"])
        self.assertEqual(spec["arm_tie_order"], ["control_f005", "energy_005"])
        self.assertEqual(spec["control_arm_id"], "control_f005")
        self.assertNotIn("uniform_005", spec["arm_ids"])
        # E0 is a runtime-local reduced scorer, not a rewrite of the sealed
        # preflight/training config or its signature.
        self.assertEqual(config_signature(config), before)
        self.assertEqual(config["arms"][-1]["id"], "uniform_005")

    def test_inner_summary_has_no_outer_metric_or_promotion_claim(self) -> None:
        policy = {
            "seal": {
                "arm_policies": {
                    arm: {
                        "advanced_weight": 0.5,
                        "inner_metrics": {
                            "inner_macro_f1_full_label_map": 0.9,
                            "known_query_macro_f1_over_observed_labels": 0.91,
                            "known_query_top1_accuracy": 0.92,
                        },
                    }
                    for arm in ("control_f005", "energy_005")
                },
                "arm_selection": {
                    "selected_arm": "energy_005",
                    "eligible_arms": ["control_f005", "energy_005"],
                    "rejected_by_known_preservation": [],
                    "minimum_known_macro_f1": 0.909,
                },
            }
        }
        summary = launcher._inner_summary(policy, outer=0)
        self.assertEqual(summary["selected_arm"], "energy_005")
        self.assertFalse(summary["outer_truth_read"])
        self.assertNotIn("macro_f1_oof", str(summary))
        self.assertNotIn("promotion", str(summary))


if __name__ == "__main__":
    unittest.main()
