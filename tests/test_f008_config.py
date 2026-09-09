"""Focused immutability checks for the declared F008 experiment surface."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from speaker_id.training.f008_config import config_signature, load_f008_config, validate_f008_config


ROOT = Path(__file__).resolve().parents[1]


class F008ConfigTests(unittest.TestCase):
    def test_checked_in_config_is_valid_and_has_stable_signature(self) -> None:
        first = load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")
        second = load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")
        self.assertEqual(first, second)
        self.assertEqual(config_signature(first), config_signature(second))
        self.assertEqual(len(config_signature(first)), 64)

    def test_unreviewed_arm_or_temperature_changes_fail_closed(self) -> None:
        config = load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")
        altered_arm = deepcopy(config)
        altered_arm["arms"][1]["lambda"] = 0.1
        with self.assertRaisesRegex(ValueError, "arms changed"):
            validate_f008_config(altered_arm)
        altered_temperature = deepcopy(config)
        altered_temperature["energy_margin"]["energy_temperature"] = 0.5
        with self.assertRaisesRegex(ValueError, "energy-margin values changed"):
            validate_f008_config(altered_temperature)

    def test_extra_or_missing_fields_fail_closed(self) -> None:
        config = load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")
        extra = deepcopy(config)
        extra["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "fields changed"):
            validate_f008_config(extra)
        missing = deepcopy(config)
        del missing["retention"]
        with self.assertRaisesRegex(ValueError, "fields changed"):
            validate_f008_config(missing)


if __name__ == "__main__":
    unittest.main()
