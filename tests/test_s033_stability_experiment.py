"""Small deterministic checks for S033 selection/configuration semantics."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from speaker_id.research.s033_stability_experiment import _select_threshold, validate_config


ROOT = Path(__file__).resolve().parents[1]


class S033ExperimentTests(unittest.TestCase):
    def test_threshold_selection_uses_inner_labels_and_prefers_less_intervention_on_ties(self):
        # Stability .0 identifies the two false accepts; .75 removes both and
        # improves F1.  It must be selected without access to an outer row.
        selected = _select_threshold(
            np.asarray([0, 1, 0, 1], dtype=np.int64),
            np.asarray([1, 1, 1, 1], dtype=np.int64),
            np.asarray([0.0, 1.0, 0.5, 1.0], dtype=np.float64),
            np.asarray([True, True, True, True]), [0.0, 0.75, 1.0],
        )
        self.assertEqual(selected["selected"]["minimum_stability"], 0.75)
        self.assertEqual(selected["selected"]["vetoed_known_rows"], 2)
        # With no benefit from a veto, threshold zero is the deterministic
        # lower-intervention tie break.
        tied = _select_threshold(
            np.asarray([1, 1], dtype=np.int64), np.asarray([1, 1], dtype=np.int64),
            np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([True, True]), [0.0, 0.5, 1.0],
        )
        self.assertEqual(tied["selected"]["minimum_stability"], 0.0)

    def test_checked_config_cannot_silently_switch_to_cpu_or_a_replacement_identity(self):
        path = ROOT / "configs/research/s033_reference_resampling_veto.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(validate_config(config), config)
        changed = copy.deepcopy(config)
        changed["sampling"]["backend"] = "cpu"
        with self.assertRaisesRegex(ValueError, "sampling protocol"):
            validate_config(changed)
        changed = copy.deepcopy(config)
        changed["retention"]["local_transfer"] = "candidate_copy"
        with self.assertRaisesRegex(ValueError, "retention"):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
