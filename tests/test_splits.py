import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.data.splits import construct_folds


class SplitTests(unittest.TestCase):
    def samples(self):
        return [{"audio_file": f"{label}-{i}", "speaker_id": label,
                 "input_sha256": f"hash-{label}-{i}", "pcm_sha256": f"pcm-{label}-{i}",
                 "usable_for_training": True, "duration_seconds": i + 1}
                for label in ("alice", "bob", "unknown") for i in range(5)]

    def test_duplicate_never_crosses_and_invalids_remain(self):
        rows = self.samples()
        rows.append({**rows[0], "audio_file": "alice-copy"})
        rows += [{"audio_file": f"empty-{label}", "speaker_id": label,
                  "input_sha256": "empty", "pcm_sha256": "empty-pcm",
                  "usable_for_training": False, "duration_seconds": 0.0000625}
                 for label in ("alice", "unknown")]
        folds, summary = construct_folds(rows, [])
        by_name = {r["audio_file"]: r for r in folds}
        self.assertEqual(len(folds), len(rows))
        self.assertEqual(by_name["alice-0"]["fold"], by_name["alice-copy"]["fold"])
        self.assertEqual(by_name["empty-alice"]["fold"], by_name["empty-unknown"]["fold"])
        self.assertFalse(by_name["empty-alice"]["train_eligible"])
        self.assertEqual(summary["actual_folds"], 5)
        self.assertTrue(all(f["validation_known_classes"] == 2 for f in summary["folds"]))
        self.assertEqual(folds, construct_folds(list(reversed(rows)), [])[0])

    def test_verified_cross_label_pair_is_quarantined(self):
        folds, summary = construct_folds(self.samples(), [("alice-0", "bob-0")])
        affected = [r for r in folds if r["audio_file"] in ("alice-0", "bob-0")]
        self.assertTrue(all(not r["train_eligible"] for r in affected))
        self.assertEqual(len({r["fold"] for r in affected}), 1)
        self.assertEqual(summary["actual_folds"], 4)

    def test_impossible_enrollment_not_hidden(self):
        rows = [r for r in self.samples() if r["speaker_id"] != "bob" or r["audio_file"] == "bob-0"]
        folds, summary = construct_folds(rows, [])
        self.assertEqual(folds, [])
        self.assertEqual(summary["status"], "infeasible")
        self.assertEqual(summary["unsupported_known_classes"], ["bob"])

    def test_capacity_protocol_can_use_five_folds_with_sparse_validation(self):
        rows = [r for r in self.samples() if r["speaker_id"] != "bob" or int(r["audio_file"].rsplit("-", 1)[1]) < 2]
        folds, summary = construct_folds(
            rows, [], requested_folds=5, require_validation_known_coverage=False,
        )
        self.assertEqual(summary["actual_folds"], 5)
        self.assertEqual(len(folds), len(rows))
        self.assertTrue(all(fold["training_known_classes"] == 2 for fold in summary["folds"]))
        self.assertTrue(any(fold["validation_known_classes"] < 2 for fold in summary["folds"]))
        default_folds, default_summary = construct_folds(rows, [], requested_folds=5)
        self.assertEqual(len(default_folds), len(rows))
        self.assertEqual(default_summary["actual_folds"], 2)


if __name__ == "__main__":
    unittest.main()
