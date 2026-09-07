"""Check forensic measurements against explicit synthetic stored waveforms."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.eda.forensics import run_lengths, sample_metrics, verify_sparse_coverage


class ForensicsTests(unittest.TestCase):
    def test_boundary_runs_and_all_zero(self):
        np.testing.assert_array_equal(run_lengths(np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)), [2, 1, 3])
        metrics, histogram = sample_metrics(np.zeros(1600, dtype=np.int16), 16000)
        self.assertEqual(metrics["zero_fraction"], 1)
        self.assertEqual(metrics["nonzero_runs"], 0)
        self.assertEqual(metrics["observed_code_entropy_bits"], 0)
        self.assertEqual(metrics["diagnostic_peak_safe_gain_db"], 0)
        self.assertIsNone(metrics["lag1_correlation"])
        self.assertEqual(histogram, {"0": 1600})

    def test_isolated_single_code_events_and_safe_gain(self):
        codes = np.zeros(16000, dtype=np.int16)
        codes[::100] = 1
        metrics, histogram = sample_metrics(codes, 16000)
        self.assertEqual(metrics["unique_pcm_codes"], 2)
        self.assertEqual(metrics["isolated_nonzero_sample_fraction"], 1)
        self.assertEqual(metrics["nonzero_abs_one_fraction"], 1)
        self.assertEqual(metrics["nonzero_run_max_samples"], 1)
        self.assertAlmostEqual(metrics["zero_fraction"], .99)
        self.assertEqual(histogram, {"0": 15840, "1": 160})
        self.assertLess(metrics["diagnostic_post_gain_rms_dbfs"], -20)
        self.assertFalse(metrics["diagnostic_target_minus20_dbfs_reached"])

    def test_sine_spectrum_and_gain_invariant_structure(self):
        codes = np.rint(500 * np.sin(2 * np.pi * 1000 * np.arange(32000) / 16000)).astype(np.int16)
        first, _ = sample_metrics(codes, 16000)
        second, _ = sample_metrics(codes * 10, 16000)
        self.assertGreater(first["spectral_power_fraction_300to3400_hz"], .99)
        self.assertAlmostEqual(first["spectral_centroid_hz"], 1000, delta=2)
        self.assertEqual(first["zero_fraction"], second["zero_fraction"])
        self.assertAlmostEqual(first["lag1_correlation"], second["lag1_correlation"])
        self.assertAlmostEqual(first["rms_dbfs"] + 20, second["rms_dbfs"])
        self.assertTrue(first["diagnostic_target_minus20_dbfs_reached"])
        self.assertAlmostEqual(first["diagnostic_post_gain_rms_dbfs"], -20)

    def test_sparse_coverage_finds_missing_and_mismatched_audits(self):
        rows = [{"status": "ok", "has_nonzero_signal": "True", "channel_zero_fraction": "[0.99]",
                 "analysis_channel_index": "0", "decoded_frames": "100", "audio_file": "one"}]
        coverage = verify_sparse_coverage(rows, [])
        self.assertFalse(coverage["all_matched_files_exactly_audited"])
        self.assertEqual(coverage["missing_files"], ["one"])
        coverage = verify_sparse_coverage(rows, [{"status": "ok", "audio_file": "one", "nonzero_samples": 2}])
        self.assertEqual(coverage["count_mismatch_files"], ["one"])
        self.assertTrue(verify_sparse_coverage(rows, [{"status": "ok", "audio_file": "one", "nonzero_samples": 1}])["all_matched_files_exactly_audited"])


if __name__ == "__main__":
    unittest.main()
