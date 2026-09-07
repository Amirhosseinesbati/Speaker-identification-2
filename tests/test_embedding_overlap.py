"""Independent waveform fixtures for bounded global-overlap verification."""

import unittest
import numpy as np
from scipy import signal
from speaker_id.eda.duplicates import FingerprintConfig
from speaker_id.eda.embedding_overlap import global_correlation, probe_positions, verify_pair


class EmbeddingOverlapTests(unittest.TestCase):
    def setUp(self):
        self.config = FingerprintConfig()
        rng = np.random.default_rng(923)
        self.a = (.08 * signal.lfilter([1, .4], [1, -.7], rng.normal(size=48000))).astype(np.float32)

    def test_unknown_shift_gain_and_polarity_are_found(self):
        shifted = np.r_[np.zeros(5371), -.3 * self.a, np.zeros(1700)].astype(np.float32)
        result = verify_pair(self.a, shifted, self.config)
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["verified_offset_seconds_b_minus_a"], 5371 / 4000, places=4)
        reverse = verify_pair(shifted, self.a, self.config)
        self.assertTrue(reverse["verified"])
        self.assertAlmostEqual(reverse["verified_offset_seconds_b_minus_a"], -5371 / 4000, places=4)

    def test_different_waveforms_and_silence_are_not_verified(self):
        other = np.random.default_rng(35).normal(0, .1, len(self.a)).astype(np.float32)
        self.assertFalse(verify_pair(self.a, other, self.config)["verified"])
        self.assertFalse(verify_pair(np.zeros(32000), np.zeros(40000), self.config)["verified"])

    def test_one_global_probe_with_four_second_excerpt_requires_local_confirmation(self):
        other = np.random.default_rng(45).normal(0, .1, 64000).astype(np.float32)
        other[8000:24000] = .4 * self.a[:16000]
        result = verify_pair(self.a, other, self.config)
        self.assertTrue(result["verified"])
        self.assertEqual(result["high_correlation_global_probes"], 1)
        self.assertGreaterEqual(result["verified_overlap_seconds"], 4)
        self.assertGreaterEqual(len(result["verification_attempts"][0]["verification_correlations"]), 2)

    def test_probes_do_not_overlap(self):
        for seconds in (8, 10, 12, 60):
            starts = probe_positions(seconds * 4000, 4000)
            self.assertLessEqual(len(starts), 3)
            self.assertTrue(all(b - a >= 16000 for a, b in zip(starts, starts[1:])))

    def test_long_silent_target_spans_cannot_win_by_roundoff(self):
        target = np.r_[self.a, np.zeros(80000)].astype(np.float32)
        unrelated = np.random.default_rng(22).normal(0, .1, 16000).astype(np.float32)
        correlation, lag = global_correlation(unrelated, target, 1e-5)
        self.assertLess(correlation, .1)
        self.assertLess(lag, len(self.a))
        self.assertEqual(global_correlation(unrelated, np.zeros(160000), 1e-5), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
