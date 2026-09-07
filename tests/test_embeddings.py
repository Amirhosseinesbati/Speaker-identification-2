import unittest
import numpy as np
from speaker_id.eda.embeddings import probe_windows, gain_probe


class EmbeddingProbeTests(unittest.TestCase):
    def test_short_and_long_windows_never_overlap_or_exceed_audio(self):
        for frames in (1, 4000, 16000, 96000, 159999, 192000, 9600000):
            windows = probe_windows(frames, 16000)
            self.assertGreater(len(windows), 0)
            self.assertLessEqual(len(windows), 3)
            for start, length in windows:
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start+length, frames)
            for (start, length), (next_start, _) in zip(windows, windows[1:]):
                self.assertLessEqual(start+length, next_start)

    def test_gain_does_not_amplify_loud_signal_or_create_signal(self):
        for x in (np.zeros(16000), np.array([-.99, .99])):
            y, db = gain_probe(x)
            np.testing.assert_array_equal(y, x)
        y, db = gain_probe(np.full(16000, 1e-5))
        self.assertAlmostEqual(db, 40.)
        self.assertAlmostEqual(float(y[0]), .001)
