"""Signal-level checks of duplicate detection, beyond implementation mirrors."""

import unittest

import numpy as np
from scipy import signal

from speaker_id.eda.duplicates import (
    FingerprintConfig, acoustic_components, exact_groups, find_candidates,
    prepare_waveform, spectral_landmarks, verify_alignment,
)


def synthetic_recording(seed=5, seconds=24, rate=4000):
    rng = np.random.default_rng(seed)
    time = np.arange(seconds * rate) / rate
    waveform = np.zeros(len(time))
    for _ in range(60):
        center = rng.uniform(0, seconds)
        width = rng.uniform(0.03, 0.4)
        frequency = rng.uniform(120, 1700)
        envelope = np.exp(-0.5 * ((time - center) / width) ** 2)
        waveform += envelope * np.sin(2 * np.pi * (frequency * time + rng.uniform(-30, 30) * time**2))
    waveform += 0.015 * signal.lfilter([1, -0.4], [1, -0.8], rng.normal(size=len(time)))
    return (0.2 * waveform / np.max(np.abs(waveform))).astype(np.float32)


class DuplicatesAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = FingerprintConfig()

    def test_gain_polarity_and_crop_verify(self):
        original = synthetic_recording()
        crop_samples = 5371  # Non-frame-aligned crop.
        crop = -0.42 * original[crop_samples:crop_samples + 15 * 4000]
        result = verify_alignment(original, crop, -crop_samples / 4000 + 0.04, self.config)
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["verified_offset_seconds"], -crop_samples / 4000, places=4)

    def test_unrelated_audio_and_silence_do_not_verify(self):
        original = synthetic_recording()
        self.assertFalse(verify_alignment(original, synthetic_recording(seed=18), 0, self.config)["verified"])
        self.assertFalse(verify_alignment(np.zeros(20000), np.zeros(20000), 0, self.config)["verified"])
        self.assertEqual(len(spectral_landmarks(np.zeros(20000), self.config)[0]), 0)

    def test_landmarks_find_shifted_crop(self):
        original = synthetic_recording()
        crop = 0.35 * original[5371:5371 + 15 * 4000]
        negative = synthetic_recording(seed=27)
        all_hashes, all_files, all_anchors = [], [], []
        for index, waveform in enumerate((original, crop, negative)):
            hashes, anchors = spectral_landmarks(waveform, self.config)
            all_hashes.append(hashes)
            all_anchors.append(anchors)
            all_files.append(np.full(len(hashes), index, dtype=np.uint16))
        candidates, _ = find_candidates(np.concatenate(all_hashes), np.concatenate(all_files), np.concatenate(all_anchors), self.config)
        expected = [row for row in candidates if (row["file_index_a"], row["file_index_b"]) == (0, 1)]
        self.assertEqual(len(expected), 1)
        self.assertLess(abs(expected[0]["candidate_offset_seconds"] + 5371 / 4000), 0.15)
        self.assertTrue(verify_alignment(original, crop, expected[0]["candidate_offset_seconds"], self.config)["verified"])

    def test_antiphase_stereo_preserves_signal(self):
        original = synthetic_recording(seconds=5)
        stereo = np.column_stack((original, -original))
        prepared = prepare_waveform(stereo, 4000, self.config)
        self.assertGreater(float(np.std(prepared)), 0.001)

    def test_shared_excerpt_with_unrelated_surroundings(self):
        original = synthetic_recording()
        other = synthetic_recording(seed=23)
        other[8 * 4000:16 * 4000] = 0.6 * original[8 * 4000:16 * 4000]
        result = verify_alignment(original, other, 0.04, self.config, 8.2, 15.7)
        self.assertTrue(result["verified"])

    def test_no_signal_conflict_is_not_a_split_edge(self):
        rows = [
            {"audio_file": "a", "speaker_id": "speaker1", "file_sha256": "zero", "_no_signal": True},
            {"audio_file": "b", "speaker_id": "speaker2", "file_sha256": "zero", "_no_signal": True},
        ]
        groups = exact_groups(rows)
        self.assertTrue(groups[0]["label_conflict"])
        self.assertFalse(groups[0]["use_for_split_grouping"])
        self.assertEqual(acoustic_components(["a", "b", "c"], [("a", "c")]), [["a", "c"]])


if __name__ == "__main__":
    unittest.main()
