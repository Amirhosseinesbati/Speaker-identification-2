"""Signal invariants that protect preprocessing from silent corruption."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from speaker_id.eda.audit import audit_audio


class AudioAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, audio, sr=16000, subtype="FLOAT"):
        path = self.directory / name
        sf.write(path, audio, sr, format="WAV", subtype=subtype)
        return path

    def test_opposite_polarity_is_signal_despite_cancelled_downmix(self):
        t = np.arange(16000) / 16000
        mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = self.write("stereo.mp3", np.column_stack((mono, -mono)))
        result = audit_audio(path, "speaker", block_frames=173)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["usable_for_training"])
        self.assertFalse(result["channel_identical"])
        self.assertAlmostEqual(result["channel_correlation"], -1, places=6)
        self.assertEqual(result["mono_zero_fraction"], 1)
        self.assertIn("downmix_loss_gt_6db", result["quality_flags"])
        self.assertGreater(result["frame_energy_p50_dbfs"], -20)
        self.assertEqual(result["detected_format"], "WAV")
        self.assertFalse(result["extension_matches_container"])

    def test_one_zero_frame_is_excluded_but_one_nonzero_frame_is_only_flagged(self):
        silent = audit_audio(self.write("silent.mp3", np.zeros((1, 2))), "a")
        tiny = audit_audio(self.write("tiny.mp3", np.ones((1, 2)) * .1), "a")
        self.assertFalse(silent["usable_for_training"])
        self.assertIn("no_signal", silent["quality_flags"])
        self.assertIsNone(silent["channel_correlation"])
        self.assertTrue(tiny["usable_for_training"])
        self.assertIn("tiny_duration_lt_025s", tiny["quality_flags"])
        self.assertAlmostEqual(tiny["energy_above_minus50_dbfs_seconds"], 1 / 16000)

    def test_pcm_hash_ignores_container_and_stream_block_boundaries(self):
        mono = np.tile(np.array([0, .25, -.25, 0], np.float32), 1000)
        wave = self.write("a.mp3", mono, subtype="PCM_16")
        flac = self.directory / "b.flac"
        sf.write(flac, mono, 16000, format="FLAC", subtype="PCM_16")
        first, second = audit_audio(wave, "a", 337), audit_audio(flac, "a", 1000)
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual(first["pcm_sha256"], second["pcm_sha256"])
        self.assertAlmostEqual(first["frame_energy_p50_dbfs"], second["frame_energy_p50_dbfs"])
        different_sr = audit_audio(self.write("c.wav", mono, sr=8000, subtype="PCM_16"), "a")
        self.assertNotEqual(first["pcm_sha256"], different_sr["pcm_sha256"])

    def test_truncated_payload_is_not_silently_accepted(self):
        path = self.write("truncated.mp3", np.ones((1000, 2)) * .2, subtype="PCM_16")
        path.write_bytes(path.read_bytes()[:-100])
        result = audit_audio(path, "a")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["usable_for_training"])
        self.assertEqual(result["wave_integrity"], "error")

    def test_real_mp3_detected_and_fully_decoded(self):
        if "MP3" not in sf.available_formats():
            self.skipTest("libsndfile built without MP3")
        path = self.directory / "actual.mp3"
        t = np.arange(24000) / 48000
        sf.write(path, .1 * np.sin(2 * np.pi * 330 * t), 48000, format="MP3")
        result = audit_audio(path, "a")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["detected_format"], "MP3")
        self.assertTrue(result["extension_matches_container"])
        self.assertEqual(result["channels"], 1)
        self.assertGreater(result["duration_seconds"], .45)
        self.assertEqual(result["wave_integrity"], "not_checked")

    def test_vad_silence_and_partial_tail_are_not_counted_as_speech(self):
        result = audit_audio(self.write("silent_vad.wav", np.zeros((16123, 2))), "a", block_frames=517)
        if result["vad_status"] != "ok":
            self.skipTest("WebRTC VAD optional package unavailable")
        self.assertEqual(result["vad_mode1_speech_seconds"], 0)
        self.assertEqual(result["vad_mode3_speech_fraction"], 0)
        self.assertAlmostEqual(result["vad_analyzed_seconds"], .99)
        self.assertAlmostEqual(result["vad_unprocessed_tail_seconds"], 283 / 16000)


if __name__ == "__main__":
    unittest.main()
