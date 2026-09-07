"""Check gain constraints and baseline preservation without changing raw inputs."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.audio.io import file_sha256
from speaker_id.eda.audit import audit_audio
from speaker_id.eda.vad_sensitivity import controlled_gain, diagnose_file


class VadSensitivityTests(unittest.TestCase):
    def test_zero_and_nonfinite_guards(self):
        self.assertEqual(controlled_gain(0, 0), 1)
        with self.assertRaises(ValueError):
            controlled_gain(float("nan"), .1)

    def test_gain_respects_target_cap_peak_and_no_attenuation(self):
        self.assertAlmostEqual(controlled_gain(.01, .04), 10)
        self.assertEqual(controlled_gain(1e-8, 1e-6), 100)
        self.assertAlmostEqual(controlled_gain(.001, .5), 1.9)
        self.assertEqual(controlled_gain(.5, 1), 1)

    def test_unity_gain_preserves_baseline_predictions_and_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loud.wav"
            samples = (.2 * np.sin(2 * np.pi * 200 * np.arange(16037) / 16000)).astype(np.float32)
            sf.write(path, samples, 16000, subtype="PCM_16")
            baseline = audit_audio(path, "speaker")
            before = file_sha256(path)
            result = diagnose_file(path, baseline)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["gain_factor"], 1)
            self.assertEqual(result["post_gain_vad_mode1_speech_seconds"], baseline["vad_mode1_speech_seconds"])
            self.assertEqual(result["post_gain_vad_mode3_speech_seconds"], baseline["vad_mode3_speech_seconds"])
            self.assertAlmostEqual(result["vad_unprocessed_tail_seconds"], baseline["vad_unprocessed_tail_seconds"])
            self.assertEqual(file_sha256(path), before)


if __name__ == "__main__":
    unittest.main()
