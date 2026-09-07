import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.eda.semantic import plan_windows, select_files, gain_factor


class SemanticTests(unittest.TestCase):
    def test_windows_are_bounded_disjoint_and_cover_short_input(self):
        for duration in (.0000625, .25, 3, 10, 11, 19, 20, 37, 100):
            windows = plan_windows(duration)
            self.assertLessEqual(len(windows), 2)
            self.assertLessEqual(sum(length for _, length in windows), min(duration, 20))
            for start, length in windows:
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + length, duration)
            if len(windows) == 2:
                self.assertLessEqual(sum(windows[0]), windows[1][0])
        self.assertEqual(plan_windows(0), [])

    def test_selection_is_order_independent_retains_anomalies_and_balances_controls(self):
        rows = [{"audio_file": f"{label}-{i}", "speaker_id": f"speaker-{i}" if label == "known" else "unknown",
                 "status": "ok", "max_channel_rms_dbfs": -20, "vad_mode3_speech_fraction": .8,
                 "duration_seconds": i + 20} for label in ("known", "unknown") for i in range(40)]
        rows += [{"audio_file": "zero", "speaker_id": "unknown", "status": "ok", "duration_seconds": 1}]
        chosen, controls = select_files(rows, [{"audio_file": "zero"}], [{"audio_file": "zero"}], 2)
        chosen2, controls2 = select_files(list(reversed(rows)), [{"audio_file": "zero"}], [{"audio_file": "zero"}], 2)
        self.assertEqual(chosen, chosen2)
        self.assertEqual(controls, controls2)
        self.assertEqual(len(controls), 16)
        self.assertIn("zero", [r["audio_file"] for r in chosen])
        self.assertEqual(len(chosen), 17)

    def test_gain_is_capped_no_silence_amplification_no_clipping(self):
        self.assertEqual(gain_factor(0, 0), 1)
        self.assertEqual(gain_factor(1e-9, 1e-8), 100)
        self.assertAlmostEqual(gain_factor(.01, .5), 1.9)
        self.assertEqual(gain_factor(.5, 1), 1)


if __name__ == "__main__":
    unittest.main()
