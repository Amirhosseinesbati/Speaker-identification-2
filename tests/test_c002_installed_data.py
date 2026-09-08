"""C002 installed-data verification never trusts a prior extraction receipt."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_installed_data_for_tests", ROOT / "scripts/infra/verify_installed_data.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class C002InstalledDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.raw = self.root / "data/raw"
        self.raw.mkdir(parents=True)
        metadata = self.root / "data/processed/eda_v1"
        metadata.mkdir(parents=True)
        self.config = self.root / "configs/train/campp_coverage_c002.json"
        self.config.parent.mkdir(parents=True)
        self.entries = {"a.mp3": ("alice", b"a"), "b.mp3": ("unknown", b"bc")}
        with (metadata / "audio_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_file", "speaker_id", "input_sha256", "file_bytes"])
            writer.writeheader()
            for name, (speaker, payload) in self.entries.items():
                writer.writerow({"audio_file": name, "speaker_id": speaker,
                                 "input_sha256": hashlib.sha256(payload).hexdigest(), "file_bytes": len(payload)})
        with (self.raw / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["speaker_id", "audio_file"])
            writer.writeheader()
            for name, (speaker, _) in self.entries.items():
                writer.writerow({"speaker_id": speaker, "audio_file": name})
        for name, (_, payload) in self.entries.items():
            (self.raw / name).write_bytes(payload)
        self.config.write_text(json.dumps({
            "data_dir": "data/raw", "manifest": "data/processed/eda_v1/audio_manifest.csv",
            "expected_source_files": 2, "evaluation_classes": 2,
        }), encoding="utf-8")
        self.report = Path("artifacts/infrastructure/C002_preparation/data_readiness.json")

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_installed_bytes_are_rehashed(self):
        result = VERIFY.verify_installed_data(workspace=self.root,
                                               config_path=Path("configs/train/campp_coverage_c002.json"),
                                               report=self.report)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["verification_scope"], "all_installed_raw_files_manifest_sha256")
        self.assertTrue(result["installed_file_sha256_verified"])
        self.assertEqual(result["output_files_verified"], 3)
        self.assertTrue(result["output_bytes_verified"])
        self.assertEqual(Path(result["output"]), self.raw)

    def test_changed_audio_fails_even_when_no_zip_is_present(self):
        (self.raw / "a.mp3").write_bytes(b"changed")
        with self.assertRaisesRegex(Exception, "size differs|SHA256"):
            VERIFY.verify_installed_data(workspace=self.root,
                                         config_path=Path("configs/train/campp_coverage_c002.json"),
                                         report=self.report)
        report = json.loads((self.root / self.report).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["installed_file_sha256_verified"])

    def test_extra_installed_file_fails(self):
        (self.raw / "unexpected.mp3").write_bytes(b"x")
        with self.assertRaisesRegex(Exception, "inventory"):
            VERIFY.verify_installed_data(workspace=self.root,
                                         config_path=Path("configs/train/campp_coverage_c002.json"),
                                         report=self.report)


if __name__ == "__main__":
    unittest.main()
