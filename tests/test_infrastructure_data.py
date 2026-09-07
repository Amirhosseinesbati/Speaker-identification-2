"""Synthetic integrity and containment checks for server data preparation."""

import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from speaker_id.infrastructure.data import DataVerificationError, sha256_file, verify_extract_data


class InfrastructureDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "data/incoming").mkdir(parents=True)
        self.archive = self.root / "data/incoming/raw.zip"
        self.manifest = self.root / "data/processed/eda_v1/audio_manifest.csv"
        self.manifest.parent.mkdir(parents=True)
        self.audio = {"one.mp3": b"RIFF fake first audio", "two.mp3": b"RIFF fake second audio"}
        self.labels = {"one.mp3": "person-a", "two.mp3": "unknown"}
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_file", "speaker_id", "input_sha256", "file_bytes"])
            writer.writeheader()
            for name, payload in self.audio.items():
                writer.writerow({"audio_file": name, "speaker_id": self.labels[name],
                                 "input_sha256": hashlib.sha256(payload).hexdigest(), "file_bytes": len(payload)})

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self, *, audio=None, labels=None, extra=None):
        label_stream = io.StringIO(newline="")
        writer = csv.writer(label_stream)
        writer.writerow(["speaker_id", "audio_file"])
        for name, speaker in (labels or self.labels).items():
            writer.writerow([speaker, name])
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as zipped:
            zipped.writestr("raw/", b"")
            for name, payload in (audio or self.audio).items():
                zipped.writestr("raw/" + name, payload)
            zipped.writestr("raw/labels.csv", label_stream.getvalue().encode())
            if extra is not None:
                zipped.writestr(*extra)
        return sha256_file(self.archive)

    def run_verifier(self, digest=None, delete=False, **kwargs):
        return verify_extract_data(workspace=self.root, archive=self.archive,
                                   expected_archive_sha256=digest or sha256_file(self.archive),
                                   delete_archive_after_verification=delete, **kwargs)

    def test_success_verifies_every_member_and_deletes_only_incoming_zip(self):
        digest = self.write_archive()
        result = self.run_verifier(digest, delete=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["audio_files_verified"], 2)
        self.assertEqual(result["archive_crc_verified_members"], 3)
        self.assertTrue(result["archive_deleted"])
        self.assertFalse(self.archive.exists())
        for name, payload in self.audio.items():
            self.assertEqual((self.root / "data/raw" / name).read_bytes(), payload)
        report = json.loads((self.root / "artifacts/infrastructure/data_readiness.json").read_text())
        self.assertTrue(report["archive_deleted"])

    def test_existing_matching_dataset_can_be_verified_again_without_overwrite(self):
        self.write_archive()
        self.run_verifier()
        sample = self.root / "data/raw/one.mp3"
        previous = sample.stat().st_mtime_ns
        result = self.run_verifier()
        self.assertEqual(result["existing_matching_files_reused"], 3)
        self.assertEqual(sample.stat().st_mtime_ns, previous)

    def test_corrupted_archive_crc_prevents_publication_and_deletion(self):
        self.write_archive()
        payload = self.archive.read_bytes()
        marker = payload.index(self.audio["one.mp3"])
        changed = bytearray(payload)
        changed[marker] ^= 1
        self.archive.write_bytes(changed)
        with self.assertRaises(zipfile.BadZipFile):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_audio_sha_mismatch_prevents_publication_and_deletion(self):
        audio = dict(self.audio)
        audio["one.mp3"] = b"x" * len(audio["one.mp3"])
        self.write_archive(audio=audio)
        with self.assertRaisesRegex(DataVerificationError, "SHA256 differs"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_archive_sha_mismatch_prevents_extraction_and_deletion(self):
        self.write_archive()
        with self.assertRaisesRegex(DataVerificationError, "Transferred ZIP SHA256"):
            self.run_verifier("0" * 64, delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_label_mismatch_prevents_publication_and_deletion(self):
        self.write_archive(labels={"one.mp3": "unknown", "two.mp3": "unknown"})
        with self.assertRaisesRegex(DataVerificationError, "names/labels differ"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_unsafe_member_paths_are_rejected_before_extraction(self):
        for name in ("../escape", "/absolute", "raw/../escape", "C:/absolute", "raw\\escape"):
            with self.subTest(name=name):
                self.write_archive(extra=(name, b"bad"))
                with self.assertRaises(DataVerificationError):
                    self.run_verifier(delete=True)
                self.assertTrue(self.archive.exists())
                self.assertFalse((self.root / "data/raw").exists())

    def test_zip_symlink_is_rejected(self):
        info = zipfile.ZipInfo("raw/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_archive(extra=(info, b"/tmp/escape"))
        with self.assertRaisesRegex(DataVerificationError, "symlink/special"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())

    def test_existing_wrong_raw_file_is_never_overwritten(self):
        self.write_archive()
        raw = self.root / "data/raw"
        raw.mkdir()
        existing = raw / "one.mp3"
        existing.write_bytes(b"unrelated user content")
        with self.assertRaisesRegex(DataVerificationError, "refusing overwrite"):
            self.run_verifier(delete=True)
        self.assertEqual(existing.read_bytes(), b"unrelated user content")
        self.assertTrue(self.archive.exists())

    def test_deletion_refuses_local_source_zip(self):
        self.write_archive()
        original = self.archive
        self.archive = self.root / "data/raw.zip"
        original.replace(self.archive)
        with self.assertRaisesRegex(DataVerificationError, "under data/incoming"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())

    def test_failure_report_cannot_overwrite_archive_or_raw_data(self):
        expected = self.write_archive()
        with self.assertRaisesRegex(DataVerificationError, "Report must be"):
            self.run_verifier(delete=True, report=self.archive)
        self.assertEqual(sha256_file(self.archive), expected)

    def test_missing_file_prevents_publication(self):
        self.write_archive(audio={"one.mp3": self.audio["one.mp3"]})
        with self.assertRaisesRegex(DataVerificationError, "missing 1"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_unrelated_raw_file_is_preserved_and_blocks_operation(self):
        self.write_archive()
        raw = self.root / "data/raw"
        raw.mkdir()
        unrelated = raw / "notes.txt"
        unrelated.write_text("user notes")
        with self.assertRaisesRegex(DataVerificationError, "Unrelated existing raw entry"):
            self.run_verifier(delete=True)
        self.assertEqual(unrelated.read_text(), "user notes")
        self.assertTrue(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
