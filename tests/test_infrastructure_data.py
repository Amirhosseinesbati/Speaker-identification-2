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
from speaker_id.infrastructure.readiness import ReadinessError, validate_archive_source


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

    def write_archive(self, *, audio=None, labels=None, extra=None, prefix="raw", labels_bytes=None):
        label_stream = io.StringIO(newline="")
        writer = csv.writer(label_stream)
        writer.writerow(["speaker_id", "audio_file"])
        for name, speaker in (labels or self.labels).items():
            writer.writerow([speaker, name])
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as zipped:
            if prefix:
                zipped.writestr(prefix + "/", b"")
            parent = prefix + "/" if prefix else ""
            for name, payload in (audio or self.audio).items():
                zipped.writestr(parent + name, payload)
            zipped.writestr(parent + "labels.csv", labels_bytes if labels_bytes is not None else label_stream.getvalue().encode())
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

    def test_explicit_training_root_matches_identical_payload_and_exact_labels(self):
        self.write_archive(prefix="training")
        with zipfile.ZipFile(self.archive) as zipped:
            labels_hash = hashlib.sha256(zipped.read("training/labels.csv")).hexdigest()
        before_size = self.archive.stat().st_size
        result = self.run_verifier(delete=True, archive_prefix="training",
                                   archive_source_id="official_original", expected_labels_sha256=labels_hash)
        self.assertTrue(result["archive_deleted"])
        self.assertEqual(result["archive_prefix"], "training")
        self.assertEqual(result["archive_source_id"], "official_original")
        self.assertEqual(result["archive_size_bytes"], before_size)
        self.assertTrue(result["labels_byte_sha256_verified"])
        self.assertEqual(result["labels_sha256"], labels_hash)
        self.assertEqual(result["archive_crc_verified_members"], 3)
        for name, payload in self.audio.items():
            self.assertEqual((self.root / "data/raw" / name).read_bytes(), payload)

    def test_default_policy_does_not_implicitly_accept_training_root(self):
        self.write_archive(prefix="training")
        with self.assertRaisesRegex(DataVerificationError, "Unexpected ZIP directory"):
            self.run_verifier(delete=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_explicit_prefix_rejects_other_roots_and_flat_members(self):
        for prefix in ("raw", "other", ""):
            with self.subTest(prefix=prefix):
                self.write_archive(prefix=prefix)
                with self.assertRaises(DataVerificationError):
                    self.run_verifier(delete=True, archive_prefix="training")
                self.assertTrue(self.archive.exists())
                self.assertFalse((self.root / "data/raw").exists())

    def test_explicit_prefix_still_rejects_traversal_and_nested_members(self):
        for name in ("training/../escape", "training/nested/one.mp3", "one.mp3", "raw/two.mp3"):
            with self.subTest(name=name):
                self.write_archive(prefix="training", extra=(name, b"unexpected"))
                with self.assertRaises(DataVerificationError):
                    self.run_verifier(delete=True, archive_prefix="training")
                self.assertTrue(self.archive.exists())
                self.assertFalse((self.root / "data/raw").exists())

    def test_explicit_prefix_itself_cannot_contain_path_components(self):
        self.write_archive(prefix="training")
        for prefix in ("../training", "/training", "training/nested", "training/", ""):
            with self.subTest(prefix=prefix):
                with self.assertRaises(DataVerificationError):
                    self.run_verifier(delete=True, archive_prefix=prefix)
                self.assertTrue(self.archive.exists())

    def test_byte_changed_labels_fail_even_when_mapping_is_identical(self):
        self.write_archive(prefix="training")
        with zipfile.ZipFile(self.archive) as zipped:
            original = zipped.read("training/labels.csv")
        original_hash = hashlib.sha256(original).hexdigest()
        changed = original.replace(b"\r\n", b"\n")
        self.assertNotEqual(original, changed)
        self.write_archive(prefix="training", labels_bytes=changed)
        with self.assertRaisesRegex(DataVerificationError, "original CSV bytes"):
            self.run_verifier(delete=True, archive_prefix="training", expected_labels_sha256=original_hash)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.root / "data/raw").exists())

    def test_official_source_readiness_binds_every_container_identity_field(self):
        self.write_archive(prefix="training")
        with zipfile.ZipFile(self.archive) as zipped:
            labels_hash = hashlib.sha256(zipped.read("training/labels.csv")).hexdigest()
        result = self.run_verifier(delete=True, archive_prefix="training",
                                   archive_source_id="official_original", expected_labels_sha256=labels_hash)
        config = self.root / "configs/infra/archive_sources.json"
        config.parent.mkdir(parents=True)
        source = {"archive_path": "data/incoming/raw.zip", "archive_size_bytes": result["archive_size_bytes"],
                  "archive_sha256": result["archive_sha256"], "member_prefix": "training", "labels_sha256": labels_hash}
        config.write_text(json.dumps({"official_original": source}))
        self.assertEqual(validate_archive_source(self.root, result)["archive_source_id"], "official_original")
        for key, replacement in {
            "archive_sha256": "0" * 64, "archive_size_bytes": result["archive_size_bytes"] + 1,
            "archive_prefix": "raw", "expected_labels_sha256": "0" * 64,
            "labels_sha256": "0" * 64, "labels_byte_sha256_verified": False,
            "archive": str(self.root / "data/incoming/another.zip"),
        }.items():
            with self.subTest(field=key):
                changed = {**result, key: replacement}
                with self.assertRaises(ReadinessError):
                    validate_archive_source(self.root, changed)
        for pending in (None, "pending", "0" * 64):
            with self.subTest(placeholder=pending):
                config.write_text(json.dumps({"official_original": {**source, "archive_sha256": pending}}))
                with self.assertRaisesRegex(ReadinessError, "placeholder"):
                    validate_archive_source(self.root, result)

    def test_unknown_archive_source_does_not_pass_readiness(self):
        with self.assertRaisesRegex(ReadinessError, "Unrecognized"):
            validate_archive_source(self.root, {"archive_source_id": "anything_else"})


if __name__ == "__main__":
    unittest.main()
