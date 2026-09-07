import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
import zipfile

from speaker_id.infrastructure.data import DataVerificationError, sha256_file
from speaker_id.infrastructure.metadata import install_metadata, prepare_metadata_transfer


class MetadataTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.local, self.remote = self.root / "local", self.root / "remote"
        self.names = ["data/processed/eda_v1/folds.csv", "data/processed/eda_v1/label_map.json"]
        self.contents = {self.names[0]: b"audio_file,fold\none.mp3,0\n", self.names[1]: b'{"unknown":0,"speaker":1}\n'}
        for workspace in (self.local, self.remote):
            (workspace / "configs/infra").mkdir(parents=True)
            (workspace / "configs/infra/deployment.json").write_text(json.dumps({"metadata_files": self.names}), encoding="utf-8")
        for name, payload in self.contents.items():
            path = self.local / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self):
        identity = prepare_metadata_transfer(workspace=self.local)
        incoming = self.remote / "data/incoming/data_metadata.zip"
        incoming.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.local / identity["archive_path"], incoming)
        return incoming, identity

    def rebuild(self, path, mutate):
        with zipfile.ZipFile(path) as archive:
            comment = archive.comment
            files = [(info, archive.read(info)) for info in archive.infolist()]
        files, comment = mutate(files, comment)
        with zipfile.ZipFile(path, "w") as archive:
            for info, payload in files:
                archive.writestr(info, payload)
            archive.comment = comment

    def test_deterministic_zip_contains_exact_allowlist_and_manifest_comment(self):
        first = prepare_metadata_transfer(workspace=self.local)
        original = (self.local / first["archive_path"]).read_bytes()
        second = prepare_metadata_transfer(workspace=self.local)
        self.assertEqual(first, second)
        self.assertEqual(original, (self.local / second["archive_path"]).read_bytes())
        with zipfile.ZipFile(self.local / first["archive_path"]) as archive:
            self.assertEqual(archive.namelist(), sorted(self.names))
            manifest = json.loads(archive.comment)
            self.assertEqual([row["path"] for row in manifest["files"]], sorted(self.names))

    def test_install_verifies_all_bytes_then_delete_and_reuses_only_exact_matches(self):
        archive, identity = self.bundle()
        result = install_metadata(workspace=self.remote, archive=archive,
                                  expected_archive_sha256=identity["archive_sha256"],
                                  delete_archive_after_verification=True)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["archive_deleted"])
        self.assertFalse(archive.exists())
        for name, payload in self.contents.items():
            self.assertEqual((self.remote / name).read_bytes(), payload)
        archive, identity = self.bundle()
        repeated = install_metadata(workspace=self.remote, archive=archive,
                                    expected_archive_sha256=identity["archive_sha256"])
        self.assertEqual(repeated["existing_matching_files_reused"], len(self.names))
        self.assertTrue(archive.exists())

    def test_wrong_trusted_archive_hash_fails_without_install_or_deletion(self):
        archive, _ = self.bundle()
        with self.assertRaisesRegex(DataVerificationError, "trusted expected"):
            install_metadata(workspace=self.remote, archive=archive, expected_archive_sha256="0" * 64,
                             delete_archive_after_verification=True)
        self.assertTrue(archive.exists())
        self.assertFalse((self.remote / self.names[0]).exists())

    def test_existing_mismatch_is_never_overwritten_and_nothing_new_is_published(self):
        archive, identity = self.bundle()
        existing = self.remote / self.names[-1]
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"local-different-file")
        with self.assertRaisesRegex(DataVerificationError, "refusing overwrite"):
            install_metadata(workspace=self.remote, archive=archive,
                             expected_archive_sha256=identity["archive_sha256"], delete_archive_after_verification=True)
        self.assertEqual(existing.read_bytes(), b"local-different-file")
        self.assertFalse((self.remote / self.names[0]).exists())
        self.assertTrue(archive.exists())

    def test_internal_member_hash_mismatch_prevents_publication(self):
        archive, _ = self.bundle()

        def mutate(files, comment):
            info, payload = files[0]
            files[0] = info, b"x" + payload[1:]
            return files, comment

        self.rebuild(archive, mutate)
        with self.assertRaisesRegex(DataVerificationError, "member SHA256 mismatch"):
            install_metadata(workspace=self.remote, archive=archive, expected_archive_sha256=sha256_file(archive))
        self.assertFalse((self.remote / self.names[0]).exists())

    def test_extra_traversal_and_symlink_members_are_rejected(self):
        for kind in ("extra", "traversal", "symlink"):
            with self.subTest(kind=kind):
                archive, _ = self.bundle()

                def mutate(files, comment):
                    if kind == "extra":
                        files.append((zipfile.ZipInfo("data/processed/eda_v1/extra.csv"), b"extra"))
                    elif kind == "traversal":
                        files.append((zipfile.ZipInfo("../escape.csv"), b"escape"))
                    else:
                        files[0][0].create_system = 3
                        files[0][0].external_attr = (stat.S_IFLNK | 0o777) << 16
                    return files, comment

                self.rebuild(archive, mutate)
                with self.assertRaises(DataVerificationError):
                    install_metadata(workspace=self.remote, archive=archive, expected_archive_sha256=sha256_file(archive))
                self.assertFalse((self.remote / self.names[0]).exists())

    def test_deployment_cannot_allow_raw_audio_or_credentials_or_traversal(self):
        config = self.local / "configs/infra/deployment.json"
        for name in ("data/raw/audio.mp3", ".env", "data/processed/../raw/labels.csv", "data/processed/private_key.json"):
            with self.subTest(name=name):
                config.write_text(json.dumps({"metadata_files": [name]}), encoding="utf-8")
                with self.assertRaises(DataVerificationError):
                    prepare_metadata_transfer(workspace=self.local)

    def test_deletion_cannot_target_original_local_bundle(self):
        identity = prepare_metadata_transfer(workspace=self.local)
        archive = self.local / identity["archive_path"]
        with self.assertRaisesRegex(DataVerificationError, "Deletion is allowed"):
            install_metadata(workspace=self.local, archive=archive,
                             expected_archive_sha256=identity["archive_sha256"], delete_archive_after_verification=True)
        self.assertTrue(archive.exists())


if __name__ == "__main__":
    unittest.main()
