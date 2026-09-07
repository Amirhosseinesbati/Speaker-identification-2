"""Portable releases contain only deliberate runtime inputs and verified bytes."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from speaker_id.packaging.frozen import SOURCE_FILES, build_payload, execute_build, release_manifest, write_portable_zip
from speaker_id.tracking.snapshot import write_json


class ReleasePackagingTests(unittest.TestCase):
    def test_local_execution_is_blocked_before_torch_or_fitting(self):
        with mock.patch.dict("os.environ", {"VAST_INSTANCE_ID": ""}):
            with self.assertRaisesRegex(RuntimeError, "Vast 50079023"):
                execute_build(Path("."), Path("missing.json"), Path("missing-binding.json"))

    def fixture(self, root):
        for name in (*SOURCE_FILES, "submission.py"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# portable fixture\n", encoding="utf-8")
        (root / ".env").write_text("PRIVATE_FIXTURE=never_include\n")
        unwanted = root / "data/raw/private_audio.mp3"
        unwanted.parent.mkdir(parents=True)
        unwanted.write_bytes(b"not for release")
        weight = root / "artifacts/models/weight.bin"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"public-model-fixture")
        model = {"weights_path": "artifacts/models/weight.bin",
                 "weights_sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}
        final = {"gallery": {"known_embeddings": np.eye(2, dtype=np.float32),
                             "known_targets": np.array([1, 2]), "unknown_embeddings": np.array([[1, 0]], dtype=np.float32)},
                 "calibration": {"threshold": .4, "unknown_weight": .5, "margin_weight": 0,
                                 "temperature": .05, "inference": {"seconds": 180., "maximum_windows": 1}}}
        return model, final

    def test_bundle_keeps_cli_at_zip_root_and_model_paths_portable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); model, final = self.fixture(root)
            package = root / "release/package"
            manifest = build_payload(root, package, model, ["unknown", "A", "B"], final,
                                     release_id="P001_fixture", provenance={"encoder_updates": 0})
            result = write_portable_zip(package, root / "release/output.zip", manifest)
            self.assertTrue(result["crc_verified"])
            with zipfile.ZipFile(root / "release/output.zip") as archive:
                self.assertIn("submission.py", archive.namelist())
                self.assertNotIn(".env", archive.namelist())
                self.assertFalse(any(name.startswith(("data/", "tracking/", "src/")) for name in archive.namelist()))
                config = json.loads(archive.read("assets/model_config.json"))
                self.assertEqual(config["weights_path"], "assets/campplus_voxceleb.bin")
                self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"manifest.json"})

    def test_modified_payload_never_becomes_final_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder).resolve() / "package"; package.mkdir()
            (package / "submission.py").write_text("original")
            manifest = release_manifest(package, release_id="P001_fixture", provenance={})
            write_json(package / "manifest.json", manifest)
            (package / "submission.py").write_text("changed")
            target = package.parent / "release.zip"
            with self.assertRaisesRegex(ValueError, "changed"):
                write_portable_zip(package, target, manifest)
            self.assertFalse(target.exists())

    def test_unexpected_file_added_after_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder).resolve(); (package / "submission.py").write_text("fixture")
            manifest = release_manifest(package, release_id="P001_fixture", provenance={})
            write_json(package / "manifest.json", manifest)
            (package / "private_audio.mp3").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                write_portable_zip(package, package / "release.zip", manifest)

    def test_hidden_files_cannot_enter_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder).resolve(); (package / ".env").write_text("private")
            with self.assertRaisesRegex(ValueError, "Hidden"):
                release_manifest(package, release_id="P001_fixture", provenance={})


if __name__ == "__main__":
    unittest.main()
