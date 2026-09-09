"""Focused contract tests for server-only F007 embedding caches."""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training import f007_extraction as extraction


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_indices(count: int) -> str:
    return hashlib.sha256(np.arange(count, dtype="<i8").tobytes()).hexdigest()


def _vector(position: int) -> np.ndarray:
    value = np.zeros((192,), dtype=np.float32)
    value[position % 192] = 1.0
    return value


def _contract(*, source: dict | None = None, manifest: list[dict] | None = None) -> dict:
    manifest = manifest or [
        {"audio_file": "does-not-exist-a.wav", "input_sha256": "1" * 64, "has_nonzero_signal": True},
        {"audio_file": "does-not-exist-b.wav", "input_sha256": "2" * 64, "has_nonzero_signal": False},
        {"audio_file": "does-not-exist-c.wav", "input_sha256": "3" * 64, "has_nonzero_signal": True},
    ]
    source = source or {
        "schema_version": "f007-f005-source-receipt-v1",
        "source_run_directory": "/server/f005",
        "experiment_state": {"status": "complete", "experiment_signature": "b" * 64},
        "folds": [],
    }
    return {
        "signature": "a" * 64,
        "source_f005_receipt": source,
        "source_f005_receipt_sha256": extraction.canonical_sha256(source),
        "fold_ids": [0],
        "arm_ids": ["control_f005", "l2sp_001", "l2sp_01"],
        "manifest": manifest,
    }


def _checkpoint(contract: dict, root: Path, *, arm: str = "l2sp_001") -> tuple[Path, dict]:
    path = root / "validated-tail.pt"
    path.write_bytes(b"opaque checkpoint bytes validated by the worker")
    receipt = extraction.validated_tail_checkpoint_receipt(
        contract, outer_fold=0, arm_id=arm, checkpoint_path=path,
        checkpoint_metadata_sha256="c" * 64,
        validation_receipt_sha256="d" * 64,
    )
    return path, receipt


def _plan(contract: dict, root: Path, checkpoint_receipt: dict, *,
          indices=(0, 1, 2), valid=(True, False, True)):
    return extraction.build_f007_advanced_cache_plan(
        contract, output_directory=root,
        relative_cache_directory="full_scoring/fold_0/l2sp_001/embedding_cache",
        outer_fold=0, arm_id="l2sp_001", checkpoint_receipt=checkpoint_receipt,
        indices=np.asarray(indices, dtype=np.int64), scope="full_scoring",
        expected_valid=list(valid),
        inference={"endpoint": "full_utterance", "seconds": 180.0, "maximum_windows": 1},
    )


class F007ExtractionTests(unittest.TestCase):
    def test_resumable_entries_bind_contract_checkpoint_scope_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract()
            checkpoint, receipt = _checkpoint(contract, root)
            plan = _plan(contract, root, receipt)
            calls: list[int] = []

            def extractor(index, row):
                calls.append(index)
                self.assertFalse((root / row["audio_file"]).exists())
                # A stale nonzero value for invalid audio must never be retained.
                return _vector(index), index != 1

            result = extraction.extract_or_load_f007_advanced_cache(
                plan, checkpoint_path=checkpoint, extractor=extractor,
            )
            self.assertEqual(calls, [0, 1, 2])
            self.assertEqual(result["embeddings"].shape, (3, 192))
            self.assertEqual(result["embeddings"].dtype, np.float32)
            self.assertEqual(result["valid"].tolist(), [True, False, True])
            self.assertFalse(np.any(result["embeddings"][1]))
            self.assertTrue(result["server_only"])
            self.assertFalse(result["mlflow_upload_allowed"])
            self.assertFalse(result["local_transfer_allowed"])
            identity = result["identity"]
            self.assertEqual(identity["experiment_signature"], contract["signature"])
            self.assertEqual(identity["source_f005_receipt_sha256"], contract["source_f005_receipt_sha256"])
            self.assertEqual(identity["outer_fold"], 0)
            self.assertEqual(identity["arm_id"], "l2sp_001")
            self.assertEqual(identity["scope"], "full_scoring")
            self.assertEqual(identity["checkpoint"], receipt)
            self.assertEqual(identity["row_count"], 3)

            def must_not_extract(*_):
                raise AssertionError("completed cache must be reused")

            again = extraction.extract_or_load_f007_advanced_cache(
                plan, checkpoint_path=checkpoint, extractor=must_not_extract,
            )
            np.testing.assert_array_equal(again["embeddings"], result["embeddings"])
            self.assertEqual(calls, [0, 1, 2])

    def test_interrupted_cache_resumes_only_missing_rows_and_pure_load_needs_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(manifest=[
                {"audio_file": "not-on-disk-0.wav", "input_sha256": "1" * 64, "has_nonzero_signal": True},
                {"audio_file": "not-on-disk-1.wav", "input_sha256": "2" * 64, "has_nonzero_signal": True},
            ])
            checkpoint, receipt = _checkpoint(contract, root)
            plan = _plan(contract, root, receipt, indices=(0, 1), valid=(True, True))
            attempted: list[int] = []

            def interrupted(index, _row):
                attempted.append(index)
                if index == 1:
                    raise RuntimeError("simulated interruption")
                return _vector(index), True

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                extraction.extract_or_load_f007_advanced_cache(
                    plan, checkpoint_path=checkpoint, extractor=interrupted,
                )
            self.assertEqual(attempted, [0, 1])
            resumed: list[int] = []

            def finish(index, _row):
                resumed.append(index)
                return _vector(index), True

            finished = extraction.extract_or_load_f007_advanced_cache(
                plan, checkpoint_path=checkpoint, extractor=finish,
            )
            self.assertEqual(resumed, [1])
            checkpoint.unlink()
            loaded = extraction.load_f007_advanced_cache(finished_plan := plan)
            np.testing.assert_array_equal(loaded["embeddings"], finished["embeddings"])
            self.assertEqual(loaded["valid"].tolist(), [True, True])
            self.assertIs(finished_plan, plan)

    def test_checkpoint_mutation_and_corrupt_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract()
            checkpoint, receipt = _checkpoint(contract, root)
            plan = _plan(contract, root, receipt)
            checkpoint.write_bytes(b"mutated after attestation")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                extraction.extract_or_load_f007_advanced_cache(
                    plan, checkpoint_path=checkpoint,
                    extractor=lambda index, _row: (_vector(index), index != 1),
                )

            checkpoint, receipt = _checkpoint(contract, root)
            plan = _plan(contract, root, receipt)
            extraction.extract_or_load_f007_advanced_cache(
                plan, checkpoint_path=checkpoint,
                extractor=lambda index, _row: (_vector(index), index != 1),
            )
            entries = Path(extraction.load_f007_advanced_cache(plan)["cache_directory"]) / "entries"
            target = next(entries.glob("*.npz"))
            target.write_bytes(b"not an npz")
            with self.assertRaises(ValueError):
                extraction.load_f007_advanced_cache(plan)

    def test_symlinked_entries_fail_closed_when_symlinks_are_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract()
            checkpoint, receipt = _checkpoint(contract, root)
            plan = _plan(contract, root, receipt)
            extraction.extract_or_load_f007_advanced_cache(
                plan, checkpoint_path=checkpoint,
                extractor=lambda index, _row: (_vector(index), index != 1),
            )
            entries = Path(extraction.load_f007_advanced_cache(plan)["cache_directory"]) / "entries"
            target, source = sorted(entries.glob("*.npz"))[:2]
            target.unlink()
            try:
                os.symlink(source.name, target)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable on this host: {error}")
            with self.assertRaisesRegex(ValueError, "non-symlink|may not be a symlink"):
                extraction.load_f007_advanced_cache(plan)

    def test_reuses_f005_control_cache_in_place_with_no_raw_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "f005"
            source.mkdir()
            manifest = [
                {"audio_file": "no-source-a.wav", "input_sha256": "1" * 64, "has_nonzero_signal": True},
                {"audio_file": "no-source-b.wav", "input_sha256": "2" * 64, "has_nonzero_signal": False},
            ]
            cache_root = source / "full_scoring" / "fold_0" / "control"
            entries = cache_root / "embedding_cache"
            entries.mkdir(parents=True)
            checkpoint_sha = "4" * 64
            identity_body = {
                "schema_version": 1,
                "experiment_signature": "b" * 64,
                "outer_fold": 0,
                "arm_id": "control",
                "scope": "full_scoring",
                "indices_sha256": _sha_indices(len(manifest)),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_metadata_sha256": "5" * 64,
                "embedding_dimension": 192,
                "inference": "fold_checkpoint_full_utterance_180s",
                "server_only": True,
                "mlflow_upload_allowed": False,
            }
            identity = {**identity_body, "signature": extraction.canonical_sha256(identity_body)}
            identity_path = cache_root / "full_scoring_cache_identity.json"
            identity_path.write_text(__import__("json").dumps(identity, sort_keys=True), encoding="utf-8")
            records = []
            for position, row in enumerate(manifest):
                filename = f"row-{position}.npz"
                path = entries / filename
                valid = bool(row["has_nonzero_signal"])
                vector = _vector(position) if valid else np.zeros((192,), dtype=np.float32)
                with path.open("wb") as stream:
                    np.savez_compressed(
                        stream, embedding=vector, valid=np.bool_(valid),
                        signature=identity["signature"], audio_file=row["audio_file"],
                        audio_sha256=row["input_sha256"],
                    )
                records.append({
                    "audio_file": row["audio_file"], "audio_sha256": row["input_sha256"],
                    "cache_file": filename, "cache_sha256": _sha_file(path),
                    "bytes": path.stat().st_size, "valid": valid,
                })
            receipt_body = {
                "schema_version": 1, "identity": identity, "file_count": len(records),
                "files": records, "completed": True, "server_only": True,
                "embedding_artifacts_mlflow_uploaded": False,
            }
            f005_receipt = {**receipt_body, "receipt_sha256": extraction.canonical_sha256(receipt_body)}
            receipt_path = cache_root / "full_scoring_cache_receipt.json"
            receipt_path.write_text(__import__("json").dumps(f005_receipt, sort_keys=True), encoding="utf-8")
            source_receipt = {
                "schema_version": "f007-f005-source-receipt-v1",
                "source_run_directory": str(source),
                "experiment_state": {"status": "complete", "experiment_signature": "b" * 64},
                "folds": [{
                    "outer_fold": 0,
                    "control_tail": {"checkpoint": {"sha256": checkpoint_sha}},
                    "full_scoring_control": {
                        "identity": {
                            "path": "full_scoring/fold_0/control/full_scoring_cache_identity.json",
                            "sha256": _sha_file(identity_path), "signature": identity["signature"],
                        },
                        "receipt": {
                            "path": "full_scoring/fold_0/control/full_scoring_cache_receipt.json",
                            "sha256": _sha_file(receipt_path),
                            "receipt_sha256": f005_receipt["receipt_sha256"],
                            "file_count": len(records),
                        },
                    },
                }],
            }
            contract = _contract(source=source_receipt, manifest=manifest)
            result = extraction.load_reused_f005_control_cache(
                contract, f005_run_directory=source, outer_fold=0,
            )
            self.assertEqual(result["kind"], extraction.F007_REUSED_F005_CONTROL_SCHEMA)
            self.assertEqual(result["embeddings"].shape, (2, 192))
            self.assertEqual(result["valid"].tolist(), [True, False])
            self.assertFalse(np.any(result["embeddings"][1]))
            self.assertTrue(result["server_only"])
            self.assertFalse(result["mlflow_upload_allowed"])
            self.assertFalse(result["local_transfer_allowed"])
            self.assertFalse((root / "full_scoring").exists())

    def test_pure_validation_module_has_no_audio_or_model_dependency(self):
        source = inspect.getsource(extraction)
        self.assertNotIn("import torch", source)
        self.assertNotIn("extract_advanced_embedding", source)
        self.assertNotIn("load_advanced", source)
        self.assertNotIn("read_mono", source)
        self.assertNotIn("speaker_id.candidates", source)


if __name__ == "__main__":
    unittest.main()
