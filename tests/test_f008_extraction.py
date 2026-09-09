"""Synthetic contract tests for server-only F008 full-scoring caches."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training import f008_extraction as extraction


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(index: int) -> np.ndarray:
    value = np.zeros((192,), dtype=np.float32)
    value[index % 192] = 1.0
    return value


def _manifest() -> list[dict]:
    return [
        {"audio_file": "synthetic-a.wav", "input_sha256": "1" * 64, "has_nonzero_signal": True},
        {"audio_file": "synthetic-b.wav", "input_sha256": "2" * 64, "has_nonzero_signal": False},
        {"audio_file": "synthetic-c.wav", "input_sha256": "3" * 64, "has_nonzero_signal": True},
    ]


def _tail_identity(*, source_receipt_sha: str = "c" * 64) -> dict:
    body = {
        "schema_version": 1,
        "f008_signature": "a" * 64,
        "source_f005_signature": "b" * 64,
        "f005_source_receipt_sha256": source_receipt_sha,
        "outer_fold": 0,
        "arm": {
            "id": "energy_005", "kind": "energy_margin_outlier_exposure", "lambda": 0.05,
        },
        "embedding_dimension": 192,
        "checkpoint_scope": "server_only_until_promotion",
        "known_tail_plan_sha256": "d" * 64,
        "unknown_plan_sha256": "e" * 64,
    }
    return {**body, "signature": extraction.canonical_sha256(body)}


def _metadata(identity: dict) -> dict:
    body = {
        "format_version": 1,
        "checkpoint_schema": "f008-open-set-oe-tail-checkpoint-v1",
        "stage": "tail",
        "f008_stage": "open_set_oe_tail",
        "f008_signature": identity["f008_signature"],
        "source_f005_signature": identity["source_f005_signature"],
        "f005_source_receipt_sha256": identity["f005_source_receipt_sha256"],
        "arm_signature": identity["signature"],
        "outer_fold": identity["outer_fold"],
        "arm_id": identity["arm"]["id"],
        "arm_kind": identity["arm"]["kind"],
        "oe_lambda": identity["arm"]["lambda"],
        "embedding_dimension": 192,
        "completed_steps": 1100,
        "total_steps": 1100,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    return {**body, "metadata_sha256": extraction.canonical_sha256(body)}


def _checkpoint_receipt(root: Path, identity: dict) -> tuple[Path, dict]:
    checkpoint = root / "completed-f008.pt"
    checkpoint.write_bytes(b"opaque worker-validated F008 checkpoint")
    return checkpoint, extraction.validated_f008_tail_checkpoint_receipt(
        identity, checkpoint_path=checkpoint, checkpoint_metadata=_metadata(identity),
    )


class F008ExtractionTests(unittest.TestCase):
    def test_full_manifest_cache_is_resumable_and_binds_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _tail_identity()
            checkpoint, checkpoint_receipt = _checkpoint_receipt(root, identity)
            plan = extraction.build_f008_advanced_cache_plan(
                identity, output_directory=root, relative_cache_directory="full_scoring/fold_0/energy_005",
                checkpoint_receipt=checkpoint_receipt, manifest=_manifest(),
                inference={"endpoint": "full_utterance", "seconds": 180.0, "maximum_windows": 1},
            )
            calls: list[int] = []

            def extractor(index, row):
                calls.append(index)
                self.assertEqual(row["audio_file"], _manifest()[index]["audio_file"])
                return _vector(index), index != 1

            result = extraction.extract_or_load_f008_advanced_cache(
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
            self.assertEqual(result["identity"]["checkpoint"], checkpoint_receipt)
            self.assertEqual(result["identity"]["row_count"], 3)
            self.assertEqual(
                [row["embedding_sha256"] for row in result["receipt"]["files"]],
                [extraction._array_sha256(result["embeddings"][index], "<f4") for index in range(3)],
            )

            again = extraction.extract_or_load_f008_advanced_cache(
                plan, checkpoint_path=checkpoint,
                extractor=lambda *_: self.fail("completed F008 cache must be reused"),
            )
            np.testing.assert_array_equal(again["embeddings"], result["embeddings"])

    def test_checkpoint_mutation_and_entry_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _tail_identity()
            checkpoint, receipt = _checkpoint_receipt(root, identity)
            plan = extraction.build_f008_advanced_cache_plan(
                identity, output_directory=root, relative_cache_directory="cache",
                checkpoint_receipt=receipt, manifest=_manifest(), inference={"endpoint": "full"},
            )
            checkpoint.write_bytes(b"changed checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                extraction.extract_or_load_f008_advanced_cache(
                    plan, checkpoint_path=checkpoint, extractor=lambda index, _row: (_vector(index), index != 1),
                )

            checkpoint, receipt = _checkpoint_receipt(root, identity)
            plan = extraction.build_f008_advanced_cache_plan(
                identity, output_directory=root, relative_cache_directory="cache2",
                checkpoint_receipt=receipt, manifest=_manifest(), inference={"endpoint": "full"},
            )
            extraction.extract_or_load_f008_advanced_cache(
                plan, checkpoint_path=checkpoint, extractor=lambda index, _row: (_vector(index), index != 1),
            )
            entries = Path(extraction.load_f008_advanced_cache(plan)["cache_directory"]) / "entries"
            next(entries.glob("*.npz")).write_bytes(b"not-an-npz")
            with self.assertRaises(ValueError):
                extraction.load_f008_advanced_cache(plan)

    def test_reuses_authenticated_f005_control_cache_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "f005-source"
            cache_root = source / "full_scoring" / "fold_0" / "control"
            entries = cache_root / "embedding_cache"
            entries.mkdir(parents=True)
            manifest = _manifest()
            checkpoint_sha = "4" * 64
            identity_body = {
                "schema_version": 1, "experiment_signature": "b" * 64,
                "outer_fold": 0, "arm_id": "control", "scope": "full_scoring",
                "indices_sha256": extraction._array_sha256(np.arange(len(manifest), dtype=np.int64), "<i8"),
                "checkpoint_sha256": checkpoint_sha, "checkpoint_metadata_sha256": "5" * 64,
                "embedding_dimension": 192,
                "inference": {"endpoint": "full_utterance"},
                "server_only": True, "mlflow_upload_allowed": False,
            }
            f005_identity = {**identity_body, "signature": extraction.canonical_sha256(identity_body)}
            identity_path = cache_root / "full_scoring_cache_identity.json"
            identity_path.write_text(json.dumps(f005_identity, sort_keys=True), encoding="utf-8")
            records = []
            for index, row in enumerate(manifest):
                path = entries / f"row-{index}.npz"
                valid = bool(row["has_nonzero_signal"])
                vector = _vector(index) if valid else np.zeros((192,), dtype=np.float32)
                with path.open("wb") as stream:
                    np.savez_compressed(stream, embedding=vector, valid=np.bool_(valid),
                                        signature=f005_identity["signature"], audio_file=row["audio_file"],
                                        audio_sha256=row["input_sha256"])
                records.append({
                    "audio_file": row["audio_file"], "audio_sha256": row["input_sha256"],
                    "cache_file": path.name, "cache_sha256": _sha_file(path),
                    "bytes": path.stat().st_size, "valid": valid,
                })
            f005_receipt_body = {
                "schema_version": 1, "identity": f005_identity, "file_count": len(records),
                "files": records, "completed": True, "server_only": True,
                "embedding_artifacts_mlflow_uploaded": False,
            }
            f005_receipt = {**f005_receipt_body,
                            "receipt_sha256": extraction.canonical_sha256(f005_receipt_body)}
            receipt_path = cache_root / "full_scoring_cache_receipt.json"
            receipt_path.write_text(json.dumps(f005_receipt, sort_keys=True), encoding="utf-8")
            source_receipt = {
                "schema_version": "f007-f005-source-receipt-v1",
                "source_run_directory": str(source),
                "experiment_state": {"status": "complete", "experiment_signature": "b" * 64},
                "folds": [{
                    "outer_fold": 0,
                    "control_tail": {"checkpoint": {"sha256": checkpoint_sha}},
                    "full_scoring_control": {
                        "identity": {"path": "full_scoring/fold_0/control/full_scoring_cache_identity.json",
                                     "sha256": _sha_file(identity_path), "signature": f005_identity["signature"]},
                        "receipt": {"path": "full_scoring/fold_0/control/full_scoring_cache_receipt.json",
                                    "sha256": _sha_file(receipt_path),
                                    "receipt_sha256": f005_receipt["receipt_sha256"],
                                    "file_count": len(records)},
                    },
                }],
            }
            tail = _tail_identity(source_receipt_sha=extraction.canonical_sha256(source_receipt))
            result = extraction.load_reused_f005_control_cache(
                tail, source_receipt, f005_run_directory=source, manifest=manifest, outer_fold=0,
            )
            self.assertEqual(result["kind"], extraction.F008_REUSED_F005_CONTROL_SCHEMA)
            self.assertEqual(result["embeddings"].shape, (3, 192))
            self.assertEqual(result["valid"].tolist(), [True, False, True])
            self.assertFalse(np.any(result["embeddings"][1]))
            self.assertFalse((root / "copied-cache").exists())
            records[0]["cache_sha256"] = "0" * 64
            f005_receipt_body["files"] = records
            altered = {**f005_receipt_body, "receipt_sha256": extraction.canonical_sha256(f005_receipt_body)}
            receipt_path.write_text(json.dumps(altered, sort_keys=True), encoding="utf-8")
            with self.assertRaises(ValueError):
                extraction.load_reused_f005_control_cache(
                    tail, source_receipt, f005_run_directory=source, manifest=manifest, outer_fold=0,
                )

    def test_module_has_no_model_audio_mlflow_or_transfer_dependency(self):
        tree = ast.parse(inspect.getsource(extraction))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in ("torch", "mlflow", "paramiko", "scp", "speaker_id.candidates"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(name == forbidden or name.startswith(forbidden + ".")
                                     for name in imported))


if __name__ == "__main__":
    unittest.main()
