"""Pure-filesystem tests for the F007 F005-source receipt helper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speaker_id.training.f007_source import (
    F007_F005_SOURCE_RECEIPT_SCHEMA,
    load_f005_source_receipt,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class F007SourceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "F005_complete"
        self.state = self._complete_source(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _complete_source(self, root: Path) -> dict:
        signature = "a" * 64
        state = {
            "schema_version": "f005-resumable-experiment-state-v1",
            "experiment_signature": signature,
            "output_directory": str(root),
            "status": "complete",
            "heads": {},
            "tails": {},
            "full_embeddings": {},
        }
        for outer in (0, 1):
            head_dir = root / "training" / f"fold_{outer}" / "shared_head"
            head = head_dir / "shared_head.pt"
            report = head_dir / "unit_report.json"
            head.parent.mkdir(parents=True, exist_ok=True)
            head.write_bytes(f"head-{outer}".encode())
            write_json(report, {"stage": "shared_head", "outer_fold": outer})
            head_sha = sha256_file(head)
            state["heads"][f"fold_{outer}"] = {
                "complete": True,
                "checkpoint": str(head),
                "checkpoint_sha256": head_sha,
                "report": str(report),
                "report_sha256": sha256_file(report),
            }

            tail_dir = root / "training" / f"fold_{outer}" / "tails" / "control"
            tail = tail_dir / "last.pt"
            tail_report = tail_dir / "unit_report.json"
            tail.parent.mkdir(parents=True, exist_ok=True)
            tail.write_bytes(f"tail-{outer}".encode())
            write_json(tail_report, {"stage": "tail", "outer_fold": outer, "arm_id": "control"})
            tail_sha = sha256_file(tail)
            state["tails"][f"fold_{outer}/control"] = {
                "complete": True,
                "checkpoint": str(tail),
                "checkpoint_sha256": tail_sha,
                "shared_checkpoint_sha256": head_sha,
                "report": str(tail_report),
                "report_sha256": sha256_file(tail_report),
            }

            cache_dir = root / "full_scoring" / f"fold_{outer}" / "control"
            identity_body = {
                "schema_version": 1,
                "experiment_signature": signature,
                "outer_fold": outer,
                "arm_id": "control",
                "scope": "full_scoring",
                "indices_sha256": "b" * 64,
                "checkpoint_sha256": tail_sha,
                "checkpoint_metadata_sha256": "c" * 64,
                "embedding_dimension": 192,
                "inference": {"kind": "fixture"},
                "server_only": True,
                "mlflow_upload_allowed": False,
            }
            identity = {**identity_body, "signature": canonical_sha(identity_body)}
            identity_path = cache_dir / "full_scoring_cache_identity.json"
            write_json(identity_path, identity)
            rows = 3
            receipt_body = {
                "schema_version": 1,
                "identity": identity,
                "file_count": rows,
                "files": [{"audio_file": f"f{index}.wav"} for index in range(rows)],
                "completed": True,
                "server_only": True,
                "embedding_artifacts_mlflow_uploaded": False,
            }
            receipt = {**receipt_body, "receipt_sha256": canonical_sha(receipt_body)}
            receipt_path = cache_dir / "full_scoring_cache_receipt.json"
            write_json(receipt_path, receipt)
            state["full_embeddings"][f"fold_{outer}/control"] = {
                "scope": "all_manifest_rows",
                "rows": rows,
                "cache_receipt": str(receipt_path),
                "cache_receipt_sha256": sha256_file(receipt_path),
                "mlflow_uploaded": False,
            }
        write_json(root / "experiment_state.json", state)
        return state

    def _write_state(self) -> None:
        write_json(self.root / "experiment_state.json", self.state)

    def test_loads_complete_source_and_returns_portable_compact_receipt(self):
        result = load_f005_source_receipt(self.root)
        self.assertEqual(result["schema_version"], F007_F005_SOURCE_RECEIPT_SCHEMA)
        self.assertEqual(result["source_run_directory"], str(self.root.resolve()))
        self.assertEqual(result["fold_ids"], [0, 1])
        self.assertEqual([entry["outer_fold"] for entry in result["folds"]], [0, 1])
        first = result["folds"][0]
        self.assertEqual(first["shared_head"]["checkpoint"]["path"],
                         "training/fold_0/shared_head/shared_head.pt")
        self.assertEqual(first["control_tail"]["shared_head_checkpoint_sha256"],
                         self.state["heads"]["fold_0"]["checkpoint_sha256"])
        self.assertEqual(first["full_scoring_control"]["receipt"]["file_count"], 3)
        # All returned values are serializable without a custom JSON encoder.
        json.dumps(result, allow_nan=False, sort_keys=True)

    def test_rejects_noncomplete_state_before_reuse(self):
        self.state["status"] = "running"
        self._write_state()
        with self.assertRaisesRegex(ValueError, "not complete"):
            load_f005_source_receipt(self.root)

    def test_rejects_tampered_shared_head_bytes(self):
        target = self.root / "training/fold_1/shared_head/shared_head.pt"
        target.write_bytes(target.read_bytes() + b"different")
        with self.assertRaisesRegex(ValueError, "shared head checkpoint fold 1 bytes differ"):
            load_f005_source_receipt(self.root)

    def test_rejects_control_tail_bound_to_other_head(self):
        self.state["tails"]["fold_0/control"]["shared_checkpoint_sha256"] = "d" * 64
        self._write_state()
        with self.assertRaisesRegex(ValueError, "not bound to its shared head"):
            load_f005_source_receipt(self.root)

    def test_rejects_cache_identity_not_bound_to_control_tail(self):
        path = self.root / "full_scoring/fold_0/control/full_scoring_cache_identity.json"
        identity = json.loads(path.read_text(encoding="utf-8"))
        identity["checkpoint_sha256"] = "e" * 64
        body = {key: value for key, value in identity.items() if key != "signature"}
        identity["signature"] = canonical_sha(body)
        write_json(path, identity)
        receipt_path = self.root / "full_scoring/fold_0/control/full_scoring_cache_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["identity"] = identity
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt["receipt_sha256"] = canonical_sha(receipt_body)
        write_json(receipt_path, receipt)
        self.state["full_embeddings"]["fold_0/control"]["cache_receipt_sha256"] = sha256_file(receipt_path)
        self._write_state()
        with self.assertRaisesRegex(ValueError, "not bound to its control tail"):
            load_f005_source_receipt(self.root)

    def test_rejects_cache_receipt_hash_mismatch(self):
        receipt = self.root / "full_scoring/fold_1/control/full_scoring_cache_receipt.json"
        receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt for fold 1 bytes differ"):
            load_f005_source_receipt(self.root)


if __name__ == "__main__":
    unittest.main()
