"""Focused integrity tests for the cache-only C002b/F005 scoring bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.evaluation import scoring_bridge as bridge


def _sha_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ScoringBridgeTests(unittest.TestCase):
    def test_historical_policy_uses_identity_fit_not_global_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, c002b = root / "run", root / "C002b"
            (c002b / "fold_0").mkdir(parents=True)
            calibration = {
                "unknown_weight": 0.75, "margin_weight": 0.5,
                "threshold": 0.25, "inner_macro_f1_447": 0.9,
            }
            frozen = {
                "selected": {"0": {"frontend": "gain"}},
                "inner_fits": {"0": {"identity": {"selected": {
                    "advanced_weight": 0.5, "calibration": calibration,
                }}}},
            }
            run.mkdir()
            (run / "frozen_inner_choices.json").write_text(json.dumps(frozen), encoding="utf-8")
            (c002b / "fold_0" / "calibration.json").write_text(
                json.dumps({"selected": calibration, "curve": []}), encoding="utf-8",
            )
            result = bridge._load_historical_policy(run, c002b, 0)
            self.assertEqual(result["advanced_weight"], 0.5)
            self.assertEqual(result["calibration"], calibration)

    def test_control_cache_rejects_manifest_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "full_scoring" / "fold_0" / "control"
            cache = base / "embedding_cache"
            cache.mkdir(parents=True)
            manifest = [
                {"audio_file": "a.wav", "input_sha256": "a" * 64},
                {"audio_file": "b.wav", "input_sha256": "b" * 64},
            ]
            indices_sha = hashlib.sha256(np.ascontiguousarray(
                np.arange(2, dtype=np.int64), dtype="<i8",
            ).tobytes()).hexdigest()
            identity_body = {
                "schema_version": 1, "experiment_signature": "c" * 64,
                "outer_fold": 0, "arm_id": "control", "scope": "full_scoring",
                "indices_sha256": indices_sha, "checkpoint_sha256": "d" * 64,
                "checkpoint_metadata_sha256": "e" * 64, "embedding_dimension": 192,
                "inference": "full", "server_only": True, "mlflow_upload_allowed": False,
            }
            identity = {**identity_body, "signature": bridge._canonical_sha(identity_body)}
            records = []
            for index, source in enumerate(manifest):
                vector = np.zeros(192, dtype=np.float32)
                if index == 0:
                    vector[0] = 1.0
                path = cache / (Path(source["audio_file"]).stem + ".npz")
                np.savez_compressed(
                    path, embedding=vector, valid=bool(index == 0), signature=identity["signature"],
                    audio_file=source["audio_file"], audio_sha256=source["input_sha256"],
                )
                records.append({
                    "audio_file": source["audio_file"], "audio_sha256": source["input_sha256"],
                    "cache_file": path.name, "cache_sha256": _sha_file(path),
                    "bytes": path.stat().st_size, "valid": bool(index == 0),
                })
            receipt_body = {
                "schema_version": 1, "identity": identity, "file_count": 2, "files": records,
                "completed": True, "server_only": True, "embedding_artifacts_mlflow_uploaded": False,
            }
            receipt = {**receipt_body, "receipt_sha256": bridge._canonical_sha(receipt_body)}
            (base / "full_scoring_cache_identity.json").write_text(json.dumps(identity), encoding="utf-8")
            (base / "full_scoring_cache_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            values, evidence = bridge._load_f005_control_cache(root, 0, manifest, np.array([True, False], dtype=bool))
            self.assertEqual(values.shape, (2, 192))
            self.assertEqual(evidence["cache_files_verified"], 2)
            receipt["files"] = list(reversed(receipt["files"]))
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            receipt["receipt_sha256"] = bridge._canonical_sha(body)
            (base / "full_scoring_cache_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record identity"):
                bridge._load_f005_control_cache(root, 0, manifest, np.array([True, False], dtype=bool))

    def test_score_uses_sealed_policy_without_refit(self) -> None:
        policy = {
            "advanced_weight": 0.5,
            "calibration": {
                "unknown_weight": 0.75, "margin_weight": 0.5,
                "threshold": 0.2, "inner_macro_f1_447": 0.9,
            },
        }
        fixed_score = {
            "outer_known_scores": np.array([[0.8, 0.2]], dtype=np.float32),
            "outer_unknown_similarity": np.array([0.1], dtype=np.float32),
            "outer_valid": np.array([True], dtype=bool),
            "outer_indices": np.array([0], dtype=np.int64),
        }
        endpoint = {"known_labels": ["one", "two"]}
        with patch.object(bridge, "crossfit_scores", return_value=endpoint), patch.object(
                bridge, "scores_for_alpha", return_value=fixed_score) as selected:
            probabilities, names, _ = bridge._score_with_fixed_policy(
                np.zeros((1, 512), dtype=np.float32), np.zeros((1, 192), dtype=np.float32),
                np.array([True], dtype=bool), [{"audio_file": "x.wav"}], [],
                ["unknown", "one", "two"], 0, policy,
            )
        self.assertEqual(names, ["x.wav"])
        self.assertEqual(selected.call_args.args[6], 0.5)
        expected = bridge.reference_probabilities(
            fixed_score["outer_known_scores"], fixed_score["outer_unknown_similarity"],
            policy["calibration"], fixed_score["outer_valid"], 0.05,
        )
        self.assertTrue(np.array_equal(probabilities, expected))


if __name__ == "__main__":
    unittest.main()
