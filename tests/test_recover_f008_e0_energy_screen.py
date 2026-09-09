"""Focused contracts for the bounded F008 E0 recovery launcher."""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/recover_f008_e0_energy_screen.py"
SPEC = importlib.util.spec_from_file_location("f008_e0_recovery_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class F008E0RecoveryLauncherTests(unittest.TestCase):
    f008_signature = "a" * 64
    f005_signature = "b" * 64
    runtime_sha = "c" * 64
    execution = {"gpu_name_contains": "RTX 3090"}

    def _failed_root(self, directory: Path) -> Path:
        root = directory / "F008_E0_ENERGY_SCREEN_failed"
        (root / "tracking").mkdir(parents=True)
        (root / "failure.json").write_text(json.dumps({
            "status": "failed", "screen": "E0", "outer_evaluation_called": False,
            "promotion_allowed": False,
        }), encoding="utf-8")
        (root / "runtime_receipt.json").write_text(json.dumps({
            "schema_version": "f008-runtime-receipt-v1",
            "receipt_sha256": self.runtime_sha,
            "f008_signature": self.f008_signature,
            "source_f005_signature": self.f005_signature,
            "expected_vast_instance_id": 50079023,
            "vast_instance_id": 50079023,
            "device": "cuda", "gpu_name": "NVIDIA RTX 3090",
            "cuda_available": True, "cpu_threads": 8,
            "checked_once_per_logical_run": True, "execution": self.execution,
        }), encoding="utf-8")
        (root / "tracking" / "run_state.json").write_text(
            json.dumps({"run_id": "prior-parent-run"}), encoding="utf-8",
        )
        return root

    def test_failed_e0_evidence_is_read_only_and_binds_prior_runtime_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._failed_root(Path(directory))
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            failed = launcher.validate_failed_e0_evidence(
                root, f008_signature=self.f008_signature,
                f005_signature=self.f005_signature, execution=self.execution,
            )
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(failed.parent_run_id, "prior-parent-run")
            self.assertEqual(failed.previous_runtime_receipt_sha256, self.runtime_sha)
            declaration = launcher.recovery_declaration(
                failed, fresh_runtime_receipt_sha256=self.runtime_sha,
            )
            self.assertFalse(declaration["failed_root_mutated"])
            self.assertFalse(declaration["fold_0_training_executed"])
            self.assertFalse(declaration["fold_1_training_executed"])
            self.assertTrue(declaration["fold_1_training_planned"])

    def test_failed_e0_evidence_rejects_a_different_execution_or_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._failed_root(Path(directory))
            with self.assertRaises(ValueError):
                launcher.validate_failed_e0_evidence(
                    root, f008_signature=self.f008_signature,
                    f005_signature=self.f005_signature,
                    execution={"gpu_name_contains": "A100"},
                )
            runtime_path = root / "runtime_receipt.json"
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            payload["receipt_sha256"] = "not-a-sha"
            runtime_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                launcher.validate_failed_e0_evidence(
                    root, f008_signature=self.f008_signature,
                    f005_signature=self.f005_signature, execution=self.execution,
                )

    def test_verify_only_and_reuse_contract_do_not_include_fresh_work(self) -> None:
        parser = launcher._parser()
        verified = parser.parse_args([
            "--failed-e0-directory", "artifacts/F008_E0_ENERGY_SCREEN_failed", "--verify-only",
        ])
        self.assertTrue(verified.verify_only)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--failed-e0-directory", "x", "--verify-only", "--execute",
            ])
        reuse_source = inspect.getsource(launcher.verify_reusable_fold0)
        self.assertIn("verify_validated_f008_tail_checkpoint_bytes", reuse_source)
        self.assertIn("load_f008_advanced_cache", reuse_source)
        self.assertNotIn("fit_f008_tail", reuse_source)
        self.assertNotIn("_extract_energy_cache", reuse_source)
        fold0_source = inspect.getsource(launcher._reused_fold0_pretruth)
        self.assertNotIn("fit_f008_tail", fold0_source)
        self.assertNotIn("_extract_energy_cache", fold0_source)


if __name__ == "__main__":
    unittest.main()
