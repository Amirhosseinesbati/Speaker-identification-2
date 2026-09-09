"""Fault-injection tests for F005 orchestration recovery boundaries.

These tests exercise process-loss windows at durable side effects.  They assert
the public resume/fail-closed contract rather than the shape of orchestration
helpers: an authenticated result is reused, a remote run is never duplicated,
and outer truth stays unreadable until all pretruth gates pass.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import f005_experiment as experiment

# Reuse the deliberately small end-to-end fake instead of reproducing F005's
# training/scoring behavior in recovery tests.
from tests.test_f005_experiment import FakeBackend, fake_project


class _DurableSpoolBackend(FakeBackend):
    """Fake remote runs whose local spool remains discoverable after a crash."""

    def __init__(self, *, crash_run_kind: str | None = None):
        super().__init__()
        self.crash_run_kind = crash_run_kind
        self.crashed = False
        self.opened_spools: list[str] = []
        self.prepared_spools: list[str] = []

    def prepare_tracker(self, **kwargs):
        spool = Path(kwargs["spool_dir"])
        if str(spool) in self.trackers or (spool / "run_state.json").exists():
            raise AssertionError(f"resume attempted to create a duplicate run: {spool}")
        tracker = super().prepare_tracker(**kwargs)
        self.prepared_spools.append(str(spool))
        (spool / "run_state.json").write_text(
            json.dumps({
                "run_id": tracker.run_id,
                "run_kind": kwargs["run_kind"],
                "parent_run_id": kwargs.get("parent_run_id"),
            }),
            encoding="utf-8",
        )
        original_verify = tracker.verify_remote_metadata

        def verify_remote_metadata():
            if not self.crashed and kwargs["run_kind"] == self.crash_run_kind:
                self.crashed = True
                raise RuntimeError(f"power loss after {kwargs['run_kind']} spool creation")
            return original_verify()

        tracker.verify_remote_metadata = verify_remote_metadata
        return tracker

    def open_tracker(self, spool):
        self.opened_spools.append(str(Path(spool)))
        return super().open_tracker(spool)


class _RebuildablePretruthBackend(FakeBackend):
    def rebuild_policy(self, contract, outer, *, policy_reload, **_kwargs):
        self.events.append(("rebuild_policy", outer))
        score = {
            "outer_known_scores": np.ones((2, 1), dtype=np.float32),
            "outer_unknown_similarity": np.zeros(2, dtype=np.float32),
            "outer_valid": np.ones(2, dtype=np.bool_),
        }
        return {
            "kind": "fake",
            "outer_fold": outer,
            "policy_reload": policy_reload,
            "score_bundles": {
                name: {key: value.copy() for key, value in score.items()}
                for name in ("frozen_same_protocol", "fresh_control", "selected_arm")
            },
            "outer_files": [
                row["audio_file"] for row in contract["manifest"]
                if int(row["fold"]) == outer
            ],
            "outer_truth_read": False,
        }


class _ParityFailureBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.parity_failed = False
        self.truth_reads_after_failed_parity = 0

    def verify_parity(self, _pretruth):
        self.events.append(("parity",))
        self.parity_failed = True
        return {
            "status": "failed",
            "exact_cpu_cuda_prediction_parity": False,
            "outer_truth_read": False,
        }


class _TruthReadProbe(dict):
    def __init__(self, value, backend: _ParityFailureBackend):
        super().__init__(value)
        self.backend = backend

    def __getitem__(self, key):
        if key == "speaker_id" and self.backend.parity_failed:
            self.backend.truth_reads_after_failed_parity += 1
            raise AssertionError("outer truth was read after parity failed")
        return super().__getitem__(key)


class F005TransactionalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_parent_and_child_spools_are_adopted_without_duplicate_remote_runs(self):
        cases = (
            ("parent", "f005_paired_long_short_full_experiment", "tracking/parent"),
            ("child", "f005_shared_head", "tracking/children/fold_0"),
        )
        for name, run_kind, relative_spool in cases:
            with self.subTest(stage=name):
                root, config, binding, contract = fake_project(self.temp_path / name)
                backend = _DurableSpoolBackend(crash_run_kind=run_kind)
                output = root / "runs" / "interrupted"

                with self.assertRaisesRegex(RuntimeError, "power loss"):
                    experiment.execute_f005_experiment(
                        contract, root, config, binding,
                        backend=backend, output_dir=output,
                    )

                suffix = Path(relative_spool).parts
                stranded_spool = next(
                    Path(candidate) for candidate in backend.prepared_spools
                    if Path(candidate).parts[-len(suffix):] == suffix
                )
                stranded_run_id = backend.trackers[str(stranded_spool)].run_id
                recovered = experiment.execute_f005_experiment(
                    contract, root, config, binding,
                    backend=backend, resume_dir=output,
                )

                self.assertEqual(recovered["status"], "complete")
                self.assertEqual(len(backend.prepared_spools), 11)
                self.assertEqual(len(set(backend.prepared_spools)), 11)
                self.assertIn(str(stranded_spool), backend.opened_spools)
                self.assertEqual(
                    backend.trackers[str(stranded_spool)].run_id,
                    stranded_run_id,
                )

    def test_orphan_known_scores_are_authenticated_and_adopted_on_resume(self):
        root, config, binding, contract = fake_project(self.temp_path / "known")
        backend = FakeBackend()
        output = root / "runs" / "interrupted"
        original = experiment._save_known_scores
        injected = False

        def crash_after_replace(path, arm_id, current_contract, scores):
            nonlocal injected
            receipt = original(path, arm_id, current_contract, scores)
            if not injected:
                injected = True
                raise RuntimeError("power loss after known_scores replace")
            return receipt

        with patch.object(experiment, "_save_known_scores", side_effect=crash_after_replace):
            with self.assertRaisesRegex(RuntimeError, "known_scores replace"):
                experiment.execute_f005_experiment(
                    contract, root, config, binding,
                    backend=backend, output_dir=output,
                )

        orphan = output / "selection" / "fold_0" / "control" / "known_scores.npz"
        self.assertTrue(orphan.is_file())
        recovered = experiment.execute_f005_experiment(
            contract, root, config, binding,
            backend=backend, resume_dir=output,
        )

        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(sum(event[0] == "known_scores" for event in backend.events), 8)
        self.assertEqual(
            sum(event[:2] == ("extract", "known_selection") for event in backend.events),
            8,
        )
        state = json.loads((output / "experiment_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["known_scores"]["fold_0/control"]["sha256"], experiment._sha_file(orphan))

    def test_half_committed_pretruth_pair_rebuilds_before_outer_truth(self):
        root, config, binding, contract = fake_project(self.temp_path / "pretruth")
        backend = _RebuildablePretruthBackend()
        output = root / "runs" / "interrupted"
        original = experiment._write_json
        injected = False

        def crash_before_pretruth_metadata(path, value):
            nonlocal injected
            if Path(path).name == "pretruth_bundle.json" and not injected:
                injected = True
                raise RuntimeError("power loss between pretruth array and metadata commits")
            return original(path, value)

        with patch.object(experiment, "_write_json", side_effect=crash_before_pretruth_metadata):
            with self.assertRaisesRegex(RuntimeError, "between pretruth"):
                experiment.execute_f005_experiment(
                    contract, root, config, binding,
                    backend=backend, output_dir=output,
                )

        cache = output / "full_scoring" / "fold_0" / "pretruth_bundle"
        self.assertTrue(cache.with_suffix(".npz").is_file())
        self.assertFalse(cache.with_suffix(".json").exists())
        self.assertFalse(any(event[0] == "outer" for event in backend.events))

        recovered = experiment.execute_f005_experiment(
            contract, root, config, binding,
            backend=backend, resume_dir=output,
        )

        self.assertEqual(recovered["status"], "complete")
        self.assertTrue(cache.with_suffix(".npz").is_file())
        self.assertTrue(cache.with_suffix(".json").is_file())
        self.assertEqual(
            experiment._load_pretruth(cache)["outer_truth_read"],
            False,
        )
        rebuild = backend.events.index(("rebuild_policy", 0))
        parity = backend.events.index(("parity",))
        first_outer = next(
            index for index, event in enumerate(backend.events) if event[0] == "outer"
        )
        self.assertLess(rebuild, parity)
        self.assertLess(parity, first_outer)

    def test_complete_pretruth_pair_without_state_resumes_from_sealed_policy(self):
        root, config, binding, contract = fake_project(self.temp_path / "pretruth-state-gap")
        backend = _RebuildablePretruthBackend()
        output = root / "runs" / "interrupted"
        original = experiment._save_pretruth
        injected = False

        def crash_after_complete_pretruth_pair(path, pretruth):
            nonlocal injected
            receipt = original(path, pretruth)
            if Path(path).parent.name == "fold_0" and not injected:
                injected = True
                raise RuntimeError("power loss after pretruth pair before policy state")
            return receipt

        with patch.object(
                experiment, "_save_pretruth", side_effect=crash_after_complete_pretruth_pair):
            with self.assertRaisesRegex(RuntimeError, "before policy state"):
                experiment.execute_f005_experiment(
                    contract, root, config, binding,
                    backend=backend, output_dir=output,
                )

        cache = output / "full_scoring" / "fold_0" / "pretruth_bundle"
        self.assertTrue(cache.with_suffix(".npz").is_file())
        self.assertTrue(cache.with_suffix(".json").is_file())
        state = json.loads((output / "experiment_state.json").read_text(encoding="utf-8"))
        self.assertNotIn("0", state["policy_seals"])
        self.assertFalse(any(event[0] == "outer" for event in backend.events))

        recovered = experiment.execute_f005_experiment(
            contract, root, config, binding,
            backend=backend, resume_dir=output,
        )

        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(
            experiment._load_pretruth(cache)["outer_truth_read"],
            False,
        )
        self.assertIn(("rebuild_policy", 0), backend.events)
        self.assertEqual(sum(event == ("seal_policy", 0) for event in backend.events), 1)
        rebuild = backend.events.index(("rebuild_policy", 0))
        parity = backend.events.index(("parity",))
        first_outer = next(
            index for index, event in enumerate(backend.events) if event[0] == "outer"
        )
        self.assertLess(rebuild, parity)
        self.assertLess(parity, first_outer)

    def test_orphan_pretruth_arrays_must_equal_rebuilt_sealed_evidence(self):
        cache = self.temp_path / "mismatched-pretruth" / "pretruth_bundle"
        original = {
            "outer_fold": 0,
            "scores": np.asarray([[0.1, 0.9]], dtype=np.float32),
            "outer_truth_read": False,
        }
        experiment._save_pretruth(cache, original)
        cache.with_suffix(".json").unlink()
        changed = {
            **original,
            "scores": np.asarray([[0.9, 0.1]], dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "differ from the rebuilt sealed policy"):
            experiment._save_pretruth(cache, changed)
        np.testing.assert_array_equal(
            experiment._load_npz_arrays(cache.with_suffix(".npz"))["array_0000"],
            original["scores"],
        )

    def test_parity_failure_blocks_outer_truth_materialization(self):
        root, config, binding, contract = fake_project(self.temp_path / "parity")
        backend = _ParityFailureBackend()
        contract["manifest"] = [
            _TruthReadProbe(row, backend) for row in contract["manifest"]
        ]
        output = root / "runs" / "parity-failed"

        with self.assertRaisesRegex(ValueError, "(?i)parity"):
            experiment.execute_f005_experiment(
                contract, root, config, binding,
                backend=backend, output_dir=output,
            )

        state = json.loads((output / "experiment_state.json").read_text(encoding="utf-8"))
        self.assertIs(state["outer_truth_materialized"], False)
        self.assertEqual(state["outer_evaluations"], {})
        self.assertEqual(backend.truth_reads_after_failed_parity, 0)
        self.assertFalse(any(event[0] == "outer" for event in backend.events))
        self.assertFalse((output / "evaluation").exists())


if __name__ == "__main__":
    unittest.main()
