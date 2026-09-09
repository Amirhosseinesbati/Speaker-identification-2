from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from speaker_id.training.f005_experiment import (
    _safe_add_artifact,
    execute_f005_experiment,
)


ARMS = ("control", "treatment_mse0", "treatment_mse01", "treatment_mse05")


class FakeTracker:
    def __init__(self, backend, spool, run_id):
        self.backend, self.spool, self.run_id = backend, Path(spool), run_id
        self.state = {"remote_status": None}
        self.redactor = None

    def flush(self, **_):
        return True

    def verify_artifacts(self):
        return {"status": "passed"}

    def verify_remote_metadata(self):
        return {"status": "passed"}

    def add_artifact(self, source, relative_path=None):
        self.backend.uploads.append((str(source), relative_path))

    def log_metrics(self, metrics, *_args, **_kwargs):
        self.backend.logged_metrics.append(dict(metrics))
        return True

    def write_report(self, _report, markdown=None, *_args, **_kwargs):
        if markdown is not None:
            self.backend.markdown_reports.append(markdown)
        return None

    def finish(self, status="FINISHED", **_):
        self.state["remote_status"] = status
        self.backend.events.append(("finish", self.run_id, status))
        return True

    def reopen(self):
        self.state["remote_status"] = "RUNNING"
        self.backend.events.append(("reopen", self.run_id))
        return True


class FakeBackend:
    def __init__(self, *, fail_tail_once=None):
        self.events, self.uploads, self.trackers = [], [], {}
        self.logged_metrics, self.markdown_reports = [], []
        self.fail_tail_once = fail_tail_once
        self.failed = False
        self.fit_head_calls = 0

    def load_binding(self, _path, experiment_id):
        assert experiment_id == "1"
        return object()

    def prepare_tracker(self, **kwargs):
        spool = Path(kwargs["spool_dir"])
        spool.mkdir(parents=True, exist_ok=True)
        run_id = f"run-{len(self.trackers):02d}"
        tracker = FakeTracker(self, spool, run_id)
        self.trackers[str(spool)] = tracker
        self.events.append(("prepare_tracker", kwargs["run_kind"], run_id))
        return tracker

    def open_tracker(self, spool):
        return self.trackers[str(Path(spool))]

    def execution_plan(self, contract):
        return {"experiment_signature": contract["signature"], "units": 10}

    def require_environment(self, _contract, _root):
        return {"status": "ready", "vast_instance_id": "50288952"}

    def load_frozen_sources(self, contract):
        count = len(contract["manifest"])
        public = np.zeros((count, 512), dtype=np.float32); public[:, 0] = 1
        advanced = np.zeros((count, 192), dtype=np.float32); advanced[:, 0] = 1
        self.events.append(("load_frozen",))
        return {"public": public, "frozen_advanced": advanced,
                "valid": np.ones(count, dtype=np.bool_)}

    def fit_head(self, _contract, _root, outer, output, _tracker, *, resume):
        self.fit_head_calls += 1
        self.events.append(("fit_head", outer, resume))
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "shared_head.pt"
        checkpoint.write_bytes(f"head-{outer}".encode())
        return {"checkpoint": checkpoint,
                "report": {"completed_steps": 600, "outer_fold": outer}}

    def fit_tail(self, _contract, _root, outer, arm_id, _shared, output, _tracker, *, resume):
        self.events.append(("fit_tail", outer, arm_id, resume))
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "last.pt"
        checkpoint.write_bytes(f"tail-{outer}-{arm_id}".encode())
        if self.fail_tail_once == (outer, arm_id) and not self.failed:
            self.failed = True
            raise RuntimeError("simulated interruption")
        return {"checkpoint": checkpoint,
                "report": {"completed_steps": 1100, "outer_fold": outer, "arm_id": arm_id}}

    def extract_arm(self, contract, _root, outer, arm_id, _shared, _checkpoint,
                    indices, scope, expected_valid, cache_dir, progress):
        self.events.append(("extract", scope, outer, arm_id))
        cache_dir.mkdir(parents=True, exist_ok=True)
        receipt = cache_dir.parent / f"{scope}_cache_receipt.json"
        receipt.write_text(json.dumps({"scope": scope, "rows": len(indices)}), encoding="utf-8")
        embeddings = np.zeros((len(indices), 192), dtype=np.float32); embeddings[:, 0] = 1
        progress(len(indices), len(indices), 0.01)
        return {"embeddings": embeddings, "valid": expected_valid[indices],
                "receipt_path": str(receipt)}

    def known_scores(self, _embeddings, _valid, contract, outer):
        self.events.append(("known_scores", outer))
        return {
            "known_calibration_indices": np.asarray([outer], dtype=np.int64),
            "known_scores": np.asarray([[0.9]], dtype=np.float32),
            "known_labels": contract["labels"][1:],
            "reference_support": np.asarray([[1]], dtype=np.int64),
            "provenance": {"protocol": "fake-known-only", "outer_fold": outer},
        }

    def seal_arm(self, contract, outer, _scores, path):
        self.events.append(("seal_arm", outer))
        selected = "treatment_mse01"
        seal = {
            "selected_arm": selected,
            "arm_metrics": {arm: {"macro_f1_observed_known_labels": .8,
                                   "top1_accuracy": .8} for arm in ARMS},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(seal), encoding="utf-8")
        return {"seal": seal, "seal_sha256": "a" * 64,
                "file_sha256": "b" * 64, "path": str(path)}

    def prepare_policy(self, contract, outer, *, public, frozen,
                       selected_embeddings, valid, arm_reload, path):
        assert set(selected_embeddings) == {"control", "treatment_mse01"}
        self.events.append(("seal_policy", outer))
        seal = {"policies": {name: {"calibration": {}}
                             for name in ("frozen_same_protocol", "fresh_control", "selected_arm")}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seal), encoding="utf-8")
        reload = {"seal": seal, "seal_sha256": "c" * 64,
                  "file_sha256": "d" * 64, "path": str(path)}
        score = {"outer_known_scores": np.ones((2, 1), dtype=np.float32),
                 "outer_unknown_similarity": np.zeros(2, dtype=np.float32),
                 "outer_valid": np.ones(2, dtype=np.bool_)}
        return {"kind": "fake", "outer_fold": outer, "policy_reload": reload,
                "score_bundles": {name: score for name in seal["policies"]},
                "outer_files": [row["audio_file"] for row in contract["manifest"]
                                if int(row["fold"]) == outer],
                "outer_truth_read": False}

    def reload_policy(self, path, _contract, outer, _arm_reload):
        self.events.append(("reload_policy", outer))
        seal = json.loads(path.read_text(encoding="utf-8"))
        return {"seal": seal, "seal_sha256": "c" * 64,
                "file_sha256": "d" * 64, "path": str(path)}

    def rebuild_policy(self, *_args, **_kwargs):
        raise AssertionError("rebuild should not be needed in the clean fake run")

    def verify_parity(self, _pretruth):
        self.events.append(("parity",))
        return {"exact_cpu_cuda_prediction_parity": True, "outer_truth_read": False}

    def evaluate_outer(self, pretruth, _reload, rows, labels, path):
        outer = int(pretruth["outer_fold"])
        self.events.append(("outer", outer))
        comparators = {}
        for name in ("frozen_same_protocol", "fresh_control", "selected_arm"):
            predictions = [{"audio_file": row["audio_file"], "speaker_id": row["speaker_id"]}
                           for row in rows]
            comparators[name] = {
                "metrics": {"macro_f1": .96, "accuracy": .96},
                "predictions": predictions, "known_top1_predictions": predictions,
            }
        result = {"outer_fold": outer, "comparators": comparators}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    def load_historical_predictions(self, contract):
        rows = [{"audio_file": row["audio_file"], "speaker_id": row["speaker_id"]}
                for row in contract["manifest"]]
        return rows, rows

    def aggregate(self, *_args):
        self.events.append(("aggregate",))
        return {"metrics": {"selected_arm": {"macro_f1": .966, "accuracy": .97}},
                "promote_new_incumbent": True,
                "development_metric_goal_reached": True, "goal_reached": False}


def fake_project(tmp_path: Path):
    root = tmp_path / "project"
    (root / "configs/train").mkdir(parents=True)
    (root / "configs/model").mkdir(parents=True)
    (root / "artifacts/infrastructure/C002_preparation").mkdir(parents=True)
    (root / "scripts").mkdir()
    readiness = {
        "manifest": "data/manifest.csv", "folds": "data/folds.csv",
        "roles": "data/roles.csv", "label_map": "data/labels.json",
        "model_config": "configs/model/public.json",
    }
    (root / "configs/train/readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
    for relative in (*readiness.values(), "configs/model/advanced.json",
                     "scripts/run_f005_experiment.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    config = {
        "run_name": "F005-test", "readiness_config": "configs/train/readiness.json",
        "advanced_model_config": "configs/model/advanced.json", "output_root": "runs",
        "fold_ids": [0, 1], "arms": [{"id": arm} for arm in ARMS],
        "mlflow": {"experiment_id": "1"}, "retention": {"local_transfer": "promotion_only"},
    }
    config_path = root / "configs/train/f005.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    binding_path = root / "artifacts/infrastructure/C002_preparation/mlflow_state.json"
    binding_path.write_text("{}", encoding="utf-8")
    manifest = [
        {"audio_file": f"f{i}.wav", "speaker_id": "known", "fold": i % 2,
         "duration_seconds": 3.0, "has_nonzero_signal": True,
         "input_sha256": "0" * 64}
        for i in range(4)
    ]
    folds = [{"audio_file": row["audio_file"], "fold": row["fold"],
              "train_eligible": True, "group_id": f"g{i}"}
             for i, row in enumerate(manifest)]
    roles = [{"audio_file": row["audio_file"], "outer_fold": outer,
              "outer_evaluation_included": row["fold"] == outer}
             for outer in (0, 1) for row in manifest]
    contract = {
        "signature": "e" * 64, "config": config,
        "source_verification": {"status": "verified"},
        "readiness": {"summary": {"audio_hashes_checked": True}},
        "identity": {"config_sha256": "f" * 64},
        "manifest": manifest, "folds": folds, "roles": roles,
        "labels": ["unknown", "known"],
    }
    return root, config_path, binding_path, contract


class F005ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_orchestration_seals_both_folds_before_outer_and_limits_extraction(self):
        root, config, binding, contract = fake_project(self.temp_path)
        backend = FakeBackend()
        result = execute_f005_experiment(
            contract, root, config, binding, backend=backend,
            output_dir=root / "runs" / "run1",
        )
        self.assertEqual(result["status"], "complete")
        events = backend.events
        outer_positions = [i for i, item in enumerate(events) if item[0] == "outer"]
        policy_positions = [i for i, item in enumerate(events) if item[0] == "seal_policy"]
        self.assertEqual(len(policy_positions), 2)
        self.assertLess(max(policy_positions), min(outer_positions))
        self.assertEqual(sum(item[0] == "fit_head" for item in events), 2)
        self.assertEqual(sum(item[0] == "fit_tail" for item in events), 8)
        known = {(item[2], item[3]) for item in events
                 if item[:2] == ("extract", "known_selection")}
        full = {(item[2], item[3]) for item in events
                if item[:2] == ("extract", "full_scoring")}
        self.assertEqual(known, {(outer, arm) for outer in (0, 1) for arm in ARMS})
        self.assertEqual(full, {(outer, arm) for outer in (0, 1)
                                for arm in ("control", "treatment_mse01")})
        state = json.loads((Path(result["output"]) / "experiment_state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(state["children"]), 10)
        self.assertIs(state["mlflow_forbidden_payloads_uploaded"], False)
        combined_metrics = {key: value for batch in backend.logged_metrics
                            for key, value in batch.items()}
        self.assertEqual(combined_metrics["decision/development_metric_goal_reached_0_965"], 1.0)
        summary = backend.markdown_reports[-1]
        self.assertIn("## Pooled OOF metrics", summary)
        self.assertIn("## Whole-content-group bootstrap", summary)
        self.assertIn("## Promotion gates", summary)

    def test_resume_reopens_failed_runs_without_retraining_completed_heads(self):
        root, config, binding, contract = fake_project(self.temp_path)
        backend = FakeBackend(fail_tail_once=(0, "control"))
        output = root / "runs" / "interrupted"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            execute_f005_experiment(
                contract, root, config, binding, backend=backend, output_dir=output,
            )
        failed = json.loads((output / "experiment_state.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["phase"], "training_tails")
        result = execute_f005_experiment(
            contract, root, config, binding, backend=backend, resume_dir=output,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(backend.fit_head_calls, 2)
        retried = [item for item in backend.events
                   if item[:3] == ("fit_tail", 0, "control")]
        self.assertEqual(retried, [("fit_tail", 0, "control", False),
                                   ("fit_tail", 0, "control", True)])
        self.assertTrue(any(item[0] == "reopen" for item in backend.events))

    def test_resume_recovers_parent_finished_before_final_state_write(self):
        root, config, binding, contract = fake_project(self.temp_path)
        backend = FakeBackend()
        output = root / "runs" / "finalizing"
        execute_f005_experiment(
            contract, root, config, binding, backend=backend, output_dir=output,
        )
        state_path = output / "experiment_state.json"
        state = json.loads(state_path.read_text())
        state["status"], state["phase"] = "running", "finalizing_parent"
        state["parent"]["status"] = "RUNNING"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        fit_calls = sum(item[0] in {"fit_head", "fit_tail"} for item in backend.events)
        recovered = execute_f005_experiment(
            contract, root, config, binding, backend=backend, resume_dir=output,
        )
        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(
            sum(item[0] in {"fit_head", "fit_tail"} for item in backend.events),
            fit_calls,
        )

    def test_metadata_upload_guard_rejects_checkpoints_and_embedding_arrays(self):
        tracker = FakeTracker(FakeBackend(), self.temp_path, "run")
        for name in ("last.pt", "embeddings.npz", "sample.wav"):
            path = self.temp_path / name
            path.write_bytes(b"payload")
            with self.assertRaisesRegex(ValueError, "metadata-only"):
                _safe_add_artifact(tracker, path)

    def test_cli_defaults_to_no_execution(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_f005_experiment.py"
        spec = importlib.util.spec_from_file_location("f005_cli_for_test", path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        args = module.parse_args([])
        self.assertIs(args.execute, False)
        self.assertIsNone(args.resume_dir)

    def test_cli_validation_does_not_resolve_or_require_mlflow_binding(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_f005_experiment.py"
        spec = importlib.util.spec_from_file_location("f005_cli_binding_test", path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        calls = []

        def confined(value, prefix, **_):
            calls.append(prefix)
            if prefix == "artifacts/infrastructure":
                raise AssertionError("validation touched the MLflow binding")
            return Path(value)

        contract = {
            "signature": "a" * 64,
            "readiness": {"summary": {"audio_hashes_checked": False}},
            "source_verification": None,
        }
        fake_contract = types.SimpleNamespace(
            load_f005_contract=lambda *_args, **_kwargs: contract,
        )
        fake_runner = types.SimpleNamespace(
            execution_plan=lambda *_args, **_kwargs: {"units": 10},
            probe_receipt=lambda *_args, **_kwargs: {"optimizer_steps": 0},
        )
        with mock.patch.object(module, "_confined", side_effect=confined), \
                mock.patch.dict(sys.modules, {
                    "speaker_id.training.f005_contract": fake_contract,
                    "speaker_id.training.f005_runner": fake_runner,
                }), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(["--config", "fake.json"]), 0)
        self.assertEqual(calls, ["configs/train"])
