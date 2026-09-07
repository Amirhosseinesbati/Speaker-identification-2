import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from speaker_id.tracking import DurableMLflowRun, ExperimentBinding, Redactor, resolve_experiment
from speaker_id.tracking.security import safe_endpoint
from speaker_id.tracking.snapshot import source_snapshot


class FakeClient:
    """An in-memory tracking store plus separate file artifact storage."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.experiments = {}
        self.runs = {}
        self.metric_calls = []
        self.fail_metric_once = False
        self.corrupt_download = False
        self.fail_create_ack_once = False

    def get_experiment_by_name(self, name):
        return next((item for item in self.experiments.values() if item.name == name), None)

    def create_experiment(self, name, tags):
        identifier = str(len(self.experiments) + 1)
        self.experiments[identifier] = SimpleNamespace(
            experiment_id=identifier, name=name, tags=tags, lifecycle_stage="active")
        return identifier

    def get_experiment(self, identifier):
        return self.experiments.get(identifier)

    def create_run(self, experiment_id, tags):
        identifier = f"run-{len(self.runs) + 1}"
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=identifier, experiment_id=experiment_id, status="RUNNING"),
            data=SimpleNamespace(tags=dict(tags), params={}, metrics={}),
        )
        self.runs[identifier] = run
        if self.fail_create_ack_once:
            self.fail_create_ack_once = False
            raise ConnectionError("Acknowledgement lost after server created run")
        return run

    def search_runs(self, experiment_ids, filter_string, max_results):
        identifier = filter_string.split("'")[1]
        return [run for run in self.runs.values()
                if run.info.experiment_id in experiment_ids
                and run.data.tags.get("speaker_id.local_run_id") == identifier][:max_results]

    def get_run(self, run_id):
        return self.runs[run_id]

    def log_param(self, run_id, key, value):
        params = self.runs[run_id].data.params
        if key in params and params[key] != value:
            raise ValueError("Immutable parameter changed")
        params[key] = value

    def log_metric(self, run_id, key, value, timestamp, step):
        if self.fail_metric_once:
            self.fail_metric_once = False
            raise ConnectionError("token=do-not-log-this test failure")
        self.metric_calls.append((run_id, key, value, timestamp, step))
        self.runs[run_id].data.metrics[key] = value

    def log_artifact(self, run_id, local_path, artifact_path=None):
        source = Path(local_path)
        destination = self.directory / run_id / (artifact_path or "") / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def download_artifacts(self, run_id, path, dst_path):
        source = self.directory / run_id / path
        destination = Path(dst_path) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if self.corrupt_download:
            destination.write_bytes(b"corrupt")
        return str(destination)

    def set_terminated(self, run_id, status):
        self.runs[run_id].info.status = status

    def update_run(self, run_id, status):
        self.runs[run_id].info.status = status


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src/speaker_id").mkdir(parents=True)
        (self.root / "src/speaker_id/model.py").write_bytes(b"answer = 42\n")
        self.client = FakeClient(self.root / "remote")
        self.endpoint = "https://tracking.example.test/owner/project.mlflow"
        self.binding = resolve_experiment(
            client=self.client, experiment_name="new-campp", state_path=self.root / "binding.json",
            tracking_uri=self.endpoint,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, name="run", **kwargs):
        return DurableMLflowRun.prepare(
            project_root=self.root, spool_dir=self.root / name, binding=self.binding,
            run_name="CAM++ infrastructure", config={"model": "CAM++", "fold": 0, "seed": 1729},
            client=self.client, tracking_uri=self.endpoint, redactor=Redactor(environ={}), **kwargs,
        )

    def test_recursively_redacts_keys_values_and_url_credentials(self):
        redact = Redactor(environ={"DAGSHUB_USER_TOKEN": "secret-value-123", "VISIBLE": "kept"})
        result = redact({"nested": [{"password": "different", "note": "prefix secret-value-123 suffix"}],
                         "endpoint": "https://user:pw@example.test/path?token=private&mode=ok",
                         "label-secret-value-123": "safe",
                         "normal": "kept"})
        encoded = json.dumps(result)
        for forbidden in ("secret-value-123", "different", "user:pw", "private"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(result["normal"], "kept")
        self.assertIn("[REDACTED]", encoded)
        self.assertEqual(safe_endpoint("https://u:p@example.test/path/?token=x#frag"), "https://example.test/path")

    def test_snapshot_is_deterministic_exact_and_excludes_python_cache(self):
        cache = self.root / "src/speaker_id/__pycache__"
        cache.mkdir()
        (cache / "model.cpython-312.pyc").write_bytes(b"cache")
        first = source_snapshot(self.root, self.root / "one/source.tar.gz", Redactor(environ={}))
        second = source_snapshot(self.root, self.root / "two/source.tar.gz", Redactor(environ={}))
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])
        self.assertEqual(first["file_count"], 1)
        with tarfile.open(self.root / "one/source.tar.gz") as archive:
            self.assertEqual(archive.getnames(), ["src/speaker_id/model.py"])
            content = archive.extractfile("src/speaker_id/model.py").read()
        self.assertEqual(content, b"answer = 42\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(), first["files"][0]["sha256"])

    def test_snapshot_refuses_credentials_instead_of_modifying_source(self):
        model = self.root / "src/speaker_id/model.py"
        original = b'credential = "actual-private-token-123"\n'
        model.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "credential value"):
            source_snapshot(self.root, self.root / "snapshot.tar.gz", Redactor(environ={"TOKEN": "actual-private-token-123"}))
        self.assertEqual(model.read_bytes(), original)

    def test_existing_experiment_cannot_be_adopted_without_its_binding(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            resolve_experiment(client=self.client, experiment_name="new-campp",
                               state_path=self.root / "another.json", tracking_uri=self.endpoint)
        again = resolve_experiment(client=self.client, experiment_name="new-campp",
                                   state_path=self.root / "binding.json", tracking_uri=self.endpoint)
        self.assertEqual(self.binding, again)
        with self.assertRaisesRegex(ValueError, "differs"):
            resolve_experiment(client=self.client, experiment_name="new-campp", state_path=self.root / "binding.json",
                               tracking_uri="https://another.example.test")

    def test_default_or_forged_experiment_binding_is_rejected(self):
        for identifier, name in (("0", "Default"), ("", "new"), ("1", "Default")):
            with self.assertRaises(ValueError):
                ExperimentBinding(identifier, name, "scope", self.endpoint).validate()
        self.client.experiments[self.binding.experiment_id].tags["speaker_id.scope_id"] = "someone-else"
        with self.assertRaisesRegex(ValueError, "ownership scope"):
            resolve_experiment(client=self.client, experiment_name="new-campp",
                               state_path=self.root / "binding.json", tracking_uri=self.endpoint)

    def test_failed_metric_delivery_stays_durable_and_resumes_same_run(self):
        run = self.prepare()
        self.client.fail_metric_once = True
        self.assertFalse(run.log_metrics({"inner.macro_f1": .5}, step=3))
        self.assertIn('"inner.macro_f1": 0.5', (run.directory / "events.jsonl").read_text())
        self.assertNotIn("do-not-log-this", run.state_path.read_text())
        identifier = run.run_id
        resumed = DurableMLflowRun(run.directory, client=self.client, tracking_uri=self.endpoint, redactor=Redactor(environ={}))
        self.assertTrue(resumed.flush(strict=True))
        self.assertEqual(identifier, resumed.run_id)
        self.assertEqual(len(self.client.metric_calls), 1)
        resumed.flush(strict=True)
        self.assertEqual(len(self.client.metric_calls), 1)
        self.assertEqual(resumed.verify_remote_metadata()["metrics_verified"], 1)

    def test_uncertain_create_acknowledgement_recovers_run_by_local_id(self):
        run = self.prepare()
        self.client.fail_create_ack_once = True
        self.assertFalse(run.flush())
        self.assertIsNone(run.run_id)
        self.assertEqual(len(self.client.runs), 1)
        self.assertTrue(run.flush(strict=True))
        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(len(self.client.runs), 1)

    def test_full_artifact_roundtrip_and_corruption_detection(self):
        run = self.prepare()
        run.log_metrics({"preflight.training_started": 0}, sync=False)
        run.write_report({"status": "passed", "training_started": False})
        result = run.verify_artifacts()
        self.assertEqual(result["status"], "passed")
        self.assertIn("source_snapshot.tar.gz", result["artifact_paths"])
        self.assertIn("resolved_config.json", result["artifact_paths"])
        self.assertIn("report.md", result["artifact_paths"])
        run.finish("FINISHED", strict=True)
        self.assertEqual(run.verify_remote_metadata()["remote_run_status"], "FINISHED")
        self.client.corrupt_download = True
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            run.verify_artifacts()

    def test_remote_parameter_tampering_is_detected(self):
        run = self.prepare()
        run.flush(strict=True)
        self.client.runs[run.run_id].data.params["model"] = "unexpected-model"
        with self.assertRaisesRegex(RuntimeError, "parameter does not match"):
            run.verify_remote_metadata()

    def test_explicit_reopen_clears_terminal_intent_and_preserves_run(self):
        run = self.prepare()
        run.finish("FAILED", strict=True)
        identifier = run.run_id
        run.reopen()
        run.log_metrics({"loss": 1.5}, step=4, strict=True)
        self.assertEqual(run.run_id, identifier)
        self.assertIsNone(run.state["pending_status"])
        self.assertEqual(run.verify_remote_metadata()["remote_run_status"], "RUNNING")

    def test_nonfinite_metrics_and_artifact_path_escape_are_rejected(self):
        run = self.prepare()
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                run.log_metrics({"score": value})
        with self.assertRaises(ValueError):
            run.add_artifact(self.root / "src/speaker_id/model.py", "../../escape.py")
        with self.assertRaisesRegex(ValueError, "resume"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
