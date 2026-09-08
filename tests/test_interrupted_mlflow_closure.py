from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from speaker_id.tracking.interruption import close_interrupted_run
from speaker_id.tracking.mlflow import ExperimentBinding, PROJECT
from speaker_id.tracking.security import Redactor
from speaker_id.tracking.snapshot import sha256_file, write_json


class FakeClient:
    def __init__(self, root: Path, binding: ExperimentBinding):
        self.root = root
        self.experiment = SimpleNamespace(
            experiment_id=binding.experiment_id,
            name=binding.experiment_name,
            tags={"speaker_id.project": PROJECT, "speaker_id.scope_id": binding.scope_id},
            lifecycle_stage="active",
        )
        self.run = SimpleNamespace(
            info=SimpleNamespace(run_id="a" * 32, experiment_id=binding.experiment_id, status="RUNNING"),
            data=SimpleNamespace(),
        )
        self.logged_artifacts = []

    def get_experiment(self, identifier):
        return self.experiment if identifier == self.experiment.experiment_id else None

    def get_run(self, run_id):
        if run_id != self.run.info.run_id:
            raise KeyError(run_id)
        return self.run

    def log_artifact(self, run_id, local_path, artifact_path=None):
        destination = self.root / run_id / (artifact_path or "") / Path(local_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        self.logged_artifacts.append(destination)

    def download_artifacts(self, run_id, path, dst_path):
        source = self.root / run_id / path
        destination = Path(dst_path) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return str(destination)

    def set_terminated(self, run_id, status):
        self.get_run(run_id).info.status = status


class InterruptedMLflowClosureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binding = ExperimentBinding(
            experiment_id="1",
            experiment_name="speaker-id-owned",
            scope_id="scope-c002",
            tracking_endpoint="https://tracking.example.test/owner/project.mlflow",
        )
        self.tracking = self.root / "artifacts/training/cpu_gain/C001/tracking"
        self.tracking.mkdir(parents=True)
        self.client = FakeClient(self.root / "remote", self.binding)
        write_json(self.tracking / "run_state.json", {
            "binding": asdict(self.binding), "run_id": self.client.run.info.run_id,
        })

    def tearDown(self):
        self.temporary.cleanup()

    def close(self):
        return close_interrupted_run(
            tracking_dir=self.tracking,
            expected_run_id="a" * 32,
            interruption_cause="provider_instance_replacement",
            preservation_archive_sha256="b" * 64,
            preservation_archive_path="artifacts/infrastructure/migration/c001_snapshot.tar.gz",
            client=self.client,
            tracking_uri=self.binding.tracking_endpoint,
            redactor=Redactor(environ={}),
        )

    def test_closure_uploads_only_receipt_and_reads_back_killed_status(self):
        result = self.close()
        self.assertEqual(result["status"], "closed")
        self.assertEqual(self.client.run.info.status, "KILLED")
        self.assertEqual([path.name for path in self.client.logged_artifacts], ["closure.json"])
        closure = self.tracking / "interruption/closure.json"
        receipt = self.tracking / "interruption/closure_receipt.json"
        self.assertTrue(closure.is_file())
        self.assertTrue(receipt.is_file())
        self.assertEqual(result["closure_sha256"], sha256_file(closure))
        self.assertEqual(self.close()["status"], "already_closed")
        self.assertEqual(len(self.client.logged_artifacts), 1)

    def test_refuses_terminal_run_and_wrong_identity(self):
        self.client.run.info.status = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "already terminal"):
            self.close()
        self.client.run.info.status = "RUNNING"
        with self.assertRaisesRegex(ValueError, "explicitly named run"):
            close_interrupted_run(
                tracking_dir=self.tracking,
                expected_run_id="c" * 32,
                interruption_cause="provider_instance_replacement",
                preservation_archive_sha256="b" * 64,
                preservation_archive_path="archive.tar.gz",
                client=self.client,
                tracking_uri=self.binding.tracking_endpoint,
                redactor=Redactor(environ={}),
            )
