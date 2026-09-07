"""MLflow runs with explicit experiment ownership and a durable local log.

The MLflow package is imported only when a real client is requested. Metric
delivery is at least once: an interrupted acknowledgement can replay a metric,
but successful delivery is checkpointed locally and never silently discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .security import Redactor, safe_endpoint
from .snapshot import environment_versions, input_fingerprints, sha256_file, source_snapshot, write_json

PROJECT = "iaaa2026-speaker-id-v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_client(tracking_uri: str | None = None):
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    safe_endpoint(uri)
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError("Install the project's locked MLflow dependency in the server environment.") from error
    return MlflowClient(tracking_uri=uri)


@dataclass(frozen=True)
class ExperimentBinding:
    experiment_id: str
    experiment_name: str
    scope_id: str
    tracking_endpoint: str
    project: str = PROJECT

    def validate(self):
        if not str(self.experiment_id).strip() or str(self.experiment_id) == "0":
            raise ValueError("An explicit non-default experiment ID is required.")
        if not self.experiment_name.strip() or self.experiment_name.lower() == "default":
            raise ValueError("An explicit new experiment name is required.")
        if not self.scope_id or self.project != PROJECT:
            raise ValueError("The experiment must have a project ownership scope.")
        safe_endpoint(self.tracking_endpoint)


def _check_remote_binding(client, binding: ExperimentBinding):
    binding.validate()
    experiment = client.get_experiment(str(binding.experiment_id))
    if experiment is None or experiment.name != binding.experiment_name:
        raise ValueError("Persisted experiment ID/name does not match the tracking server.")
    if getattr(experiment, "lifecycle_stage", "active") != "active":
        raise ValueError("The project experiment is not active.")
    tags = getattr(experiment, "tags", {}) or {}
    if tags.get("speaker_id.project") != PROJECT or tags.get("speaker_id.scope_id") != binding.scope_id:
        raise ValueError("The experiment does not match the persisted project ownership scope.")


def resolve_experiment(*, client, experiment_name: str, state_path: Path,
                       tracking_uri: str) -> ExperimentBinding:
    """Create a new experiment, or reuse only this project's persisted binding.

    An existing name without a matching local state file is deliberately an
    error; this prevents accidentally sending runs into the user's old work.
    """
    if not experiment_name or not experiment_name.strip() or experiment_name.lower() == "default":
        raise ValueError("Provide a fresh explicit experiment name, never Default.")
    endpoint = safe_endpoint(tracking_uri)
    state_path = Path(state_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        binding = ExperimentBinding(**state["binding"])
        if binding.experiment_name != experiment_name or binding.tracking_endpoint != endpoint:
            raise ValueError("Requested experiment/backend differs from the persisted binding.")
        _check_remote_binding(client, binding)
        return binding
    if client.get_experiment_by_name(experiment_name) is not None:
        raise ValueError("Experiment name already exists without this project's persisted state; choose a fresh name.")
    scope = uuid.uuid4().hex
    identifier = client.create_experiment(experiment_name, tags={
        "speaker_id.project": PROJECT, "speaker_id.scope_id": scope,
        "speaker_id.purpose": "CAM++ development and infrastructure; separate from prior experiments",
    })
    binding = ExperimentBinding(str(identifier), experiment_name, scope, endpoint)
    _check_remote_binding(client, binding)
    write_json(state_path, {"schema_version": 1, "created_at": utc_now(), "binding": asdict(binding)})
    return binding


def _flatten_params(config, prefix="") -> dict[str, str]:
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_flatten_params(value, name))
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
            # Full values always remain in resolved_config.json.
            if len(text) > 5500:
                text = text[:5400] + "...[see resolved_config.json]"
            if len(name) > 240:
                name = name[:220] + "_" + hashlib.sha256(name.encode()).hexdigest()[:16]
            result[name] = text
    return result


class DurableMLflowRun:
    """One run whose source, report, metrics, and unsent work survive process exit."""

    def __init__(self, directory: Path, *, client=None, tracking_uri=None, redactor=None, entity_factory=None):
        self.directory = Path(directory).resolve()
        self.artifacts = self.directory / "artifacts"
        self.redactor = redactor or Redactor()
        self.state_path = self.directory / "run_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.binding = ExperimentBinding(**self.state["binding"])
        self.binding.validate()
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or self.binding.tracking_endpoint
        if safe_endpoint(uri) != self.binding.tracking_endpoint:
            raise ValueError("Run resume attempted against a different MLflow backend.")
        self.client = client if client is not None else make_client(uri)
        self.entity_factory = entity_factory

    @classmethod
    def prepare(cls, *, project_root: Path, spool_dir: Path, binding: ExperimentBinding,
                run_name: str, config: dict, input_paths: dict[str, Path] | None = None,
                run_kind="infrastructure_preflight", training_started=False,
                client=None, tracking_uri=None, redactor=None, parent_run_id=None, entity_factory=None):
        binding.validate()
        redactor = redactor or Redactor()
        directory = Path(spool_dir).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Run spool already contains files; use DurableMLflowRun(...) to resume it.")
        artifacts = directory / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        resolved = redactor(config)
        write_json(artifacts / "resolved_config.json", resolved)
        write_json(artifacts / "environment_versions.json", redactor(environment_versions()))
        inputs = redactor(input_fingerprints(input_paths or {}))
        write_json(artifacts / "inputs_manifest.json", inputs)
        source = source_snapshot(Path(project_root), artifacts / "source_snapshot.zip", redactor)
        tags = {
            "mlflow.runName": redactor.text(run_name), "speaker_id.project": PROJECT,
            "speaker_id.scope_id": binding.scope_id, "speaker_id.run_kind": run_kind,
            "speaker_id.training_started": str(bool(training_started)).lower(),
            "speaker_id.local_run_id": uuid.uuid4().hex,
            "speaker_id.source_sha256": source["archive_sha256"],
            "speaker_id.source_git_dirty": str(source["src_dirty"]).lower(),
        }
        if source["git_commit"]:
            tags["mlflow.source.git.commit"] = source["git_commit"]
        if parent_run_id:
            tags["mlflow.parentRunId"] = str(parent_run_id)
        state = {
            "schema_version": 1, "binding": asdict(binding), "run_id": None,
            "created_at": utc_now(), "tags": redactor(tags), "params": _flatten_params(resolved),
            "params_sent": False, "sent_event_sequence": -1, "uploaded_artifacts": {},
            "pending_status": None, "remote_status": None, "last_sync_error": None,
            "delivery_semantics": "at_least_once_with_durable_local_events",
        }
        write_json(directory / "run_state.json", state)
        (directory / "events.jsonl").touch()
        run = cls(directory, client=client, tracking_uri=tracking_uri, redactor=redactor,
                  entity_factory=entity_factory)
        run.write_report({
            "status": "prepared", "run_kind": run_kind, "training_started": bool(training_started),
            "source_git_commit": source["git_commit"], "source_sha256": source["archive_sha256"],
            "input_fingerprints": inputs,
        })
        return run

    @property
    def run_id(self):
        return self.state["run_id"]

    def _save(self):
        write_json(self.state_path, self.redactor(self.state))

    def _entity(self, name: str, **fields):
        if self.entity_factory is not None:
            return self.entity_factory(name, **fields)
        # Import only on real batch delivery; pure local/fake-client use needs no MLflow package.
        from mlflow import entities
        return getattr(entities, name)(**fields)

    def _send_metric_batch(self, records, acknowledged_sequence):
        if records:
            self.client.log_batch(self.run_id, metrics=records, synchronous=True)
        self.state["sent_event_sequence"] = acknowledged_sequence
        self._save()

    def log_metrics(self, metrics: dict[str, float], *, step: int = 0, sync=True, strict=False):
        if not isinstance(step, int) or step < 0:
            raise ValueError("Metric step must be a non-negative integer.")
        cleaned = {}
        for key, value in metrics.items():
            if self.redactor.text(str(key)) != str(key):
                raise ValueError("Metric keys must not contain credential values.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Metric {key!r} must be finite.")
            if self.redactor.secret_key(str(key)):
                raise ValueError("Credential-like metric keys are not allowed.")
            cleaned[str(key)] = number
        event = {"kind": "metrics", "metrics": cleaned, "step": step, "timestamp_ms": int(time.time() * 1000)}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if sync:
            return self.flush(strict=strict)
        return False

    def write_report(self, report: dict, markdown: str | None = None):
        safe = self.redactor(report)
        write_json(self.artifacts / "report.json", safe)
        if markdown is None:
            lines = [f"# {self.state['tags']['mlflow.runName']}", "", "## Run summary", ""]
            for key, value in safe.items():
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                lines.append(f"- **{key}**: {rendered}")
            markdown = "\n".join(lines) + "\n"
        (self.artifacts / "report.md").write_text(self.redactor.text(markdown), encoding="utf-8")

    def add_artifact(self, source: Path, relative_path: str | None = None):
        source = Path(source)
        relative = Path(relative_path or source.name)
        destination = (self.artifacts / relative).resolve()
        if relative.is_absolute() or not destination.is_relative_to(self.artifacts.resolve()):
            raise ValueError("Artifact destination must stay within the run artifact directory.")
        if destination.name.startswith(".env") or destination.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise ValueError("Credential-like artifact files are forbidden.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".yml", ".toml", ".py"}:
            self.redactor.assert_no_secret_bytes(source.read_bytes(), source.name)
        shutil.copyfile(source, destination)

    def flush(self, *, strict=False) -> bool:
        """Replay pending local work; failures stay on disk and are never a success."""
        try:
            _check_remote_binding(self.client, self.binding)
            if self.run_id is None:
                local_id = self.state["tags"]["speaker_id.local_run_id"]
                recovered = self.client.search_runs(
                    [str(self.binding.experiment_id)],
                    filter_string=f"tags.`speaker_id.local_run_id` = '{local_id}'", max_results=2,
                )
                if len(recovered) > 1:
                    raise ValueError("Multiple remote runs claim this local run ID; explicit reconciliation is required.")
                created = recovered[0] if recovered else self.client.create_run(
                    str(self.binding.experiment_id), tags=self.state["tags"])
                self.state["run_id"] = created.info.run_id
                self._save()
            if not self.state["params_sent"]:
                params = [self._entity("Param", key=key, value=value) for key, value in self.state["params"].items()]
                for start in range(0, len(params), 100):
                    self.client.log_batch(self.run_id, params=params[start:start + 100], synchronous=True)
                self.state["params_sent"] = True
                self._save()
            pending_metrics = []
            pending_sequence = self.state["sent_event_sequence"]
            for sequence, line in enumerate((self.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()):
                if sequence <= self.state["sent_event_sequence"]:
                    continue
                event = json.loads(line)
                metrics = [self._entity("Metric", key=key, value=value, timestamp=event["timestamp_ms"], step=event["step"])
                           for key, value in event["metrics"].items()]
                if len(metrics) > 500:
                    if pending_sequence > self.state["sent_event_sequence"]:
                        self._send_metric_batch(pending_metrics, pending_sequence)
                        pending_metrics = []
                    # A single oversized event is acknowledged only after every chunk succeeds.
                    for start in range(0, len(metrics), 500):
                        self.client.log_batch(self.run_id, metrics=metrics[start:start + 500], synchronous=True)
                    self._send_metric_batch([], sequence)
                    pending_sequence = sequence
                    continue
                if len(pending_metrics) + len(metrics) > 500:
                    self._send_metric_batch(pending_metrics, pending_sequence)
                    pending_metrics = []
                pending_metrics.extend(metrics)
                pending_sequence = sequence
            if pending_sequence > self.state["sent_event_sequence"]:
                self._send_metric_batch(pending_metrics, pending_sequence)
            for path in sorted(self.artifacts.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.artifacts).as_posix()
                digest = sha256_file(path)
                if self.state["uploaded_artifacts"].get(relative) == digest:
                    continue
                parent = path.parent.relative_to(self.artifacts).as_posix()
                self.client.log_artifact(self.run_id, str(path), artifact_path=None if parent == "." else parent)
                self.state["uploaded_artifacts"][relative] = digest
                self._save()
            if self.state["pending_status"]:
                self.client.set_terminated(self.run_id, status=self.state["pending_status"])
                self.state["remote_status"] = self.state["pending_status"]
                self.state["pending_status"] = None
            self.state["last_sync_error"] = None
            self.state["last_successful_sync_at"] = utc_now()
            self._save()
            return True
        except Exception as error:
            message = self.redactor.text(f"{type(error).__name__}: {error}")
            self.state["last_sync_error"] = message
            self._save()
            if strict:
                raise RuntimeError(f"MLflow synchronization failed; durable local run preserved: {message}") from None
            return False

    def verify_artifacts(self) -> dict:
        """Download from this run's configured backend and compare every byte hash."""
        self.flush(strict=True)
        verified = []
        with tempfile.TemporaryDirectory(prefix="speaker-id-mlflow-roundtrip-") as temporary:
            for relative, expected in sorted(self.state["uploaded_artifacts"].items()):
                downloaded = Path(self.client.download_artifacts(self.run_id, relative, dst_path=temporary))
                if not downloaded.is_file() or sha256_file(downloaded) != expected:
                    raise RuntimeError(f"MLflow artifact roundtrip hash mismatch: {relative}")
                verified.append(relative)
        return {"status": "passed", "files_verified": len(verified), "artifact_paths": verified,
                "backend": self.binding.tracking_endpoint, "run_id": self.run_id}

    def verify_remote_metadata(self) -> dict:
        """Read the recorded run back, rather than assuming accepted writes are visible."""
        self.flush(strict=True)
        remote = self.client.get_run(self.run_id)
        if str(remote.info.experiment_id) != self.binding.experiment_id:
            raise RuntimeError("Remote run belongs to an unexpected experiment.")
        for key, value in self.state["tags"].items():
            if remote.data.tags.get(key) != value:
                raise RuntimeError(f"Remote run tag does not match: {key}")
        for key, value in self.state["params"].items():
            if remote.data.params.get(key) != value:
                raise RuntimeError(f"Remote parameter does not match: {key}")
        latest = {}
        for line in (self.directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            for key, value in event["metrics"].items():
                rank = (event["step"], event["timestamp_ms"])
                if key not in latest or rank >= latest[key][0]:
                    latest[key] = (rank, value)
        for key, (_, value) in latest.items():
            if key not in remote.data.metrics or not math.isclose(remote.data.metrics[key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"Remote metric does not match: {key}")
        if self.state["remote_status"] and remote.info.status != self.state["remote_status"]:
            raise RuntimeError("Remote run termination status does not match.")
        return {"status": "passed", "parameters_verified": len(self.state["params"]),
                "metrics_verified": len(latest), "tags_verified": len(self.state["tags"]),
                "remote_run_status": remote.info.status}

    def finish(self, status="FINISHED", *, strict=False) -> bool:
        if status not in {"FINISHED", "FAILED", "KILLED"}:
            raise ValueError("Unsupported terminal run status.")
        self.state["pending_status"] = status
        self._save()
        return self.flush(strict=strict)

    def reopen(self):
        """Explicitly resume the same captured config/source after caller checks its checkpoint.

        Changing code or configuration requires a new run with provenance linking
        it to the prior run; this method never replaces the original snapshot.
        """
        _check_remote_binding(self.client, self.binding)
        if self.run_id is None:
            raise ValueError("Only an existing remote run can be reopened.")
        self.client.update_run(self.run_id, status="RUNNING")
        self.state["pending_status"] = None
        self.state["remote_status"] = "RUNNING"
        self.state["resume_count"] = int(self.state.get("resume_count", 0)) + 1
        self.state["last_resumed_at"] = utc_now()
        self._save()
        return self.flush(strict=True)
