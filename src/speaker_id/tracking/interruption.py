"""Close an interrupted MLflow run without replaying its pending artifacts.

This is deliberately separate from :class:`DurableMLflowRun`.  A partially
completed training run can have large local artifacts that were intentionally
not approved for upload.  Calling ``DurableMLflowRun.flush`` merely to mark it
terminal would replay every unsent artifact.  This module records a small,
auditable interruption receipt and terminates only the explicitly named run.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from pathlib import Path
import tempfile

from .mlflow import ExperimentBinding, _check_remote_binding, make_client, utc_now
from .security import Redactor, safe_endpoint
from .snapshot import sha256_file, write_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_TERMINAL = {"FINISHED", "FAILED", "KILLED"}


def close_interrupted_run(
    *,
    tracking_dir: Path,
    expected_run_id: str,
    interruption_cause: str,
    preservation_archive_sha256: str,
    preservation_archive_path: str,
    client=None,
    tracking_uri: str | None = None,
    redactor: Redactor | None = None,
) -> dict:
    """Upload one closure receipt and set an owned, still-running run to KILLED.

    ``tracking_dir`` must be the existing directory which owns ``run_state``.
    No pending metrics or pre-existing artifacts are flushed.  The operation is
    idempotent only after a matching local receipt and remote ``KILLED`` status
    are both visible.
    """
    directory = Path(tracking_dir).resolve(strict=True)
    state_path = directory / "run_state.json"
    if not state_path.is_file():
        raise ValueError("Interrupted run tracking directory has no run_state.json.")
    if not expected_run_id or not re.fullmatch(r"[0-9a-f]{32}", expected_run_id):
        raise ValueError("An explicit 32-character MLflow run ID is required.")
    if not _SHA256.fullmatch(preservation_archive_sha256):
        raise ValueError("Preservation archive SHA-256 must be lowercase hexadecimal.")
    if not interruption_cause.strip():
        raise ValueError("An explicit interruption cause is required.")
    if not preservation_archive_path.strip():
        raise ValueError("An explicit preservation archive path is required.")

    redactor = redactor or Redactor()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if str(state.get("run_id")) != expected_run_id:
        raise ValueError("Tracking state does not belong to the explicitly named run.")
    binding = ExperimentBinding(**state["binding"])
    binding.validate()
    uri = tracking_uri or safe_endpoint(__import__("os").environ.get("MLFLOW_TRACKING_URI", ""))
    if safe_endpoint(uri) != binding.tracking_endpoint:
        raise ValueError("Interrupted run cannot be closed against a different MLflow backend.")
    client = client if client is not None else make_client(uri)
    _check_remote_binding(client, binding)

    closure_dir = directory / "interruption"
    closure_path = closure_dir / "closure.json"
    receipt_path = closure_dir / "closure_receipt.json"
    remote = client.get_run(expected_run_id)
    if str(remote.info.experiment_id) != binding.experiment_id:
        raise ValueError("Explicit run does not belong to the owned MLflow experiment.")

    if remote.info.status == "KILLED":
        if not closure_path.is_file() or not receipt_path.is_file():
            raise RuntimeError("Run is already KILLED without a matching local closure receipt.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("run_id") != expected_run_id or receipt.get("remote_status") != "KILLED":
            raise RuntimeError("Existing closure receipt does not match the requested run.")
        return {"status": "already_closed", "run_id": expected_run_id, "remote_status": "KILLED"}
    if remote.info.status in _TERMINAL:
        raise RuntimeError(f"Run is already terminal with status {remote.info.status!r}; it will not be rewritten.")
    if remote.info.status != "RUNNING":
        raise RuntimeError(f"Run must be RUNNING before it can be safely closed, got {remote.info.status!r}.")

    closure_dir.mkdir(parents=True, exist_ok=True)
    closure = redactor({
        "schema_version": 1,
        "status": "interrupted",
        "training_started": True,
        "run_id": expected_run_id,
        "experiment_id": binding.experiment_id,
        "prior_remote_status": remote.info.status,
        "interruption_cause": interruption_cause,
        "preservation_archive": {
            "path": preservation_archive_path,
            "sha256": preservation_archive_sha256,
        },
        "delivery_policy": (
            "Only this interruption receipt is uploaded. Existing pending metrics and artifacts "
            "are intentionally not replayed by this operation."
        ),
        "closed_at": utc_now(),
    })
    write_json(closure_path, closure)
    closure_sha256 = sha256_file(closure_path)
    client.log_artifact(expected_run_id, str(closure_path), artifact_path="interruption")
    with tempfile.TemporaryDirectory(prefix="speaker-id-interruption-readback-") as temporary:
        downloaded = Path(client.download_artifacts(
            expected_run_id, "interruption/closure.json", dst_path=temporary
        ))
        if not downloaded.is_file() or sha256_file(downloaded) != closure_sha256:
            raise RuntimeError("MLflow interruption receipt roundtrip hash mismatch.")
    client.set_terminated(expected_run_id, status="KILLED")
    observed = client.get_run(expected_run_id)
    if observed.info.status != "KILLED":
        raise RuntimeError("MLflow did not read back the expected KILLED status.")
    receipt = redactor({
        "schema_version": 1,
        "run_id": expected_run_id,
        "binding": asdict(binding),
        "closure_artifact": "interruption/closure.json",
        "closure_sha256": closure_sha256,
        "remote_status": observed.info.status,
        "completed_at": utc_now(),
    })
    write_json(receipt_path, receipt)
    return {"status": "closed", "run_id": expected_run_id, "remote_status": observed.info.status,
            "closure_sha256": closure_sha256}
