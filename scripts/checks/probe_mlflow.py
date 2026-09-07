"""Create a separate infrastructure run and verify MLflow artifact roundtrips.

This command performs no model fitting and does not start training. Credentials
come only from standard MLflow environment variables and are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.tracking.mlflow import DurableMLflowRun, make_client, resolve_experiment, utc_now
from speaker_id.tracking.security import Redactor, safe_endpoint
from speaker_id.tracking.snapshot import write_json
from speaker_id.infrastructure.data import confined_path, sha256_file


def load_evidence_reports(arguments, root: Path = ROOT) -> dict:
    """Validate explicitly selected infrastructure JSON before contacting MLflow."""
    root = root.resolve(strict=True)
    reports, sources = {}, set()
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("Evidence reports require NAME=PATH.")
        name, raw_path = argument.split("=", 1)
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in reports:
            raise ValueError("Evidence names must be unique and use only lowercase letters, digits, and underscores.")
        path = confined_path(root, raw_path)
        if (not path.is_relative_to(root / "artifacts/infrastructure")
                or path.suffix.lower() != ".json" or not path.is_file()):
            raise ValueError("Evidence must be a regular JSON file under artifacts/infrastructure.")
        if path in sources:
            raise ValueError("The same evidence file cannot be supplied more than once.")
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("training_started") is not False:
            raise ValueError("Evidence JSON must be an object explicitly recording training_started=false.")
        if not isinstance(payload.get("status"), str):
            raise ValueError("Evidence JSON must record a string status.")
        reports[name] = {"path": path, "payload": payload, "sha256": hashlib.sha256(raw).hexdigest()}
        sources.add(path)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True, help="A new explicit name; existing unowned names are rejected.")
    parser.add_argument("--state-path", type=Path, default=ROOT / "artifacts/infrastructure/mlflow_state.json")
    parser.add_argument("--spool-dir", type=Path, required=True, help="An empty run directory under ignored artifacts/.")
    parser.add_argument("--config", type=Path, help="Resolved JSON or TOML training/preflight configuration.")
    parser.add_argument("--fingerprint", action="append", default=[], metavar="[NAME=]PATH")
    parser.add_argument("--evidence-report", action="append", default=[], metavar="NAME=PATH",
                        help="Upload a measured infrastructure JSON report as checks/NAME.json.")
    parser.add_argument("--run-name", default="infrastructure-preflight-campp")
    args = parser.parse_args(argv)
    redactor = Redactor()
    run = None
    started = time.monotonic()
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        safe_endpoint(uri)
        config = {}
        if args.config:
            config = tomllib.loads(args.config.read_text(encoding="utf-8")) if args.config.suffix == ".toml" else json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Configuration root must be an object.")
        config = {**config, "infrastructure_preflight": {"training_started": False, "base_model": "CAM++"}}
        paths = {}
        if args.config:
            paths["resolved_config_input"] = args.config
        for argument in args.fingerprint:
            if "=" in argument:
                name, raw_path = argument.split("=", 1)
            else:
                raw_path, name = argument, Path(argument).name
            if not name or name in paths:
                raise ValueError("Fingerprint names must be non-empty and unique.")
            paths[name] = Path(raw_path)
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"Fingerprint input does not exist: {path.name}")
        evidence_reports = load_evidence_reports(args.evidence_report)
        for name, evidence in evidence_reports.items():
            fingerprint_name = f"evidence_{name}"
            if fingerprint_name in paths:
                raise ValueError("Evidence fingerprint name conflicts with an explicit fingerprint.")
            paths[fingerprint_name] = evidence["path"]
        client = make_client(uri)
        binding = resolve_experiment(client=client, experiment_name=args.experiment_name,
                                     state_path=args.state_path, tracking_uri=uri)
        run = DurableMLflowRun.prepare(
            project_root=ROOT, spool_dir=args.spool_dir, binding=binding, run_name=args.run_name,
            config=config, input_paths=paths, client=client, tracking_uri=uri,
            redactor=redactor, run_kind="infrastructure_preflight", training_started=False,
        )
        infrastructure_checks = {}
        metrics = {"preflight.training_started": 0., "preflight.input_files_fingerprinted": float(len(paths))}
        for name, evidence in evidence_reports.items():
            artifact_path = f"checks/{name}.json"
            run.add_artifact(evidence["path"], artifact_path)
            if sha256_file(run.artifacts / artifact_path) != evidence["sha256"]:
                raise ValueError("Infrastructure evidence changed during probe preparation.")
            status = evidence["payload"]["status"]
            passed = status in {"passed", "passed_forward_only"}
            infrastructure_checks[name] = {"status": status, "passed": passed,
                                            "artifact_path": artifact_path, "sha256": evidence["sha256"]}
            metrics[f"checks.{name}.passed"] = float(passed)
        run.log_metrics(metrics, sync=False)
        first_roundtrip = run.verify_artifacts()
        metadata_readback = run.verify_remote_metadata()
        elapsed = time.monotonic() - started
        report = {
            "status": "passed", "run_kind": "infrastructure_preflight", "training_started": False,
            "base_model": "CAM++", "checked_at": utc_now(), "elapsed_seconds": elapsed,
            "experiment_id": binding.experiment_id, "experiment_name": binding.experiment_name,
            "run_id": run.run_id, "tracking_backend": binding.tracking_endpoint,
            "infrastructure_checks": infrastructure_checks,
            "checks": {"explicit_new_project_experiment": True, "run_creation": True,
                       "parameter_logging": True, "metric_logging": True,
                       "run_metadata_readback": metadata_readback,
                       "source_snapshot_upload": True, "artifact_upload_and_download_hashes": first_roundtrip},
            "scope": "Infrastructure only. No fitting, training, optimizer steps, or competition score is produced.",
            "remaining_condition": "Training requires the user's explicit start instruction.",
        }
        run.write_report(report)
        run.log_metrics({"preflight.passed": 1., "preflight.elapsed_seconds": elapsed,
                         "preflight.artifacts_verified": float(first_roundtrip["files_verified"])}, sync=False)
        # The final detailed report itself must also be uploaded and downloaded.
        final_roundtrip = run.verify_artifacts()
        run.finish("FINISHED", strict=True)
        final_metadata = run.verify_remote_metadata()
        result = {**report, "final_artifact_roundtrip": final_roundtrip,
                  "final_metadata_readback": final_metadata,
                  "local_run_directory": str(run.directory)}
        write_json(run.directory / "preflight_result.json", redactor(result))
        print(json.dumps(redactor(result), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        message = redactor.text(f"{type(error).__name__}: {error}")
        if run is not None:
            run.write_report({"status": "failed", "run_kind": "infrastructure_preflight",
                              "training_started": False, "error": message, "checked_at": utc_now()})
            run.finish("FAILED", strict=False)
        print(json.dumps({"status": "failed", "training_started": False, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
