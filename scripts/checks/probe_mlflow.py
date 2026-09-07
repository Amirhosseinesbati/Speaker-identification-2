"""Create a separate infrastructure run and verify MLflow artifact roundtrips.

This command performs no model fitting and does not start training. Credentials
come only from standard MLflow environment variables and are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.tracking.mlflow import DurableMLflowRun, make_client, resolve_experiment, utc_now
from speaker_id.tracking.security import Redactor, safe_endpoint
from speaker_id.tracking.snapshot import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True, help="A new explicit name; existing unowned names are rejected.")
    parser.add_argument("--state-path", type=Path, default=ROOT / "artifacts/infrastructure/mlflow_state.json")
    parser.add_argument("--spool-dir", type=Path, required=True, help="An empty run directory under ignored artifacts/.")
    parser.add_argument("--config", type=Path, help="Resolved JSON or TOML training/preflight configuration.")
    parser.add_argument("--fingerprint", action="append", default=[], metavar="[NAME=]PATH")
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
        client = make_client(uri)
        binding = resolve_experiment(client=client, experiment_name=args.experiment_name,
                                     state_path=args.state_path, tracking_uri=uri)
        run = DurableMLflowRun.prepare(
            project_root=ROOT, spool_dir=args.spool_dir, binding=binding, run_name=args.run_name,
            config=config, input_paths=paths, client=client, tracking_uri=uri,
            redactor=redactor, run_kind="infrastructure_preflight", training_started=False,
        )
        run.log_metrics({"preflight.training_started": 0., "preflight.input_files_fingerprinted": float(len(paths))}, sync=False)
        first_roundtrip = run.verify_artifacts()
        metadata_readback = run.verify_remote_metadata()
        elapsed = time.monotonic() - started
        report = {
            "status": "passed", "run_kind": "infrastructure_preflight", "training_started": False,
            "base_model": "CAM++", "checked_at": utc_now(), "elapsed_seconds": elapsed,
            "experiment_id": binding.experiment_id, "experiment_name": binding.experiment_name,
            "run_id": run.run_id, "tracking_backend": binding.tracking_endpoint,
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
