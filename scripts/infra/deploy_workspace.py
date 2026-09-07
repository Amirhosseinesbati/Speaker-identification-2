"""Deploy the public Git workspace to the explicitly selected running Vast instance.

This local control script only transfers assets/configuration. All server source
code must arrive through git, and this script has no training command.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "deploy", "bootstrap", "upload-assets", "verify", "download-evidence"])
    parser.add_argument("--identity-file", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/infra/deployment.json").read_text())
    subprocess.run([sys.executable, str(ROOT / "scripts/infra/vast_control.py"), "show"], check=True)
    instance = json.loads((ROOT / "artifacts/infrastructure/vast_instance.json").read_text())
    if instance["id"] != config["instance_id"] or instance["actual_status"] != "running":
        raise SystemExit("Selected instance is not confirmed running. Refresh it using vast_control.py show.")
    if not args.identity_file.is_file():
        raise SystemExit("SSH identity file does not exist or is not accessible.")
    destination = "root@" + instance["ssh_host"]
    options = ["-i", str(args.identity_file.resolve()), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
               "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new"]
    ssh = ["ssh", *options, "-p", str(instance["ssh_port"]), destination]
    scp = ["scp", *options, "-P", str(instance["ssh_port"])]
    workspace = config["remote_workspace"]
    q = shlex.quote
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, env=environment, text=True).strip()
    def remote(command, capture=False):
        return subprocess.run([*ssh, command], check=True, text=True, capture_output=capture)
    def upload(source, target):
        # OpenSSH scp defaults to SFTP; target paths are absolute and contain no spaces.
        subprocess.run([*scp, str(Path(source).resolve()), destination + ":" + target], check=True)
    def check_checkout():
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, env=environment, text=True)
        if dirty.strip():
            raise SystemExit("Local code must be committed before provisioning the server.")
        remote(f"set -eu; cd {q(workspace)}; test \"$(git rev-parse HEAD)\" = {q(revision)}; "
               f"test \"$(git remote get-url origin)\" = {q(config['repository'])}; "
               f"test \"$(git branch --show-current)\" = {q(config['branch'])}; "
               "test -z \"$(git status --porcelain --untracked-files=normal)\"")
        marker = json.loads((ROOT / "artifacts/infrastructure/instance.json").read_text())
        if marker.get("instance_id") != config["instance_id"] or marker.get("git_commit") != revision or marker.get("remote_workspace") != workspace:
            raise SystemExit("Run deploy for this exact instance, workspace and commit before provisioning.")
        if remote("hostname", capture=True).stdout.strip() != marker.get("hostname"):
            raise SystemExit("Remote hostname changed; deploy and attest the instance again.")
    if args.action not in {"inspect", "deploy"}:
        check_checkout()
    if args.action == "inspect":
        remote("hostname; python3 --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace")
    elif args.action == "deploy":
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, env=environment, text=True)
        if dirty.strip():
            raise SystemExit("Commit and push all code before deploying.")
        command = (
            f"set -eu; if [ ! -e {q(workspace)} ]; then git clone --branch {q(config['branch'])} -- {q(config['repository'])} {q(workspace)}; fi; "
            f"cd {q(workspace)}; test -d .git; test -z \"$(git status --porcelain --untracked-files=normal)\"; "
            f"test \"$(git remote get-url origin)\" = {q(config['repository'])}; "
            f"test \"$(git branch --show-current)\" = {q(config['branch'])}; "
            f"git fetch origin {q(config['branch'])}; git merge --ff-only origin/{q(config['branch'])}; "
            f"test \"$(git rev-parse HEAD)\" = {q(revision)}; git rev-parse HEAD"
        )
        remote(command)
        hostname = remote("hostname", capture=True).stdout.strip()
        from datetime import datetime, timezone
        marker = {"instance_id": config["instance_id"], "verified_via": "vast_api_and_ssh", "hostname": hostname,
                  "checked_at_utc": datetime.now(timezone.utc).isoformat(), "remote_workspace": workspace,
                  "status": "verified", "git_commit": revision}
        marker_path = ROOT / "artifacts/infrastructure/instance.json"
        marker_path.write_text(json.dumps(marker, indent=2))
        remote(f"mkdir -p {q(workspace + '/artifacts/infrastructure')}")
        upload(marker_path, workspace + "/artifacts/infrastructure/instance.json")
    elif args.action == "bootstrap":
        remote(f"cd {q(workspace)} && bash scripts/infra/bootstrap_server.sh")
    elif args.action == "upload-assets":
        assets = [
            (ROOT / "artifacts/infrastructure/data_metadata.zip", "data/incoming/data_metadata.zip"),
            (ROOT / "artifacts/models/campp/campplus_voxceleb.bin", "artifacts/models/campp/campplus_voxceleb.bin"),
            (ROOT / "artifacts/infrastructure/mlflow_state.json", "artifacts/infrastructure/mlflow_state.json"),
            (ROOT / "data/raw.zip", "data/incoming/raw.zip"),
        ]
        for source, _ in assets:
            if not source.is_file():
                raise SystemExit(f"Required prepared asset is missing: {source.relative_to(ROOT)}")
        from with_project_env import mlflow_environment
        values = mlflow_environment()
        remote(f"mkdir -p {q(workspace + '/data/incoming')} {q(workspace + '/artifacts/models/campp')}")
        # Only MLflow credentials reach the training server. Vast and Git tokens stay local.
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/infrastructure") as directory:
            credentials = Path(directory) / ".env"
            credentials.write_text("\n".join(k + "=" + json.dumps(values[k]) for k in
                                   ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]) + "\n")
            credential_target = q(workspace + "/.env")
            remote(f"set -eu; test ! -L {credential_target}; if [ -e {credential_target} ]; then test -f {credential_target}; fi; umask 077; touch {credential_target}; chmod 600 {credential_target}")
            upload(credentials, workspace + "/.env")
            remote(f"chmod 600 {credential_target}")
        for source, target in assets:
            upload(source, workspace + "/" + target)
        print("Assets transferred. Run metadata/data verification from committed scripts before deleting any archive.")
    elif args.action == "verify":
        import csv
        from datetime import datetime, timezone
        metadata_identity = json.loads((ROOT / "artifacts/infrastructure/data_metadata.identity.json").read_text())
        with (ROOT / config["metadata_files"][0]).open(encoding="utf-8-sig", newline="") as handle:
            audio = next(row["audio_file"] for row in csv.DictReader(handle) if row["usable_for_training"].lower() == "true")
        spool = "artifacts/infrastructure/mlflow_probe/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        commands = [
            [".venv/bin/python", "scripts/infra/install_metadata.py", "--archive", "data/incoming/data_metadata.zip", "--expected-archive-sha256", metadata_identity["archive_sha256"], "--delete-archive-after-verification"],
            [".venv/bin/python", "scripts/infra/verify_extract_data.py", "--archive", config["archive_path"], "--expected-archive-sha256", config["archive_sha256"], "--delete-archive-after-verification"],
            [".venv/bin/python", "scripts/infra/preflight_runtime.py", "--require-cuda", "--expected-gpu", "RTX3090", "--min-free-disk-gb", "10"],
            [".venv/bin/python", "scripts/checks/probe_campp.py", "--audio", "data/raw/" + audio, "--device", "cuda", "--config", config["training_config"]],
            [".venv/bin/python", "scripts/infra/with_project_env.py", ".venv/bin/python", "scripts/checks/probe_mlflow.py", "--experiment-name", config["mlflow_experiment_name"], "--spool-dir", spool, "--config", config["training_config"],
             "--fingerprint", "model_config=configs/model/campp.json", "--fingerprint", "manifest=data/processed/eda_v1/audio_manifest.csv", "--fingerprint", "folds=data/processed/eda_v1/folds.csv", "--fingerprint", "roles=data/processed/eda_v1/calibration_roles.csv", "--fingerprint", "label_map=data/processed/eda_v1/label_map.json", "--fingerprint", "weights=artifacts/models/campp/campplus_voxceleb.bin"],
            [".venv/bin/python", "scripts/infra/check_readiness.py", "--config", config["training_config"], "--mlflow-report", spool + "/preflight_result.json"],
        ]
        for index, command in enumerate(commands):
            invocation = shlex.join(command)
            if index < 2:
                # Successful extraction already removed its incoming archive.
                # Preserve its evidence and let aggregate checks validate it.
                archive = "data/incoming/data_metadata.zip" if index == 0 else config["archive_path"]
                evidence = "metadata_readiness.json" if index == 0 else "data_readiness.json"
                invocation = f"if [ -f {q(archive)} ]; then {invocation}; else test -f artifacts/infrastructure/{evidence}; fi"
            remote(f"set -eu; cd {q(workspace)}; export VAST_INSTANCE_ID={config['instance_id']}; " + invocation)
        (ROOT / "artifacts/infrastructure/remote_probe_path.json").write_text(json.dumps({"spool": spool}))
        print("All server readiness checks passed. No training has started.")
    else:
        output = ROOT / "artifacts/infrastructure/server_evidence"
        output.mkdir(parents=True, exist_ok=True)
        for name in ["instance.json", "data_readiness.json", "runtime_readiness.json", "campp_probe.json", "readiness.json"]:
            subprocess.run([*scp, destination + ":" + workspace + "/artifacts/infrastructure/" + name, str(output / name)], check=True)
        print("Server evidence downloaded; training was not started.")


if __name__ == "__main__":
    main()
