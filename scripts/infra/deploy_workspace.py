"""Deploy the public Git workspace to the explicitly selected running Vast instance.

This local control script only transfers assets/configuration. All server source
code must arrive through git, and this script has no training command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT_CONFIG = Path("configs/infra/deployment.json")


def resolve_deployment_config(root: Path, requested: Path | None) -> tuple[dict, str, Path]:
    """Load a committed deployment identity without mutating its legacy state."""
    root = root.resolve(strict=True)
    candidate = DEFAULT_DEPLOYMENT_CONFIG if requested is None else Path(requested)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise ValueError("Deployment config cannot contain parent traversal components")
    absolute = candidate if candidate.is_absolute() else root / candidate
    absolute = Path(os.path.abspath(absolute))
    allowed = root / "configs/infra"
    if not absolute.is_relative_to(allowed):
        raise ValueError("Deployment config must be inside this workspace's configs/infra directory")
    for part in (absolute, *absolute.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Deployment config paths cannot contain symlinks or junctions")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ValueError("Deployment config must be an existing JSON file inside configs/infra")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or type(config.get("instance_id")) is not int:
        raise ValueError("Deployment config must be a JSON object with an integer instance_id")
    raw_evidence = config.get("evidence_root", "artifacts/infrastructure")
    if (not isinstance(raw_evidence, str) or not raw_evidence or Path(raw_evidence).is_absolute()
            or ".." in Path(raw_evidence).parts):
        raise ValueError("Deployment evidence_root must be a relative path without parent traversal")
    evidence = Path(os.path.abspath(root / raw_evidence))
    allowed_evidence = root / "artifacts/infrastructure"
    if not evidence.is_relative_to(allowed_evidence):
        raise ValueError("Deployment evidence_root must be under artifacts/infrastructure")
    for part in (evidence, *evidence.parents):
        if part == root:
            break
        if part.exists() and (part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction())):
            raise ValueError("Deployment evidence_root cannot contain symlinks or junctions")
    return config, resolved.relative_to(root).as_posix(), evidence


def data_verification_mode(config: dict) -> str:
    """Return the only supported C002 installed-data verification mode."""
    policy = config.get("data_verification")
    if policy is None:
        return "archive_extract"
    if not isinstance(policy, dict) or set(policy) != {"mode"}:
        raise ValueError("data_verification must contain only a mode")
    mode = policy["mode"]
    if mode != "installed_manifest_sha256":
        raise ValueError("Unsupported data_verification mode")
    return mode


def resolve_training_config(root: Path, action: str, requested: Path | None,
                            default_path: str) -> str:
    """Validate a verify-only override without rewriting deployment identity."""
    if requested is None:
        return default_path
    if action != "verify":
        raise ValueError("--training-config is supported only by the verify action")
    root = root.resolve(strict=True)
    requested = Path(requested)
    if ".." in requested.parts:
        raise ValueError("Training config cannot contain parent traversal components")
    candidate = requested if requested.is_absolute() else root / requested
    absolute = Path(os.path.abspath(candidate))
    allowed = root / "configs/train"
    if not absolute.is_relative_to(allowed):
        raise ValueError("Training config must be inside this workspace's configs/train directory")
    for part in (absolute, *absolute.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Training config paths cannot contain symlinks or junctions")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ValueError("Training config must be an existing JSON file inside configs/train")
    if not isinstance(json.loads(resolved.read_text(encoding="utf-8")), dict):
        raise ValueError("Training config JSON must contain an object")
    return resolved.relative_to(root).as_posix()


def prefix_sha256(source: Path, length: int) -> str:
    """Hash exactly the bytes already present on the remote upload target."""
    if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= source.stat().st_size:
        raise ValueError("Remote upload length must lie within the local source file")
    digest = hashlib.sha256()
    remaining = length
    with source.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Local archive became shorter while checking its prefix")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def validate_raw_resume(source: Path, expected_bytes: int, expected_sha256: str, snapshot: dict) -> int:
    """Resume only a matching prefix of the exact deployment archive."""
    before = source.stat()
    if not source.is_file() or source.is_symlink() or before.st_size != expected_bytes:
        raise ValueError("Local raw.zip is not the regular archive of the deployment's expected size")
    count = snapshot["size"]
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= expected_bytes:
        raise ValueError("Remote raw.zip size is invalid or exceeds the local source")
    if snapshot["sha256"] != prefix_sha256(source, count):
        raise ValueError("Remote raw.zip does not match the local prefix; refusing corrupt resume")
    if count == expected_bytes and snapshot["sha256"] != expected_sha256:
        raise ValueError("Complete remote raw.zip does not match the deployment SHA256")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Local archive changed while validating resume")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "deploy", "bootstrap", "upload-assets", "resume-raw", "verify", "download-evidence"])
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--deployment-config", type=Path,
                        help="Committed configs/infra/*.json deployment identity; defaults to the historical config")
    parser.add_argument("--archive-source", choices=["local_upload", "official_original"], default="local_upload",
                        help="Select the explicitly pinned ZIP container for full data verification")
    parser.add_argument("--training-config", type=Path,
                        help="Verify only: existing configs/train/*.json training contract; deployment metadata stays unchanged")
    args = parser.parse_args()
    try:
        config, deployment_config, evidence_root = resolve_deployment_config(ROOT, args.deployment_config)
        verification_mode = data_verification_mode(config)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    evidence_relative = evidence_root.relative_to(ROOT).as_posix()
    try:
        training_config = resolve_training_config(ROOT, args.action, args.training_config, config["training_config"])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    subprocess.run([sys.executable, str(ROOT / "scripts/infra/vast_control.py"), "show",
                    "--deployment-config", deployment_config], check=True)
    instance = json.loads((evidence_root / "vast_instance.json").read_text())
    if instance["id"] != config["instance_id"] or instance["actual_status"] != "running":
        raise SystemExit("Selected instance is not confirmed running. Refresh it using vast_control.py show.")
    if not args.identity_file.is_file():
        raise SystemExit("SSH identity file does not exist or is not accessible.")
    direct_ports = (instance.get("ports") or {}).get("22/tcp", [])
    if direct_ports and instance.get("public_ipaddr"):
        host, port = instance["public_ipaddr"], direct_ports[0]["HostPort"]
    else:
        host, port = instance["ssh_host"], instance["ssh_port"]
    destination = "root@" + host
    options = ["-i", str(args.identity_file.resolve()), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=accept-new"]
    ssh = ["ssh", *options, "-p", str(port), destination]
    scp = ["scp", *options, "-P", str(port)]
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
        marker = json.loads((evidence_root / "instance.json").read_text())
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
        marker_path = evidence_root / "instance.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker, indent=2))
        remote(f"mkdir -p {q(workspace + '/' + evidence_relative)}")
        upload(marker_path, workspace + "/" + evidence_relative + "/instance.json")
    elif args.action == "bootstrap":
        remote(f"cd {q(workspace)} && EXPECTED_GIT_BRANCH={q(config['branch'])} bash scripts/infra/bootstrap_server.sh")
    elif args.action == "upload-assets":
        if verification_mode == "installed_manifest_sha256":
            # C002 receives only its model and a copied binding.  Raw data is
            # adopted inside the server and must be rehashed in the new checkout.
            source_state = ROOT / config["mlflow_state_source"]
            assets = [
                (ROOT / "artifacts/models/campp/campplus_voxceleb.bin", "artifacts/models/campp/campplus_voxceleb.bin"),
                (source_state, config["mlflow_state_path"]),
            ]
        else:
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
        remote(f"mkdir -p {q(workspace + '/data/incoming')} {q(workspace + '/artifacts/models/campp')} {q(workspace + '/' + evidence_relative)}")
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
    elif args.action == "resume-raw":
        if verification_mode == "installed_manifest_sha256":
            raise SystemExit("C002 installed-data deployment forbids ZIP resume; run verify for a full installed-file SHA256 check.")
        # Call only after the previous writer has exited. This action never kills
        # another transfer, removes a partial upload, or starts model execution.
        source = ROOT / "data/raw.zip"
        if config["archive_path"] != "data/incoming/raw.zip":
            raise SystemExit("Resume is restricted to the fixed incoming raw.zip target")
        target = workspace + "/data/incoming/raw.zip"
        inspect_source = "\n".join([
            "import hashlib,json,pathlib,stat,sys",
            "root=pathlib.Path(sys.argv[1]).resolve(strict=True)",
            "target=pathlib.Path(sys.argv[2])",
            "assert target.is_absolute() and target.resolve().is_relative_to(root), 'Upload target escaped workspace'",
            "assert target.parent.is_dir(), 'Incoming directory is missing'",
            "assert not any(p.is_symlink() for p in (target,*target.parents)), 'Upload path contains a symlink'",
            "if target.exists():",
            "    before=target.lstat()",
            "    assert stat.S_ISREG(before.st_mode), 'Upload target is not a regular file'",
            "    with target.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()",
            "    after=target.lstat()",
            "    assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino), 'Another writer changed raw.zip during hashing'",
            "    result={'exists':True,'size':after.st_size,'sha256':digest,'mtime_ns':after.st_mtime_ns,'inode':after.st_ino}",
            "else:",
            "    result={'exists':False,'size':0,'sha256':hashlib.sha256(b'').hexdigest(),'mtime_ns':None,'inode':None}",
            "print(json.dumps(result))",
        ])

        def inspect_raw():
            result = remote(shlex.join([workspace + "/.venv/bin/python", "-c", inspect_source, workspace, target]), capture=True)
            return json.loads(result.stdout)

        snapshot = inspect_raw()
        resumed_from = validate_raw_resume(source, config["archive_size_bytes"], config["archive_sha256"], snapshot)
        print(json.dumps({"stage": "raw_resume_prefix_verified", "existing_bytes": resumed_from,
                          "total_bytes": config["archive_size_bytes"]}), flush=True)
        if resumed_from < config["archive_size_bytes"]:
            # Relative local path avoids Windows drive-colon parsing and spaces.
            # Both paths are fixed controlled paths; SFTP receives its own quoted
            # batch language directly through stdin, never through a local shell.
            batch = 'reput "data/raw.zip" "' + target + '"\n'
            subprocess.run(["sftp", *options, "-P", str(port), "-b", "-", "-N", destination],
                           input=batch, cwd=ROOT, text=True, check=True)
            completed = inspect_raw()
        else:
            completed = snapshot
        if completed["size"] != config["archive_size_bytes"] or completed["sha256"] != config["archive_sha256"]:
            raise SystemExit("Resumed archive failed final size/SHA256 verification; extraction is blocked")
        report = {"status": "passed", "training_started": False, "resumed_from_bytes": resumed_from,
                  "archive_size_bytes": completed["size"], "archive_sha256": completed["sha256"],
                  "archive_path": config["archive_path"], "archive_deleted": False,
                  "next_step": "Run verify for complete ZIP CRC, extraction and raw-file verification"}
        (ROOT / "artifacts/infrastructure/raw_upload.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.action == "verify":
        if verification_mode == "installed_manifest_sha256":
            # A transferred C001 receipt or a now-deleted ZIP cannot authorize
            # C002.  This path rehashes every installed byte in the new checkout.
            from datetime import datetime, timezone
            import csv
            with (ROOT / config["metadata_files"][0]).open(encoding="utf-8-sig", newline="") as handle:
                audio = next(row["audio_file"] for row in csv.DictReader(handle)
                             if row["usable_for_training"].lower() == "true")
            spool = evidence_relative + "/mlflow_probe/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            data_report = evidence_relative + "/data_readiness.json"
            runtime_report = evidence_relative + "/runtime_readiness.json"
            campp_report = evidence_relative + "/campp_probe.json"
            instance_report = evidence_relative + "/instance.json"
            mlflow_report = spool + "/preflight_result.json"
            readiness_report = evidence_relative + "/readiness.json"
            commands = [
                [".venv/bin/python", "scripts/infra/verify_installed_data.py", "--config", training_config,
                 "--report", data_report],
                [".venv/bin/python", "scripts/infra/preflight_runtime.py", "--require-cuda", "--expected-gpu",
                 config["gpu"], "--min-free-disk-gb", "10", "--report", runtime_report],
                [".venv/bin/python", "scripts/checks/probe_campp.py", "--audio", "data/raw/" + audio,
                 "--device", "cuda", "--config", training_config, "--report", campp_report],
                [".venv/bin/python", "scripts/infra/with_project_env.py", ".venv/bin/python",
                 "scripts/checks/probe_mlflow.py", "--experiment-name", config["mlflow_experiment_name"],
                 "--state-path", config["mlflow_state_path"], "--spool-dir", spool, "--config", training_config,
                 "--fingerprint", "model_config=configs/model/campp.json",
                 "--fingerprint", "manifest=data/processed/eda_v1/audio_manifest.csv",
                 "--fingerprint", "folds=data/processed/eda_v1/folds.csv",
                 "--fingerprint", "roles=data/processed/eda_v1/calibration_roles.csv",
                 "--fingerprint", "label_map=data/processed/eda_v1/label_map.json",
                 "--fingerprint", "weights=artifacts/models/campp/campplus_voxceleb.bin",
                 "--evidence-report", "data=" + data_report,
                 "--evidence-report", "runtime=" + runtime_report,
                 "--evidence-report", "campp=" + campp_report],
                [".venv/bin/python", "scripts/infra/check_readiness.py", "--config", training_config,
                 "--report", readiness_report, "--data-report", data_report, "--runtime-report", runtime_report,
                 "--campp-report", campp_report, "--mlflow-report", mlflow_report,
                 "--instance-report", instance_report],
            ]
            for command in commands:
                invocation = shlex.join(command)
                remote(f"set -eu; cd {q(workspace)}; export VAST_INSTANCE_ID={config['instance_id']}; " + invocation)
            local_probe = evidence_root / "remote_probe_path.json"
            local_probe.parent.mkdir(parents=True, exist_ok=True)
            local_probe.write_text(json.dumps({"spool": spool}, indent=2), encoding="utf-8")
            print("All C002 server readiness checks passed. This verification did not start training.")
            return
        import csv
        from datetime import datetime, timezone
        archive_config = config
        archive_options = []
        if args.archive_source == "official_original":
            archive_config = json.loads((ROOT / "configs/infra/archive_sources.json").read_text())[args.archive_source]
            if archive_config["archive_path"] != "data/incoming/competition_parallel.zip" or archive_config["member_prefix"] != "training":
                raise SystemExit("Official source must use its explicitly verified incoming path and archive prefix")
            archive_options = ["--archive-prefix", archive_config["member_prefix"],
                               "--expected-labels-sha256", archive_config["labels_sha256"],
                               "--archive-source-id", args.archive_source]
        metadata_identity = json.loads((ROOT / "artifacts/infrastructure/data_metadata.identity.json").read_text())
        with (ROOT / config["metadata_files"][0]).open(encoding="utf-8-sig", newline="") as handle:
            audio = next(row["audio_file"] for row in csv.DictReader(handle) if row["usable_for_training"].lower() == "true")
        spool = "artifacts/infrastructure/mlflow_probe/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        commands = [
            [".venv/bin/python", "scripts/infra/install_metadata.py", "--archive", "data/incoming/data_metadata.zip", "--expected-archive-sha256", metadata_identity["archive_sha256"], "--delete-archive-after-verification"],
            [".venv/bin/python", "scripts/infra/verify_extract_data.py", "--archive", archive_config["archive_path"], "--expected-archive-sha256", archive_config["archive_sha256"], *archive_options, "--delete-archive-after-verification"],
            [".venv/bin/python", "scripts/infra/preflight_runtime.py", "--require-cuda", "--expected-gpu", "RTX3090", "--min-free-disk-gb", "10"],
            [".venv/bin/python", "scripts/checks/probe_campp.py", "--audio", "data/raw/" + audio, "--device", "cuda", "--config", training_config],
            [".venv/bin/python", "scripts/infra/with_project_env.py", ".venv/bin/python", "scripts/checks/probe_mlflow.py", "--experiment-name", config["mlflow_experiment_name"], "--spool-dir", spool, "--config", training_config,
             "--fingerprint", "model_config=configs/model/campp.json", "--fingerprint", "manifest=data/processed/eda_v1/audio_manifest.csv", "--fingerprint", "folds=data/processed/eda_v1/folds.csv", "--fingerprint", "roles=data/processed/eda_v1/calibration_roles.csv", "--fingerprint", "label_map=data/processed/eda_v1/label_map.json", "--fingerprint", "weights=artifacts/models/campp/campplus_voxceleb.bin",
             "--evidence-report", "data=artifacts/infrastructure/data_readiness.json",
             "--evidence-report", "runtime=artifacts/infrastructure/runtime_readiness.json",
             "--evidence-report", "campp=artifacts/infrastructure/campp_probe.json"],
            [".venv/bin/python", "scripts/infra/check_readiness.py", "--config", training_config, "--mlflow-report", spool + "/preflight_result.json"],
        ]
        for index, command in enumerate(commands):
            invocation = shlex.join(command)
            if index < 2:
                # Successful extraction already removed its incoming archive.
                # Reuse only proof for this exact successfully installed archive.
                archive = "data/incoming/data_metadata.zip" if index == 0 else archive_config["archive_path"]
                evidence = "metadata_readiness.json" if index == 0 else "data_readiness.json"
                invocation = f"if [ -f {q(archive)} ]; then {invocation}; else cat artifacts/infrastructure/{evidence}; fi"
                result = remote(f"set -eu; cd {q(workspace)}; " + invocation, capture=True)
                proof = json.loads(result.stdout)
                expected = metadata_identity["archive_sha256"] if index == 0 else archive_config["archive_sha256"]
                if proof.get("status") != "passed" or proof.get("archive_deleted") is not True or proof.get("archive_sha256") != expected:
                    raise SystemExit("Transferred asset verification is missing, failed, or belongs to another archive.")
                if index == 1 and proof.get("archive_source_id", "local_upload") != args.archive_source:
                    raise SystemExit("Data evidence belongs to a different selected archive source")
                if index == 0:
                    expected_files = {item["path"]: item["sha256"] for item in metadata_identity["files"]}
                    measured = remote(f"cd {q(workspace)} && sha256sum -- " + shlex.join(sorted(expected_files)), capture=True)
                    actual = {line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0] for line in measured.stdout.splitlines()}
                    if actual != expected_files:
                        raise SystemExit("Installed metadata differs from the local prepared bundle.")
                print(f"Verified transferred archive and installed evidence: {archive}")
            else:
                remote(f"set -eu; cd {q(workspace)}; export VAST_INSTANCE_ID={config['instance_id']}; " + invocation)
        (evidence_root / "remote_probe_path.json").write_text(json.dumps({"spool": spool}))
        print("All server readiness checks passed. This verification did not start training.")
    else:
        output = evidence_root / "server_evidence"
        output.mkdir(parents=True, exist_ok=True)
        names = ["instance.json", "data_readiness.json", "runtime_readiness.json", "campp_probe.json", "readiness.json"]
        if verification_mode == "archive_extract":
            names.insert(1, "metadata_readiness.json")
        for name in names:
            subprocess.run([*scp, destination + ":" + workspace + "/" + evidence_relative + "/" + name,
                            str(output / name)], check=True)
        spool = json.loads((evidence_root / "remote_probe_path.json").read_text())["spool"]
        expected_spool_prefix = evidence_relative + "/mlflow_probe/"
        if (not isinstance(spool, str) or not spool.startswith(expected_spool_prefix)
                or "\\" in spool or any(part in {"", ".", ".."} for part in spool.split("/"))):
            raise SystemExit("Recorded remote MLflow spool is outside the expected infrastructure directory")
        subprocess.run([*scp, destination + ":" + workspace + "/" + spool + "/preflight_result.json",
                        str(output / "mlflow_preflight_result.json")], check=True)
        print("Server evidence downloaded; this download did not start training.")


if __name__ == "__main__":
    main()
