"""Use the official Vast CLI with project-scoped configuration and secret handling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT_CONFIG = Path("configs/infra/deployment.json")


def resolve_deployment_config(root: Path, requested: Path | None) -> tuple[dict, str, Path]:
    """Load a checked deployment configuration and its isolated evidence root."""
    root = root.resolve(strict=True)
    candidate = DEFAULT_DEPLOYMENT_CONFIG if requested is None else Path(requested)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise ValueError("Deployment config cannot contain parent traversal components")
    absolute = candidate if candidate.is_absolute() else root / candidate
    absolute = Path(os.path.abspath(absolute))
    allowed = root / "configs/infra"
    if not absolute.is_relative_to(allowed):
        raise ValueError("Deployment config must be inside configs/infra")
    for part in (absolute, *absolute.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Deployment config paths cannot contain symlinks or junctions")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ValueError("Deployment config must be an existing JSON file inside configs/infra")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("instance_id")) is not int:
        raise ValueError("Deployment config must be a JSON object with an integer instance_id")
    raw_root = payload.get("evidence_root", "artifacts/infrastructure")
    if not isinstance(raw_root, str) or not raw_root or Path(raw_root).is_absolute() or ".." in Path(raw_root).parts:
        raise ValueError("Deployment evidence_root must be a relative path without parent traversal")
    evidence = Path(os.path.abspath(root / raw_root))
    allowed_evidence = root / "artifacts/infrastructure"
    if not evidence.is_relative_to(allowed_evidence):
        raise ValueError("Deployment evidence_root must be under artifacts/infrastructure")
    for part in (evidence, *evidence.parents):
        if part == root:
            break
        if part.exists() and (part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction())):
            raise ValueError("Deployment evidence_root cannot contain symlinks or junctions")
    return payload, resolved.relative_to(root).as_posix(), evidence


def project_environment() -> dict[str, str]:
    env = os.environ.copy()
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["XDG_CONFIG_HOME"] = str(ROOT / "artifacts/tooling/config")
    env["XDG_CACHE_HOME"] = str(ROOT / "artifacts/tooling/cache")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["show", "start", "logs", "ssh-keys", "attach-key"])
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--deployment-config", type=Path,
                        help="Committed configs/infra/*.json deployment identity; defaults to the historical config")
    args = parser.parse_args()
    try:
        config, _, evidence_root = resolve_deployment_config(ROOT, args.deployment_config)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    instance_id = str(config["instance_id"])
    env = project_environment()
    # Windows resolves executable names using the parent's PATH.
    os.environ["PATH"] = env["PATH"]
    commands = {
        "show": ["show", "instance", instance_id],
        "start": ["start", "instance", instance_id],
        "logs": ["logs", instance_id, "--tail", "100"],
        "ssh-keys": ["show", "ssh-keys"],
    }
    if args.action == "attach-key":
        if not args.public_key:
            parser.error("--public-key required")
        public = args.public_key.read_text().strip()
        if not public.startswith(("ssh-ed25519 ", "ssh-rsa ")):
            parser.error("Expected an OpenSSH public key, never a private key")
        command = ["attach", "ssh", instance_id, public]
    else:
        command = commands[args.action]
    result = subprocess.run(["vastai", *command, "--raw"], env=env, capture_output=True, text=True, timeout=90)
    secret = env.get("VAST_API_KEY", "")
    def safe(value: str) -> str:
        return value.replace(secret, "[REDACTED]") if secret else value
    if result.returncode:
        print(safe(result.stderr or result.stdout), file=sys.stderr)
        return result.returncode
    if args.action == "show":
        data = json.loads(result.stdout)
        keys = ["id", "actual_status", "intended_status", "next_state", "gpu_name", "num_gpus", "ssh_host", "ssh_port", "public_ipaddr", "ports", "disk_space", "dph_total", "cur_state"]
        data = {key: data.get(key) for key in keys}
        target = evidence_root / "vast_instance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps(data))
    else:
        print(safe(result.stdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
