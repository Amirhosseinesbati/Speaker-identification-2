"""Use the official Vast CLI with project-scoped configuration and secret handling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "50079023"


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
    args = parser.parse_args()
    env = project_environment()
    # Windows resolves executable names using the parent's PATH.
    os.environ["PATH"] = env["PATH"]
    commands = {
        "show": ["show", "instance", INSTANCE_ID],
        "start": ["start", "instance", INSTANCE_ID],
        "logs": ["logs", INSTANCE_ID, "--tail", "100"],
        "ssh-keys": ["show", "ssh-keys"],
    }
    if args.action == "attach-key":
        if not args.public_key:
            parser.error("--public-key required")
        public = args.public_key.read_text().strip()
        if not public.startswith(("ssh-ed25519 ", "ssh-rsa ")):
            parser.error("Expected an OpenSSH public key, never a private key")
        command = ["attach", "ssh", INSTANCE_ID, public]
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
        target = ROOT / "artifacts/infrastructure/vast_instance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps(data))
    else:
        print(safe(result.stdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
