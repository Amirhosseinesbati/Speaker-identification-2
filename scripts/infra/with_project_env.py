"""Run a command using project MLflow credentials without printing or shell expansion."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def mlflow_environment(env_file: Path = ROOT / ".env") -> dict[str, str]:
    from dotenv import dotenv_values
    values = dotenv_values(env_file)
    env = os.environ.copy()
    for destination, source in {
        "MLFLOW_TRACKING_URI": "DAGSHUB_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME": "DAGSHUB_REPO_OWNER",
        "MLFLOW_TRACKING_PASSWORD": "DAGSHUB_USER_TOKEN",
    }.items():
        value = values.get(destination) or values.get(source) or env.get(destination)
        if not value:
            raise ValueError(f"Missing required configuration: {destination}")
        env[destination] = value
    # A transient service failure must not hold a readiness check for minutes.
    env.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    env.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/infra/with_project_env.py <command> [arguments...]")
    raise SystemExit(subprocess.run(sys.argv[1:], env=mlflow_environment(), cwd=ROOT).returncode)
