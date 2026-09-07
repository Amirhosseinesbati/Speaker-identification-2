"""Reproducible source snapshots and non-secret execution provenance."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

from .security import Redactor

_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_provenance(root: Path) -> dict:
    environment = dict(os.environ)
    # Read-only git inspection must work even when a host's global config is unreadable.
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"

    def read(*arguments):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments], capture_output=True,
                text=True, encoding="utf-8", errors="replace", env=environment,
                timeout=15, check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = read("status", "--porcelain", "--untracked-files=normal", "--", "src")
    return {"git_commit": read("rev-parse", "HEAD"), "src_dirty": None if status is None else bool(status)}


def source_snapshot(root: Path, destination: Path, redactor: Redactor | None = None) -> dict:
    """Archive every regular src file in deterministic POSIX tar order."""
    root, destination = root.resolve(), destination.resolve()
    source = root / "src"
    if not source.is_dir():
        raise ValueError("The project must have a src directory.")
    redactor = redactor or Redactor()
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_DIRS for part in relative.parts) or path.suffix in _EXCLUDED_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f"Source snapshot refuses symlinks: {relative.as_posix()}")
        if not path.is_file():
            continue
        if path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise ValueError(f"Source snapshot refuses credential-like files: {relative.as_posix()}")
        payload = path.read_bytes()
        redactor.assert_no_secret_bytes(payload, relative.as_posix())
        files.append((relative.as_posix(), payload))
    if not files:
        raise ValueError("An empty source snapshot is not allowed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
                for name, payload in files:
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(payload), 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    manifest = {
        "schema_version": 1, **git_provenance(root), "archive_sha256": sha256_file(destination),
        "file_count": len(files), "excluded_generated_directories": sorted(_EXCLUDED_DIRS),
        "files": [{"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                  for name, payload in files],
    }
    write_json(destination.with_name("source_manifest.json"), manifest)
    return manifest


def environment_versions() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return {
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "environment_variables_recorded": False,
    }


def input_fingerprints(paths: dict[str, Path]) -> dict:
    return {str(name): {"file_name": Path(path).name, "bytes": Path(path).stat().st_size,
                        "sha256": sha256_file(Path(path))}
            for name, path in sorted(paths.items())}
