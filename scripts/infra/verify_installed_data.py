"""Rehash every already-installed raw input without relying on a ZIP receipt.

This is the C002 migration path for data that was copied between Vast
workspaces.  It performs no extraction, network operation, model loading, or
training; it verifies the bytes actually available to the new checkout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.data import (  # noqa: E402
    DataVerificationError,
    _validate_labels,
    confined_path,
    load_manifest,
    sha256_file,
    write_json_atomic,
)


def _read_config(workspace: Path, config_path: Path) -> tuple[dict, Path]:
    path = confined_path(workspace, config_path)
    if not path.is_file() or path.suffix.lower() != ".json":
        raise DataVerificationError("Training configuration must be an existing JSON file")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataVerificationError("Training configuration is not valid JSON") from error
    if not isinstance(config, dict):
        raise DataVerificationError("Training configuration must be a JSON object")
    required = {"data_dir", "manifest", "expected_source_files", "evaluation_classes"}
    if not required <= set(config):
        raise DataVerificationError("Training configuration is missing installed-data requirements")
    return config, path


def verify_installed_data(*, workspace: Path, config_path: Path,
                          report: Path) -> dict:
    """Verify every file in ``data/raw`` against the committed manifest.

    The caller must use a freshly committed C002 config.  A prior data-readiness
    receipt, an archive hash, or a successful extraction on another workspace
    cannot substitute for this check.
    """
    workspace = workspace.resolve(strict=True)
    report = confined_path(workspace, report)
    if not report.is_relative_to(workspace / "artifacts/infrastructure"):
        raise DataVerificationError("Installed-data report must stay under artifacts/infrastructure")

    result: dict = {
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "training_started": False,
        "verification_scope": "all_installed_raw_files_manifest_sha256",
        "installed_file_sha256_verified": False,
    }
    try:
        config, config_path = _read_config(workspace, config_path)
        output = confined_path(workspace, config["data_dir"])
        manifest_path = confined_path(workspace, config["manifest"])
        if output != workspace / "data/raw":
            raise DataVerificationError("Installed data must be the workspace's data/raw directory")
        if not output.is_dir():
            raise DataVerificationError("Installed data/raw directory is missing")
        manifest = load_manifest(manifest_path)
        if len(manifest) != config["expected_source_files"]:
            raise DataVerificationError("Manifest file count differs from the training configuration")
        classes = {row["speaker_id"] for row in manifest.values()}
        if len(classes) != config["evaluation_classes"]:
            raise DataVerificationError("Manifest class count differs from the training configuration")

        expected_names = set(manifest) | {"labels.csv"}
        observed_entries = list(output.iterdir())
        actual_names = {entry.name for entry in observed_entries}
        if actual_names != expected_names:
            raise DataVerificationError("Installed raw dataset inventory differs from the trusted manifest")
        if any(entry.is_symlink() or not entry.is_file() for entry in observed_entries):
            raise DataVerificationError("Installed raw dataset contains a symlink or non-file entry")

        verified_bytes = 0
        for name, row in manifest.items():
            audio = confined_path(workspace, output / name)
            if not audio.is_file() or audio.stat().st_size != int(row["file_bytes"]):
                raise DataVerificationError(f"Installed audio size differs from manifest: {name}")
            if sha256_file(audio) != row["input_sha256"].lower():
                raise DataVerificationError(f"Installed audio SHA256 differs from manifest: {name}")
            verified_bytes += audio.stat().st_size

        labels = confined_path(workspace, output / "labels.csv")
        labels_bytes = labels.read_bytes()
        _validate_labels(labels_bytes, manifest)
        verified_bytes += labels.stat().st_size
        result.update({
            "status": "passed",
            "config_path": config_path.relative_to(workspace).as_posix(),
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "output": str(output),
            "audio_files_verified": len(manifest),
            "output_files_verified": len(expected_names),
            "output_bytes_verified": True,
            "output_bytes": verified_bytes,
            "class_count": len(classes),
            "labels_verified": True,
            "labels_sha256": sha256_file(labels),
            "installed_file_sha256_verified": True,
        })
    except Exception as error:
        result.update({"error_type": type(error).__name__, "error": str(error)})
        write_json_atomic(report, result)
        raise
    write_json_atomic(report, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = vars(parser.parse_args())
    try:
        result = verify_installed_data(**arguments)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
