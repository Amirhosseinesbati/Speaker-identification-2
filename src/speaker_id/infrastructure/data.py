"""Verify a transferred dataset before publishing files or deleting its ZIP.

The manifest is the trusted inventory tracked with the project. Archive member
names are deliberately restricted to the competition's flat ``raw/`` layout.
No existing raw file is overwritten, and no raw file is ever deleted.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import zipfile


BLOCK_BYTES = 1024 * 1024


class DataVerificationError(ValueError):
    """The supplied archive or extracted dataset did not pass verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def confined_path(workspace: Path, value: str | Path) -> Path:
    """Resolve a workspace child and reject symlinks in its existing ancestry."""
    workspace = workspace.resolve(strict=True)
    value = Path(value)
    candidate = value if value.is_absolute() else workspace / value
    absolute = Path(os.path.abspath(candidate))
    if not absolute.is_relative_to(workspace) or absolute == workspace:
        raise DataVerificationError(f"Path must be a child of the workspace: {value}")
    for part in (absolute, *absolute.parents):
        if part == workspace:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise DataVerificationError(f"Symlink/junction paths are not permitted: {part}")
    resolved = absolute.resolve()
    if not resolved.is_relative_to(workspace):
        raise DataVerificationError(f"Resolved path escaped the workspace: {value}")
    return resolved


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _flat_filename(name: str) -> str:
    if (not name or name in {".", ".."} or "/" in name or "\\" in name
            or ":" in name or "\x00" in name or name != name.strip()):
        raise DataVerificationError(f"Unsafe flat filename: {name!r}")
    return name


def load_manifest(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"audio_file", "speaker_id", "input_sha256", "file_bytes"}
        if not required <= set(reader.fieldnames or []):
            raise DataVerificationError(f"Manifest missing columns: {sorted(required)}")
        result = {}
        for row in reader:
            name = _flat_filename(row["audio_file"])
            if name in result or name == "labels.csv":
                raise DataVerificationError(f"Repeated or reserved manifest filename: {name}")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", row["input_sha256"]):
                raise DataVerificationError(f"Invalid manifest SHA256 for {name}")
            if not row["speaker_id"] or int(row["file_bytes"]) < 0:
                raise DataVerificationError(f"Invalid manifest label/length for {name}")
            result[name] = row
    if not result:
        raise DataVerificationError("Manifest is empty")
    return result


def _validate_labels(raw: bytes, manifest: dict[str, dict]) -> None:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if (len(reader.fieldnames or []) != 2
            or set(reader.fieldnames or []) != {"speaker_id", "audio_file"}):
        raise DataVerificationError("labels.csv must have exactly speaker_id,audio_file columns")
    observed = {}
    for row in reader:
        if None in row or not row.get("audio_file") or not row.get("speaker_id"):
            raise DataVerificationError("Malformed labels.csv row")
        name = _flat_filename(row["audio_file"])
        if name in observed:
            raise DataVerificationError(f"Duplicate labels.csv row: {name}")
        observed[name] = row["speaker_id"]
    expected = {name: row["speaker_id"] for name, row in manifest.items()}
    if observed != expected:
        raise DataVerificationError("labels.csv names/labels differ from the trusted manifest")


def _archive_inventory(archive: zipfile.ZipFile, expected: set[str]) -> dict[str, zipfile.ZipInfo]:
    members = {}
    seen_raw_names = set()
    for info in archive.infolist():
        name = info.filename
        # Check before normalization: PurePosixPath would silently remove './'.
        components = name.rstrip("/").split("/")
        if (not name or name.startswith("/") or "\\" in name or ":" in name
                or "\x00" in name or any(p in {"", ".", ".."} for p in components)
                or PurePosixPath(name).is_absolute()):
            raise DataVerificationError(f"Unsafe ZIP member: {name!r}")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise DataVerificationError(f"ZIP symlink/special member forbidden: {name}")
        if info.flag_bits & 1:
            raise DataVerificationError(f"Encrypted ZIP member forbidden: {name}")
        if name in seen_raw_names:
            raise DataVerificationError(f"Duplicate ZIP member: {name}")
        seen_raw_names.add(name)
        if info.is_dir():
            if name != "raw/":
                raise DataVerificationError(f"Unexpected ZIP directory: {name}")
            continue
        if len(components) == 2 and components[0] == "raw":
            basename = _flat_filename(components[1])
        elif len(components) == 1:
            basename = _flat_filename(components[0])
        else:
            raise DataVerificationError(f"ZIP must contain a flat dataset: {name}")
        if basename in members:
            raise DataVerificationError(f"ZIP filename collision: {basename}")
        if basename not in expected:
            raise DataVerificationError(f"Unexpected ZIP file: {name}")
        members[basename] = info
    if set(members) != expected:
        raise DataVerificationError(f"ZIP missing {len(expected - set(members))} required files")
    return members


def verify_extract_data(*, workspace: Path, archive: Path,
                        expected_archive_sha256: str | None,
                        manifest: Path = Path("data/processed/eda_v1/audio_manifest.csv"),
                        output: Path = Path("data/raw"),
                        report: Path = Path("artifacts/infrastructure/data_readiness.json"),
                        delete_archive_after_verification: bool = False) -> dict:
    """Full archive CRC + audio SHA256 + labels verification, then safe publish.

    Deletion is opt-in and restricted to a ZIP under ``data/incoming``. A local
    source ZIP such as ``data/raw.zip`` cannot be deleted by this operation.
    Existing matching files are reused; any unrelated or mismatching file fails
    verification. Partial publication after an interrupted process is resumable.
    """
    started = time.monotonic()
    workspace = workspace.resolve(strict=True)
    report = confined_path(workspace, report)
    archive = confined_path(workspace, archive)
    manifest = confined_path(workspace, manifest)
    output = confined_path(workspace, output)
    # Validate reporting destinations before exception handling can write a report.
    if (not report.is_relative_to(workspace / "artifacts/infrastructure")
            or report in {archive, manifest}):
        raise DataVerificationError("Report must be a separate file under artifacts/infrastructure")
    result = {"status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "workspace": str(workspace), "archive_deleted": False,
              "training_started": False, "verification_scope": "all archive members and all output files"}
    staged: dict[str, Path] = {}
    staging = None
    try:
        if output != workspace / "data/raw":
            raise DataVerificationError("Dataset output must be the workspace's data/raw directory")
        if report.is_relative_to(output) or report == archive or report == manifest:
            raise DataVerificationError("Report must not overwrite an archive, manifest, or raw data")
        if delete_archive_after_verification:
            if (not expected_archive_sha256 or archive.suffix.lower() != ".zip"
                    or not archive.is_relative_to(workspace / "data/incoming")):
                raise DataVerificationError("Deletion requires expected SHA256 and a ZIP under data/incoming")
        if expected_archive_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_archive_sha256):
            raise DataVerificationError("Expected archive SHA256 must be exactly 64 hexadecimal characters")
        expected = load_manifest(manifest)
        result.update({"archive": str(archive), "output": str(output),
                       "manifest_sha256": sha256_file(manifest), "audio_files_expected": len(expected),
                       "class_count": len({row["speaker_id"] for row in expected.values()})})
        archive_hash = sha256_file(archive)
        result["archive_sha256"] = archive_hash
        if expected_archive_sha256 and archive_hash != expected_archive_sha256.lower():
            raise DataVerificationError("Transferred ZIP SHA256 does not match the local source")
        archive_before = archive.stat()
        wanted = set(expected) | {"labels.csv"}
        if output.exists():
            if not output.is_dir():
                raise DataVerificationError("data/raw already exists and is not a directory")
            for item in output.iterdir():
                confined_path(workspace, item)
                if item.name not in wanted or not item.is_file():
                    raise DataVerificationError(f"Unrelated existing raw entry: {item.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".raw-verification-", dir=output.parent))
        confined_path(workspace, staging)
        archived_hashes = {}
        reused = 0
        with zipfile.ZipFile(archive) as zipped:
            members = _archive_inventory(zipped, wanted)
            for name, info in members.items():
                if name in expected and info.file_size != int(expected[name]["file_bytes"]):
                    raise DataVerificationError(f"ZIP member size disagrees with manifest: {name}")
                if name == "labels.csv" and info.file_size > 16 * 1024 * 1024:
                    raise DataVerificationError("Unexpectedly large labels.csv")
                destination = confined_path(workspace, output / name)
                digest = hashlib.sha256()
                staged_path = staging / name
                with zipped.open(info) as source, staged_path.open("xb") as target:
                    for block in iter(lambda: source.read(BLOCK_BYTES), b""):
                        digest.update(block)
                        target.write(block)
                # Reading every member to EOF also verifies its ZIP CRC.
                staged[name] = staged_path
                observed = digest.hexdigest()
                archived_hashes[name] = observed
                if name in expected and observed != expected[name]["input_sha256"].lower():
                    raise DataVerificationError(f"ZIP audio SHA256 differs from manifest: {name}")
                if name == "labels.csv":
                    _validate_labels(staged_path.read_bytes(), expected)
                if destination.exists():
                    if sha256_file(destination) != observed:
                        raise DataVerificationError(f"Existing raw file differs; refusing overwrite: {name}")
                    staged_path.unlink()
                    del staged[name]
                    reused += 1
            result["archive_crc_verified_members"] = len(members)
        output.mkdir(exist_ok=True)
        for name, source in list(staged.items()):
            destination = confined_path(workspace, output / name)
            # Hard-link publication is atomic and cannot replace an existing file.
            os.link(source, destination)
            source.unlink()
            del staged[name]
        verified_bytes = 0
        for name, digest in archived_hashes.items():
            destination = confined_path(workspace, output / name)
            if not destination.is_file() or sha256_file(destination) != digest:
                raise DataVerificationError(f"Final extracted file SHA256 mismatch: {name}")
            verified_bytes += destination.stat().st_size
        _validate_labels((output / "labels.csv").read_bytes(), expected)
        result.update({"status": "passed", "audio_files_verified": len(expected),
                       "labels_verified": True, "labels_sha256": archived_hashes["labels.csv"],
                       "output_files_verified": len(archived_hashes), "output_bytes_verified": verified_bytes,
                       "existing_matching_files_reused": reused,
                       "elapsed_seconds": round(time.monotonic() - started, 3)})
        # Persist proof before any deletion. The second report records its outcome.
        write_json_atomic(report, result)
        if delete_archive_after_verification:
            confined_path(workspace, archive)
            archive_after = archive.stat()
            if (archive_after.st_size, archive_after.st_mtime_ns, archive_after.st_ino) != (
                    archive_before.st_size, archive_before.st_mtime_ns, archive_before.st_ino):
                raise DataVerificationError("ZIP changed during verification; refusing deletion")
            archive.unlink()
            result["archive_deleted"] = True
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json_atomic(report, result)
        return result
    except Exception as error:
        result.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "elapsed_seconds": round(time.monotonic() - started, 3)})
        write_json_atomic(report, result)
        raise
    finally:
        if staging is not None:
            # Only generated staging files are removed; raw files are never removed.
            for path in staging.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            staging.rmdir()
