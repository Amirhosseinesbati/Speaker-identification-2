"""Deterministic, allowlisted metadata transfer independent of the public Git repository."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

from speaker_id.tracking.security import Redactor
from .data import DataVerificationError, confined_path, sha256_file, write_json_atomic

MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
COMMENT_KIND = "speaker-identification-data-metadata"


def _metadata_name(value: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or "\\" in value or ":" in value or "\x00" in value
            or any(part in {"", ".", ".."} or part.startswith(".") for part in value.split("/"))):
        raise DataVerificationError("Metadata paths must be normalized relative POSIX paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts[:2] != ("data", "processed") or path.suffix.lower() not in {".csv", ".json"}:
        raise DataVerificationError("Metadata files must be CSV/JSON files under data/processed.")
    if re.search(r"(?:password|credential|secret|token|private.?key)", path.name, re.IGNORECASE):
        raise DataVerificationError("Credential-like metadata filenames are forbidden.")
    return path.as_posix()


def _deployment_files(path: Path) -> tuple[dict, list[str]]:
    deployment = json.loads(path.read_text(encoding="utf-8"))
    names = deployment.get("metadata_files")
    if not isinstance(names, list) or not names or len(names) > 128:
        raise DataVerificationError("Deployment must explicitly allowlist metadata_files.")
    names = [_metadata_name(name) for name in names]
    if len(set(name.casefold() for name in names)) != len(names):
        raise DataVerificationError("Deployment metadata paths contain duplicates/case collisions.")
    return deployment, sorted(names)


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _manifest(comment: bytes, expected_names: list[str]) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DataVerificationError("Duplicate JSON key in ZIP metadata manifest.")
            result[key] = value
        return result

    try:
        manifest = json.loads(comment.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataVerificationError("ZIP must contain its metadata manifest in its comment.") from error
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("bundle_kind") != COMMENT_KIND or not isinstance(manifest.get("files"), list)):
        raise DataVerificationError("Unsupported metadata manifest schema.")
    observed = {}
    for record in manifest["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise DataVerificationError("Malformed metadata manifest file entry.")
        name = _metadata_name(record["path"])
        if name in observed:
            raise DataVerificationError("Repeated metadata manifest filename.")
        if (not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool)
                or not 0 <= record["bytes"] <= MAX_MEMBER_BYTES
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise DataVerificationError("Invalid metadata member size or SHA256.")
        observed[name] = record
    if set(observed) != set(expected_names):
        raise DataVerificationError("ZIP manifest files differ from deployment metadata_files.")
    if sum(record["bytes"] for record in observed.values()) > MAX_TOTAL_BYTES:
        raise DataVerificationError("Metadata bundle exceeds the allowed total size.")
    return observed


def prepare_metadata_transfer(*, workspace: Path, deployment: Path = Path("configs/infra/deployment.json"),
                              output: Path = Path("artifacts/infrastructure/data_metadata.zip"),
                              identity: Path = Path("artifacts/infrastructure/data_metadata.identity.json")) -> dict:
    workspace = workspace.resolve(strict=True)
    deployment, output, identity = [confined_path(workspace, item) for item in (deployment, output, identity)]
    allowed_output = workspace / "artifacts/infrastructure"
    if (not output.is_relative_to(allowed_output) or not identity.is_relative_to(allowed_output)
            or output == identity or output.suffix.lower() != ".zip" or identity.suffix.lower() != ".json"):
        raise DataVerificationError("Bundle and identity outputs must be separate ZIP/JSON files under artifacts/infrastructure.")
    _, names = _deployment_files(deployment)
    redactor = Redactor()
    payloads, files = {}, []
    for name in names:
        source = confined_path(workspace, name)
        if not source.is_file() or source.stat().st_size > MAX_MEMBER_BYTES:
            raise DataVerificationError(f"Metadata input missing or too large: {name}")
        payload = source.read_bytes()
        redactor.assert_no_secret_bytes(payload, name)
        payloads[name] = payload
        files.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    if sum(record["bytes"] for record in files) > MAX_TOTAL_BYTES:
        raise DataVerificationError("Metadata bundle exceeds the allowed total size.")
    manifest = {"schema_version": 1, "bundle_kind": COMMENT_KIND, "files": files,
                "deployment_sha256": sha256_file(deployment)}
    comment = _json_bytes(manifest)
    if len(comment) > 65535:
        raise DataVerificationError("Metadata ZIP manifest exceeds the ZIP comment limit.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".metadata-build-", suffix=".zip", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            archive.comment = comment
        for record in files:
            if sha256_file(confined_path(workspace, record["path"])) != record["sha256"]:
                raise DataVerificationError("Metadata inputs changed while preparing the bundle.")
        result = {**manifest, "archive_sha256": sha256_file(temporary),
                  "archive_size_bytes": temporary.stat().st_size,
                  "archive_path": output.relative_to(workspace).as_posix(),
                  "metadata_file_count": len(files), "training_started": False}
        os.replace(temporary, output)
        write_json_atomic(identity, result)
        return result
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def install_metadata(*, workspace: Path, archive: Path,
                     expected_archive_sha256: str,
                     deployment: Path = Path("configs/infra/deployment.json"),
                     report: Path = Path("artifacts/infrastructure/metadata_readiness.json"),
                     delete_archive_after_verification=False) -> dict:
    """Verify all bytes, stage all members, then publish without replacing any file."""
    workspace = workspace.resolve(strict=True)
    archive, deployment, report = [confined_path(workspace, item) for item in (archive, deployment, report)]
    if not report.is_relative_to(workspace / "artifacts/infrastructure") or report in {archive, deployment}:
        raise DataVerificationError("Readiness report must be separate under artifacts/infrastructure.")
    if not isinstance(expected_archive_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_archive_sha256):
        raise DataVerificationError("A trusted expected archive SHA256 is required.")
    if delete_archive_after_verification and (
            archive.suffix.lower() != ".zip" or not archive.is_relative_to(workspace / "data/incoming")):
        raise DataVerificationError("Deletion is allowed only for a transferred ZIP under data/incoming.")
    result = {"status": "failed", "training_started": False, "archive_deleted": False,
              "verification_scope": "trusted ZIP SHA256, exact allowlist, every member CRC/SHA, all installed files"}
    staging = None
    try:
        _, names = _deployment_files(deployment)
        observed_sha = sha256_file(archive)
        if observed_sha != expected_archive_sha256.lower():
            raise DataVerificationError("Metadata ZIP SHA256 differs from the trusted expected SHA256.")
        archive_before = archive.stat()
        result["archive_sha256"] = observed_sha
        staging_parent = confined_path(workspace, "artifacts/infrastructure")
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".metadata-install-", dir=staging_parent))
        staged, reused = {}, 0
        with zipfile.ZipFile(archive) as zipped:
            manifest = _manifest(zipped.comment, names)
            members = {}
            for info in zipped.infolist():
                name = _metadata_name(info.filename)
                kind = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
                if name in members or name not in manifest or info.is_dir():
                    raise DataVerificationError("Unexpected or duplicate metadata ZIP member.")
                if kind not in {0, stat.S_IFREG} or info.flag_bits & 1:
                    raise DataVerificationError("ZIP symlinks, special files, and encrypted members are forbidden.")
                if info.file_size != manifest[name]["bytes"]:
                    raise DataVerificationError("ZIP member size differs from the metadata manifest.")
                members[name] = info
            if set(members) != set(names):
                raise DataVerificationError("ZIP must contain exactly deployment metadata_files.")
            for index, name in enumerate(names):
                destination = confined_path(workspace, name)
                staged_path = staging / str(index)
                digest = hashlib.sha256()
                with zipped.open(members[name]) as source, staged_path.open("xb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                # The complete ZIP member read also verifies CRC.
                if digest.hexdigest() != manifest[name]["sha256"]:
                    raise DataVerificationError(f"Metadata member SHA256 mismatch: {name}")
                if destination.exists():
                    if not destination.is_file() or sha256_file(destination) != manifest[name]["sha256"]:
                        raise DataVerificationError(f"Existing metadata differs; refusing overwrite: {name}")
                    staged_path.unlink()
                    reused += 1
                else:
                    staged[name] = staged_path
        # No publication occurs until every archive member and existing destination passes.
        for name, source in staged.items():
            destination = confined_path(workspace, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = confined_path(workspace, name)
            os.link(source, destination)  # Atomic no-replace publication on the workspace filesystem.
            source.unlink()
        for name in names:
            destination = confined_path(workspace, name)
            if not destination.is_file() or sha256_file(destination) != manifest[name]["sha256"]:
                raise DataVerificationError(f"Installed metadata verification failed: {name}")
        result.update({"status": "passed", "metadata_file_count": len(names),
                       "existing_matching_files_reused": reused, "files": [manifest[name] for name in names]})
        write_json_atomic(report, result)
        if delete_archive_after_verification:
            confined_path(workspace, archive)
            after = archive.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != (
                    archive_before.st_size, archive_before.st_mtime_ns, archive_before.st_ino):
                raise DataVerificationError("Metadata ZIP changed during verification; refusing deletion.")
            archive.unlink()
            result["archive_deleted"] = True
        write_json_atomic(report, result)
        return result
    except Exception as error:
        result.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json_atomic(report, result)
        raise
    finally:
        if staging is not None:
            for path in staging.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            staging.rmdir()
