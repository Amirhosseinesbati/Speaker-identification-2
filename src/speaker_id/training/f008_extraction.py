"""Server-only full-scoring embedding caches for completed F008 tails.

This module has a deliberately narrow job.  It authenticates an already
completed F008 tail checkpoint *through its metadata receipt*, extracts one
embedding for every row in the caller's complete manifest, and seals the
result as a resumable server-only cache.  It also reads the corresponding F005
control cache in place for a later scorer.  It never imports Torch, a model
loader, an audio decoder, MLflow, or a file-transfer client.

The caller must first use the F008 worker to authenticate and load the
checkpoint, then supply a small extractor callback.  That keeps model loading
and raw-audio access out of this cache/provenance layer while still binding
every cached vector to the completed encoder checkpoint, its metadata, the
manifest order, and the source-audio digest recorded in the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np


F008_ADVANCED_CACHE_IDENTITY_SCHEMA = "f008-advanced-embedding-cache-identity-v1"
F008_ADVANCED_CACHE_RECEIPT_SCHEMA = "f008-advanced-embedding-cache-receipt-v1"
F008_VALIDATED_TAIL_CHECKPOINT_SCHEMA = "f008-validated-tail-checkpoint-v1"
F008_REUSED_F005_CONTROL_SCHEMA = "f008-reused-f005-control-cache-v1"
ADVANCED_EMBEDDING_DIMENSION = 192

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SCOPE = "full_scoring"


def canonical(value: object) -> bytes:
    """Canonical finite JSON used by all F008 extraction receipts."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in _SHA256_CHARACTERS for character in value),
             f"F008 {label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_json(value: object) -> Any:
    return json.loads(canonical(value).decode("utf-8"))


def _safe_relative_posix(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"F008 {label} is missing")
    parsed = PurePosixPath(value)
    _require(not parsed.is_absolute() and parsed.as_posix() == value
             and "." not in parsed.parts and ".." not in parsed.parts
             and "\\" not in value and ":" not in value,
             f"F008 {label} must be a safe relative POSIX path")
    return value


def _safe_file_name(value: object, label: str) -> str:
    value = _safe_relative_posix(value, label)
    _require("/" not in value, f"F008 {label} must name one file")
    return value


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"F008 {label} must be a regular non-symlink file")
    return path


def _regular_directory(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_dir() and not path.is_symlink(),
             f"F008 {label} must be a regular non-symlink directory")
    return path


def _directory_below(root: Path, relative: str, label: str, *, create: bool) -> Path:
    """Return a non-symlink directory safely below an existing root."""
    current = _regular_directory(Path(root), f"{label} output root")
    for part in PurePosixPath(relative).parts:
        candidate = current / part
        if candidate.exists() or candidate.is_symlink():
            _regular_directory(candidate, label)
        else:
            _require(create, f"F008 {label} directory is missing")
            candidate.mkdir()
            _regular_directory(candidate, label)
        current = candidate
    return current


def _confined_regular_file(root: Path, relative: object, label: str) -> Path:
    relative = _safe_relative_posix(relative, label)
    current = _regular_directory(Path(root), f"{label} source root")
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        current = _regular_directory(current / part, label)
    return _regular_file(current / parts[-1], label)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, label)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F008 {label} must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), f"F008 {label} must contain a JSON object")
    return value, payload


def _write_new_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    """Atomically create immutable metadata and never overwrite evidence."""
    path = Path(path)
    _regular_directory(path.parent, f"{label} parent")
    _require(not path.exists() and not path.is_symlink(),
             f"F008 refuses to replace {label}")
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         indent=2).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _require(not path.exists() and not path.is_symlink(),
                 f"F008 refuses to replace {label}")
        temporary.replace(path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _scalar_string(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    _require(array.shape == (), f"F008 {label} must be scalar")
    item = array.item()
    _require(isinstance(item, (str, np.str_)), f"F008 {label} must be a string")
    return str(item)


def _scalar_bool(value: np.ndarray, label: str) -> bool:
    array = np.asarray(value)
    _require(array.shape == () and array.dtype == np.bool_,
             f"F008 {label} must be a bool scalar")
    return bool(array.item())


def _scalar_index(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    _require(array.shape == () and array.dtype.kind in "iu",
             f"F008 {label} must be an integer scalar")
    result = int(array.item())
    _require(result >= 0, f"F008 {label} must be nonnegative")
    return result


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_json(value: object, label: str) -> Any:
    try:
        return _clone_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"F008 {label} must be finite JSON") from error


def _validate_f008_tail_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stable subset of a worker-produced F008 tail identity.

    The worker owns its full evolving identity schema.  The extractor does not
    import it; it verifies the canonical signature over *all* supplied fields
    and the fields that are material to a full-scoring cache.
    """
    _require(isinstance(identity, Mapping), "F008 tail identity must be an object")
    body = {key: value for key, value in identity.items() if key != "signature"}
    required = {
        "schema_version", "f008_signature", "source_f005_signature",
        "f005_source_receipt_sha256", "outer_fold", "arm",
        "embedding_dimension", "checkpoint_scope",
    }
    _require(required.issubset(body) and set(identity) == set(body) | {"signature"},
             "F008 tail identity is missing cache-critical fields")
    _require(identity.get("signature") == canonical_sha256(body),
             "F008 tail identity signature changed")
    for key in ("f008_signature", "source_f005_signature", "f005_source_receipt_sha256"):
        _sha256(body.get(key), f"tail identity {key}")
    _require(type(body.get("outer_fold")) is int and body["outer_fold"] >= 0,
             "F008 tail identity outer fold is invalid")
    arm = body.get("arm")
    _require(isinstance(arm, Mapping) and isinstance(arm.get("id"), str) and arm["id"]
             and isinstance(arm.get("kind"), str) and arm["kind"]
             and type(arm.get("lambda")) in (int, float)
             and not isinstance(arm.get("lambda"), bool)
             and math.isfinite(float(arm["lambda"])) and float(arm["lambda"]) > 0,
             "F008 tail identity arm is invalid")
    _require(body.get("embedding_dimension") == ADVANCED_EMBEDDING_DIMENSION
             and body.get("checkpoint_scope") == "server_only_until_promotion",
             "F008 tail identity endpoint or retention changed")
    return _clone_json(identity)


def _validate_completed_checkpoint_metadata(metadata: Mapping[str, Any],
                                            tail_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate worker-validated final-tail metadata without opening Torch.

    The caller gets this mapping from ``load_authenticated_f008_tail``.  The
    cache layer checks its canonical self-hash and every identity link, while
    leaving tensor deserialization to that worker.
    """
    tail = _validate_f008_tail_identity(tail_identity)
    _require(isinstance(metadata, Mapping), "F008 checkpoint metadata must be an object")
    body = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    required = {
        "format_version", "checkpoint_schema", "stage", "f008_stage",
        "f008_signature", "source_f005_signature", "f005_source_receipt_sha256",
        "arm_signature", "outer_fold", "arm_id", "arm_kind", "oe_lambda",
        "embedding_dimension", "completed_steps", "total_steps",
        "mlflow_upload_allowed", "local_transfer_allowed",
    }
    _require(required.issubset(body) and set(metadata) == set(body) | {"metadata_sha256"},
             "F008 checkpoint metadata is missing cache-critical fields")
    _require(metadata.get("metadata_sha256") == canonical_sha256(body),
             "F008 checkpoint metadata hash changed")
    _require(body.get("format_version") == 1
             and body.get("checkpoint_schema") == "f008-open-set-oe-tail-checkpoint-v1"
             and body.get("stage") == "tail" and body.get("f008_stage") == "open_set_oe_tail"
             and body.get("f008_signature") == tail["f008_signature"]
             and body.get("source_f005_signature") == tail["source_f005_signature"]
             and body.get("f005_source_receipt_sha256") == tail["f005_source_receipt_sha256"]
             and body.get("arm_signature") == tail["signature"]
             and body.get("outer_fold") == tail["outer_fold"]
             and body.get("arm_id") == tail["arm"]["id"]
             and body.get("arm_kind") == tail["arm"]["kind"]
             and body.get("oe_lambda") == tail["arm"]["lambda"]
             and body.get("embedding_dimension") == ADVANCED_EMBEDDING_DIMENSION
             and type(body.get("completed_steps")) is int
             and type(body.get("total_steps")) is int
             and body["completed_steps"] > 0 and body["completed_steps"] == body["total_steps"]
             and body.get("mlflow_upload_allowed") is False
             and body.get("local_transfer_allowed") is False,
             "F008 checkpoint metadata is not the completed declared tail")
    return _clone_json(metadata)


def validated_f008_tail_checkpoint_receipt(
        tail_identity: Mapping[str, Any], *, checkpoint_path: Path,
        checkpoint_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a worker-validated completed checkpoint to its immutable bytes.

    This function intentionally does not deserialize Torch tensors.  The
    caller must obtain ``tail_identity`` and ``checkpoint_metadata`` from the
    F008 worker's authenticated completed-tail loader before constructing this
    receipt.  The extracted cache rehashes ``checkpoint_path`` before any
    callback can run.
    """
    identity = _validate_f008_tail_identity(tail_identity)
    metadata = _validate_completed_checkpoint_metadata(checkpoint_metadata, identity)
    checkpoint = _regular_file(Path(checkpoint_path), "completed F008 checkpoint")
    body = {
        "schema_version": F008_VALIDATED_TAIL_CHECKPOINT_SCHEMA,
        "tail_identity": identity,
        "tail_identity_signature": identity["signature"],
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "checkpoint_metadata_sha256": metadata["metadata_sha256"],
        "completed": True,
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    return {**body, "signature": canonical_sha256(body)}


def validate_f008_tail_checkpoint_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(receipt, Mapping), "F008 checkpoint receipt must be an object")
    required = {
        "schema_version", "tail_identity", "tail_identity_signature", "checkpoint_sha256",
        "checkpoint_metadata", "checkpoint_metadata_sha256", "completed", "server_only",
        "mlflow_upload_allowed", "local_transfer_allowed", "signature",
    }
    _require(set(receipt) == required, "F008 checkpoint receipt schema changed")
    body = {key: receipt[key] for key in required if key != "signature"}
    identity = _validate_f008_tail_identity(body["tail_identity"])
    metadata = _validate_completed_checkpoint_metadata(body["checkpoint_metadata"], identity)
    _require(body["schema_version"] == F008_VALIDATED_TAIL_CHECKPOINT_SCHEMA
             and body["tail_identity_signature"] == identity["signature"]
             and _sha256(body["checkpoint_sha256"], "checkpoint receipt checkpoint hash")
             and body["checkpoint_metadata_sha256"] == metadata["metadata_sha256"]
             and body["completed"] is True and body["server_only"] is True
             and body["mlflow_upload_allowed"] is False
             and body["local_transfer_allowed"] is False
             and receipt["signature"] == canonical_sha256(body),
             "F008 checkpoint receipt identity changed")
    return _clone_json(receipt)


def verify_validated_f008_tail_checkpoint_bytes(checkpoint_path: Path,
                                                checkpoint_receipt: Mapping[str, Any]) -> None:
    receipt = validate_f008_tail_checkpoint_receipt(checkpoint_receipt)
    checkpoint = _regular_file(Path(checkpoint_path), "completed F008 checkpoint")
    _require(_file_sha256(checkpoint) == receipt["checkpoint_sha256"],
             "F008 checkpoint bytes changed after worker validation")


def _manifest_rows(manifest: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, str, str], ...]:
    _require(isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes))
             and len(manifest) > 0,
             "F008 full-scoring manifest must be a nonempty sequence")
    values: list[tuple[int, str, str]] = []
    names: set[str] = set()
    for index, row in enumerate(manifest):
        _require(isinstance(row, Mapping), "F008 full-scoring manifest row is invalid")
        name = _safe_file_name(row.get("audio_file"), "manifest audio filename")
        _require(name not in names, "F008 full-scoring manifest has duplicate audio files")
        names.add(name)
        values.append((index, name, _sha256(row.get("input_sha256"), "manifest input hash")))
    return tuple(values)


def _manifest_valid(manifest: Sequence[Mapping[str, Any]]) -> tuple[bool, ...]:
    return tuple(_truth(row.get("has_nonzero_signal")) for row in manifest)


@dataclass(frozen=True)
class F008AdvancedCachePlan:
    """One immutable full-manifest F008 extraction plan.

    There is intentionally no ``indices`` argument.  A caller can only create
    a cache for every row of the manifest it supplied, preserving its exact
    order for downstream scoring.
    """

    output_directory: Path
    relative_cache_directory: str
    input_rows: tuple[tuple[int, str, str], ...]
    expected_valid: tuple[bool, ...]
    _identity_json: str

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)


def _validate_cache_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(identity, Mapping), "F008 cache identity must be an object")
    required = {
        "schema_version", "tail_identity", "tail_identity_signature", "checkpoint",
        "scope", "indices_sha256", "manifest_inputs_sha256", "expected_valid_sha256",
        "row_count", "embedding_dimension", "inference", "server_only",
        "mlflow_upload_allowed", "local_transfer_allowed", "signature",
    }
    _require(set(identity) == required, "F008 cache identity schema changed")
    body = {key: identity[key] for key in required if key != "signature"}
    tail = _validate_f008_tail_identity(body["tail_identity"])
    checkpoint = validate_f008_tail_checkpoint_receipt(body["checkpoint"])
    _require(body["schema_version"] == F008_ADVANCED_CACHE_IDENTITY_SCHEMA
             and body["tail_identity_signature"] == tail["signature"]
             and checkpoint["tail_identity"] == tail
             and body["scope"] == _SCOPE
             and _sha256(body["indices_sha256"], "cache indices hash")
             and _sha256(body["manifest_inputs_sha256"], "cache manifest-input hash")
             and _sha256(body["expected_valid_sha256"], "cache expected-valid hash")
             and type(body["row_count"]) is int and body["row_count"] > 0
             and body["embedding_dimension"] == ADVANCED_EMBEDDING_DIMENSION
             and body["server_only"] is True
             and body["mlflow_upload_allowed"] is False
             and body["local_transfer_allowed"] is False
             and identity["signature"] == canonical_sha256(body),
             "F008 cache identity changed")
    _finite_json(body["inference"], "cache inference")
    return _clone_json(identity)


def build_f008_advanced_cache_plan(
        tail_identity: Mapping[str, Any], *, output_directory: Path,
        relative_cache_directory: str, checkpoint_receipt: Mapping[str, Any],
        manifest: Sequence[Mapping[str, Any]], inference: object) -> F008AdvancedCachePlan:
    """Build an all-row full-scoring cache plan without audio/model access."""
    tail = _validate_f008_tail_identity(tail_identity)
    receipt = validate_f008_tail_checkpoint_receipt(checkpoint_receipt)
    _require(receipt["tail_identity"] == tail,
             "F008 cache checkpoint receipt belongs to another tail identity")
    output = _regular_directory(Path(output_directory), "cache output root")
    relative = _safe_relative_posix(relative_cache_directory, "cache directory")
    rows = _manifest_rows(manifest)
    valid = _manifest_valid(manifest)
    evidence = [
        {"index": index, "audio_file": name, "input_sha256": digest}
        for index, name, digest in rows
    ]
    body = {
        "schema_version": F008_ADVANCED_CACHE_IDENTITY_SCHEMA,
        "tail_identity": tail,
        "tail_identity_signature": tail["signature"],
        "checkpoint": receipt,
        "scope": _SCOPE,
        "indices_sha256": _array_sha256(np.arange(len(rows), dtype=np.int64), "<i8"),
        "manifest_inputs_sha256": canonical_sha256(evidence),
        "expected_valid_sha256": _array_sha256(np.asarray(valid, dtype=np.bool_), "bool"),
        "row_count": len(rows),
        "embedding_dimension": ADVANCED_EMBEDDING_DIMENSION,
        "inference": _finite_json(inference, "cache inference"),
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    identity = {**body, "signature": canonical_sha256(body)}
    _validate_cache_identity(identity)
    return F008AdvancedCachePlan(
        output_directory=output, relative_cache_directory=relative,
        input_rows=rows, expected_valid=valid,
        _identity_json=canonical(identity).decode("utf-8"),
    )


def _validate_plan(plan: F008AdvancedCachePlan) -> dict[str, Any]:
    _require(isinstance(plan, F008AdvancedCachePlan), "F008 cache plan has the wrong type")
    identity = _validate_cache_identity(plan.identity)
    _regular_directory(plan.output_directory, "cache output root")
    _safe_relative_posix(plan.relative_cache_directory, "cache directory")
    _require(len(plan.input_rows) == identity["row_count"] == len(plan.expected_valid),
             "F008 full-scoring cache plan length changed")
    evidence: list[dict[str, Any]] = []
    for position, ((index, name, digest), valid) in enumerate(
            zip(plan.input_rows, plan.expected_valid, strict=True)):
        _require(index == position and _safe_file_name(name, "plan audio filename") == name
                 and _sha256(digest, "plan input hash") and isinstance(valid, bool),
                 "F008 full-scoring cache plan row is malformed")
        evidence.append({"index": index, "audio_file": name, "input_sha256": digest})
    _require(identity["indices_sha256"]
             == _array_sha256(np.arange(len(plan.input_rows), dtype=np.int64), "<i8")
             and identity["manifest_inputs_sha256"] == canonical_sha256(evidence)
             and identity["expected_valid_sha256"]
             == _array_sha256(np.asarray(plan.expected_valid, dtype=np.bool_), "bool"),
             "F008 cache plan no longer matches its identity")
    return identity


def _cache_directory(plan: F008AdvancedCachePlan, *, create: bool) -> Path:
    _validate_plan(plan)
    return _directory_below(plan.output_directory, plan.relative_cache_directory,
                            "cache", create=create)


def _entry_name(position: int, audio_file: str) -> str:
    digest = hashlib.sha256(audio_file.encode("utf-8")).hexdigest()[:20]
    return f"{position:05d}-{digest}.npz"


def _cache_paths(plan: F008AdvancedCachePlan, *, create: bool) -> tuple[Path, Path, Path, Path]:
    root = _cache_directory(plan, create=create)
    identity, receipt, entries = (root / "cache_identity.json", root / "cache_receipt.json",
                                  root / "entries")
    if create:
        if entries.exists() or entries.is_symlink():
            _regular_directory(entries, "cache entries")
        else:
            entries.mkdir()
            _regular_directory(entries, "cache entries")
    else:
        _regular_directory(entries, "cache entries")
    return root, identity, receipt, entries


def _write_or_verify_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        existing, _ = _read_json(path, "cache identity")
        _require(existing == identity, "F008 stored cache identity differs from this plan")
        _validate_cache_identity(existing)
    else:
        _write_new_json(path, identity, "cache identity")


def _embedding_geometry(vector: object, valid: bool, expected: bool, label: str) -> np.ndarray:
    array = np.asarray(vector)
    _require(array.shape == (ADVANCED_EMBEDDING_DIMENSION,) and array.dtype == np.float32
             and bool(np.isfinite(array).all()),
             f"F008 {label} must be finite float32 [{ADVANCED_EMBEDDING_DIMENSION}]")
    _require(valid == expected, f"F008 {label} validity differs from the manifest")
    if valid:
        _require(np.isclose(np.linalg.norm(array), 1.0, atol=1e-5),
                 f"F008 valid {label} must be unit-normalized")
    else:
        _require(not np.any(array), f"F008 invalid {label} must be exactly zero")
    return array.copy()


def _load_f008_entry(path: Path, identity: Mapping[str, Any], *, position: int,
                     index: int, audio_file: str, input_sha256: str,
                     expected_valid: bool) -> tuple[np.ndarray, bool, str]:
    path = _regular_file(path, "cache entry")
    try:
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {
                "embedding", "valid", "identity_signature", "index", "audio_file",
                "input_sha256", "embedding_sha256",
            }, "F008 cache entry schema changed")
            _require(_scalar_string(saved["identity_signature"], "cache identity signature")
                     == identity["signature"]
                     and _scalar_index(saved["index"], "cache index") == index
                     and _scalar_string(saved["audio_file"], "cache audio filename") == audio_file
                     and _scalar_string(saved["input_sha256"], "cache input hash") == input_sha256,
                     "F008 cache entry identity changed")
            valid = _scalar_bool(saved["valid"], "cache valid")
            vector = saved["embedding"].copy()
            embedding_sha = _scalar_string(saved["embedding_sha256"], "cache embedding hash")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("F008 cache entry cannot be read") from error
    array = _embedding_geometry(vector, valid, expected_valid,
                                f"cache embedding row {position}")
    _require(embedding_sha == _array_sha256(array, "<f4"),
             "F008 cache entry embedding hash changed")
    return array, valid, embedding_sha


def _result_records(plan: F008AdvancedCachePlan, entries_directory: Path,
                    identity: Mapping[str, Any], *, verify_cache_sha256: bool) -> tuple[
                        list[dict[str, Any]], list[np.ndarray], list[bool]]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    validity: list[bool] = []
    for position, ((index, name, input_hash), expected) in enumerate(
            zip(plan.input_rows, plan.expected_valid, strict=True)):
        filename = _entry_name(position, name)
        target = _regular_file(entries_directory / filename, "cache entry")
        vector, valid, embedding_sha = _load_f008_entry(
            target, identity, position=position, index=index, audio_file=name,
            input_sha256=input_hash, expected_valid=expected,
        )
        byte_count = target.stat().st_size
        _require(byte_count > 0, "F008 cache entry is empty")
        records.append({
            "index": index, "audio_file": name, "input_sha256": input_hash,
            "cache_file": filename, "embedding_sha256": embedding_sha,
            "cache_sha256": _file_sha256(target) if verify_cache_sha256 else None,
            "bytes": byte_count, "valid": valid,
        })
        vectors.append(vector); validity.append(valid)
    return records, vectors, validity


def _receipt_body(plan: F008AdvancedCachePlan, identity: Mapping[str, Any],
                  records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": F008_ADVANCED_CACHE_RECEIPT_SCHEMA,
        "identity": _clone_json(identity),
        "file_count": len(records),
        "files": [_clone_json(record) for record in records],
        "completed": True,
        "server_only": True,
        "embedding_artifacts_mlflow_uploaded": False,
        "local_transfer_allowed": False,
    }


def _validate_receipt(receipt: Mapping[str, Any], plan: F008AdvancedCachePlan,
                      identity: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "identity", "file_count", "files", "completed", "server_only",
        "embedding_artifacts_mlflow_uploaded", "local_transfer_allowed", "receipt_sha256",
    }
    _require(isinstance(receipt, Mapping) and set(receipt) == required,
             "F008 cache receipt schema changed")
    body = {key: receipt[key] for key in required if key != "receipt_sha256"}
    _require(body["schema_version"] == F008_ADVANCED_CACHE_RECEIPT_SCHEMA
             and body["identity"] == identity
             and type(body["file_count"]) is int and body["file_count"] == len(plan.input_rows)
             and isinstance(body["files"], list) and len(body["files"]) == len(plan.input_rows)
             and body["completed"] is True and body["server_only"] is True
             and body["embedding_artifacts_mlflow_uploaded"] is False
             and body["local_transfer_allowed"] is False
             and receipt["receipt_sha256"] == canonical_sha256(body),
             "F008 cache receipt identity changed")
    return _clone_json(receipt)


def _recover_partial(target: Path, loader: Callable[[Path], tuple[np.ndarray, bool, str]]) -> None:
    partial = target.with_suffix(target.suffix + ".partial")
    if not partial.exists() and not partial.is_symlink():
        return
    _regular_file(partial, "cache partial")
    _require(not target.exists() and not target.is_symlink(),
             "F008 cache has both final and staged entry")
    loader(partial)
    partial.replace(target)


def _write_f008_entry(target: Path, identity: Mapping[str, Any], *, index: int,
                      audio_file: str, input_sha256: str, vector: np.ndarray,
                      valid: bool) -> None:
    _require(not target.exists() and not target.is_symlink(),
             "F008 refuses to replace a cache entry")
    partial = target.with_suffix(target.suffix + ".partial")
    _require(not partial.exists() and not partial.is_symlink(),
             "F008 cache partial must be recovered before writing")
    digest = _array_sha256(vector, "<f4")
    try:
        with partial.open("xb") as stream:
            np.savez_compressed(
                stream, embedding=vector, valid=np.bool_(valid),
                identity_signature=identity["signature"], index=np.int64(index),
                audio_file=audio_file, input_sha256=input_sha256,
                embedding_sha256=digest,
            )
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(target)
    finally:
        if partial.exists() and not partial.is_symlink():
            partial.unlink()


def _extractor_result(value: object, label: str) -> tuple[object, bool]:
    if isinstance(value, tuple) and len(value) == 2:
        vector, valid = value
    elif isinstance(value, Mapping) and set(value) == {"embedding", "valid"}:
        vector, valid = value["embedding"], value["valid"]
    else:
        raise ValueError("F008 extractor must return (embedding, valid) or {'embedding', 'valid'}")
    _require(isinstance(valid, (bool, np.bool_)), f"F008 {label} valid flag must be bool")
    return vector, bool(valid)


def load_f008_advanced_cache(plan: F008AdvancedCachePlan) -> dict[str, Any]:
    """Purely authenticate one completed F008 cache; no callback is invoked."""
    identity = _validate_plan(plan)
    root, identity_path, receipt_path, entries = _cache_paths(plan, create=False)
    saved_identity, _ = _read_json(identity_path, "cache identity")
    _require(saved_identity == identity, "F008 stored cache identity differs from this plan")
    _validate_cache_identity(saved_identity)
    receipt, _ = _read_json(receipt_path, "cache receipt")
    receipt = _validate_receipt(receipt, plan, identity)
    records, vectors, valid = _result_records(plan, entries, identity, verify_cache_sha256=True)
    _require(receipt["files"] == records,
             "F008 cache receipt does not authenticate its cache entries")
    return {
        "kind": "f008_advanced_cache",
        "indices": np.arange(len(plan.input_rows), dtype=np.int64),
        "embeddings": np.asarray(vectors, dtype=np.float32),
        "valid": np.asarray(valid, dtype=np.bool_),
        "identity": identity,
        "receipt": receipt,
        "cache_directory": str(root),
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }


def extract_or_load_f008_advanced_cache(
        plan: F008AdvancedCachePlan, *, checkpoint_path: Path,
        extractor: Callable[[int, Mapping[str, str]], object],
        progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Extract/resume every manifest row using an already-authenticated model.

    This cache layer does not locate or open raw audio.  The callback owns that
    server-local step and sees only the authenticated row identifier/hash.  A
    changed checkpoint, source hash, vector geometry, or cache file fails
    closed before the cache can be marked complete.
    """
    _require(callable(extractor), "F008 cache extractor must be callable")
    _require(progress is None or callable(progress), "F008 cache progress callback must be callable")
    identity = _validate_plan(plan)
    verify_validated_f008_tail_checkpoint_bytes(checkpoint_path, identity["checkpoint"])
    _, identity_path, receipt_path, entries = _cache_paths(plan, create=True)
    _write_or_verify_identity(identity_path, identity)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_f008_advanced_cache(plan)

    total = len(plan.input_rows)
    for position, ((index, name, input_hash), expected) in enumerate(
            zip(plan.input_rows, plan.expected_valid, strict=True)):
        target = entries / _entry_name(position, name)
        loader = lambda candidate, p=position, i=index, n=name, h=input_hash, e=expected: _load_f008_entry(
            candidate, identity, position=p, index=i, audio_file=n, input_sha256=h,
            expected_valid=e,
        )
        _recover_partial(target, loader)
        if target.exists() or target.is_symlink():
            loader(target)
        else:
            supplied, valid = _extractor_result(
                extractor(index, {"audio_file": name, "input_sha256": input_hash}),
                f"extractor result for row {position}",
            )
            _require(valid == expected,
                     "F008 extractor validity differs from the authenticated manifest")
            vector = (_embedding_geometry(supplied, True, True, f"extractor row {position}")
                      if valid else np.zeros((ADVANCED_EMBEDDING_DIMENSION,), dtype=np.float32))
            _write_f008_entry(target, identity, index=index, audio_file=name,
                              input_sha256=input_hash, vector=vector, valid=valid)
            loader(target)
        if progress is not None:
            progress(position + 1, total)

    records, _vectors, _valid = _result_records(plan, entries, identity, verify_cache_sha256=True)
    body = _receipt_body(plan, identity, records)
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _write_new_json(receipt_path, receipt, "cache receipt")
    return load_f008_advanced_cache(plan)


def _f005_source_fold(tail_identity: Mapping[str, Any], source_receipt: Mapping[str, Any], *,
                      f005_run_directory: Path, outer_fold: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    tail = _validate_f008_tail_identity(tail_identity)
    _require(type(outer_fold) is int and outer_fold == tail["outer_fold"],
             "F008 reused-control outer fold differs from the F008 tail")
    _require(isinstance(source_receipt, Mapping)
             and source_receipt.get("schema_version") == "f007-f005-source-receipt-v1",
             "F008 reused-control F005 source receipt schema is invalid")
    _require(canonical_sha256(source_receipt) == tail["f005_source_receipt_sha256"],
             "F008 reused-control source receipt differs from the F008 tail")
    root = _regular_directory(Path(f005_run_directory), "F005 source run")
    recorded_root = source_receipt.get("source_run_directory")
    _require(isinstance(recorded_root, str) and Path(recorded_root).resolve() == root.resolve(),
             "F008 reused-control source directory differs from its receipt")
    state = source_receipt.get("experiment_state")
    _require(isinstance(state, Mapping) and state.get("status") == "complete"
             and state.get("experiment_signature") == tail["source_f005_signature"],
             "F008 reused-control F005 source state differs from the tail")
    rows = source_receipt.get("folds")
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
             "F008 reused-control F005 source folds are missing")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("outer_fold") == outer_fold]
    _require(len(selected) == 1, "F008 reused-control F005 source fold is missing or duplicated")
    fold = dict(selected[0])
    control, full = fold.get("control_tail"), fold.get("full_scoring_control")
    _require(isinstance(control, Mapping) and isinstance(full, Mapping),
             "F008 reused-control source fold is malformed")
    checkpoint = control.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "F008 reused-control control checkpoint is malformed")
    _sha256(checkpoint.get("sha256"), "F005 control checkpoint hash")
    _require(isinstance(full.get("identity"), Mapping) and isinstance(full.get("receipt"), Mapping),
             "F008 reused-control cache declaration is malformed")
    return fold, dict(full), root


def _validate_f005_identity(identity: Mapping[str, Any], *, source_signature: str,
                            outer_fold: int, checkpoint_sha256: str,
                            expected_indices_sha256: str) -> dict[str, Any]:
    required = {
        "schema_version", "experiment_signature", "outer_fold", "arm_id", "scope",
        "indices_sha256", "checkpoint_sha256", "checkpoint_metadata_sha256",
        "embedding_dimension", "inference", "server_only", "mlflow_upload_allowed", "signature",
    }
    _require(isinstance(identity, Mapping) and set(identity) == required,
             "F008 reused-control F005 identity schema changed")
    body = {key: identity[key] for key in required if key != "signature"}
    _require(body["schema_version"] == 1
             and body["experiment_signature"] == source_signature
             and body["outer_fold"] == outer_fold and body["arm_id"] == "control"
             and body["scope"] == _SCOPE and body["indices_sha256"] == expected_indices_sha256
             and body["checkpoint_sha256"] == checkpoint_sha256
             and _sha256(body["checkpoint_metadata_sha256"], "F005 control metadata hash")
             and body["embedding_dimension"] == ADVANCED_EMBEDDING_DIMENSION
             and body["server_only"] is True and body["mlflow_upload_allowed"] is False
             and identity["signature"] == canonical_sha256(body),
             "F008 reused-control F005 identity changed")
    _finite_json(body["inference"], "F005 control inference")
    return _clone_json(identity)


def _load_f005_entry(path: Path, identity: Mapping[str, Any], *, audio_file: str,
                     input_sha256: str, expected_valid: bool, position: int) -> tuple[np.ndarray, bool]:
    path = _regular_file(path, "reused F005 cache entry")
    try:
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {
                "embedding", "valid", "signature", "audio_file", "audio_sha256",
            }, "F008 reused-control F005 cache entry schema changed")
            _require(_scalar_string(saved["signature"], "F005 cache signature") == identity["signature"]
                     and _scalar_string(saved["audio_file"], "F005 cache audio filename") == audio_file
                     and _scalar_string(saved["audio_sha256"], "F005 cache audio hash") == input_sha256,
                     "F008 reused-control F005 cache entry identity changed")
            valid = _scalar_bool(saved["valid"], "F005 cache valid")
            vector = saved["embedding"].copy()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("F008 reused-control F005 cache entry cannot be read") from error
    return _embedding_geometry(vector, valid, expected_valid,
                               f"reused F005 embedding row {position}"), valid


def load_reused_f005_control_cache(
        tail_identity: Mapping[str, Any], source_receipt: Mapping[str, Any], *,
        f005_run_directory: Path, manifest: Sequence[Mapping[str, Any]],
        outer_fold: int) -> dict[str, Any]:
    """Authenticate/load F005's full-scoring control cache in place.

    No F005 embeddings, raw audio, checkpoints, or cache files are copied.
    The returned arrays remain process-local on the server and every file is
    checked against the source receipt, cache identity, manifest order, input
    hash, vector hash, and cache-file hash.
    """
    tail = _validate_f008_tail_identity(tail_identity)
    fold, full, root = _f005_source_fold(
        tail, source_receipt, f005_run_directory=f005_run_directory, outer_fold=outer_fold,
    )
    identity_declared, receipt_declared = full["identity"], full["receipt"]
    _require(set(identity_declared) == {"path", "sha256", "signature"}
             and set(receipt_declared) == {"path", "sha256", "receipt_sha256", "file_count"},
             "F008 reused-control source cache declaration changed")
    identity_path = _confined_regular_file(root, identity_declared["path"], "F005 cache identity")
    receipt_path = _confined_regular_file(root, receipt_declared["path"], "F005 cache receipt")
    identity_payload, receipt_payload = identity_path.read_bytes(), receipt_path.read_bytes()
    _require(_file_sha256(identity_path) == _sha256(identity_declared["sha256"], "F005 identity file hash")
             and _file_sha256(receipt_path) == _sha256(receipt_declared["sha256"], "F005 receipt file hash"),
             "F008 reused-control F005 identity or receipt bytes changed")
    identity, _ = _read_json(identity_path, "F005 cache identity")
    receipt, _ = _read_json(receipt_path, "F005 cache receipt")
    _require(identity.get("signature") == identity_declared["signature"],
             "F008 reused-control declared F005 identity signature changed")

    rows = _manifest_rows(manifest)
    valid_mask = _manifest_valid(manifest)
    expected_indices_sha256 = _array_sha256(np.arange(len(rows), dtype=np.int64), "<i8")
    control_checkpoint = fold["control_tail"]["checkpoint"]
    validated_identity = _validate_f005_identity(
        identity, source_signature=tail["source_f005_signature"], outer_fold=outer_fold,
        checkpoint_sha256=_sha256(control_checkpoint["sha256"], "F005 control checkpoint hash"),
        expected_indices_sha256=expected_indices_sha256,
    )
    required_receipt = {
        "schema_version", "identity", "file_count", "files", "completed", "server_only",
        "embedding_artifacts_mlflow_uploaded", "receipt_sha256",
    }
    _require(isinstance(receipt, Mapping) and set(receipt) == required_receipt,
             "F008 reused-control F005 receipt schema changed")
    body = {key: receipt[key] for key in required_receipt if key != "receipt_sha256"}
    _require(body["schema_version"] == 1 and body["identity"] == validated_identity
             and type(body["file_count"]) is int and body["file_count"] == len(rows)
             and body["file_count"] == receipt_declared["file_count"]
             and isinstance(body["files"], list) and len(body["files"]) == len(rows)
             and body["completed"] is True and body["server_only"] is True
             and body["embedding_artifacts_mlflow_uploaded"] is False
             and receipt["receipt_sha256"] == canonical_sha256(body)
             and receipt["receipt_sha256"]
             == _sha256(receipt_declared["receipt_sha256"], "F005 receipt self hash"),
             "F008 reused-control F005 receipt identity changed")
    cache_directory = _regular_directory(identity_path.parent / "embedding_cache", "F005 cache entries")
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    validity: list[bool] = []
    for position, ((index, name, input_hash), expected) in enumerate(
            zip(rows, valid_mask, strict=True)):
        record = receipt["files"][position]
        required_record = {
            "audio_file", "audio_sha256", "cache_file", "cache_sha256", "bytes", "valid",
        }
        _require(isinstance(record, Mapping) and set(record) == required_record,
                 "F008 reused-control F005 file record schema changed")
        filename = _safe_file_name(record["cache_file"], "F005 cache file")
        target = _regular_file(cache_directory / filename, "F005 cache entry")
        _require(record["audio_file"] == name and record["audio_sha256"] == input_hash
                 and type(record["bytes"]) is int and record["bytes"] == target.stat().st_size
                 and _file_sha256(target) == _sha256(record["cache_sha256"], "F005 cache file hash")
                 and isinstance(record["valid"], bool),
                 "F008 reused-control F005 file record changed")
        vector, valid = _load_f005_entry(target, validated_identity, audio_file=name,
                                         input_sha256=input_hash, expected_valid=expected,
                                         position=position)
        _require(valid == record["valid"], "F008 reused-control F005 valid flag changed")
        records.append(dict(record)); vectors.append(vector); validity.append(valid)
    _require(receipt["files"] == records, "F008 reused-control F005 records changed during load")
    return {
        "kind": F008_REUSED_F005_CONTROL_SCHEMA,
        "outer_fold": outer_fold,
        "indices": np.arange(len(rows), dtype=np.int64),
        "embeddings": np.asarray(vectors, dtype=np.float32),
        "valid": np.asarray(validity, dtype=np.bool_),
        "source_f005_receipt_sha256": tail["f005_source_receipt_sha256"],
        "identity": validated_identity,
        "receipt": _clone_json(receipt),
        "source_identity_file_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "source_receipt_file_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
