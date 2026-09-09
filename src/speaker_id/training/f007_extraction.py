"""Server-only, resumable advanced-embedding caches for F007.

This module deliberately owns cache identity and bytes only.  It does not
import Torch, a CAM++ loader, an audio decoder, MLflow, or an F005/F007
trainer.  A caller supplies an already-loaded extractor callback after the
F007 worker has validated the tail checkpoint.  The pure load/validation
functions never invoke that callback and never touch a raw source file.

There are two deliberately separate paths:

* :func:`extract_or_load_f007_advanced_cache` creates or resumes an F007-tail
  cache.  Its identity binds the F007 contract, the authenticated F005 source
  receipt, checkpoint evidence, ordered row scope, manifest input hashes and
  inference recipe.
* :func:`load_reused_f005_control_cache` reads the completed F005 control
  cache in place.  It never copies embeddings into the F007 run directory.

Embedding files, checkpoints, and raw inputs remain server-only.  The only
portable evidence emitted by this module is JSON identity/receipt metadata;
the module exposes no MLflow or local-transfer operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np


F007_ADVANCED_CACHE_IDENTITY_SCHEMA = "f007-advanced-embedding-cache-identity-v1"
F007_ADVANCED_CACHE_RECEIPT_SCHEMA = "f007-advanced-embedding-cache-receipt-v1"
F007_VALIDATED_CHECKPOINT_SCHEMA = "f007-validated-tail-checkpoint-v1"
F007_REUSED_F005_CONTROL_SCHEMA = "f007-reused-f005-control-cache-v1"
ADVANCED_EMBEDDING_DIMENSION = 192

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SCOPE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def canonical(value: object) -> bytes:
    """Encode finite JSON in the canonical form used by all F007 receipts."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value),
        f"F007 {label} must be a lowercase SHA-256",
    )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: object) -> bool:
    """Match the project manifest's stable serialized boolean convention."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def _clone_json(value: object) -> Any:
    return json.loads(canonical(value).decode("utf-8"))


def _safe_relative_posix(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"F007 {label} is missing")
    parsed = PurePosixPath(value)
    _require(
        not parsed.is_absolute() and parsed.as_posix() == value
        and "." not in parsed.parts and ".." not in parsed.parts
        and "\\" not in value and ":" not in value,
        f"F007 {label} must be a safe relative POSIX path",
    )
    return value


def _safe_file_name(value: object, label: str) -> str:
    value = _safe_relative_posix(value, label)
    _require("/" not in value, f"F007 {label} must name one file")
    return value


def _scope(value: object) -> str:
    _require(isinstance(value, str) and _SCOPE_PATTERN.fullmatch(value) is not None,
             "F007 cache scope must be a simple stable identifier")
    return value


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"F007 {label} must be a regular non-symlink file")
    return path


def _regular_directory(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_dir() and not path.is_symlink(),
             f"F007 {label} must be a directory and may not be a symlink")
    return path


def _directory_below(root: Path, relative: str, label: str, *, create: bool) -> Path:
    """Return one non-symlink directory safely below an existing output root."""
    current = _regular_directory(Path(root), f"{label} output root")
    for part in PurePosixPath(relative).parts:
        candidate = current / part
        if candidate.exists() or candidate.is_symlink():
            _regular_directory(candidate, label)
        else:
            _require(create, f"F007 {label} directory is missing")
            candidate.mkdir()
            _regular_directory(candidate, label)
        current = candidate
    return current


def _confined_regular_file(root: Path, relative: object, label: str) -> Path:
    """Open a declared source file without accepting traversal or symlinks."""
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
        raise ValueError(f"F007 {label} must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), f"F007 {label} must contain a JSON object")
    return value, payload


def _write_new_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    """Atomically create immutable metadata; never overwrite an existing receipt."""
    path = Path(path)
    _regular_directory(path.parent, f"{label} parent")
    _require(not path.exists() and not path.is_symlink(),
             f"F007 refuses to replace {label}")
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _require(not path.exists() and not path.is_symlink(),
                 f"F007 refuses to replace {label}")
        temporary.replace(path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _scalar_string(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    _require(array.shape == (), f"F007 {label} must be scalar")
    item = array.item()
    _require(isinstance(item, str), f"F007 {label} must be a string")
    return item


def _scalar_bool(value: np.ndarray, label: str) -> bool:
    array = np.asarray(value)
    _require(array.shape == () and array.dtype == np.bool_,
             f"F007 {label} must be a bool scalar")
    return bool(array.item())


def _scalar_index(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    _require(array.shape == () and array.dtype.kind in "iu",
             f"F007 {label} must be an integer scalar")
    result = int(array.item())
    _require(result >= 0, f"F007 {label} must be nonnegative")
    return result


def _normalise_contract_binding(contract: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...], tuple[str, ...]]:
    """Read only the immutable F007 provenance needed by cache identity."""
    _require(isinstance(contract, Mapping), "F007 cache requires a contract object")
    signature = _sha256(contract.get("signature"), "contract signature")
    source = contract.get("source_f005_receipt")
    _require(isinstance(source, Mapping), "F007 cache requires an F005 source receipt")
    source_sha = _sha256(contract.get("source_f005_receipt_sha256"), "source F005 receipt hash")
    _require(source_sha == canonical_sha256(source),
             "F007 cache source F005 receipt hash does not match its receipt")
    fold_values = contract.get("fold_ids")
    arm_values = contract.get("arm_ids")
    _require(isinstance(fold_values, Sequence) and not isinstance(fold_values, (str, bytes)),
             "F007 cache contract fold IDs are missing")
    _require(isinstance(arm_values, Sequence) and not isinstance(arm_values, (str, bytes)),
             "F007 cache contract arm IDs are missing")
    folds = tuple(int(value) for value in fold_values)
    _require(
        len(folds) == len(set(folds)) and all(type(value) is int for value in fold_values),
        "F007 cache contract fold IDs are malformed",
    )
    arms = tuple(arm_values)
    _require(
        len(arms) == len(set(arms)) and all(isinstance(value, str) and value for value in arms),
        "F007 cache contract arm IDs are malformed",
    )
    return signature, source_sha, folds, arms


def _manifest_inputs(contract: Mapping[str, Any], indices: Sequence[int]) -> tuple[tuple[int, str, str], ...]:
    manifest = contract.get("manifest")
    _require(isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)),
             "F007 cache contract manifest is missing")
    normalized: list[tuple[int, str, str]] = []
    for index in indices:
        _require(type(index) is int and 0 <= index < len(manifest),
                 "F007 cache index is outside the manifest")
        row = manifest[index]
        _require(isinstance(row, Mapping), "F007 cache manifest row is malformed")
        name = _safe_relative_posix(row.get("audio_file"), "manifest audio filename")
        digest = _sha256(row.get("input_sha256"), "manifest input hash")
        normalized.append((index, name, digest))
    _require(len(normalized) == len({index for index, _, _ in normalized}),
             "F007 cache indices must be unique")
    return tuple(normalized)


def _normalise_indices(indices: object, row_count: int) -> tuple[int, ...]:
    raw = np.asarray(indices)
    _require(raw.ndim == 1 and raw.dtype.kind in "iu" and len(raw) > 0,
             "F007 cache indices must be one nonempty integer vector")
    values = tuple(int(value) for value in raw.tolist())
    _require(len(values) == len(set(values)) and all(0 <= value < row_count for value in values),
             "F007 cache indices are duplicated or outside the manifest")
    return values


def _normalise_expected_valid(expected_valid: object, row_count: int,
                              indices: tuple[int, ...]) -> tuple[bool, ...]:
    if isinstance(expected_valid, np.ndarray):
        _require(expected_valid.ndim == 1 and expected_valid.dtype == np.bool_,
                 "F007 cache expected-valid ndarray must be one-dimensional bool")
        values = expected_valid.tolist()
    else:
        _require(isinstance(expected_valid, Sequence) and not isinstance(expected_valid, (str, bytes)),
                 "F007 cache expected-valid mask must be a sequence")
        values = list(expected_valid)
    _require(len(values) == row_count,
             "F007 cache expected-valid mask must align to the full manifest")
    _require(all(isinstance(value, (bool, np.bool_)) for value in values),
             "F007 cache expected-valid mask must contain bool values")
    return tuple(bool(values[index]) for index in indices)


def _checkpoint_body(contract: Mapping[str, Any], *, outer_fold: int, arm_id: str,
                     checkpoint_sha256: object, checkpoint_metadata_sha256: object,
                     validation_receipt_sha256: object) -> dict[str, Any]:
    signature, source_sha, folds, arms = _normalise_contract_binding(contract)
    _require(type(outer_fold) is int and outer_fold in folds,
             "F007 checkpoint outer fold is not configured")
    _require(isinstance(arm_id, str) and arm_id in arms,
             "F007 checkpoint arm is not configured")
    return {
        "schema_version": F007_VALIDATED_CHECKPOINT_SCHEMA,
        "experiment_signature": signature,
        "source_f005_receipt_sha256": source_sha,
        "outer_fold": outer_fold,
        "arm_id": arm_id,
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint hash"),
        "checkpoint_metadata_sha256": _sha256(
            checkpoint_metadata_sha256, "checkpoint metadata hash"),
        "validation_receipt_sha256": _sha256(
            validation_receipt_sha256, "checkpoint validation receipt hash"),
        "validated": True,
    }


def validated_tail_checkpoint_receipt(
        contract: Mapping[str, Any], *, outer_fold: int, arm_id: str,
        checkpoint_path: Path, checkpoint_metadata_sha256: object,
        validation_receipt_sha256: object) -> dict[str, Any]:
    """Bind a regular, already worker-validated F007 checkpoint to its bytes.

    ``validation_receipt_sha256`` must come from the worker's prior structural
    checkpoint validation report.  This function intentionally hashes bytes
    but never imports a model loader or inspects model tensors.
    """
    path = _regular_file(Path(checkpoint_path), "validated tail checkpoint")
    body = _checkpoint_body(
        contract, outer_fold=outer_fold, arm_id=arm_id,
        checkpoint_sha256=_file_sha256(path),
        checkpoint_metadata_sha256=checkpoint_metadata_sha256,
        validation_receipt_sha256=validation_receipt_sha256,
    )
    return {**body, "signature": canonical_sha256(body)}


def validate_validated_tail_checkpoint_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable checkpoint evidence without opening a checkpoint."""
    _require(isinstance(receipt, Mapping), "F007 checkpoint receipt must be an object")
    required = {
        "schema_version", "experiment_signature", "source_f005_receipt_sha256",
        "outer_fold", "arm_id", "checkpoint_sha256", "checkpoint_metadata_sha256",
        "validation_receipt_sha256", "validated", "signature",
    }
    _require(set(receipt) == required, "F007 checkpoint receipt schema changed")
    body = {key: receipt[key] for key in required if key != "signature"}
    _require(
        body["schema_version"] == F007_VALIDATED_CHECKPOINT_SCHEMA
        and _sha256(body["experiment_signature"], "checkpoint experiment signature")
        and _sha256(body["source_f005_receipt_sha256"], "checkpoint source receipt hash")
        and type(body["outer_fold"]) is int
        and isinstance(body["arm_id"], str) and bool(body["arm_id"])
        and _sha256(body["checkpoint_sha256"], "checkpoint hash")
        and _sha256(body["checkpoint_metadata_sha256"], "checkpoint metadata hash")
        and _sha256(body["validation_receipt_sha256"], "checkpoint validation receipt hash")
        and body["validated"] is True
        and receipt["signature"] == canonical_sha256(body),
        "F007 checkpoint receipt identity changed",
    )
    return _clone_json(receipt)


def verify_validated_tail_checkpoint_bytes(checkpoint_path: Path,
                                            checkpoint_receipt: Mapping[str, Any]) -> None:
    """Confirm a regular checkpoint still has the bytes named by its receipt."""
    receipt = validate_validated_tail_checkpoint_receipt(checkpoint_receipt)
    path = _regular_file(Path(checkpoint_path), "validated tail checkpoint")
    _require(_file_sha256(path) == receipt["checkpoint_sha256"],
             "F007 checkpoint bytes changed after worker validation")


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(identity, Mapping), "F007 cache identity must be an object")
    required = {
        "schema_version", "experiment_signature", "source_f005_receipt_sha256",
        "outer_fold", "arm_id", "scope", "indices_sha256",
        "manifest_inputs_sha256", "expected_valid_sha256", "row_count",
        "checkpoint", "embedding_dimension", "inference", "server_only",
        "mlflow_upload_allowed", "local_transfer_allowed", "signature",
    }
    _require(set(identity) == required, "F007 cache identity schema changed")
    body = {key: identity[key] for key in required if key != "signature"}
    receipt = validate_validated_tail_checkpoint_receipt(body["checkpoint"])
    _require(
        body["schema_version"] == F007_ADVANCED_CACHE_IDENTITY_SCHEMA
        and _sha256(body["experiment_signature"], "cache experiment signature")
        and _sha256(body["source_f005_receipt_sha256"], "cache source receipt hash")
        and type(body["outer_fold"]) is int
        and isinstance(body["arm_id"], str) and bool(body["arm_id"])
        and _scope(body["scope"])
        and _sha256(body["indices_sha256"], "cache indices hash")
        and _sha256(body["manifest_inputs_sha256"], "cache manifest-input hash")
        and _sha256(body["expected_valid_sha256"], "cache expected-valid hash")
        and type(body["row_count"]) is int and body["row_count"] > 0
        and body["embedding_dimension"] == ADVANCED_EMBEDDING_DIMENSION
        and body["server_only"] is True
        and body["mlflow_upload_allowed"] is False
        and body["local_transfer_allowed"] is False
        and receipt["experiment_signature"] == body["experiment_signature"]
        and receipt["source_f005_receipt_sha256"] == body["source_f005_receipt_sha256"]
        and receipt["outer_fold"] == body["outer_fold"]
        and receipt["arm_id"] == body["arm_id"]
        and identity["signature"] == canonical_sha256(body),
        "F007 cache identity changed",
    )
    try:
        canonical(body["inference"])
    except (TypeError, ValueError) as error:
        raise ValueError("F007 cache inference recipe must be finite JSON") from error
    return _clone_json(identity)


@dataclass(frozen=True)
class F007AdvancedCachePlan:
    """Immutable in-memory description of one ordered F007 cache scope."""

    output_directory: Path
    relative_cache_directory: str
    indices: tuple[int, ...]
    input_rows: tuple[tuple[int, str, str], ...]
    expected_valid: tuple[bool, ...]
    _identity_json: str

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)


def build_f007_advanced_cache_plan(
        contract: Mapping[str, Any], *, output_directory: Path,
        relative_cache_directory: str, outer_fold: int, arm_id: str,
        checkpoint_receipt: Mapping[str, Any], indices: object, scope: str,
        expected_valid: Sequence[bool], inference: object) -> F007AdvancedCachePlan:
    """Build a cache plan without reading a checkpoint, cache, model, or input file."""
    signature, source_sha, folds, arms = _normalise_contract_binding(contract)
    _require(type(outer_fold) is int and outer_fold in folds,
             "F007 cache outer fold is not configured")
    _require(isinstance(arm_id, str) and arm_id in arms,
             "F007 cache arm is not configured")
    scope = _scope(scope)
    output = _regular_directory(Path(output_directory), "cache output root")
    relative = _safe_relative_posix(relative_cache_directory, "cache directory")
    manifest = contract.get("manifest")
    _require(isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)),
             "F007 cache contract manifest is missing")
    ordered_indices = _normalise_indices(indices, len(manifest))
    inputs = _manifest_inputs(contract, ordered_indices)
    valid = _normalise_expected_valid(expected_valid, len(manifest), ordered_indices)
    checkpoint = validate_validated_tail_checkpoint_receipt(checkpoint_receipt)
    _require(
        checkpoint["experiment_signature"] == signature
        and checkpoint["source_f005_receipt_sha256"] == source_sha
        and checkpoint["outer_fold"] == outer_fold
        and checkpoint["arm_id"] == arm_id,
        "F007 cache checkpoint receipt is not bound to this contract/fold/arm",
    )
    input_evidence = [
        {"index": index, "audio_file": name, "input_sha256": digest}
        for index, name, digest in inputs
    ]
    try:
        normalized_inference = _clone_json(inference)
    except (TypeError, ValueError) as error:
        raise ValueError("F007 cache inference recipe must be finite JSON") from error
    body = {
        "schema_version": F007_ADVANCED_CACHE_IDENTITY_SCHEMA,
        "experiment_signature": signature,
        "source_f005_receipt_sha256": source_sha,
        "outer_fold": outer_fold,
        "arm_id": arm_id,
        "scope": scope,
        "indices_sha256": _array_sha256(np.asarray(ordered_indices), "<i8"),
        "manifest_inputs_sha256": canonical_sha256(input_evidence),
        "expected_valid_sha256": _array_sha256(np.asarray(valid), "bool"),
        "row_count": len(ordered_indices),
        "checkpoint": checkpoint,
        "embedding_dimension": ADVANCED_EMBEDDING_DIMENSION,
        "inference": normalized_inference,
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
    identity = {**body, "signature": canonical_sha256(body)}
    _validate_identity(identity)
    return F007AdvancedCachePlan(
        output_directory=output,
        relative_cache_directory=relative,
        indices=ordered_indices,
        input_rows=inputs,
        expected_valid=valid,
        _identity_json=canonical(identity).decode("utf-8"),
    )


def _validate_plan(plan: F007AdvancedCachePlan) -> dict[str, Any]:
    _require(isinstance(plan, F007AdvancedCachePlan), "F007 cache plan has the wrong type")
    identity = _validate_identity(plan.identity)
    _regular_directory(plan.output_directory, "cache output root")
    _safe_relative_posix(plan.relative_cache_directory, "cache directory")
    _require(
        len(plan.indices) == identity["row_count"] == len(plan.input_rows) == len(plan.expected_valid)
        and len(plan.indices) == len(set(plan.indices)),
        "F007 cache plan rows are malformed",
    )
    evidence = []
    for index, source, expected in zip(plan.indices, plan.input_rows, plan.expected_valid, strict=True):
        _require(source[0] == index and isinstance(source[1], str) and _sha256(source[2], "plan input hash"),
                 "F007 cache plan input rows are malformed")
        _require(isinstance(expected, bool), "F007 cache plan validity is malformed")
        evidence.append({"index": source[0], "audio_file": source[1], "input_sha256": source[2]})
    _require(
        identity["indices_sha256"] == _array_sha256(np.asarray(plan.indices), "<i8")
        and identity["manifest_inputs_sha256"] == canonical_sha256(evidence)
        and identity["expected_valid_sha256"]
        == _array_sha256(np.asarray(plan.expected_valid), "bool"),
        "F007 cache plan no longer matches its identity",
    )
    return identity


def _cache_directory(plan: F007AdvancedCachePlan, *, create: bool) -> Path:
    _validate_plan(plan)
    return _directory_below(
        plan.output_directory, plan.relative_cache_directory, "cache", create=create,
    )


def _entry_name(position: int, audio_file: str) -> str:
    digest = hashlib.sha256(audio_file.encode("utf-8")).hexdigest()[:20]
    return f"{position:05d}-{digest}.npz"


def _cache_paths(plan: F007AdvancedCachePlan, *, create: bool) -> tuple[Path, Path, Path, Path]:
    root = _cache_directory(plan, create=create)
    identity = root / "cache_identity.json"
    receipt = root / "cache_receipt.json"
    entries = root / "entries"
    if create:
        if entries.exists() or entries.is_symlink():
            _regular_directory(entries, "cache entries")
        else:
            entries.mkdir()
            _regular_directory(entries, "cache entries")
    else:
        _regular_directory(entries, "cache entries")
    return root, identity, receipt, entries


def _write_or_verify_identity(identity_path: Path, identity: Mapping[str, Any]) -> None:
    if identity_path.exists() or identity_path.is_symlink():
        existing, _ = _read_json(identity_path, "cache identity")
        _require(existing == identity, "F007 cache identity differs from this plan")
        _validate_identity(existing)
    else:
        _write_new_json(identity_path, identity, "cache identity")


def _embedding_geometry(vector: object, valid: bool, expected: bool, label: str) -> np.ndarray:
    array = np.asarray(vector)
    _require(
        array.shape == (ADVANCED_EMBEDDING_DIMENSION,) and array.dtype == np.float32
        and bool(np.isfinite(array).all()),
        f"F007 {label} must be finite float32 [{ADVANCED_EMBEDDING_DIMENSION}]",
    )
    _require(valid == expected, f"F007 {label} validity differs from the authenticated manifest")
    if not valid:
        _require(not np.any(array), f"F007 invalid {label} must be exactly zero")
    else:
        _require(np.isclose(np.linalg.norm(array), 1.0, atol=1e-5),
                 f"F007 valid {label} must be unit-normalized")
    return array.copy()


def _load_f007_entry(path: Path, identity: Mapping[str, Any], *, position: int,
                      index: int, audio_file: str, input_sha256: str,
                      expected_valid: bool) -> tuple[np.ndarray, bool]:
    path = _regular_file(path, "cache entry")
    with np.load(path, allow_pickle=False) as saved:
        _require(
            set(saved.files) == {
                "embedding", "valid", "identity_signature", "index", "audio_file", "input_sha256",
            }, "F007 cache entry schema changed")
        _require(
            _scalar_string(saved["identity_signature"], "cache entry identity signature")
            == identity["signature"]
            and _scalar_index(saved["index"], "cache entry index") == index
            and _scalar_string(saved["audio_file"], "cache entry audio filename") == audio_file
            and _scalar_string(saved["input_sha256"], "cache entry input hash") == input_sha256,
            "F007 cache entry identity changed",
        )
        valid = _scalar_bool(saved["valid"], "cache entry valid")
        vector = saved["embedding"].copy()
    return _embedding_geometry(vector, valid, expected_valid, f"cache embedding row {position}"), valid


def _result_records(plan: F007AdvancedCachePlan, entries_directory: Path,
                    identity: Mapping[str, Any], *, verify_cache_sha256: bool) -> tuple[list[dict[str, Any]], list[np.ndarray], list[bool]]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    valid: list[bool] = []
    for position, (index, name, input_hash) in enumerate(plan.input_rows):
        filename = _entry_name(position, name)
        target = entries_directory / filename
        vector, is_valid = _load_f007_entry(
            target, identity, position=position, index=index, audio_file=name,
            input_sha256=input_hash, expected_valid=plan.expected_valid[position],
        )
        byte_count = target.stat().st_size
        _require(byte_count > 0, "F007 cache entry is empty")
        record = {
            "index": index,
            "audio_file": name,
            "input_sha256": input_hash,
            "cache_file": filename,
            "cache_sha256": _file_sha256(target) if verify_cache_sha256 else None,
            "bytes": byte_count,
            "valid": is_valid,
        }
        records.append(record); vectors.append(vector); valid.append(is_valid)
    return records, vectors, valid


def _receipt_body(plan: F007AdvancedCachePlan, identity: Mapping[str, Any],
                  records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": F007_ADVANCED_CACHE_RECEIPT_SCHEMA,
        "identity": _clone_json(identity),
        "file_count": len(records),
        "files": records,
        "completed": True,
        "server_only": True,
        "embedding_artifacts_mlflow_uploaded": False,
        "local_transfer_allowed": False,
    }


def _validate_receipt(receipt: Mapping[str, Any], plan: F007AdvancedCachePlan,
                      identity: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "identity", "file_count", "files", "completed",
        "server_only", "embedding_artifacts_mlflow_uploaded", "local_transfer_allowed",
        "receipt_sha256",
    }
    _require(isinstance(receipt, Mapping) and set(receipt) == required,
             "F007 cache receipt schema changed")
    body = {key: receipt[key] for key in required if key != "receipt_sha256"}
    _require(
        body["schema_version"] == F007_ADVANCED_CACHE_RECEIPT_SCHEMA
        and body["identity"] == identity
        and type(body["file_count"]) is int and body["file_count"] == len(plan.indices)
        and isinstance(body["files"], list) and len(body["files"]) == len(plan.indices)
        and body["completed"] is True and body["server_only"] is True
        and body["embedding_artifacts_mlflow_uploaded"] is False
        and body["local_transfer_allowed"] is False
        and receipt["receipt_sha256"] == canonical_sha256(body),
        "F007 cache receipt identity changed",
    )
    return _clone_json(receipt)


def _recover_partial(target: Path, loader: Callable[[Path], tuple[np.ndarray, bool]]) -> None:
    """Promote only a complete, self-authenticating staged entry after interruption."""
    partial = target.with_suffix(target.suffix + ".partial")
    if not partial.exists() and not partial.is_symlink():
        return
    _regular_file(partial, "cache partial")
    _require(not target.exists() and not target.is_symlink(),
             "F007 cache has both final and staged entry")
    loader(partial)
    partial.replace(target)


def _write_f007_entry(target: Path, identity: Mapping[str, Any], *, index: int,
                      audio_file: str, input_sha256: str, vector: np.ndarray,
                      valid: bool) -> None:
    _require(not target.exists() and not target.is_symlink(),
             "F007 refuses to replace a cache entry")
    partial = target.with_suffix(target.suffix + ".partial")
    _require(not partial.exists() and not partial.is_symlink(),
             "F007 cache partial must be recovered before writing")
    try:
        with partial.open("xb") as stream:
            np.savez_compressed(
                stream, embedding=vector, valid=np.bool_(valid),
                identity_signature=identity["signature"], index=np.int64(index),
                audio_file=audio_file, input_sha256=input_sha256,
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
        raise ValueError(
            "F007 extractor must return (embedding, valid) or {'embedding', 'valid'}")
    _require(isinstance(valid, (bool, np.bool_)), f"F007 {label} valid flag must be bool")
    return vector, bool(valid)


def load_f007_advanced_cache(plan: F007AdvancedCachePlan) -> dict[str, Any]:
    """Purely validate and load one completed F007 cache; no extractor is called."""
    identity = _validate_plan(plan)
    _, identity_path, receipt_path, entries = _cache_paths(plan, create=False)
    saved_identity, _ = _read_json(identity_path, "cache identity")
    _require(saved_identity == identity, "F007 stored cache identity differs from this plan")
    _validate_identity(saved_identity)
    receipt, _ = _read_json(receipt_path, "cache receipt")
    receipt = _validate_receipt(receipt, plan, identity)
    records, vectors, valid = _result_records(
        plan, entries, identity, verify_cache_sha256=True,
    )
    _require(receipt["files"] == records,
             "F007 cache receipt does not authenticate its cache entries")
    return {
        "kind": "f007_advanced_cache",
        "indices": np.asarray(plan.indices, dtype=np.int64),
        "embeddings": np.asarray(vectors, dtype=np.float32),
        "valid": np.asarray(valid, dtype=np.bool_),
        "identity": identity,
        "receipt": receipt,
        "cache_directory": str(_cache_directory(plan, create=False)),
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }


def extract_or_load_f007_advanced_cache(
        plan: F007AdvancedCachePlan, *, checkpoint_path: Path,
        extractor: Callable[[int, Mapping[str, str]], object],
        progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Resume an F007 cache with an externally supplied model/audio callback.

    The checkpoint bytes are checked before an extractor is allowed to run.
    Existing valid entries are reused; an interrupted run writes only missing
    rows and seals the receipt after every requested row is authenticated.
    Invalid-signal rows are always stored as zero vectors.
    """
    _require(callable(extractor), "F007 cache extractor must be callable")
    _require(progress is None or callable(progress), "F007 cache progress callback must be callable")
    identity = _validate_plan(plan)
    verify_validated_tail_checkpoint_bytes(checkpoint_path, identity["checkpoint"])
    _, identity_path, receipt_path, entries = _cache_paths(plan, create=True)
    _write_or_verify_identity(identity_path, identity)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_f007_advanced_cache(plan)

    total = len(plan.indices)
    for position, (index, name, input_hash) in enumerate(plan.input_rows):
        target = entries / _entry_name(position, name)
        load = lambda candidate, p=position, i=index, n=name, h=input_hash: _load_f007_entry(
            candidate, identity, position=p, index=i, audio_file=n,
            input_sha256=h, expected_valid=plan.expected_valid[p],
        )
        _recover_partial(target, load)
        if target.exists() or target.is_symlink():
            load(target)
        else:
            supplied, supplied_valid = _extractor_result(
                extractor(index, {"audio_file": name, "input_sha256": input_hash}),
                f"extractor result for row {position}",
            )
            _require(supplied_valid == plan.expected_valid[position],
                     "F007 extractor validity differs from the authenticated manifest")
            if supplied_valid:
                vector = _embedding_geometry(
                    supplied, True, True, f"extractor embedding row {position}")
            else:
                # Do not retain accidental stale/nonzero values for a row the
                # authenticated manifest marks invalid.
                vector = np.zeros((ADVANCED_EMBEDDING_DIMENSION,), dtype=np.float32)
            _write_f007_entry(
                target, identity, index=index, audio_file=name, input_sha256=input_hash,
                vector=vector, valid=supplied_valid,
            )
            load(target)
        if progress is not None:
            progress(position + 1, total)

    records, _, _ = _result_records(plan, entries, identity, verify_cache_sha256=True)
    body = _receipt_body(plan, identity, records)
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _write_new_json(receipt_path, receipt, "cache receipt")
    return load_f007_advanced_cache(plan)


def _f005_source_fold(contract: Mapping[str, Any], outer_fold: int) -> tuple[dict[str, Any], dict[str, Any], str]:
    signature, source_sha, folds, _ = _normalise_contract_binding(contract)
    _require(type(outer_fold) is int and outer_fold in folds,
             "F007 reused-control outer fold is not configured")
    source = contract["source_f005_receipt"]
    _require(isinstance(source, Mapping), "F007 reused-control source receipt is malformed")
    state = source.get("experiment_state")
    _require(
        isinstance(state, Mapping) and state.get("status") == "complete"
        and _sha256(state.get("experiment_signature"), "F005 source experiment signature"),
        "F007 reused-control source F005 state is malformed",
    )
    rows = source.get("folds")
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
             "F007 reused-control source F005 folds are missing")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("outer_fold") == outer_fold]
    _require(len(selected) == 1, "F007 reused-control source F005 fold is missing or duplicated")
    fold = selected[0]
    control = fold.get("control_tail")
    full = fold.get("full_scoring_control")
    _require(isinstance(control, Mapping) and isinstance(full, Mapping),
             "F007 reused-control source F005 fold is malformed")
    checkpoint = control.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "F007 reused-control control checkpoint is malformed")
    _sha256(checkpoint.get("sha256"), "F005 control checkpoint hash")
    _require(
        isinstance(full.get("identity"), Mapping) and isinstance(full.get("receipt"), Mapping),
        "F007 reused-control cache receipt declaration is malformed",
    )
    # ``signature`` is intentionally read so an invalid contract fails before
    # opening the cache, while source_sha becomes the returned provenance.
    _sha256(signature, "F007 contract signature")
    return dict(fold), dict(full), source_sha


def _validate_f005_identity(identity: Mapping[str, Any], *, source_signature: str,
                            outer_fold: int, checkpoint_sha256: str,
                            expected_indices_sha256: str) -> dict[str, Any]:
    required = {
        "schema_version", "experiment_signature", "outer_fold", "arm_id", "scope",
        "indices_sha256", "checkpoint_sha256", "checkpoint_metadata_sha256",
        "embedding_dimension", "inference", "server_only", "mlflow_upload_allowed", "signature",
    }
    _require(isinstance(identity, Mapping) and set(identity) == required,
             "F007 reused-control F005 identity schema changed")
    body = {key: identity[key] for key in required if key != "signature"}
    _require(
        body["schema_version"] == 1
        and body["experiment_signature"] == source_signature
        and body["outer_fold"] == outer_fold
        and body["arm_id"] == "control"
        and body["scope"] == "full_scoring"
        and body["indices_sha256"] == expected_indices_sha256
        and body["checkpoint_sha256"] == checkpoint_sha256
        and _sha256(body["checkpoint_metadata_sha256"], "F005 control checkpoint metadata hash")
        and body["embedding_dimension"] == ADVANCED_EMBEDDING_DIMENSION
        and body["server_only"] is True and body["mlflow_upload_allowed"] is False
        and identity["signature"] == canonical_sha256(body),
        "F007 reused-control F005 identity changed",
    )
    try:
        canonical(body["inference"])
    except (TypeError, ValueError) as error:
        raise ValueError("F007 reused-control F005 inference receipt is invalid") from error
    return _clone_json(identity)


def _load_f005_entry(path: Path, identity: Mapping[str, Any], *, audio_file: str,
                     input_sha256: str, expected_valid: bool, position: int) -> tuple[np.ndarray, bool]:
    path = _regular_file(path, "reused F005 cache entry")
    with np.load(path, allow_pickle=False) as saved:
        _require(
            set(saved.files) == {
                "embedding", "valid", "signature", "audio_file", "audio_sha256",
            }, "F007 reused-control F005 cache entry schema changed")
        _require(
            _scalar_string(saved["signature"], "F005 cache entry signature") == identity["signature"]
            and _scalar_string(saved["audio_file"], "F005 cache entry audio filename") == audio_file
            and _scalar_string(saved["audio_sha256"], "F005 cache entry input hash") == input_sha256,
            "F007 reused-control F005 cache entry identity changed",
        )
        valid = _scalar_bool(saved["valid"], "F005 cache entry valid")
        vector = saved["embedding"].copy()
    return _embedding_geometry(vector, valid, expected_valid, f"reused F005 embedding row {position}"), valid


def load_reused_f005_control_cache(
        contract: Mapping[str, Any], *, f005_run_directory: Path,
        outer_fold: int) -> dict[str, Any]:
    """Load an authenticated F005 full-scoring control cache in place.

    The function hashes cache artifacts named by the authenticated F005 source
    receipt, but it never opens or hashes raw inputs, and it never creates a
    copy under the F007 output.  Call it once per fold and retain the returned
    arrays for subsequent F007 stages.
    """
    fold, full, source_receipt_sha = _f005_source_fold(contract, outer_fold)
    root = _regular_directory(Path(f005_run_directory), "F005 source run")
    identity_declared, receipt_declared = full["identity"], full["receipt"]
    _require(
        set(identity_declared) == {"path", "sha256", "signature"}
        and set(receipt_declared) == {"path", "sha256", "receipt_sha256", "file_count"},
        "F007 reused-control source cache declaration changed",
    )
    identity_path = _confined_regular_file(root, identity_declared["path"], "F005 control cache identity")
    receipt_path = _confined_regular_file(root, receipt_declared["path"], "F005 control cache receipt")
    identity_payload = identity_path.read_bytes()
    receipt_payload = receipt_path.read_bytes()
    _require(
        _file_sha256(identity_path) == _sha256(identity_declared["sha256"], "F005 control identity file hash")
        and _file_sha256(receipt_path) == _sha256(receipt_declared["sha256"], "F005 control receipt file hash"),
        "F007 reused-control F005 identity or receipt bytes changed",
    )
    identity, _ = _read_json(identity_path, "F005 control cache identity")
    receipt, _ = _read_json(receipt_path, "F005 control cache receipt")
    _require(identity.get("signature") == identity_declared["signature"],
             "F007 reused-control declared F005 identity signature changed")

    manifest = contract.get("manifest")
    _require(isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)) and len(manifest) > 0,
             "F007 reused-control contract manifest is missing")
    expected_inputs = _manifest_inputs(contract, tuple(range(len(manifest))))
    expected_indices_sha256 = _array_sha256(np.arange(len(manifest), dtype=np.int64), "<i8")
    control_checkpoint = fold["control_tail"]["checkpoint"]
    validated_identity = _validate_f005_identity(
        identity,
        source_signature=contract["source_f005_receipt"]["experiment_state"]["experiment_signature"],
        outer_fold=outer_fold,
        checkpoint_sha256=_sha256(control_checkpoint["sha256"], "F005 control checkpoint hash"),
        expected_indices_sha256=expected_indices_sha256,
    )
    required_receipt = {
        "schema_version", "identity", "file_count", "files", "completed", "server_only",
        "embedding_artifacts_mlflow_uploaded", "receipt_sha256",
    }
    _require(isinstance(receipt, Mapping) and set(receipt) == required_receipt,
             "F007 reused-control F005 cache receipt schema changed")
    body = {key: receipt[key] for key in required_receipt if key != "receipt_sha256"}
    _require(
        body["schema_version"] == 1 and body["identity"] == validated_identity
        and type(body["file_count"]) is int and body["file_count"] == len(manifest)
        and body["file_count"] == receipt_declared["file_count"]
        and isinstance(body["files"], list) and len(body["files"]) == len(manifest)
        and body["completed"] is True and body["server_only"] is True
        and body["embedding_artifacts_mlflow_uploaded"] is False
        and receipt["receipt_sha256"] == canonical_sha256(body)
        and receipt["receipt_sha256"] == _sha256(receipt_declared["receipt_sha256"], "F005 control receipt self hash"),
        "F007 reused-control F005 cache receipt identity changed",
    )
    cache_directory = _regular_directory(identity_path.parent / "embedding_cache", "F005 control cache entries")
    vectors: list[np.ndarray] = []
    validity: list[bool] = []
    records: list[dict[str, Any]] = []
    for position, ((index, name, input_hash), record) in enumerate(zip(expected_inputs, receipt["files"], strict=True)):
        required_record = {
            "audio_file", "audio_sha256", "cache_file", "cache_sha256", "bytes", "valid",
        }
        _require(isinstance(record, Mapping) and set(record) == required_record,
                 "F007 reused-control F005 cache file record schema changed")
        filename = _safe_file_name(record["cache_file"], "F005 control cache file")
        target = _regular_file(cache_directory / filename, "F005 control cache entry")
        _require(
            record["audio_file"] == name and record["audio_sha256"] == input_hash
            and type(record["bytes"]) is int and record["bytes"] == target.stat().st_size
            and _file_sha256(target) == _sha256(record["cache_sha256"], "F005 control cache file hash")
            and isinstance(record["valid"], bool),
            "F007 reused-control F005 cache file record changed",
        )
        row = manifest[index]
        expected_valid = _truth(row.get("has_nonzero_signal"))
        vector, valid = _load_f005_entry(
            target, validated_identity, audio_file=name, input_sha256=input_hash,
            expected_valid=expected_valid, position=position,
        )
        _require(valid == record["valid"], "F007 reused-control F005 cache valid flag changed")
        vectors.append(vector); validity.append(valid); records.append(dict(record))
    _require(receipt["files"] == records, "F007 reused-control F005 records changed during load")
    return {
        "kind": F007_REUSED_F005_CONTROL_SCHEMA,
        "outer_fold": outer_fold,
        "indices": np.arange(len(manifest), dtype=np.int64),
        "embeddings": np.asarray(vectors, dtype=np.float32),
        "valid": np.asarray(validity, dtype=np.bool_),
        "source_f005_receipt_sha256": source_receipt_sha,
        "identity": validated_identity,
        "receipt": _clone_json(receipt),
        "source_identity_file_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "source_receipt_file_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "server_only": True,
        "mlflow_upload_allowed": False,
        "local_transfer_allowed": False,
    }
