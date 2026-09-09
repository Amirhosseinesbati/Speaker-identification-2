"""Read-only provenance receipt for reusing F005 checkpoints in F007.

This module deliberately has no dependency on Torch, audio decoding, MLflow, or
the F005 trainer.  It authenticates only the small set of already-completed
F005 artifacts that an F007 tail experiment may name as immutable sources.
The embedding cache itself is not opened: its sealed identity and receipt are
checked without repeating a full cache audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


F007_F005_SOURCE_RECEIPT_SCHEMA = "f007-f005-source-receipt-v1"
_F005_STATE_SCHEMA = "f005-resumable-experiment-state-v1"
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value),
        f"F005 {label} must be a lowercase SHA-256",
    )
    return value


def _source_root(f005_dir: Path) -> Path:
    path = Path(f005_dir)
    _require(path.is_dir() and not path.is_symlink(),
             "F005 source run must be a non-symlink directory")
    return path.resolve()


def _regular_file(root: Path, relative: Path, label: str) -> Path:
    """Resolve a source file only if every relevant component is non-symlink."""
    candidate = root / relative
    _require(candidate.is_relative_to(root), f"F005 {label} path escapes the source run")
    _require(candidate.is_file() and not candidate.is_symlink(),
             f"F005 {label} must be a regular file")
    current = candidate.parent
    while current != root:
        _require(not current.is_symlink(), f"F005 {label} parent may not be a symlink")
        current = current.parent
    return candidate


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F005 {label} must be a readable JSON object") from error
    _require(isinstance(value, dict), f"F005 {label} must be a JSON object")
    return value


def _recorded_path(root: Path, recorded: object, expected: Path, label: str) -> None:
    _require(isinstance(recorded, str) and recorded,
             f"F005 {label} record lacks its path")
    value = Path(recorded)
    if not value.is_absolute():
        value = root / value
    try:
        matches = value.resolve() == expected.resolve()
    except OSError as error:
        raise ValueError(f"F005 {label} recorded path cannot be resolved") from error
    _require(matches, f"F005 {label} record points outside its canonical path")


def _file_entry(root: Path, relative: Path, recorded: object, expected_sha: object,
                label: str, *, parse_json: bool = False) -> dict[str, str]:
    path = _regular_file(root, relative, label)
    _recorded_path(root, recorded, path, label)
    expected = _sha256(expected_sha, f"{label} recorded hash")
    actual = _file_sha256(path)
    _require(actual == expected, f"F005 {label} bytes differ from its recorded hash")
    if parse_json:
        _read_object(path, label)
    return {"path": relative.as_posix(), "sha256": actual}


def _fold_ids(heads: object) -> list[int]:
    _require(isinstance(heads, dict) and heads, "F005 complete state lacks shared-head records")
    result: list[int] = []
    for key in heads:
        _require(isinstance(key, str) and key.startswith("fold_"),
                 "F005 shared-head key is malformed")
        suffix = key.removeprefix("fold_")
        _require(suffix.isdecimal(), "F005 shared-head key has an invalid fold id")
        result.append(int(suffix))
    result.sort()
    _require(len(result) == len(set(result)), "F005 shared-head fold ids are duplicated")
    return result


def _head_entry(root: Path, state: dict[str, Any], outer: int) -> tuple[dict[str, Any], dict[str, Any]]:
    key = f"fold_{outer}"
    heads = state["heads"]
    record = heads.get(key) if isinstance(heads, dict) else None
    _require(isinstance(record, dict) and record.get("complete") is True,
             f"F005 shared head for fold {outer} is incomplete")
    checkpoint = _file_entry(
        root,
        Path("training") / key / "shared_head" / "shared_head.pt",
        record.get("checkpoint"), record.get("checkpoint_sha256"),
        f"shared head checkpoint fold {outer}",
    )
    report = _file_entry(
        root,
        Path("training") / key / "shared_head" / "unit_report.json",
        record.get("report"), record.get("report_sha256"),
        f"shared head report fold {outer}", parse_json=True,
    )
    return record, {"checkpoint": checkpoint, "unit_report": report}


def _control_tail_entry(root: Path, state: dict[str, Any], outer: int,
                        shared_head_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    key = f"fold_{outer}/control"
    tails = state.get("tails")
    record = tails.get(key) if isinstance(tails, dict) else None
    _require(isinstance(record, dict) and record.get("complete") is True,
             f"F005 control tail for fold {outer} is incomplete")
    _require(record.get("shared_checkpoint_sha256") == shared_head_sha256,
             f"F005 control tail for fold {outer} is not bound to its shared head")
    checkpoint = _file_entry(
        root,
        Path("training") / f"fold_{outer}" / "tails" / "control" / "last.pt",
        record.get("checkpoint"), record.get("checkpoint_sha256"),
        f"control tail checkpoint fold {outer}",
    )
    report = _file_entry(
        root,
        Path("training") / f"fold_{outer}" / "tails" / "control" / "unit_report.json",
        record.get("report"), record.get("report_sha256"),
        f"control tail report fold {outer}", parse_json=True,
    )
    return record, {
        "checkpoint": checkpoint,
        "unit_report": report,
        "shared_head_checkpoint_sha256": shared_head_sha256,
    }


def _control_cache_entry(root: Path, state: dict[str, Any], outer: int,
                         tail_checkpoint_sha256: str,
                         experiment_signature: str) -> dict[str, Any]:
    key = f"fold_{outer}/control"
    full_embeddings = state.get("full_embeddings")
    record = full_embeddings.get(key) if isinstance(full_embeddings, dict) else None
    _require(isinstance(record, dict),
             f"F005 full-scoring control record for fold {outer} is missing")
    _require(record.get("scope") == "all_manifest_rows" and record.get("mlflow_uploaded") is False,
             f"F005 full-scoring control record for fold {outer} changed scope or retention")
    rows = record.get("rows")
    _require(type(rows) is int and rows > 0,
             f"F005 full-scoring control record for fold {outer} has invalid row count")

    base = Path("full_scoring") / f"fold_{outer}" / "control"
    identity_path = _regular_file(root, base / "full_scoring_cache_identity.json",
                                  f"control cache identity fold {outer}")
    receipt_path = _regular_file(root, base / "full_scoring_cache_receipt.json",
                                 f"control cache receipt fold {outer}")
    _recorded_path(root, record.get("cache_receipt"), receipt_path,
                   f"control cache receipt fold {outer}")
    expected_receipt_sha = _sha256(record.get("cache_receipt_sha256"),
                                   f"control cache receipt fold {outer} recorded hash")
    actual_receipt_sha = _file_sha256(receipt_path)
    _require(actual_receipt_sha == expected_receipt_sha,
             f"F005 control cache receipt for fold {outer} bytes differ from its recorded hash")

    identity = _read_object(identity_path, f"control cache identity fold {outer}")
    identity_body = {name: value for name, value in identity.items() if name != "signature"}
    expected_identity_fields = {
        "schema_version", "experiment_signature", "outer_fold", "arm_id", "scope",
        "indices_sha256", "checkpoint_sha256", "checkpoint_metadata_sha256",
        "embedding_dimension", "inference", "server_only", "mlflow_upload_allowed",
    }
    _require(
        set(identity_body) == expected_identity_fields
        and identity.get("signature") == _canonical_sha(identity_body)
        and identity.get("schema_version") == 1
        and identity.get("experiment_signature") == experiment_signature
        and identity.get("outer_fold") == outer
        and identity.get("arm_id") == "control"
        and identity.get("scope") == "full_scoring"
        and identity.get("embedding_dimension") == 192
        and identity.get("server_only") is True
        and identity.get("mlflow_upload_allowed") is False,
        f"F005 control cache identity for fold {outer} changed",
    )
    for field in ("indices_sha256", "checkpoint_sha256", "checkpoint_metadata_sha256"):
        _sha256(identity.get(field), f"control cache identity fold {outer}/{field}")
    _require(identity["checkpoint_sha256"] == tail_checkpoint_sha256,
             f"F005 control cache identity for fold {outer} is not bound to its control tail")

    receipt = _read_object(receipt_path, f"control cache receipt fold {outer}")
    receipt_body = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    expected_receipt_fields = {
        "schema_version", "identity", "file_count", "files", "completed", "server_only",
        "embedding_artifacts_mlflow_uploaded", "receipt_sha256",
    }
    _require(
        set(receipt) == expected_receipt_fields
        and receipt.get("receipt_sha256") == _canonical_sha(receipt_body)
        and receipt.get("schema_version") == 1
        and receipt.get("identity") == identity
        and receipt.get("file_count") == rows
        and isinstance(receipt.get("files"), list)
        and len(receipt["files"]) == rows
        and receipt.get("completed") is True
        and receipt.get("server_only") is True
        and receipt.get("embedding_artifacts_mlflow_uploaded") is False,
        f"F005 control cache receipt for fold {outer} changed",
    )
    _sha256(receipt.get("receipt_sha256"), f"control cache receipt fold {outer} self hash")
    return {
        "identity": {
            "path": (base / "full_scoring_cache_identity.json").as_posix(),
            "sha256": _file_sha256(identity_path),
            "signature": identity["signature"],
        },
        "receipt": {
            "path": (base / "full_scoring_cache_receipt.json").as_posix(),
            "sha256": actual_receipt_sha,
            "receipt_sha256": receipt["receipt_sha256"],
            "file_count": rows,
        },
    }


def load_f005_source_receipt(f005_dir: Path) -> dict[str, Any]:
    """Authenticate a complete F005 source run and return its compact receipt.

    The returned paths are relative to ``source_run_directory``.  This lets a
    later F007 orchestrator persist a portable source receipt while resolving
    only the explicitly verified F005 files.  The function never opens a
    checkpoint, an embedding cache, audio, or an MLflow client.
    """
    root = _source_root(f005_dir)
    state_path = _regular_file(root, Path("experiment_state.json"), "experiment state")
    state = _read_object(state_path, "experiment state")
    _require(state.get("schema_version") == _F005_STATE_SCHEMA,
             "F005 source state schema is not supported")
    _require(state.get("status") == "complete", "F005 source run is not complete")
    _require(state.get("output_directory") is not None,
             "F005 complete state lacks its output directory")
    _recorded_path(root, state.get("output_directory"), root, "source output directory")
    experiment_signature = _sha256(state.get("experiment_signature"), "experiment signature")
    fold_ids = _fold_ids(state.get("heads"))

    folds: list[dict[str, Any]] = []
    for outer in fold_ids:
        head_record, shared_head = _head_entry(root, state, outer)
        tail_record, control_tail = _control_tail_entry(
            root, state, outer, shared_head["checkpoint"]["sha256"],
        )
        control_cache = _control_cache_entry(
            root, state, outer, control_tail["checkpoint"]["sha256"], experiment_signature,
        )
        # Accessing these values above validates the state fields and documents
        # their binding for readers of the compact receipt.
        _require(head_record.get("checkpoint_sha256") == shared_head["checkpoint"]["sha256"],
                 f"F005 shared head state hash for fold {outer} changed")
        _require(tail_record.get("checkpoint_sha256") == control_tail["checkpoint"]["sha256"],
                 f"F005 control tail state hash for fold {outer} changed")
        folds.append({
            "outer_fold": outer,
            "shared_head": shared_head,
            "control_tail": control_tail,
            "full_scoring_control": control_cache,
        })

    return {
        "schema_version": F007_F005_SOURCE_RECEIPT_SCHEMA,
        "source_run_directory": str(root),
        "experiment_state": {
            "path": "experiment_state.json",
            "sha256": _file_sha256(state_path),
            "status": "complete",
            "experiment_signature": experiment_signature,
        },
        "fold_ids": fold_ids,
        "folds": folds,
    }
