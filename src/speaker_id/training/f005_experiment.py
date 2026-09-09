"""Resumable, leak-resistant orchestration for the complete F005 experiment.

This module owns ordering and persistence, not the model objective or scoring
math.  The ordering is deliberately stronger than a conventional fold loop:
both shared heads and all eight tails finish, both known-only arm choices are
sealed, and both open-set policies are sealed before the first outer label is
materialized.  Embedding caches and checkpoints are server-only working files.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


STATE_SCHEMA = "f005-resumable-experiment-state-v1"
PRETRUTH_SCHEMA = "f005-server-only-pretruth-cache-v1"
UPLOADABLE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
FORBIDDEN_UPLOAD_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npz", ".npy", ".wav", ".mp3", ".flac",
    ".ogg", ".m4a", ".pem", ".key", ".p12", ".pfx",
}
FORBIDDEN_UPLOAD_PARTS = {
    "embedding_cache", "checkpoints", "optimizer", "raw_audio",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value, stream, ensure_ascii=False, allow_nan=False,
                sort_keys=True, indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"F005 expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _confined_output(root: Path, configured_root: str, path: Path, *, existing: bool) -> Path:
    root = Path(root).resolve()
    base = (root / configured_root).resolve()
    candidate = Path(path).resolve(strict=existing)
    if not candidate.is_relative_to(base) or candidate == base:
        raise ValueError("F005 run directory must stay below its configured output root")
    unresolved = Path(path).absolute()
    if unresolved.is_symlink() or any(
            parent.is_symlink() for parent in unresolved.parents
            if parent != root.parent and parent.is_relative_to(root)):
        raise ValueError("F005 run directories may not cross symlinks")
    return candidate


def _safe_add_artifact(tracker, source: Path, relative_path: str | None = None) -> None:
    """Upload metadata/report bytes only; fail closed on model/data payloads."""
    source = Path(source)
    lowered_parts = {part.lower() for part in source.parts}
    if (not source.is_file() or source.is_symlink()
            or source.suffix.lower() not in UPLOADABLE_SUFFIXES
            or source.suffix.lower() in FORBIDDEN_UPLOAD_SUFFIXES
            or lowered_parts & FORBIDDEN_UPLOAD_PARTS):
        raise ValueError(f"F005 artifact is outside the metadata-only MLflow boundary: {source}")
    tracker.add_artifact(source, relative_path)


def _fresh_state(contract: dict, output: Path) -> dict:
    return {
        "schema_version": STATE_SCHEMA,
        "experiment_signature": contract["signature"],
        "output_directory": str(output),
        "status": "initialized",
        "phase": "initialized",
        "parent": None,
        "children": {},
        "heads": {},
        "tails": {},
        "known_scores": {},
        "arm_seals": {},
        "full_embeddings": {},
        "policy_seals": {},
        "parity": None,
        "outer_truth_materialized": False,
        "outer_evaluations": {},
        "aggregate": None,
        "resume_supported": True,
        "checkpoint_and_embedding_caches_server_only": True,
        "mlflow_forbidden_payloads_uploaded": False,
    }


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, state)


def _validate_state(state: dict, contract: dict, output: Path) -> None:
    _require(
        state.get("schema_version") == STATE_SCHEMA
        and state.get("experiment_signature") == contract["signature"]
        and Path(state.get("output_directory", "")).resolve() == output
        and state.get("resume_supported") is True,
        "F005 resume state belongs to different code, data, configuration, or output",
    )
    _require(
        state.get("mlflow_forbidden_payloads_uploaded") is False
        and state.get("checkpoint_and_embedding_caches_server_only") is True,
        "F005 resume state violates the retention boundary",
    )


def _unit_key(outer: int, arm_id: str | None = None) -> str:
    return f"fold_{outer}" if arm_id is None else f"fold_{outer}/{arm_id}"


def _read_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _pack_arrays(value: object, arrays: dict[str, np.ndarray]) -> object:
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):04d}"
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "O":
            raise ValueError("F005 pretruth cache refuses object arrays")
        arrays[key] = array
        return {"__ndarray__": key, "dtype": array.dtype.str, "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _pack_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack_arrays(item, arrays) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported F005 pretruth cache type: {type(value).__name__}")


def _unpack_arrays(value: object, arrays: dict[str, np.ndarray]) -> object:
    if isinstance(value, dict) and set(value) == {"__ndarray__", "dtype", "shape"}:
        key = value["__ndarray__"]
        if key not in arrays:
            raise ValueError("F005 pretruth cache array is missing")
        array = np.asarray(arrays[key])
        if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
            raise ValueError("F005 pretruth cache array metadata changed")
        return array.copy()
    if isinstance(value, dict):
        return {key: _unpack_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_arrays(item, arrays) for item in value]
    return value


def _save_pretruth(path: Path, pretruth: dict) -> dict:
    """Persist score arrays locally so a sealed policy can resume without resealing."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    structure = _pack_arrays(pretruth, arrays)
    metadata = {
        "schema_version": PRETRUTH_SCHEMA,
        "structure": structure,
        "array_keys": sorted(arrays),
        "structure_sha256": _sha(structure),
    }
    array_path = path.with_suffix(".npz")
    metadata_path = path.with_suffix(".json")
    for target in (array_path, metadata_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"F005 refuses to overwrite pretruth cache: {target}")
    temporary = array_path.with_suffix(".npz.partial")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(array_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata["arrays_file_sha256"] = _sha_file(array_path)
    _write_json(metadata_path, metadata)
    return {
        "metadata_path": str(metadata_path), "arrays_path": str(array_path),
        "metadata_sha256": _sha_file(metadata_path),
        "arrays_sha256": metadata["arrays_file_sha256"],
        "mlflow_uploaded": False,
    }


def _load_pretruth(path: Path) -> dict:
    metadata_path = Path(path).with_suffix(".json")
    array_path = Path(path).with_suffix(".npz")
    metadata = _read_json(metadata_path)
    _require(
        metadata.get("schema_version") == PRETRUTH_SCHEMA
        and metadata.get("structure_sha256") == _sha(metadata.get("structure"))
        and metadata.get("arrays_file_sha256") == _sha_file(array_path),
        "F005 pretruth cache identity changed",
    )
    with np.load(array_path, allow_pickle=False) as saved:
        _require(set(saved.files) == set(metadata["array_keys"]), "F005 pretruth arrays changed")
        arrays = {key: saved[key].copy() for key in saved.files}
    return _unpack_arrays(metadata["structure"], arrays)


class DefaultF005Backend:
    """Lazy real implementation; tests replace this object with pure fakes."""

    def load_binding(self, path: Path, experiment_id: str):
        from speaker_id.tracking import ExperimentBinding
        binding = ExperimentBinding(**_read_json(path)["binding"])
        binding.validate()
        _require(binding.experiment_id == experiment_id, "F005 requires MLflow experiment 1")
        return binding

    def prepare_tracker(self, **kwargs):
        from speaker_id.tracking import DurableMLflowRun
        return DurableMLflowRun.prepare(**kwargs)

    def open_tracker(self, spool: Path):
        from speaker_id.tracking import DurableMLflowRun
        return DurableMLflowRun(spool)

    def execution_plan(self, contract: dict) -> dict:
        from speaker_id.training.f005_runner import execution_plan
        return execution_plan(contract, include_plan_hashes=True)

    def require_environment(self, contract: dict, root: Path) -> dict:
        from speaker_id.infrastructure.readiness import validate_readiness_for_execution
        from speaker_id.training.f005_contract import require_execution_environment
        readiness = contract["readiness"]
        _require(
            readiness.get("summary", {}).get("audio_hashes_checked") is True,
            "F005 full execution requires a fresh full-audio hash verification",
        )
        report = validate_readiness_for_execution(
            Path(root), readiness,
            Path("artifacts/infrastructure/C002_preparation/readiness.json"),
        )
        environment = require_execution_environment(contract)
        return {"readiness_status": report["status"], **environment}

    def load_frozen_sources(self, contract: dict) -> dict:
        from speaker_id.training.gain_suite import verify_gain_cache
        source = contract["config"]["source_c002b"]
        directory = Path(source["run_dir"])
        identity = _read_json(directory / "identity_cache_identity.json")
        receipt = _read_json(directory / "identity_cache_manifest.json")
        vectors, valid = verify_gain_cache(
            directory / "identity_embedding_cache", identity,
            contract["manifest"], receipt,
        )
        _require(
            vectors["public"].shape == (len(contract["manifest"]), 512)
            and vectors["advanced"].shape == (len(contract["manifest"]), 192)
            and valid.dtype == np.bool_,
            "F005 frozen C002b cache dimensions changed",
        )
        return {
            "public": vectors["public"], "frozen_advanced": vectors["advanced"],
            "valid": valid, "identity": identity, "receipt": receipt,
        }

    def fit_head(self, contract, root, outer, output, tracker, *, resume):
        from speaker_id.training.f005_worker import fit_shared_head
        return fit_shared_head(contract, root, outer, output, tracker, resume=resume)

    def fit_tail(self, contract, root, outer, arm_id, shared, output, tracker, *, resume):
        from speaker_id.training.f005_worker import fit_tail_arm
        return fit_tail_arm(
            contract, root, outer, arm_id, shared, output, tracker, resume=resume,
        )

    def _load_tail_encoder(self, contract: dict, root: Path, outer: int,
                           arm_id: str, shared: Path, checkpoint: Path):
        import torch
        from speaker_id.training.f005_runner import (
            arm_identity, plan_range_sha256, validate_resume_payload,
        )
        from speaker_id.training.f005_worker import _load_advanced_trainable
        from speaker_id.training.schedules import adaptation_total_steps
        fit = contract["config"]["fit"]
        head_steps, total = fit["adaptation_schedule"]["head_only_steps"], adaptation_total_steps(fit)
        identity = arm_identity(
            contract, outer, arm_id,
            shared_head_checkpoint_sha256=_sha_file(shared),
            tail_plan_sha256=plan_range_sha256(contract, outer, head_steps, total),
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = validate_resume_payload(payload, identity, fit)
        _require(metadata["completed_steps"] == total, "F005 extraction requires a complete tail")
        encoder, _ = _load_advanced_trainable(contract, root, "cuda")
        encoder.load_state_dict(payload["encoder"], strict=True)
        encoder.requires_grad_(False).eval()
        return encoder, metadata

    def extract_arm(self, contract: dict, root: Path, outer: int, arm_id: str,
                    shared: Path, checkpoint: Path, indices: np.ndarray,
                    scope: str, expected_valid: np.ndarray, cache_dir: Path,
                    progress) -> dict:
        encoder, metadata = self._load_tail_encoder(
            contract, root, outer, arm_id, shared, checkpoint,
        )
        try:
            return _extract_advanced_cache(
                encoder, contract, root, outer, arm_id, checkpoint, metadata,
                indices, scope, expected_valid, cache_dir, progress,
            )
        finally:
            del encoder
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def known_scores(self, embeddings, valid, contract, outer):
        from speaker_id.training.f005_runner import known_selection_scores
        return known_selection_scores(embeddings, valid, contract, outer)

    def seal_arm(self, contract: dict, outer: int, scores: dict, path: Path) -> dict:
        from speaker_id.training.f005_runner import select_arm, write_and_reload_seal
        from speaker_id.training.f005_scoring import reload_arm_selection_seal
        if path.exists():
            return reload_arm_selection_seal(path, contract, outer)
        seal = select_arm(contract, outer, scores)
        write_and_reload_seal(path, seal)
        return reload_arm_selection_seal(path, contract, outer)

    def prepare_policy(self, contract: dict, outer: int, *, public, frozen,
                       selected_embeddings: dict, valid, arm_reload, path: Path) -> dict:
        from speaker_id.training.f005_scoring import prepare_and_seal_inner_policies
        return prepare_and_seal_inner_policies(
            contract, outer, public_embeddings=public,
            frozen_advanced_embeddings=frozen,
            advanced_embeddings_by_arm=selected_embeddings,
            valid=valid, arm_selection_reload=arm_reload,
            policy_seal_path=path,
        )

    def rebuild_policy(self, contract: dict, outer: int, *, public, frozen,
                       selected_embeddings: dict, valid, arm_reload,
                       policy_reload: dict) -> dict:
        from speaker_id.training.f005_scoring import rebuild_pretruth_from_policy_seal
        return rebuild_pretruth_from_policy_seal(
            contract, outer, public_embeddings=public,
            frozen_advanced_embeddings=frozen,
            advanced_embeddings_by_arm=selected_embeddings,
            valid=valid, arm_selection_reload=arm_reload,
            policy_reload=policy_reload,
        )

    def reload_policy(self, path: Path, contract: dict, outer: int, arm_reload: dict):
        from speaker_id.training.f005_scoring import reload_policy_seal
        return reload_policy_seal(path, contract, outer, arm_reload=arm_reload)

    def verify_parity(self, pretruth_by_fold: dict[int, dict]) -> dict:
        """Recompute every sealed decision on CUDA without touching true labels."""
        import torch
        from speaker_id.training.reference_scoring import reference_probabilities
        comparisons = []
        for outer, pretruth in sorted(pretruth_by_fold.items()):
            seal = pretruth["policy_reload"]["seal"]
            for comparator, score in pretruth["score_bundles"].items():
                calibration = seal["policies"][comparator]["calibration"]
                known = np.asarray(score["outer_known_scores"], dtype=np.float64)
                unknown = np.asarray(score["outer_unknown_similarity"], dtype=np.float64)
                valid = np.asarray(score["outer_valid"], dtype=np.bool_)
                cpu = reference_probabilities(known, unknown, calibration, valid, 0.05).argmax(axis=1)
                known_cuda = torch.as_tensor(known, device="cuda", dtype=torch.float64)
                unknown_cuda = torch.as_tensor(unknown, device="cuda", dtype=torch.float64)
                values, _ = torch.topk(known_cuda, k=2, dim=1, largest=True, sorted=True)
                top, second = values[:, 0], values[:, 1]
                gate = (
                    top - float(calibration["unknown_weight"]) * unknown_cuda
                    + float(calibration["margin_weight"]) * (top - second)
                )
                relative = known_cuda - known_cuda.max(dim=1, keepdim=True).values
                logits = torch.cat((float(calibration["threshold"]) - gate[:, None], relative), dim=1) / 0.05
                cuda = logits.argmax(dim=1).cpu().numpy()
                cuda[~valid] = 0
                exact = bool(np.array_equal(cpu, cuda))
                comparisons.append({
                    "outer_fold": outer, "comparator": comparator,
                    "rows": len(cpu), "exact_prediction_parity": exact,
                    "mismatches": int(np.sum(cpu != cuda)),
                })
        return {
            "status": "passed" if all(row["exact_prediction_parity"] for row in comparisons) else "failed",
            "exact_cpu_cuda_prediction_parity": all(row["exact_prediction_parity"] for row in comparisons),
            "comparisons": comparisons,
            "outer_truth_read": False,
        }

    def evaluate_outer(self, pretruth, policy_reload, rows, labels, path):
        from speaker_id.training.f005_scoring import evaluate_outer_once
        return evaluate_outer_once(
            pretruth, policy_reload, rows, labels,
            evaluation_path=path, probability_temperature=0.05,
        )

    def aggregate(self, contract, fold_results, historical_predictions,
                  historical_top1, parity):
        from speaker_id.training.f005_scoring import aggregate_oof_and_decide
        return aggregate_oof_and_decide(
            contract, fold_results,
            historical_c002b_predictions=historical_predictions,
            historical_c002b_known_top1_predictions=historical_top1,
            exact_cpu_cuda_prediction_parity=parity,
        )

    def load_historical_predictions(self, contract: dict) -> tuple[list[dict], list[dict]]:
        source = contract["config"]["source_c002b"]
        directory = Path(source["run_dir"]) / "C002b"
        predictions = _read_rows(directory / "oof_predictions.csv")
        top1 = []
        expected_labels = contract["labels"]
        for outer in contract["config"]["fold_ids"]:
            path = directory / f"fold_{outer}" / "outer_probabilities.npz"
            with np.load(path, allow_pickle=False) as saved:
                _require(
                    set(saved.files) >= {"probabilities", "audio_files", "labels"}
                    and saved["probabilities"].shape[1] == len(expected_labels)
                    and saved["labels"].tolist() == expected_labels,
                    "Historical C002b probability identity changed",
                )
                guesses = saved["probabilities"][:, 1:].argmax(axis=1) + 1
                top1.extend({
                    "audio_file": str(name), "speaker_id": expected_labels[int(guess)],
                } for name, guess in zip(saved["audio_files"], guesses, strict=True))
        return predictions, top1


def _extract_advanced_cache(
        encoder, contract: dict, root: Path, outer: int, arm_id: str,
        checkpoint: Path, checkpoint_metadata: dict, indices: np.ndarray,
        scope: str, expected_valid: np.ndarray, cache_dir: Path, progress) -> dict:
    """Extract one explicit row scope into a resumable, server-only cache."""
    from speaker_id.candidates.campp_advanced import extract_advanced_embedding
    from speaker_id.data.splits import truth
    manifest, config = contract["manifest"], contract["config"]
    values = np.asarray(indices)
    _require(
        values.ndim == 1 and values.dtype.kind in "iu" and len(values)
        and len(set(int(item) for item in values)) == len(values)
        and np.all(values >= 0) and np.all(values < len(manifest)),
        "F005 extraction scope indices are malformed",
    )
    values = values.astype(np.int64, copy=False)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _require(not cache_dir.is_symlink(), "F005 cache directory may not be a symlink")
    identity_body = {
        "schema_version": 1, "experiment_signature": contract["signature"],
        "outer_fold": outer, "arm_id": arm_id, "scope": scope,
        "indices_sha256": hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest(),
        "checkpoint_sha256": _sha_file(checkpoint),
        "checkpoint_metadata_sha256": checkpoint_metadata["metadata_sha256"],
        "embedding_dimension": 192, "inference": config["scoring"]["advanced_inference"],
        "server_only": True, "mlflow_upload_allowed": False,
    }
    identity = {**identity_body, "signature": _sha(identity_body)}
    identity_path = cache_dir.parent / f"{scope}_cache_identity.json"
    if identity_path.exists():
        _require(_read_json(identity_path) == identity, "F005 extraction cache identity changed")
    else:
        _write_json(identity_path, identity)
    vectors, mask, records = [], [], []
    started = time.monotonic()
    for position, index in enumerate(values):
        row = manifest[int(index)]
        name = row["audio_file"]
        _require(Path(name).name == name and "/" not in name and "\\" not in name,
                 "F005 audio/cache names must be flat")
        target = cache_dir / (Path(name).stem + ".npz")
        if target.exists():
            _require(target.is_file() and not target.is_symlink(), "F005 cache entry is not regular")
            with np.load(target, allow_pickle=False) as saved:
                _require(
                    set(saved.files) == {"embedding", "valid", "signature", "audio_file", "audio_sha256"}
                    and str(saved["signature"]) == identity["signature"]
                    and str(saved["audio_file"]) == name
                    and str(saved["audio_sha256"]) == row["input_sha256"],
                    "F005 resumed cache entry identity changed",
                )
                vector, valid = saved["embedding"].copy(), bool(saved["valid"])
        else:
            audio = Path(root) / contract["readiness"]["config"]["data_dir"] / name
            _require(_sha_file(audio) == row["input_sha256"], "F005 audio changed during extraction")
            vector, info = extract_advanced_embedding(
                encoder, audio, device="cuda", seconds=180.0, maximum_windows=1,
            )
            valid = bool(info["nonzero_signal"])
            temporary = target.with_suffix(".npz.partial")
            if temporary.exists():
                raise FileExistsError("F005 partial embedding cache requires forensic handling")
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream, embedding=vector, valid=valid,
                    signature=identity["signature"], audio_file=name,
                    audio_sha256=row["input_sha256"],
                )
            temporary.replace(target)
        expected = bool(expected_valid[int(index)])
        _require(
            valid == expected == truth(row["has_nonzero_signal"])
            and vector.shape == (192,) and vector.dtype == np.float32
            and np.isfinite(vector).all()
            and ((np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)) if valid else not np.any(vector)),
            "F005 adapted embedding validity or geometry changed",
        )
        vectors.append(vector); mask.append(valid)
        records.append({
            "audio_file": name, "audio_sha256": row["input_sha256"],
            "cache_file": target.name, "cache_sha256": _sha_file(target),
            "bytes": target.stat().st_size, "valid": valid,
        })
        if (position + 1) % 50 == 0 or position + 1 == len(values):
            progress(position + 1, len(values), time.monotonic() - started)
    receipt_body = {
        "schema_version": 1, "identity": identity, "file_count": len(records),
        "files": records, "completed": True, "server_only": True,
        "embedding_artifacts_mlflow_uploaded": False,
    }
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    receipt_path = cache_dir.parent / f"{scope}_cache_receipt.json"
    if receipt_path.exists():
        _require(_read_json(receipt_path) == receipt, "F005 cache receipt changed")
    else:
        _write_json(receipt_path, receipt)
    return {
        "indices": values.copy(), "embeddings": np.asarray(vectors, dtype=np.float32),
        "valid": np.asarray(mask, dtype=np.bool_), "identity": identity,
        "receipt": receipt, "receipt_path": str(receipt_path),
    }


def _selection_indices(contract: dict, outer: int, valid: np.ndarray) -> np.ndarray:
    from speaker_id.data.splits import truth
    known = set(contract["labels"][1:])
    split = {row["audio_file"]: row for row in contract["folds"]}
    _require(
        len(split) == len(contract["manifest"])
        and set(split) == {row["audio_file"] for row in contract["manifest"]},
        "F005 split rows are not aligned with the manifest",
    )
    result = [
        index for index, row in enumerate(contract["manifest"])
        if int(split[row["audio_file"]]["fold"]) != outer
        and truth(split[row["audio_file"]]["train_eligible"])
        and row["speaker_id"] in known and bool(valid[index])
    ]
    _require(bool(result), "F005 known-only extraction scope is empty")
    return np.asarray(result, dtype=np.int64)


def _save_known_scores(path: Path, arm_id: str, contract: dict, scores: dict) -> dict:
    path = Path(path)
    provenance = json.dumps(scores["provenance"], ensure_ascii=False, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(".npz.partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("F005 refuses to overwrite known-only scores")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            experiment_signature=contract["signature"], arm_id=arm_id,
            known_calibration_indices=scores["known_calibration_indices"],
            known_scores=scores["known_scores"],
            known_labels=np.asarray(scores["known_labels"]),
            reference_support=scores["reference_support"],
            provenance=provenance,
        )
    temporary.replace(path)
    return {"path": str(path), "sha256": _sha_file(path), "mlflow_uploaded": False}


def _load_known_scores(path: Path, arm_id: str, contract: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        _require(
            set(saved.files) == {
                "experiment_signature", "arm_id", "known_calibration_indices",
                "known_scores", "known_labels", "reference_support", "provenance",
            }
            and str(saved["experiment_signature"]) == contract["signature"]
            and str(saved["arm_id"]) == arm_id,
            "F005 known-only score cache identity changed",
        )
        return {
            "known_calibration_indices": saved["known_calibration_indices"].copy(),
            "known_scores": saved["known_scores"].copy(),
            "known_labels": saved["known_labels"].tolist(),
            "reference_support": saved["reference_support"].copy(),
            "provenance": json.loads(str(saved["provenance"])),
        }


def _outer_truth_rows(contract: dict, outer: int) -> list[dict]:
    """This is the sole orchestrator site that materializes outer class labels."""
    from speaker_id.data.splits import truth
    role_names = [
        row["audio_file"] for row in contract["roles"]
        if int(row["outer_fold"]) == outer and truth(row["outer_evaluation_included"])
    ]
    manifest = {row["audio_file"]: row for row in contract["manifest"]}
    split = {row["audio_file"]: row for row in contract["folds"]}
    _require(
        len(role_names) == len(set(role_names)) and set(role_names).issubset(manifest)
        and set(role_names).issubset(split),
        "F005 outer role, manifest, and split rows are not aligned",
    )
    return [
        {
            "audio_file": name, "speaker_id": manifest[name]["speaker_id"],
            "group_id": split[name]["group_id"],
            "duration_seconds": manifest[name]["duration_seconds"],
            "has_nonzero_signal": manifest[name]["has_nonzero_signal"],
        }
        for name in role_names
    ]


def _tracker_live(tracker, *, resume: bool) -> None:
    status = tracker.state.get("remote_status")
    if resume and status in {"FAILED", "KILLED"}:
        tracker.reopen()
    elif status == "FINISHED":
        raise RuntimeError("F005 cannot resume an incomplete stage from a FINISHED run")
    else:
        tracker.flush(strict=True)
    tracker.verify_artifacts()
    tracker.verify_remote_metadata()


def _finish_quietly(tracker, status: str) -> None:
    if tracker is None:
        return
    try:
        tracker.finish(status, strict=False)
    except Exception:
        pass


def _input_paths(root: Path, config_path: Path, binding_path: Path, config: dict) -> dict[str, Path]:
    readiness = json.loads((root / config["readiness_config"]).read_text(encoding="utf-8"))
    paths = {
        "f005_config": config_path,
        "advanced_model_config": root / config["advanced_model_config"],
        "readiness_config": root / config["readiness_config"],
        "mlflow_binding": binding_path,
        "launcher": root / "scripts/run_f005_experiment.py",
    }
    for key in ("manifest", "folds", "roles", "label_map", "model_config"):
        paths["readiness_" + key] = root / readiness[key]
    return paths


def _experiment_markdown(report: dict) -> str:
    """Render the full decision evidence for the MLflow parent summary."""
    result = report.get("result", {})
    lines = [
        "# F005 paired long/short CAM++ experiment",
        "",
        "## Execution contract",
        "",
        ("Two shared-head and eight forked-tail runs completed. Both fold-specific "
         "known-only arm choices and all heldout open-set policies were sealed before "
         "the first outer label was materialized. Checkpoints and embedding caches "
         "remained on the training server and were not uploaded to MLflow."),
        "",
        "## Selection and decision",
        "",
        f"- Selected arms by outer fold: `{json.dumps(result.get('selected_arms_by_outer', {}), sort_keys=True)}`",
        f"- Selection kind: `{result.get('selection_kind', 'unavailable')}`",
        f"- Promotion verdict: `{result.get('decision', 'unavailable')}`",
        f"- All promotion gates passed: `{bool(result.get('all_conditions_passed', False))}`",
        f"- Development OOF target: `{result.get('goal_target_oof_macro_f1', 0.965)}`",
        f"- Development metric goal reached: `{bool(result.get('development_metric_goal_reached', False))}`",
        f"- Project goal reached: `{bool(result.get('goal_reached', False))}`",
        f"- Project goal status: `{result.get('goal_status', 'unavailable')}`",
        "",
        "## Pooled OOF metrics",
        "",
        "| Comparator | Macro-F1 | Accuracy | K→U | U→K | K→K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result.get("metrics", {}).items():
        errors = metrics.get("errors", {})
        lines.append(
            f"| {name} | {metrics.get('macro_f1', float('nan')):.9f} | "
            f"{metrics.get('accuracy', float('nan')):.9f} | "
            f"{errors.get('known_to_unknown', 'n/a')} | "
            f"{errors.get('unknown_to_known', 'n/a')} | "
            f"{errors.get('known_to_other_known', 'n/a')} |"
        )
    lines.extend(["", "## Outer-fold metrics", "",
                  "| Fold | Comparator | Macro-F1 | Accuracy |", "|---:|---|---:|---:|"])
    for outer, comparators in sorted(result.get("fold_metrics", {}).items()):
        for name, metrics in comparators.items():
            lines.append(
                f"| {outer} | {name} | {metrics.get('macro_f1', float('nan')):.9f} | "
                f"{metrics.get('accuracy', float('nan')):.9f} |"
            )
    lines.extend(["", "## Deltas", ""])
    for name, value in result.get("deltas", {}).items():
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        lines.append(f"- {name}: `{rendered}`")
    lines.extend(["", "## Whole-content-group bootstrap", "",
                  "| Comparison | Point delta | 95% lower | 95% upper | Groups | Mixed-label groups |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, bootstrap in result.get("bootstraps", {}).items():
        point = bootstrap.get("point_estimate", {})
        interval = bootstrap.get("confidence_interval", {})
        counts = bootstrap.get("counts", {})
        lines.append(
            f"| {name} | {point.get('delta_macro_f1', float('nan')):.9f} | "
            f"{interval.get('lower_delta_macro_f1', float('nan')):.9f} | "
            f"{interval.get('upper_delta_macro_f1', float('nan')):.9f} | "
            f"{counts.get('groups', 'n/a')} | {counts.get('mixed_label_groups', 'n/a')} |"
        )
    lines.extend(["", "## Promotion gates", ""])
    for name, passed in result.get("conditions", {}).items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    short = result.get("short_known_top1", {})
    if short:
        lines.extend([
            "", "## Short-known ranking diagnostic", "",
            f"`{json.dumps(short, ensure_ascii=False, sort_keys=True, allow_nan=False)}`",
        ])
    note = result.get("historical_comparator_note")
    if note:
        lines.extend(["", "## Comparator interpretation", "", str(note)])
    return "\n".join(lines) + "\n"


def execute_f005_experiment(
        contract: dict, root: Path, config_path: Path, binding_path: Path, *,
        resume_dir: Path | None = None, output_dir: Path | None = None,
        backend=None) -> dict:
    """Execute or resume F005 after the caller explicitly requested execution."""
    root, config_path, binding_path = Path(root).resolve(), Path(config_path).resolve(), Path(binding_path).resolve()
    config = contract["config"]
    backend = backend or DefaultF005Backend()
    _require(contract.get("source_verification") is not None,
             "F005 execution requires verified pinned model and C002b source receipts")
    _require(contract["readiness"].get("summary", {}).get("audio_hashes_checked") is True,
             "F005 execution requires verify_audio=True")
    _require(not (resume_dir is not None and output_dir is not None),
             "F005 accepts either resume_dir or a fresh output_dir")
    if resume_dir is not None:
        output = _confined_output(root, config["output_root"], resume_dir, existing=True)
        state_path = output / "experiment_state.json"
        state = _read_json(state_path)
        _validate_state(state, contract, output)
        if state["status"] == "complete":
            report = _read_json(output / "experiment_report.json")
            return {"status": "complete", "resumed": True, "output": str(output), "report": report}
        resumed = True
    else:
        base = (root / config["output_root"]).resolve()
        base.mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = base / f"F005_{stamp}_{uuid.uuid4().hex}"
        output = _confined_output(root, config["output_root"], output_dir, existing=False)
        output.mkdir(parents=True, exist_ok=False)
        state_path = output / "experiment_state.json"
        state = _fresh_state(contract, output)
        _save_state(state_path, state)
        resumed = False

    environment = backend.require_environment(contract, root)
    plan_path, source_path, resolved_path = (
        output / "execution_plan.json", output / "source_verification.json",
        output / "resolved_config.json",
    )
    if not plan_path.exists():
        _write_json(plan_path, backend.execution_plan(contract))
    if not source_path.exists():
        _write_json(source_path, contract["source_verification"])
    resolved = {
        "schema_version": 1, "experiment": config,
        "experiment_signature": contract["signature"],
        "identity": contract["identity"], "source_verification": contract["source_verification"],
        "environment": environment, "execution_plan_sha256": _sha_file(plan_path),
        "retention": config["retention"], "resume_attempt": resumed,
        "model_or_optimizer_mlflow_upload": False,
        "embedding_cache_mlflow_upload": False,
    }
    if resolved_path.exists():
        previous = _read_json(resolved_path)
        _require(
            previous["experiment_signature"] == contract["signature"]
            and previous["execution_plan_sha256"] == resolved["execution_plan_sha256"],
            "F005 resolved execution changed on resume",
        )
    else:
        _write_json(resolved_path, resolved)

    binding = backend.load_binding(binding_path, config["mlflow"]["experiment_id"])
    parent_spool = output / "tracking" / "parent"
    if state["parent"] is None:
        parent = backend.prepare_tracker(
            project_root=root, spool_dir=parent_spool, binding=binding,
            run_name=config["run_name"], config=resolved,
            input_paths=_input_paths(root, config_path, binding_path, config),
            run_kind="f005_paired_long_short_full_experiment",
            training_started=True,
        )
        _tracker_live(parent, resume=False)
        state["parent"] = {"spool": str(parent_spool), "run_id": parent.run_id, "status": "RUNNING"}
        state["status"], state["phase"] = "running", "parent_verified"
        _save_state(state_path, state)
    else:
        parent = backend.open_tracker(Path(state["parent"]["spool"]))
        if state["status"] != "complete":
            if (parent.state.get("remote_status") == "FINISHED"
                    and state.get("aggregate") is not None
                    and (output / "experiment_report.json").is_file()):
                parent.verify_artifacts(); parent.verify_remote_metadata()
                state["status"], state["phase"] = "complete", "complete"
                state["parent"]["status"] = "FINISHED"
                _save_state(state_path, state)
                return {
                    "status": "complete", "resumed": True, "output": str(output),
                    "parent_run_id": parent.run_id,
                    "report": _read_json(output / "experiment_report.json"),
                }
            _tracker_live(parent, resume=True)
            state["parent"]["status"] = "RUNNING"
            state["status"] = "running"
            _save_state(state_path, state)
    for path, relative in (
        (plan_path, "receipts/execution_plan.json"),
        (source_path, "receipts/source_verification.json"),
        (resolved_path, "receipts/resolved_config.json"),
    ):
        _safe_add_artifact(parent, path, relative)
    parent.flush(strict=True); parent.verify_artifacts(); parent.verify_remote_metadata()

    active_child = None
    try:
        # Frozen public/advanced arrays are verified source caches.  Loading them
        # has no optimizer effect and exposes no new outer truth.
        sources = backend.load_frozen_sources(contract)
        valid = np.asarray(sources["valid"])
        _require(valid.shape == (len(contract["manifest"]),) and valid.dtype == np.bool_,
                 "F005 source validity mask changed")

        def child_tracker(key: str, run_name: str, run_kind: str, unit_config: dict):
            nonlocal active_child
            spool = output / "tracking" / "children" / key.replace("/", "__")
            record = state["children"].get(key)
            if record is None:
                tracker = backend.prepare_tracker(
                    project_root=root, spool_dir=spool, binding=binding,
                    run_name=run_name, config={**resolved, "unit": unit_config},
                    input_paths=_input_paths(root, config_path, binding_path, config),
                    run_kind=run_kind, training_started=True,
                    parent_run_id=parent.run_id,
                )
                _tracker_live(tracker, resume=False)
                state["children"][key] = {"spool": str(spool), "run_id": tracker.run_id, "status": "RUNNING"}
                _save_state(state_path, state)
            else:
                tracker = backend.open_tracker(Path(record["spool"]))
                unit_complete = key in state["heads"] or key in state["tails"]
                if tracker.state.get("remote_status") == "FINISHED" and unit_complete:
                    tracker.verify_artifacts(); tracker.verify_remote_metadata()
                    record["status"] = "FINISHED"
                    _save_state(state_path, state)
                elif record["status"] != "FINISHED":
                    _tracker_live(tracker, resume=True)
                    record["status"] = "RUNNING"
                    _save_state(state_path, state)
            active_child = tracker
            return tracker

        # All two heads are complete before any tail is fitted.
        state["phase"] = "training_shared_heads"; _save_state(state_path, state)
        for outer in config["fold_ids"]:
            key = _unit_key(outer)
            unit_dir = output / "training" / key / "shared_head"
            child = child_tracker(
                key, f"F005-shared-head-fold{outer}", "f005_shared_head",
                {"stage": "shared_head", "outer_fold": outer, "arm": None},
            )
            if key not in state["heads"]:
                result = backend.fit_head(
                    contract, root, outer, unit_dir, child,
                    resume=(unit_dir / "shared_head.pt").exists(),
                )
                report_path = unit_dir / "unit_report.json"
                _write_json(report_path, result["report"])
                _safe_add_artifact(child, report_path, "training/unit_report.json")
                child.log_metrics({"fit/completed_steps": result["report"]["completed_steps"]}, sync=False)
                child.write_report(result["report"])
                state["heads"][key] = {
                    "checkpoint": str(result["checkpoint"]),
                    "checkpoint_sha256": _sha_file(result["checkpoint"]),
                    "report": str(report_path), "complete": True,
                }
                _save_state(state_path, state)
            child.finish("FINISHED", strict=True); child.verify_remote_metadata()
            state["children"][key]["status"] = "FINISHED"; _save_state(state_path, state)
            active_child = None

        # All eight tails start from their fold's exact same shared checkpoint.
        state["phase"] = "training_tails"; _save_state(state_path, state)
        for outer in config["fold_ids"]:
            shared = Path(state["heads"][_unit_key(outer)]["checkpoint"])
            for arm in config["arms"]:
                arm_id, key = arm["id"], _unit_key(outer, arm["id"])
                unit_dir = output / "training" / f"fold_{outer}" / "tails" / arm_id
                child = child_tracker(
                    key, f"F005-tail-fold{outer}-{arm_id}", "f005_tail_arm",
                    {"stage": "tail", "outer_fold": outer, "arm": arm},
                )
                if key not in state["tails"]:
                    result = backend.fit_tail(
                        contract, root, outer, arm_id, shared, unit_dir, child,
                        resume=(unit_dir / "last.pt").exists(),
                    )
                    report_path = unit_dir / "unit_report.json"
                    _write_json(report_path, result["report"])
                    _safe_add_artifact(child, report_path, "training/unit_report.json")
                    child.log_metrics({"fit/completed_steps": result["report"]["completed_steps"]}, sync=False)
                    child.write_report(result["report"])
                    state["tails"][key] = {
                        "checkpoint": str(result["checkpoint"]),
                        "checkpoint_sha256": _sha_file(result["checkpoint"]),
                        "shared_checkpoint_sha256": _sha_file(shared),
                        "report": str(report_path), "complete": True,
                    }
                    _save_state(state_path, state)
                child.finish("FINISHED", strict=True); child.verify_remote_metadata()
                state["children"][key]["status"] = "FINISHED"; _save_state(state_path, state)
                active_child = None

        _require(len(state["heads"]) == 2 and len(state["tails"]) == 8,
                 "F005 requires exactly two complete heads and eight complete tails")

        # Adapt only known, permitted non-outer rows for arm selection.  All
        # other dense rows remain the verified frozen advanced endpoint; the
        # known-only scorer never consumes them for an arm comparison.
        state["phase"] = "known_only_arm_scoring"; _save_state(state_path, state)
        for outer in config["fold_ids"]:
            indices = _selection_indices(contract, outer, valid)
            shared = Path(state["heads"][_unit_key(outer)]["checkpoint"])
            for arm in config["arms"]:
                arm_id, key = arm["id"], _unit_key(outer, arm["id"])
                score_path = output / "selection" / f"fold_{outer}" / arm_id / "known_scores.npz"
                if key not in state["known_scores"]:
                    extracted = backend.extract_arm(
                        contract, root, outer, arm_id, shared,
                        Path(state["tails"][key]["checkpoint"]), indices,
                        "known_selection", valid,
                        output / "selection" / f"fold_{outer}" / arm_id / "embedding_cache",
                        lambda done, total, elapsed, o=outer, a=arm_id: parent.log_metrics(
                            {f"selection/fold{o}/{a}/files": done,
                             f"selection/fold{o}/{a}/elapsed_seconds": elapsed},
                            step=done, sync=(done == total), strict=False,
                        ),
                    )
                    dense = np.asarray(sources["frozen_advanced"]).copy()
                    dense[indices] = extracted["embeddings"]
                    scores = backend.known_scores(dense, valid, contract, outer)
                    receipt = _save_known_scores(score_path, arm_id, contract, scores)
                    state["known_scores"][key] = {
                        **receipt, "adapted_rows": len(indices),
                        "scope": "known_nonouter_only",
                        "unknown_or_outer_arm_embeddings_extracted": False,
                    }
                    parent.log_metrics({
                        f"selection/fold{outer}/{arm_id}/known_rows": len(scores["known_calibration_indices"]),
                    }, sync=False)
                    _save_state(state_path, state)

        # Both arm choices are immutable and disk-reloaded before unknown
        # calibration similarities or any arm-specific outer embedding exists.
        state["phase"] = "sealing_arms"; _save_state(state_path, state)
        arm_reloads = {}
        for outer in config["fold_ids"]:
            score_sets = {
                arm["id"]: _load_known_scores(
                    Path(state["known_scores"][_unit_key(outer, arm["id"])]["path"]),
                    arm["id"], contract,
                ) for arm in config["arms"]
            }
            seal_path = output / "selection" / f"fold_{outer}" / "arm_selection_seal.json"
            reload = backend.seal_arm(contract, outer, score_sets, seal_path)
            arm_reloads[outer] = reload
            state["arm_seals"][str(outer)] = {
                "path": str(seal_path), "selected_arm": reload["seal"]["selected_arm"],
                "seal_sha256": reload["seal_sha256"], "file_sha256": reload["file_sha256"],
                "disk_reloaded": True,
            }
            _safe_add_artifact(parent, seal_path, f"fold_{outer}/arm_selection_seal.json")
            selected = reload["seal"]["selected_arm"]
            parent.log_metrics({
                f"selection/fold{outer}/selected_control": float(selected == "control"),
                **{f"selection/fold{outer}/{arm}/known_macro_f1":
                   reload["seal"]["arm_metrics"][arm]["macro_f1_observed_known_labels"]
                   for arm in reload["seal"]["arm_metrics"]},
            }, sync=False)
            _save_state(state_path, state)
        parent.flush(strict=True); parent.verify_artifacts(); parent.verify_remote_metadata()

        # Full extraction is limited to control and the already sealed winner.
        state["phase"] = "selected_full_extraction"; _save_state(state_path, state)
        full_arrays: dict[int, dict[str, np.ndarray]] = {}
        for outer in config["fold_ids"]:
            selected = state["arm_seals"][str(outer)]["selected_arm"]
            required = tuple(dict.fromkeys(("control", selected)))
            _require(set(required).issubset({arm["id"] for arm in config["arms"]}),
                     "F005 selected arm is unknown")
            full_arrays[outer] = {}
            shared = Path(state["heads"][_unit_key(outer)]["checkpoint"])
            indices = np.arange(len(contract["manifest"]), dtype=np.int64)
            for arm_id in required:
                key = _unit_key(outer, arm_id)
                cache_root = output / "full_scoring" / f"fold_{outer}" / arm_id
                extracted = backend.extract_arm(
                    contract, root, outer, arm_id, shared,
                    Path(state["tails"][key]["checkpoint"]), indices,
                    "full_scoring", valid, cache_root / "embedding_cache",
                    lambda done, total, elapsed, o=outer, a=arm_id: parent.log_metrics(
                        {f"full/fold{o}/{a}/files": done,
                         f"full/fold{o}/{a}/elapsed_seconds": elapsed},
                        step=done, sync=(done == total), strict=False,
                    ),
                )
                full_arrays[outer][arm_id] = extracted["embeddings"]
                state["full_embeddings"][key] = {
                    "scope": "all_manifest_rows", "rows": len(indices),
                    "cache_receipt": extracted["receipt_path"],
                    "cache_receipt_sha256": _sha_file(Path(extracted["receipt_path"])),
                    "mlflow_uploaded": False,
                }
                _save_state(state_path, state)
            rejected = {arm["id"] for arm in config["arms"]} - set(required)
            state["full_embeddings"][f"fold_{outer}/rejected_arms"] = {
                "arms": sorted(rejected), "outer_or_unknown_embeddings_extracted": False,
            }
            _save_state(state_path, state)

        # Seal policies for both folds, persist score arrays server-side, then
        # verify disk reloads.  No outer truth is accepted by this API.
        state["phase"] = "sealing_inner_policies"; _save_state(state_path, state)
        pretruth_by_fold, policy_reloads = {}, {}
        for outer in config["fold_ids"]:
            seal_path = output / "full_scoring" / f"fold_{outer}" / "policy_seal.json"
            cache_base = output / "full_scoring" / f"fold_{outer}" / "pretruth_bundle"
            if cache_base.with_suffix(".json").exists():
                pretruth = _load_pretruth(cache_base)
                policy_reload = backend.reload_policy(seal_path, contract, outer, arm_reloads[outer])
                pretruth["policy_reload"] = policy_reload
            elif seal_path.exists():
                policy_reload = backend.reload_policy(seal_path, contract, outer, arm_reloads[outer])
                pretruth = backend.rebuild_policy(
                    contract, outer, public=sources["public"], frozen=sources["frozen_advanced"],
                    selected_embeddings=full_arrays[outer], valid=valid,
                    arm_reload=arm_reloads[outer], policy_reload=policy_reload,
                )
                _save_pretruth(cache_base, pretruth)
            else:
                pretruth = backend.prepare_policy(
                    contract, outer, public=sources["public"], frozen=sources["frozen_advanced"],
                    selected_embeddings=full_arrays[outer], valid=valid,
                    arm_reload=arm_reloads[outer], path=seal_path,
                )
                _save_pretruth(cache_base, pretruth)
                policy_reload = pretruth["policy_reload"]
            pretruth_by_fold[outer], policy_reloads[outer] = pretruth, policy_reload
            state["policy_seals"][str(outer)] = {
                "path": str(seal_path), "seal_sha256": policy_reload["seal_sha256"],
                "file_sha256": policy_reload["file_sha256"], "disk_reloaded": True,
                "pretruth_cache": str(cache_base.with_suffix(".json")),
                "pretruth_cache_mlflow_uploaded": False,
            }
            _safe_add_artifact(parent, seal_path, f"fold_{outer}/policy_seal.json")
            _save_state(state_path, state)
        _require(len(state["policy_seals"]) == len(config["fold_ids"]),
                 "F005 all fold policies must be sealed before outer truth")
        parent.flush(strict=True); parent.verify_artifacts(); parent.verify_remote_metadata()

        if state["parity"] is None:
            parity = backend.verify_parity(pretruth_by_fold)
            parity_path = output / "cpu_cuda_prediction_parity.json"
            _write_json(parity_path, parity)
            _safe_add_artifact(parent, parity_path, "receipts/cpu_cuda_prediction_parity.json")
            state["parity"] = {**parity, "path": str(parity_path), "file_sha256": _sha_file(parity_path)}
            _save_state(state_path, state)
        else:
            parity = state["parity"]

        # This flag changes only after both policy seals and parity evidence are
        # durable.  Outer evaluation paths are immutable one-shot receipts.
        state["phase"] = "one_shot_outer_evaluation"
        state["outer_truth_materialized"] = True
        _save_state(state_path, state)
        fold_results = []
        for outer in config["fold_ids"]:
            evaluation_path = output / "evaluation" / f"fold_{outer}.json"
            if evaluation_path.exists():
                result = _read_json(evaluation_path)
            else:
                rows = _outer_truth_rows(contract, outer)
                result = backend.evaluate_outer(
                    pretruth_by_fold[outer], policy_reloads[outer], rows,
                    contract["labels"], evaluation_path,
                )
            fold_results.append(result)
            state["outer_evaluations"][str(outer)] = {
                "path": str(evaluation_path), "file_sha256": _sha_file(evaluation_path),
                "one_shot": True,
            }
            _safe_add_artifact(parent, evaluation_path, f"fold_{outer}/outer_evaluation.json")
            for comparator, row in result["comparators"].items():
                parent.log_metrics({
                    f"fold_{outer}/{comparator}/macro_f1": row["metrics"]["macro_f1"],
                    f"fold_{outer}/{comparator}/accuracy": row["metrics"]["accuracy"],
                }, sync=False)
            _save_state(state_path, state)

        state["phase"] = "oof_bootstrap_promotion"; _save_state(state_path, state)
        historical, historical_top1 = backend.load_historical_predictions(contract)
        report = backend.aggregate(
            contract, fold_results, historical, historical_top1,
            bool(parity["exact_cpu_cuda_prediction_parity"]),
        )
        report = {
            "status": "complete", "parent_run_id": parent.run_id,
            "experiment_signature": contract["signature"],
            "child_run_count": len(state["children"]),
            "shared_head_child_runs": 2, "tail_child_runs": 8,
            "outer_truth_materialized_after_all_policy_seals": True,
            "rejected_arm_full_extraction": False,
            "checkpoint_and_embedding_caches_server_only": True,
            "mlflow_forbidden_payloads_uploaded": False,
            "result": report,
        }
        report_path = output / "experiment_report.json"
        _write_json(report_path, report)
        _safe_add_artifact(parent, report_path, "experiment_report.json")
        oof = report["result"].get("metrics", {}).get("selected_arm", {})
        metrics = {}
        if "macro_f1" in oof:
            metrics["oof/selected_macro_f1_447"] = oof["macro_f1"]
        if "accuracy" in oof:
            metrics["oof/selected_accuracy"] = oof["accuracy"]
        metrics.update({
            "decision/promote_new_incumbent": float(bool(report["result"].get("promote_new_incumbent"))),
            "decision/development_metric_goal_reached_0_965": float(bool(
                report["result"].get("development_metric_goal_reached")
            )),
            "decision/goal_reached_0_965": float(bool(report["result"].get("goal_reached"))),
            "execution/child_runs": len(state["children"]),
        })
        parent.log_metrics(metrics, sync=False)
        parent.write_report(report, markdown=_experiment_markdown(report))
        state["aggregate"] = {"path": str(report_path), "sha256": _sha_file(report_path)}
        state["phase"] = "finalizing_parent"
        _save_state(state_path, state)
        parent.finish("FINISHED", strict=True); parent.verify_remote_metadata()
        state["status"], state["phase"] = "complete", "complete"
        state["parent"]["status"] = "FINISHED"
        _save_state(state_path, state)
        return {
            "status": "complete", "resumed": resumed, "output": str(output),
            "parent_run_id": parent.run_id, "report": report,
        }
    except BaseException as error:
        failure = {
            "status": "failed", "phase": state.get("phase"),
            "error_type": type(error).__name__,
            "error": getattr(parent, "redactor", None).text(str(error))
            if getattr(parent, "redactor", None) is not None else str(error),
            "resume_directory": str(output), "resume_supported": True,
            "outer_truth_materialized": state.get("outer_truth_materialized", False),
            "model_or_optimizer_mlflow_upload": False,
            "embedding_cache_mlflow_upload": False,
        }
        failure_path = output / "failure.json"
        _write_json(failure_path, failure)
        try:
            _safe_add_artifact(parent, failure_path, "failure.json")
            parent.write_report(failure)
        except Exception:
            pass
        _finish_quietly(active_child, "FAILED")
        _finish_quietly(parent, "FAILED")
        state["status"] = "failed"
        state["failure"] = failure
        if active_child is not None:
            for record in state["children"].values():
                if record.get("run_id") == active_child.run_id:
                    record["status"] = "FAILED"
        if state.get("parent"):
            state["parent"]["status"] = "FAILED"
        _save_state(state_path, state)
        raise
