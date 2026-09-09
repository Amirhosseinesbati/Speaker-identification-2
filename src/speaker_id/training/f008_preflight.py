"""Role-safe source-energy preflight for F008 outlier exposure.

The preflight is intentionally inference-only.  It starts from each
authenticated F005 shared-head checkpoint, reads only the role-safe known and
unknown encoder-fit pools, and seals the two energy margins before an F008
tail worker may take an optimizer step.  It never reads calibration or outer
labels, and it writes only scalar summaries and provenance receipts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from speaker_id.training.f008_energy_margin_plan import (
    build_energy_margin_plan,
    verify_energy_margin_plan,
)
from speaker_id.training.f008_protocol import canonical, role_pools


F008_PREFLIGHT_SCHEMA = "f008-shared-head-energy-preflight-v1"
_SHA256 = frozenset("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(item in _SHA256 for item in value),
        f"F008 {label} must be a lowercase SHA-256",
    )
    return value


def _finite_positive(value: object, label: str) -> float:
    _require(type(value) in (int, float) and not isinstance(value, bool),
             f"F008 {label} must be a finite positive number")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0,
             f"F008 {label} must be a finite positive number")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_f005_control_selection(source_run_directory: Path,
                                    source_config: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Authenticate F005's sealed control choice before reusing its comparator.

    F008 forks the shared head, but it also names F005's control tail as its
    scientific comparator.  This small metadata-only check binds that claim to
    the immutable F005 state/seal bytes without opening a checkpoint or cache.
    """
    root = Path(source_run_directory)
    _require(root.is_dir() and not root.is_symlink(),
             "F008 F005 source run must be a regular directory")
    _require(isinstance(source_config, Mapping), "F008 F005 source config is invalid")
    expected = source_config.get("arm_selection_seals")
    selected = source_config.get("required_selected_arm_by_outer_fold")
    _require(isinstance(expected, Mapping) and set(expected) == {"0", "1"}
             and selected == {"0": "control", "1": "control"},
             "F008 expected F005 control seals are invalid")
    state_path = root / "experiment_state.json"
    _require(state_path.is_file() and not state_path.is_symlink(),
             "F008 F005 experiment state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _require(isinstance(state, Mapping) and state.get("status") == "complete"
             and isinstance(state.get("arm_seals"), Mapping),
             "F008 F005 source state is incomplete")
    summaries: dict[str, dict[str, str]] = {}
    for fold in ("0", "1"):
        expected_fold = expected[fold]
        _require(isinstance(expected_fold, Mapping) and set(expected_fold) == {
            "file_sha256", "seal_sha256", "selected_arm"
        } and expected_fold.get("selected_arm") == "control",
                 "F008 expected F005 control seal is malformed")
        record = state["arm_seals"].get(fold)
        _require(isinstance(record, Mapping)
                 and record.get("selected_arm") == selected[fold]
                 and record.get("file_sha256") == expected_fold["file_sha256"]
                 and record.get("seal_sha256") == expected_fold["seal_sha256"],
                 "F008 F005 selected arm differs from its pinned control seal")
        seal_path = root / "selection" / f"fold_{fold}" / "arm_selection_seal.json"
        _require(seal_path.is_file() and not seal_path.is_symlink()
                 and _file_sha256(seal_path) == expected_fold["file_sha256"],
                 "F008 F005 arm-selection seal bytes changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        _require(isinstance(seal, Mapping) and seal.get("selected_arm") == "control"
                 and seal.get("seal_sha256") == expected_fold["seal_sha256"],
                 "F008 F005 arm-selection seal content changed")
        summaries[fold] = {
            "selected_arm": "control",
            "file_sha256": expected_fold["file_sha256"],
            "seal_sha256": expected_fold["seal_sha256"],
        }
    return summaries


def source_crop_seed(role_pool_signature: str, *, outer_fold: int, stream: str,
                     audio_file: str) -> int:
    """Return the one fixed paired-crop seed used only by source preflight."""
    _sha256(role_pool_signature, "role-pool signature")
    _require(type(outer_fold) is int and outer_fold >= 0,
             "F008 preflight outer fold must be a nonnegative integer")
    _require(stream in {"known", "unknown"}, "F008 preflight stream is invalid")
    _require(isinstance(audio_file, str) and audio_file and audio_file == audio_file.strip(),
             "F008 preflight audio file is invalid")
    digest = hashlib.sha256(
        canonical(("F008-source-energy-crop-v1", role_pool_signature, outer_fold, stream, audio_file))
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _source_plan(rows: Sequence[Mapping[str, object]], role_pool_signature: str,
                 outer_fold: int, stream: str) -> list[dict[str, object]]:
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows,
             "F008 preflight source rows must be nonempty")
    names: list[str] = []
    plan: list[dict[str, object]] = []
    for slot, row in enumerate(rows):
        _require(isinstance(row, Mapping), "F008 preflight source row must be an object")
        name = row.get("audio_file")
        _require(isinstance(name, str) and name and name == name.strip(),
                 "F008 preflight source audio file is invalid")
        names.append(name)
        plan.append({
            "slot": slot,
            "audio_file": name,
            # _microbatch needs known AAM targets but this inference-only
            # preflight never evaluates AAM; a fixed placeholder is safe.
            "target": 0,
            "group_id": row.get("group_id"),
            "crop_seed": source_crop_seed(role_pool_signature, outer_fold=outer_fold,
                                           stream=stream, audio_file=name),
        })
    _require(len(names) == len(set(names)), "F008 preflight source rows have duplicate files")
    return plan


def _energy_values(encoder: object, head: object, contract: Mapping[str, object], root: Path,
                   plan: Sequence[Mapping[str, object]], *, energy_temperature: float,
                   microbatch_pairs: int) -> np.ndarray:
    """Extract both paired-view source energies without targets or gradients."""
    import torch

    from speaker_id.adaptation.open_set_oe import raw_cosine_logits
    from speaker_id.training.f005_worker import _microbatch

    _finite_positive(energy_temperature, "energy temperature")
    _require(type(microbatch_pairs) is int and microbatch_pairs > 0,
             "F008 preflight microbatch size must be positive")
    _require(bool(plan), "F008 preflight plan must be nonempty")
    values: list[np.ndarray] = []
    device = next(encoder.parameters()).device
    _require(device.type == "cuda", "F008 source-energy preflight requires CUDA")
    encoder.eval(); head.eval()
    with torch.no_grad():
        for start in range(0, len(plan), microbatch_pairs):
            batch = _microbatch(dict(contract), list(plan[start:start + microbatch_pairs]), Path(root))
            short = batch["short"].to(device, dtype=torch.float32, non_blocking=False)
            long = batch["long"].to(device, dtype=torch.float32, non_blocking=False)
            short_h, long_h = encoder(short), encoder(long)
            logits = torch.cat((
                raw_cosine_logits(short_h, head.weight),
                raw_cosine_logits(long_h, head.weight),
            ), dim=0)
            energy = -energy_temperature * torch.logsumexp(logits / energy_temperature, dim=1)
            _require(energy.dtype == torch.float32 and bool(torch.isfinite(energy).all().item()),
                     "F008 source energy is nonfinite")
            values.append(energy.detach().cpu().numpy().astype(np.float64, copy=True))
    result = np.concatenate(values)
    _require(result.shape == (2 * len(plan),) and np.isfinite(result).all(),
             "F008 source-energy vector is malformed")
    return result


def build_preflight_receipt(*, f008_signature: str, f005_signature: str,
                            f005_source_receipt: Mapping[str, object], outer_fold: int,
                            role_pool: Mapping[str, object], shared_head_checkpoint: Path,
                            energy_margin_config: Mapping[str, object],
                            known_energies: Sequence[float], unknown_energies: Sequence[float],
                            source_view_algorithm: str) -> dict[str, Any]:
    """Seal source provenance, margins, and scalar diagnostics for one fold."""
    _sha256(f008_signature, "experiment signature")
    _sha256(f005_signature, "F005 signature")
    _require(type(outer_fold) is int and outer_fold >= 0,
             "F008 preflight outer fold must be nonnegative")
    _require(isinstance(f005_source_receipt, Mapping), "F008 F005 source receipt is invalid")
    _require(Path(shared_head_checkpoint).is_file() and not Path(shared_head_checkpoint).is_symlink(),
             "F008 shared-head checkpoint must be a regular file")
    _require(isinstance(source_view_algorithm, str) and source_view_algorithm,
             "F008 source-view algorithm is required")
    pools = dict(role_pool)
    # ``role_pools`` has already canonicalized the input.  Validate its digest
    # through the counter-derived unknown-plan path, which refuses malformed pools.
    from speaker_id.training.f008_protocol import unknown_exposure_plan
    unknown_exposure_plan(pools, 0, seed=0, samples_per_step=1)
    _require(pools.get("outer_fold") == outer_fold,
             "F008 role-pool fold differs from the preflight fold")
    margin = build_energy_margin_plan(known_energies, unknown_energies, energy_margin_config)
    verify_energy_margin_plan(margin)
    known = np.asarray(known_energies, dtype=np.float64)
    unknown = np.asarray(unknown_energies, dtype=np.float64)
    _require(known.ndim == unknown.ndim == 1 and len(known) and len(unknown)
             and np.isfinite(known).all() and np.isfinite(unknown).all(),
             "F008 preflight energy inputs are malformed")
    source_receipt_sha = _sha(dict(f005_source_receipt))
    checkpoint_sha = _file_sha256(Path(shared_head_checkpoint))
    body = {
        "schema_version": F008_PREFLIGHT_SCHEMA,
        "f008_signature": f008_signature,
        "f005_signature": f005_signature,
        "f005_source_receipt_sha256": source_receipt_sha,
        "outer_fold": outer_fold,
        "role_pool_signature": pools["signature"],
        "shared_head_checkpoint_sha256": checkpoint_sha,
        "source_view_algorithm": source_view_algorithm,
        "known_energy_views": int(len(known)),
        "unknown_energy_views": int(len(unknown)),
        "energy_margin_plan": margin,
        "checkpoint_scope": "server_only_until_promotion",
        "mlflow_payload": "scalar_receipt_only_no_audio_embeddings_weights_optimizer",
    }
    return {**body, "receipt_sha256": _sha(body)}


def verify_preflight_receipt(receipt: Mapping[str, object]) -> dict[str, Any]:
    """Validate a persisted margin receipt before an F008 tail can use it."""
    required = {
        "schema_version", "f008_signature", "f005_signature", "f005_source_receipt_sha256",
        "outer_fold", "role_pool_signature", "shared_head_checkpoint_sha256",
        "source_view_algorithm", "known_energy_views", "unknown_energy_views",
        "energy_margin_plan", "checkpoint_scope", "mlflow_payload", "receipt_sha256",
    }
    _require(isinstance(receipt, Mapping) and set(receipt) == required,
             "F008 preflight receipt schema changed")
    body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    _require(
        receipt.get("schema_version") == F008_PREFLIGHT_SCHEMA
        and receipt.get("receipt_sha256") == _sha(body)
        and receipt.get("checkpoint_scope") == "server_only_until_promotion"
        and receipt.get("mlflow_payload") == "scalar_receipt_only_no_audio_embeddings_weights_optimizer",
        "F008 preflight receipt identity changed",
    )
    for key in (
        "f008_signature", "f005_signature", "f005_source_receipt_sha256",
        "role_pool_signature", "shared_head_checkpoint_sha256",
    ):
        _sha256(receipt[key], key)
    _require(type(receipt["outer_fold"]) is int and receipt["outer_fold"] >= 0,
             "F008 preflight receipt outer fold is invalid")
    _require(isinstance(receipt["source_view_algorithm"], str) and receipt["source_view_algorithm"],
             "F008 preflight receipt source-view algorithm changed")
    for key in ("known_energy_views", "unknown_energy_views"):
        _require(type(receipt[key]) is int and receipt[key] > 0,
                 f"F008 preflight receipt {key} is invalid")
    verify_energy_margin_plan(receipt["energy_margin_plan"])
    return dict(receipt)


def run_source_energy_preflight(f005_contract: Mapping[str, object], *, root: Path,
                                f008_signature: str, source_receipt: Mapping[str, object],
                                outer_fold: int, energy_margin_config: Mapping[str, object],
                                unknown_microbatch_pairs: int) -> dict[str, Any]:
    """Compute and seal one fold's shared-head source margins; zero optimizer steps."""
    import torch

    from speaker_id.training.f007_worker import _load_authenticated_shared_head, _make_components

    _sha256(f008_signature, "experiment signature")
    config = f005_contract.get("config")
    _require(isinstance(config, Mapping) and outer_fold in config.get("fold_ids", ()),
             "F008 preflight F005 contract/fold is invalid")
    pools = role_pools(f005_contract.get("roles", ()), outer_fold)
    # Its F007-named schema is an implementation detail of the authenticated
    # F005 reader, not F008 state.  The caller has already read and verified it.
    expected_source = source_receipt
    _require(isinstance(expected_source, Mapping), "F008 preflight source receipt is invalid")
    source_root = Path(expected_source.get("source_run_directory", ""))
    _require(source_root.is_dir(), "F008 preflight source run is unavailable")
    candidate = next(
        (row for row in expected_source.get("folds", ()) if row.get("outer_fold") == outer_fold),
        None,
    )
    _require(isinstance(candidate, Mapping), "F008 preflight source fold is unavailable")
    checkpoint = source_root / Path(candidate["shared_head"]["checkpoint"]["path"])
    payload, checkpoint_sha = _load_authenticated_shared_head(
        dict(f005_contract), dict(expected_source), Path(root), outer_fold, checkpoint,
    )
    encoder, head, _optimizer, _trainable, _seed = _make_components(
        dict(f005_contract), Path(root), outer_fold,
    )
    encoder.load_state_dict(payload["encoder"], strict=True)
    head.load_state_dict(payload["head"], strict=True)
    _require(checkpoint_sha == candidate["shared_head"]["checkpoint"]["sha256"],
             "F008 preflight shared-head checksum changed")
    temperature = _finite_positive(energy_margin_config.get("energy_temperature"),
                                   "energy temperature")
    known_plan = _source_plan(pools["known_rows"], pools["signature"], outer_fold, "known")
    unknown_plan = _source_plan(pools["unknown_rows"], pools["signature"], outer_fold, "unknown")
    known = _energy_values(encoder, head, f005_contract, Path(root), known_plan,
                           energy_temperature=temperature,
                           microbatch_pairs=unknown_microbatch_pairs)
    unknown = _energy_values(encoder, head, f005_contract, Path(root), unknown_plan,
                             energy_temperature=temperature,
                             microbatch_pairs=unknown_microbatch_pairs)
    receipt = build_preflight_receipt(
        f008_signature=f008_signature,
        f005_signature=f005_contract["signature"], f005_source_receipt=expected_source,
        outer_fold=outer_fold, role_pool=pools, shared_head_checkpoint=checkpoint,
        energy_margin_config=energy_margin_config, known_energies=known, unknown_energies=unknown,
        source_view_algorithm="f005_paired_nested_views_one_hash_seed_per_fit_file_v1",
    )
    verify_preflight_receipt(receipt)
    # Explicitly free the source models before a tail worker can allocate its own copy.
    del encoder, head, payload
    torch.cuda.empty_cache()
    return receipt
