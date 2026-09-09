"""Run F008 E0: the energy-only, calibration-only open-set screen.

This launcher deliberately runs *one* new arm, ``energy_005``, from the
authenticated F005 shared heads.  It reuses the already completed F005 control
cache and the already sealed F008 source-energy preflight.  The only decision
made here is an inner, role-safe calibration signal comparing
``control_f005`` and ``energy_005``.  It never calls the outer-evaluation API,
never reads outer labels, never treats a calibration result as a promotion,
and explicitly defers ``uniform_005``.

All checkpoints and embedding caches remain in this run directory on the GPU
server.  MLflow receives only text/JSON/JSONL receipts, scalar metrics, and
the automatic source-code snapshot produced by ``DurableMLflowRun``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PREFLIGHT_DIRECTORY = Path(
    "artifacts/training/f008_unknown_oe/F008_PREFLIGHT_20260909T185639Z_78964e7c3f9e"
)
_METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".py", ".txt"})
_FORBIDDEN_ARTIFACT_PARTS = frozenset({"embedding_cache", "entries", "checkpoints", "optimizer", "raw_audio"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"F008 {label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F008 {label} is unreadable") from error
    _require(isinstance(value, dict), f"F008 {label} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a metadata receipt owned by this new screen run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _confined(path: Path, prefix: str, *, must_exist: bool) -> Path:
    """Resolve an input/output path without allowing symlink or root escape."""
    root = ROOT.resolve()
    raw = Path(path)
    unresolved = raw.absolute() if raw.is_absolute() else (root / raw).absolute()
    resolved = unresolved.resolve(strict=must_exist)
    base = (root / prefix).resolve()
    _require(not unresolved.is_symlink() and resolved.is_relative_to(base),
             f"F008 path must stay below {prefix}")
    return resolved


def _binding(path: Path):
    from speaker_id.tracking import ExperimentBinding

    payload = _read_json(path, "MLflow binding state")
    binding_payload = payload.get("binding")
    _require(isinstance(binding_payload, dict), "F008 MLflow binding state is malformed")
    binding = ExperimentBinding(**binding_payload)
    binding.validate()
    _require(binding.experiment_id == "1", "F008 requires MLflow experiment 1")
    return binding


def reduced_energy_screen_spec(config: Mapping[str, object]) -> dict[str, object]:
    """Make E0's explicit active-arm scoring surface without changing F008 config.

    The fixed F008 config still declares uniform OE for a later, separate
    experiment.  E0 must not fabricate a uniform cache or let the scorer pick
    it, so its temporary scoring spec contains only the authenticated F005
    control and the one actually trained energy arm.
    """
    from speaker_id.training.f008_scoring import (
        normalize_scoring_spec,
        scoring_spec_from_f008_config,
    )

    full = scoring_spec_from_f008_config(config)
    active = ["control_f005", "energy_005"]
    return normalize_scoring_spec({
        **full,
        "arm_ids": active,
        "control_arm_id": "control_f005",
        "arm_tie_order": active,
    })


def _safe_add_metadata_artifact(tracker, path: Path, relative_path: str) -> None:
    """Add only reviewable metadata; fail before any cache/checkpoint upload."""
    source = Path(path)
    lowered = {part.lower() for part in source.parts}
    _require(source.is_file() and not source.is_symlink(), "F008 MLflow artifact must be a regular file")
    _require(source.suffix.lower() in _METADATA_SUFFIXES,
             "F008 MLflow accepts only metadata/text artifacts in E0")
    _require(not (lowered & _FORBIDDEN_ARTIFACT_PARTS),
             "F008 refuses to upload a cache/checkpoint/raw-audio artifact")
    tracker.add_artifact(source, relative_path)


def _source_fold(source_receipt: Mapping[str, object], outer: int) -> dict[str, object]:
    rows = source_receipt.get("folds") if isinstance(source_receipt, Mapping) else None
    matches = [row for row in rows or () if isinstance(row, Mapping) and row.get("outer_fold") == outer]
    _require(len(matches) == 1, "F008 source receipt fold is missing or ambiguous")
    return dict(matches[0])


def _source_shared_head(source_root: Path, source_receipt: Mapping[str, object], outer: int) -> Path:
    fold = _source_fold(source_receipt, outer)
    try:
        entry = fold["shared_head"]["checkpoint"]
        relative, expected = entry["path"], entry["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("F008 source receipt lacks a shared-head checkpoint") from error
    _require(isinstance(relative, str) and isinstance(expected, str),
             "F008 source shared-head declaration is malformed")
    path = (source_root / Path(relative)).resolve(strict=True)
    _require(path.is_relative_to(source_root.resolve()) and path.is_file() and not path.is_symlink(),
             "F008 source shared-head path escapes its authenticated run")
    _require(_sha256_file(path) == expected, "F008 source shared-head bytes changed")
    return path


def _preflight_receipts(preflight_directory: Path, *, config_signature: str,
                        f005_signature: str, source_receipt: Mapping[str, object],
                        f005_contract: Mapping[str, object], source_root: Path,
                        fold_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Reload the existing immutable preflight receipts, never recompute them."""
    from speaker_id.training.f008_preflight import verify_preflight_receipt
    from speaker_id.training.f008_protocol import role_pools

    receipt_root = Path(preflight_directory) / "preflight"
    _require(receipt_root.is_dir() and not receipt_root.is_symlink(),
             "F008 preflight receipt directory is unavailable")
    source_digest = _canonical_sha256(dict(source_receipt))
    result: dict[int, dict[str, Any]] = {}
    for outer in fold_ids:
        path = receipt_root / f"fold_{outer}_energy_margin_receipt.json"
        receipt = verify_preflight_receipt(_read_json(path, f"preflight fold {outer}"))
        pool = role_pools(f005_contract.get("roles", ()), outer)
        head = _source_shared_head(source_root, source_receipt, outer)
        _require(
            receipt["f008_signature"] == config_signature
            and receipt["f005_signature"] == f005_signature
            and receipt["f005_source_receipt_sha256"] == source_digest
            and receipt["outer_fold"] == outer
            and receipt["role_pool_signature"] == pool["signature"]
            and receipt["shared_head_checkpoint_sha256"] == _sha256_file(head),
            "F008 existing preflight receipt belongs to different source/configuration",
        )
        result[outer] = receipt
    return result


def _runtime_summary(runtime_receipt: Mapping[str, object]) -> dict[str, object]:
    """Keep the runtime attestation readable and free of environment secrets."""
    keys = (
        "schema_version", "receipt_sha256", "f008_signature", "source_f005_signature",
        "expected_vast_instance_id", "vast_instance_id", "device", "gpu_name",
        "cuda_available", "cpu_threads",
        "checked_once_per_logical_run", "execution",
    )
    return {key: runtime_receipt[key] for key in keys if key in runtime_receipt}


def _log_probe_metrics(tracker, probe: Mapping[str, object], outer: int) -> None:
    metrics: dict[str, float] = {}
    for name in (
        "raw_encoder_gradient_norm", "raw_head_gradient_norm", "raw_total_gradient_norm",
        "task_loss", "oe_loss", "combined_loss", "oe_full_batch_views",
    ):
        value = probe.get(name)
        if type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value)):
            metrics[f"probe/{name}"] = float(value)
    metrics["probe/optimizer_steps_persisted"] = float(probe.get("optimizer_steps_persisted", 0))
    tracker.log_metrics(metrics, step=outer, sync=False)


def _log_history_metrics(tracker, history_path: Path, *, flush_every: int = 25) -> None:
    """Replay scalar worker history for a resumed/old worker without tensors."""
    for line in Path(history_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        _require(isinstance(event, dict) and type(event.get("step")) is int,
                 "F008 fit history event is malformed")
        metrics = {
            str(name): float(value)
            for name, value in event.items()
            if name not in {"step", "phase"}
            and type(value) in (int, float) and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        tracker.log_metrics(metrics, step=event["step"], sync=False)
        if event["step"] % flush_every == 0:
            tracker.flush(strict=False)


def _inner_summary(policy_reload: Mapping[str, object], *, outer: int) -> dict[str, object]:
    seal = policy_reload.get("seal") if isinstance(policy_reload, Mapping) else None
    _require(isinstance(seal, Mapping), "F008 pretruth policy reload lacks its seal")
    arms = seal.get("arm_policies")
    selection = seal.get("arm_selection")
    _require(isinstance(arms, Mapping) and isinstance(selection, Mapping),
             "F008 pretruth policy seal is malformed")
    compact: dict[str, object] = {}
    for arm in ("control_f005", "energy_005"):
        policy = arms.get(arm)
        _require(isinstance(policy, Mapping) and isinstance(policy.get("inner_metrics"), Mapping),
                 "F008 pretruth policy lacks active-arm metrics")
        metrics = policy["inner_metrics"]
        compact[arm] = {
            "inner_macro_f1_full_label_map": float(metrics["inner_macro_f1_full_label_map"]),
            "known_query_macro_f1_over_observed_labels": float(
                metrics["known_query_macro_f1_over_observed_labels"]
            ),
            "known_query_top1_accuracy": float(metrics["known_query_top1_accuracy"]),
            "advanced_weight": float(policy["advanced_weight"]),
        }
    return {
        "outer_fold": outer,
        "selected_arm": selection["selected_arm"],
        "eligible_arms": list(selection["eligible_arms"]),
        "rejected_by_known_preservation": list(selection["rejected_by_known_preservation"]),
        "minimum_known_macro_f1": float(selection["minimum_known_macro_f1"]),
        "arms": compact,
        "outer_truth_read": False,
    }


def _log_inner_metrics(tracker, summary: Mapping[str, object], outer: int) -> None:
    arms = summary["arms"]
    metrics: dict[str, float] = {}
    for arm, values in arms.items():
        for name, value in values.items():
            if type(value) in (int, float) and not isinstance(value, bool):
                metrics[f"inner/{arm}/{name}"] = float(value)
    metrics["inner/energy_selected"] = float(summary["selected_arm"] == "energy_005")
    metrics["inner/energy_eligible"] = float("energy_005" in summary["eligible_arms"])
    tracker.log_metrics(metrics, step=outer, sync=False)


def _extract_energy_cache(*, loaded_tail: Mapping[str, object], checkpoint: Path,
                          checkpoint_receipt: Mapping[str, object], f005_contract: Mapping[str, object],
                          fold_directory: Path, runtime_receipt: Mapping[str, object],
                          tracker) -> dict[str, Any]:
    """Make the server-only all-manifest energy cache with per-consumed-file hashes."""
    import numpy as np

    from speaker_id.candidates.campp_advanced import extract_advanced_embedding
    from speaker_id.training.f008_extraction import (
        build_f008_advanced_cache_plan,
        extract_or_load_f008_advanced_cache,
    )

    manifest = f005_contract["manifest"]
    readiness = f005_contract["readiness"]
    data_dir = (ROOT / readiness["config"]["data_dir"]).resolve(strict=True)
    _require(data_dir.is_dir() and not data_dir.is_symlink(), "F008 data directory is unavailable")
    plan = build_f008_advanced_cache_plan(
        loaded_tail["identity"], output_directory=fold_directory,
        relative_cache_directory="full_scoring/energy_005",
        checkpoint_receipt=checkpoint_receipt,
        manifest=manifest,
        inference={
            "algorithm": f005_contract["config"]["scoring"]["advanced_inference"],
            "input_sha256_verified_immediately_before_each_server_local_extraction": True,
            "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
        },
    )
    encoder = loaded_tail["encoder"]

    def extractor(index: int, declared: Mapping[str, str]) -> tuple[np.ndarray, bool]:
        row = manifest[index]
        name, digest = row.get("audio_file"), row.get("input_sha256")
        _require(
            declared == {"audio_file": name, "input_sha256": digest}
            and isinstance(name, str) and Path(name).name == name
            and isinstance(digest, str) and len(digest) == 64,
            "F008 extraction callback input declaration changed",
        )
        audio = (data_dir / name).resolve(strict=True)
        _require(audio.is_relative_to(data_dir) and audio.is_file() and not audio.is_symlink(),
                 "F008 extraction audio path escapes the configured data directory")
        # This is not a separate broad audit: each source file is checked only
        # at the instant the server-local callback is about to consume it.
        _require(_sha256_file(audio) == digest, "F008 extraction input audio hash changed")
        vector, info = extract_advanced_embedding(
            encoder, audio, device="cuda", seconds=180.0, maximum_windows=1,
        )
        return vector, bool(info["nonzero_signal"])

    started = time.monotonic()

    def progress(completed: int, total: int) -> None:
        if completed % 100 == 0 or completed == total:
            tracker.log_metrics({
                "cache/rows_completed": float(completed),
                "cache/rows_total": float(total),
                "cache/fraction": float(completed / total),
                "cache/elapsed_seconds": time.monotonic() - started,
            }, step=completed, sync=False)
            tracker.flush(strict=False)

    return extract_or_load_f008_advanced_cache(
        plan, checkpoint_path=checkpoint, extractor=extractor, progress=progress,
    )


def _new_output(config: Mapping[str, object], requested: Path | None) -> Path:
    if requested is not None:
        output = _confined(requested, str(config["output_root"]), must_exist=False)
        _require(not output.exists(), "F008 E0 output directory already exists; use a fresh screen run")
        output.mkdir(parents=True, exist_ok=False)
        return output
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / config["output_root"] / f"F008_E0_ENERGY_SCREEN_{stamp}_{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/campp_f008_unknown_oe.json"))
    parser.add_argument("--binding-state", type=Path,
                        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"))
    parser.add_argument("--preflight-directory", type=Path,
                        default=DEFAULT_PREFLIGHT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None,
                        help="Fresh directory below F008 output_root; omitted creates a timestamped directory.")
    parser.add_argument("--execute", action="store_true",
                        help="Run only energy_005 tail folds and calibration-only E0 screening.")
    args = parser.parse_args()

    config_path = _confined(args.config, "configs/train", must_exist=True)
    from speaker_id.training.f008_config import config_signature, load_f008_config

    config = load_f008_config(config_path)
    signature = config_signature(config)
    active_spec = reduced_energy_screen_spec(config)
    if not args.execute:
        print(json.dumps({
            "status": "f008_e0_config_validated_no_audio_no_cuda_no_mlflow",
            "f008_config_signature": signature,
            "active_arms": active_spec["arm_ids"],
            "deferred_uniform": True,
            "outer_evaluation_called": False,
            "promotion_allowed": False,
        }, ensure_ascii=False, indent=2), flush=True)
        return

    f005_config_path = _confined(Path(config["source_f005"]["config_path"]),
                                 "configs/train", must_exist=True)
    _require(_sha256_file(f005_config_path) == config["source_f005"]["config_sha256"],
             "F008 pinned F005 config bytes changed")
    binding_path = _confined(args.binding_state, "artifacts/infrastructure", must_exist=True)
    preflight_directory = _confined(args.preflight_directory, str(config["output_root"]), must_exist=True)

    from speaker_id.training.f005_contract import load_f005_contract
    from speaker_id.training.f007_source import load_f005_source_receipt
    from speaker_id.training.f008_preflight import (
        authenticated_f005_source_contract,
        validate_f005_control_selection,
    )

    # This reconstructs/fingerprints metadata and source identity, but does
    # not repeat the historical all-audio verification sweep.
    current_f005 = load_f005_contract(f005_config_path, ROOT, verify_audio=False,
                                      verify_sources=False)
    source_root = Path(config["source_f005"]["run_dir"])
    f005_contract, source_bridge = authenticated_f005_source_contract(
        current_f005, source_root, ROOT, config["source_f005"],
    )
    source_receipt = load_f005_source_receipt(source_root)
    _require(source_receipt["experiment_state"]["experiment_signature"] == f005_contract["signature"],
             "F008 source receipt does not match the bridged F005 contract")
    control_seals = validate_f005_control_selection(source_root, config["source_f005"])
    preflight = _preflight_receipts(
        preflight_directory, config_signature=signature,
        f005_signature=f005_contract["signature"], source_receipt=source_receipt,
        f005_contract=f005_contract, source_root=source_root,
        fold_ids=list(config["fold_ids"]),
    )
    output = _new_output(config, args.output)

    from speaker_id.tracking import DurableMLflowRun

    resolved = {
        "f008_config": config,
        "f008_config_signature": signature,
        "screen": {
            "id": "E0", "active_arms": active_spec["arm_ids"],
            "deferred_uniform": True,
            "outer_evaluation_called": False,
            "outer_labels_used_for_selection": False,
            "selection_or_promotion_allowed": False,
            "purpose": "calibration_only_causal_signal_before_any_uniform_arm",
        },
        "f005_contract_signature": f005_contract["signature"],
        "current_reconstructed_f005_signature": current_f005["signature"],
        "source_contract_bridge": source_bridge,
        "control_selection_seals": control_seals,
        "preflight_receipt_sha256_by_outer": {
            str(outer): receipt["receipt_sha256"] for outer, receipt in preflight.items()
        },
        "verify_audio": False,
        "mlflow_forbidden_payloads_uploaded": False,
    }
    parent = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=output / "tracking", binding=_binding(binding_path),
        run_name="F008-E0-energy-only-calibration-screen", config=resolved,
        input_paths={
            "f008_config": config_path,
            "f005_config": f005_config_path,
            "launcher": Path(__file__),
            "worker": ROOT / "src/speaker_id/training/f008_worker.py",
            "extraction": ROOT / "src/speaker_id/training/f008_extraction.py",
            "scoring": ROOT / "src/speaker_id/training/f008_scoring.py",
            "preflight": ROOT / "src/speaker_id/training/f008_preflight.py",
        },
        run_kind="f008_e0_energy_only_calibration_screen", training_started=True,
    )
    children = []
    active_child = None
    try:
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.verify_remote_metadata()
        bridge_path = output / "source_contract_bridge.json"
        seals_path = output / "f005_control_selection_seals.json"
        _write_json(bridge_path, source_bridge)
        _write_json(seals_path, control_seals)
        _safe_add_metadata_artifact(parent, bridge_path, "provenance/source_contract_bridge.json")
        _safe_add_metadata_artifact(parent, seals_path, "provenance/f005_control_selection_seals.json")
        for outer, receipt in preflight.items():
            path = output / "preflight" / f"fold_{outer}_energy_margin_receipt.json"
            _write_json(path, receipt)
            _safe_add_metadata_artifact(parent, path, f"preflight/{path.name}")
            plan = receipt["energy_margin_plan"]
            parent.log_metrics({
                f"preflight/fold_{outer}/known_energy_mean": float(plan["known_diagnostics"]["mean"]),
                f"preflight/fold_{outer}/unknown_energy_mean": float(plan["unknown_diagnostics"]["mean"]),
                f"preflight/fold_{outer}/unknown_margin_satisfied_fraction": float(
                    plan["unknown_diagnostics"]["boundary_satisfied_fraction"]),
            }, step=outer, sync=False)

        # One CUDA/determinism check feeds both tail units.  The worker rejects
        # a missing or cross-run receipt rather than silently checking per fold.
        from speaker_id.training.f008_worker import attest_f008_runtime

        runtime_receipt = attest_f008_runtime(config, f005_contract)
        runtime_path = output / "runtime_receipt.json"
        _write_json(runtime_path, _runtime_summary(runtime_receipt))
        _safe_add_metadata_artifact(parent, runtime_path, "runtime/runtime_receipt.json")
        parent.log_metrics({"runtime/attested_once": 1.0}, step=0, sync=False)
        parent.flush(strict=True)

        import numpy as np
        from speaker_id.evaluation.scoring_bridge import _load_c002_identity_cache
        from speaker_id.training.f008_extraction import (
            load_reused_f005_control_cache,
            validated_f008_tail_checkpoint_receipt,
        )
        from speaker_id.training.f008_scoring import (
            bind_authenticated_f005_control_embeddings,
            prepare_and_seal_pretruth,
            reload_pretruth_seal,
        )
        from speaker_id.training.f008_worker import (
            fit_f008_tail,
            load_authenticated_f008_tail,
        )

        c002_root = Path(f005_contract["config"]["source_c002b"]["run_dir"])
        frozen_vectors, frozen_valid, frozen_receipt = _load_c002_identity_cache(
            c002_root, f005_contract["manifest"],
        )
        public = frozen_vectors["public"]
        frozen_advanced = frozen_vectors["advanced"]
        fold_results: list[dict[str, object]] = []
        for outer in config["fold_ids"]:
            fold_directory = output / f"fold_{outer}" / "energy_005"
            fold_directory.mkdir(parents=True, exist_ok=False)
            active_child = DurableMLflowRun.prepare(
                project_root=ROOT, spool_dir=fold_directory / "tracking",
                binding=_binding(binding_path), parent_run_id=parent.run_id,
                run_name=f"F008-E0-energy_005-fold{outer}",
                config={
                    "screen": {"id": "E0", "deferred_uniform": True,
                               "outer_evaluation_called": False,
                               "promotion_allowed": False},
                    "outer_fold": outer, "active_arm": "energy_005",
                    "f008_config_signature": signature,
                    "f005_contract_signature": f005_contract["signature"],
                    "preflight_receipt_sha256": preflight[outer]["receipt_sha256"],
                    "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
                },
                input_paths={"launcher": Path(__file__), "worker": ROOT / "src/speaker_id/training/f008_worker.py"},
                run_kind="f008_e0_energy_tail_fold", training_started=True,
            )
            children.append(active_child)
            active_child.flush(strict=True)
            shared_head = _source_shared_head(source_root, source_receipt, outer)

            def on_step(event: Mapping[str, object], tracker=active_child) -> None:
                _require(type(event.get("step")) is int and event.get("phase") == "tail",
                         "F008 worker emitted a malformed live metric event")
                metrics = {
                    str(name): float(value)
                    for name, value in event.items()
                    if name not in {"step", "phase"}
                    and type(value) in (int, float) and not isinstance(value, bool)
                    and math.isfinite(float(value))
                }
                tracker.log_metrics(metrics, step=event["step"], sync=False)
                if event["step"] % 25 == 0:
                    tracker.flush(strict=False)

            fit = fit_f008_tail(
                config, f005_contract, ROOT, outer, "energy_005", shared_head,
                source_receipt, preflight[outer], fold_directory / "tail",
                runtime_receipt=runtime_receipt, resume=False, on_step=on_step,
            )
            _log_probe_metrics(active_child, fit["probe"], outer)
            history_path = fold_directory / "tail" / "fit_history.jsonl"
            # The callback makes real-time logging normal; this fallback keeps
            # an interrupted/recovered worker history fully represented too.
            if not (active_child.directory / "events.jsonl").read_text(encoding="utf-8").strip():
                _log_history_metrics(active_child, history_path)

            loaded = load_authenticated_f008_tail(
                config, f005_contract, ROOT, outer, "energy_005", shared_head,
                source_receipt, preflight[outer], fit["checkpoint"], runtime_receipt=runtime_receipt,
            )
            try:
                _require(loaded["identity"] == fit["identity"], "F008 completed-tail identity changed")
                checkpoint_receipt = validated_f008_tail_checkpoint_receipt(
                    loaded["identity"], checkpoint_path=fit["checkpoint"],
                    checkpoint_metadata=loaded["metadata"],
                )
                checkpoint_receipt_path = fold_directory / "checkpoint_receipt.json"
                _write_json(checkpoint_receipt_path, checkpoint_receipt)
                cache = _extract_energy_cache(
                    loaded_tail=loaded, checkpoint=fit["checkpoint"],
                    checkpoint_receipt=checkpoint_receipt, f005_contract=f005_contract,
                    fold_directory=fold_directory, runtime_receipt=runtime_receipt,
                    tracker=active_child,
                )
            finally:
                # Tensor values never leave this scope or enter MLflow.
                for name in ("encoder", "head", "optimizer"):
                    if name in loaded:
                        del loaded[name]
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            control = load_reused_f005_control_cache(
                fit["identity"], source_receipt, f005_run_directory=source_root,
                manifest=f005_contract["manifest"], outer_fold=outer,
            )
            _require(np.array_equal(frozen_valid, control["valid"])
                     and np.array_equal(frozen_valid, cache["valid"]),
                     "F008 frozen/control/energy cache validity masks differ")
            binding = bind_authenticated_f005_control_embeddings(
                source_receipt, outer, embeddings=control["embeddings"], valid=control["valid"],
            )
            policy_path = fold_directory / "scoring" / "pretruth_e0_energy_only_seal.json"
            if policy_path.exists() or policy_path.is_symlink():
                policy_reload = reload_pretruth_seal(
                    policy_path, f005_contract, outer, scoring_spec=active_spec,
                    f005_source_receipt=source_receipt, f005_control_binding=binding,
                )
            else:
                prepared = prepare_and_seal_pretruth(
                    f005_contract, outer, public_embeddings=public,
                    frozen_advanced_embeddings=frozen_advanced,
                    f005_control_embeddings=control["embeddings"],
                    f005_source_receipt=source_receipt, f005_control_binding=binding,
                    f008_advanced_embeddings_by_arm={
                        "control_f005": control["embeddings"],
                        "energy_005": cache["embeddings"],
                    },
                    valid=frozen_valid, scoring_spec=active_spec,
                    policy_seal_path=policy_path,
                )
                policy_reload = prepared["policy_reload"]
            inner = _inner_summary(policy_reload, outer=outer)
            _log_inner_metrics(active_child, inner, outer)

            fold_report = {
                "status": "complete",
                "screen": {
                    "id": "E0", "active_arms": active_spec["arm_ids"],
                    "deferred_uniform": True, "outer_evaluation_called": False,
                    "outer_labels_used_for_selection": False,
                    "selection_or_promotion_allowed": False,
                },
                "outer_fold": outer,
                "fit": fit["report"],
                "gradient_probe": fit["probe"],
                "checkpoint_receipt": checkpoint_receipt,
                "cache": {
                    "identity_signature": cache["identity"]["signature"],
                    "receipt_sha256": cache["receipt"]["receipt_sha256"],
                    "row_count": int(len(cache["embeddings"])),
                    "server_only": True,
                    "mlflow_upload_allowed": False,
                    "local_transfer_allowed": False,
                },
                "source_control_binding": binding,
                "frozen_c002_cache": frozen_receipt,
                "inner_calibration": inner,
                "outer_truth_read": False,
                "model_weights_uploaded": False,
                "optimizer_state_uploaded": False,
                "embeddings_uploaded": False,
                "raw_audio_uploaded": False,
            }
            report_path = fold_directory / "fold_report.json"
            _write_json(report_path, fold_report)
            for source, relative in (
                (fold_directory / "tail" / "zero_optimizer_gradient_probe.json", "probe/gradient_probe.json"),
                (history_path, "training/fit_history.jsonl"),
                (checkpoint_receipt_path, "checkpoint/checkpoint_receipt.json"),
                (fold_directory / "full_scoring" / "energy_005" / "cache_receipt.json", "cache/cache_receipt.json"),
                (policy_path, "scoring/pretruth_e0_energy_only_seal.json"),
                (report_path, "fold_report.json"),
            ):
                _safe_add_metadata_artifact(active_child, source, relative)
            active_child.write_report(fold_report, markdown=(
                f"# F008 E0 energy-only fold {outer}\n\n"
                "This fold trained only energy_005 from the authenticated F005 shared head. "
                "Its policy was fitted on calibration roles only; no outer evaluation, "
                "selection, promotion, uniform arm, raw audio, embeddings, checkpoint, or optimizer state was uploaded.\n"
            ))
            active_child.flush(strict=True)
            active_child.verify_artifacts()
            active_child.finish("FINISHED", strict=True)
            active_child.verify_remote_metadata()
            fold_results.append({
                "outer_fold": outer, "child_run_id": active_child.run_id,
                "gradient_probe_sha256": fit["probe"]["probe_sha256"],
                "checkpoint_receipt_sha256": checkpoint_receipt["signature"],
                "cache_receipt_sha256": cache["receipt"]["receipt_sha256"],
                "pretruth_seal_sha256": policy_reload["seal_sha256"],
                "inner_calibration": inner,
            })
            active_child = None

        screen = {
            "status": "complete",
            "screen": {
                "id": "E0", "active_arms": active_spec["arm_ids"],
                "deferred_uniform": True, "outer_evaluation_called": False,
                "outer_labels_used_for_selection": False,
                "selection_or_promotion_allowed": False,
                "interpretation": "calibration_only_causal_screen_not_oof_or_leaderboard_claim",
            },
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "runtime": _runtime_summary(runtime_receipt),
            "preflight_receipt_sha256_by_outer": {
                str(outer): receipt["receipt_sha256"] for outer, receipt in preflight.items()
            },
            "folds": fold_results,
            "uniform_next_step_allowed_only_after_review": True,
            "outer_evaluation_called": False,
            "promotion_decision": "forbidden_for_E0_calibration_only_screen",
            "raw_audio_uploaded": False,
            "embeddings_uploaded": False,
            "model_weights_uploaded": False,
            "optimizer_state_uploaded": False,
            "local_model_transfer": False,
        }
        screen_path = output / "screen_report.json"
        _write_json(screen_path, screen)
        _safe_add_metadata_artifact(parent, screen_path, "screen_report.json")
        parent.write_report(screen, markdown=(
            "# F008 E0: energy-only calibration screen\n\n"
            "Only energy_005 was trained, with F005 source controls and existing F008 preflight receipts. "
            "Both fold policies were sealed using calibration roles only. Outer evaluation was deliberately not called, "
            "uniform_005 remains deferred, and this run cannot promote or transfer a model.\n"
        ))
        parent.flush(strict=True)
        parent.verify_artifacts()
        parent.finish("FINISHED", strict=True)
        parent.verify_remote_metadata()
        print(json.dumps({
            "status": "complete", "run_id": parent.run_id, "output": str(output),
            "active_arms": active_spec["arm_ids"], "deferred_uniform": True,
            "outer_evaluation_called": False, "promotion_allowed": False,
            "folds": fold_results,
        }, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        if active_child is not None:
            child_failure = {
                "status": "failed", "screen": "E0", "error_type": type(error).__name__,
                "error": active_child.redactor.text(str(error)), "outer_evaluation_called": False,
                "model_weights_uploaded": False, "optimizer_state_uploaded": False,
                "embeddings_uploaded": False, "raw_audio_uploaded": False,
            }
            failure_path = active_child.directory / "e0_failure.json"
            _write_json(failure_path, child_failure)
            try:
                _safe_add_metadata_artifact(active_child, failure_path, "e0_failure.json")
                active_child.write_report(child_failure)
                active_child.finish("FAILED", strict=False)
            except Exception:
                pass
        failure = {
            "status": "failed", "screen": "E0", "error_type": type(error).__name__,
            "error": parent.redactor.text(str(error)), "outer_evaluation_called": False,
            "promotion_allowed": False, "deferred_uniform": True,
            "model_weights_uploaded": False, "optimizer_state_uploaded": False,
            "embeddings_uploaded": False, "raw_audio_uploaded": False,
        }
        failure_path = output / "failure.json"
        _write_json(failure_path, failure)
        try:
            _safe_add_metadata_artifact(parent, failure_path, "failure.json")
            parent.write_report(failure)
            parent.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
