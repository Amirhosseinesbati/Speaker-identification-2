"""Evaluate a *completed* F008 E0 screen once, after its calibration review.

This is not a training launcher.  It requires ``--execute`` so an operator
cannot accidentally materialize outer labels while E0 is incomplete.  The
stage authenticates server-local caches and policies, recomputes the E0
pretruth bundles without outer labels, reloads both fold seals, then evaluates
the named control and energy arms exactly once.  It does not change F008
configuration, train a model, extract audio, package a submission, or transfer
any checkpoint/cache off the GPU server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".py", ".txt"})
_FORBIDDEN_ARTIFACT_PARTS = frozenset({"embedding_cache", "entries", "checkpoints", "optimizer", "raw_audio"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"F008 E0 {label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"F008 E0 {label} is unreadable") from error
    _require(isinstance(value, dict), f"F008 E0 {label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"F008 E0 refuses to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"F008 E0 refuses to replace output: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _confined(path: Path, prefix: str, *, must_exist: bool) -> Path:
    root = ROOT.resolve()
    raw = Path(path)
    unresolved = raw.absolute() if raw.is_absolute() else (root / raw).absolute()
    resolved = unresolved.resolve(strict=must_exist)
    base = (root / prefix).resolve()
    _require(not unresolved.is_symlink() and resolved.is_relative_to(base),
             f"F008 E0 path must stay below {prefix}")
    return resolved


def _binding(path: Path):
    from speaker_id.tracking import ExperimentBinding

    payload = _read_json(path, "MLflow binding state")
    binding_payload = payload.get("binding")
    _require(isinstance(binding_payload, dict), "F008 E0 MLflow binding state is malformed")
    binding = ExperimentBinding(**binding_payload)
    binding.validate()
    _require(binding.experiment_id == "1", "F008 E0 requires MLflow experiment 1")
    return binding


def _safe_add_metadata_artifact(tracker, path: Path, relative_path: str) -> None:
    source = Path(path)
    lowered = {part.lower() for part in source.parts}
    _require(source.is_file() and not source.is_symlink(),
             "F008 E0 MLflow artifact must be a regular file")
    _require(source.suffix.lower() in _METADATA_SUFFIXES,
             "F008 E0 MLflow accepts only metadata/text artifacts")
    _require(not (lowered & _FORBIDDEN_ARTIFACT_PARTS),
             "F008 E0 refuses to upload a cache/checkpoint/raw-audio artifact")
    tracker.add_artifact(source, relative_path)


def _compact_for_markdown(report: Mapping[str, object]) -> str:
    metrics = report["metrics"]
    deltas = report["deltas"]
    control, energy = metrics["control_f005"], metrics["energy_005"]
    return (
        "# F008 E0 post-screen outer evaluation\n\n"
        "This is a one-shot, post-screen comparison of the predeclared control and energy arms. "
        "It does not select or promote a package.\n\n"
        f"- Control Macro-F1: **{float(control['macro_f1']):.9f}**\n"
        f"- Energy Macro-F1: **{float(energy['macro_f1']):.9f}**\n"
        f"- Energy minus control: **{float(deltas['macro_f1_energy_minus_control']):+.9f}**\n"
        f"- Control accuracy: **{float(control['accuracy']):.9f}**\n"
        f"- Energy accuracy: **{float(energy['accuracy']):.9f}**\n"
        f"- Energy minus control accuracy: **{float(deltas['accuracy_energy_minus_control']):+.9f}**\n"
        "\nNo raw audio, embeddings, weights, optimizer states, checkpoints, or cache files were uploaded.\n"
    )


def _log_metrics(tracker, report: Mapping[str, object]) -> None:
    metrics: dict[str, float] = {}
    for arm, values in report["metrics"].items():
        metrics[f"oof/{arm}/macro_f1_447"] = float(values["macro_f1"])
        metrics[f"oof/{arm}/accuracy"] = float(values["accuracy"])
        for error, count in values["errors"].items():
            metrics[f"oof/{arm}/errors/{error}"] = float(count)
    deltas = report["deltas"]
    metrics["delta/energy_minus_control/macro_f1_447"] = float(
        deltas["macro_f1_energy_minus_control"]
    )
    metrics["delta/energy_minus_control/accuracy"] = float(
        deltas["accuracy_energy_minus_control"]
    )
    for error, values in deltas["directional_errors"].items():
        metrics[f"delta/energy_minus_control/errors/{error}"] = float(
            values["delta_energy_minus_control"]
        )
    for outer, by_arm in report["fold_metrics"].items():
        for arm, values in by_arm.items():
            metrics[f"fold_{outer}/{arm}/macro_f1_447"] = float(values["macro_f1"])
            metrics[f"fold_{outer}/{arm}/accuracy"] = float(values["accuracy"])
    _require(all(math.isfinite(value) for value in metrics.values()),
             "F008 E0 report has a nonfinite scalar metric")
    tracker.log_metrics(metrics, sync=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/campp_f008_unknown_oe.json"))
    parser.add_argument("--binding-state", type=Path,
                        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"))
    parser.add_argument("--e0-directory", type=Path, required=True,
                        help="Completed F008_E0_ENERGY_SCREEN_* directory below F008 output_root.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Fresh subdirectory below the E0 screen; default is outer_evaluation.")
    parser.add_argument("--execute", action="store_true",
                        help="Perform the guarded one-shot outer evaluation.")
    args = parser.parse_args()

    config_path = _confined(args.config, "configs/train", must_exist=True)
    from speaker_id.training.f008_config import config_signature, load_f008_config
    from speaker_id.training.f008_e0_evaluation import (
        e0_active_scoring_spec,
        evaluate_rebuilt_e0_outer,
        mlflow_safe_outer_report,
        prepared_e0_metadata,
        rebuild_e0_pretruth_bundles,
        validate_completed_e0_screen,
    )

    config = load_f008_config(config_path)
    signature = config_signature(config)
    active_spec = e0_active_scoring_spec(config)
    e0_directory = _confined(args.e0_directory, str(config["output_root"]), must_exist=True)
    _require(e0_directory.is_dir() and not e0_directory.is_symlink(),
             "F008 E0 screen directory is unavailable")
    screen_path = e0_directory / "screen_report.json"
    screen = _read_json(screen_path, "screen report")
    if not args.execute:
        print(json.dumps({
            "status": "f008_e0_outer_evaluation_not_executed",
            "f008_config_signature": signature,
            "e0_directory": str(e0_directory),
            "active_arms": active_spec["arm_ids"],
            "outer_labels_materialized": False,
            "selection_or_promotion_allowed": False,
        }, ensure_ascii=False, indent=2), flush=True)
        return

    f005_config_path = _confined(Path(config["source_f005"]["config_path"]),
                                 "configs/train", must_exist=True)
    _require(_sha256_file(f005_config_path) == config["source_f005"]["config_sha256"],
             "F008 E0 pinned F005 config bytes changed")
    binding_path = _confined(args.binding_state, "artifacts/infrastructure", must_exist=True)

    from speaker_id.training.f005_contract import load_f005_contract
    from speaker_id.training.f007_source import load_f005_source_receipt
    from speaker_id.training.f008_preflight import (
        authenticated_f005_source_contract,
        validate_f005_control_selection,
    )

    current_f005 = load_f005_contract(f005_config_path, ROOT, verify_audio=False,
                                      verify_sources=False)
    source_root = Path(config["source_f005"]["run_dir"])
    f005_contract, source_bridge = authenticated_f005_source_contract(
        current_f005, source_root, ROOT, config["source_f005"],
    )
    source_receipt = load_f005_source_receipt(source_root)
    _require(
        source_receipt["experiment_state"]["experiment_signature"] == f005_contract["signature"],
        "F008 E0 source receipt does not match the bridged F005 contract",
    )
    control_seals = validate_f005_control_selection(source_root, config["source_f005"])
    checked_screen = validate_completed_e0_screen(
        screen, f008_signature=signature, f005_signature=f005_contract["signature"],
        fold_ids=config["fold_ids"],
    )
    # E0's immutable screen report intentionally contains no tracking handle.
    # Recover the parent only from DurableMLflowRun's local state, after the
    # screen itself has passed its independent completion checks.
    e0_tracking_state = _read_json(e0_directory / "tracking" / "run_state.json", "E0 tracking state")
    parent_run_id = e0_tracking_state.get("run_id")
    _require(isinstance(parent_run_id, str) and parent_run_id,
             "F008 E0 completed screen has no MLflow parent run identifier")

    # This touches only authenticated caches and guarded calibration scoring.
    # Its scorer turns outer labels into raising sentinels until all policies
    # have been recomputed and compared with the existing seals.
    prepared = rebuild_e0_pretruth_bundles(
        e0_directory=e0_directory, config=config, f008_signature=signature,
        f005_contract=f005_contract, source_receipt=source_receipt,
        source_root=source_root, screen=checked_screen["screen"],
    )
    output = (
        _confined(args.output, str(config["output_root"]), must_exist=False)
        if args.output is not None else (e0_directory / "outer_evaluation").resolve()
    )
    _require(output.is_relative_to(e0_directory) and not output.exists(),
             "F008 E0 outer evaluation needs a fresh directory below its completed screen")
    output.mkdir(parents=True, exist_ok=False)

    from speaker_id.tracking import DurableMLflowRun

    tracker = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=output / "tracking", binding=_binding(binding_path),
        parent_run_id=parent_run_id, run_name="F008-E0-post-screen-one-shot-outer-evaluation",
        config={
            "stage": "E0_post_screen_outer_evaluation",
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "e0_screen_run_id": parent_run_id,
            "e0_screen_report_sha256": _sha256_file(screen_path),
            "active_arms": active_spec["arm_ids"],
            "selection_or_promotion_allowed": False,
            "verify_audio": False,
            "raw_audio_uploaded": False,
            "embeddings_uploaded": False,
            "model_weights_uploaded": False,
            "optimizer_state_uploaded": False,
            "local_model_transfer": False,
        },
        input_paths={
            "f008_config": config_path, "f005_config": f005_config_path,
            "e0_screen_report": screen_path, "launcher": Path(__file__),
            "evaluator": ROOT / "src/speaker_id/training/f008_e0_evaluation.py",
            "scorer": ROOT / "src/speaker_id/training/f008_scoring.py",
        },
        run_kind="f008_e0_post_screen_one_shot_outer_evaluation", training_started=False,
    )
    try:
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.verify_remote_metadata()
        rebuild_path = output / "pretruth_rebuild_receipts.json"
        _write_json(rebuild_path, {
            "stage": "E0_post_screen_outer_evaluation",
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "source_contract_bridge": source_bridge,
            "f005_control_selection_seals": control_seals,
            "folds": prepared_e0_metadata(prepared),
            "outer_truth_read_during_rebuild": False,
            "cache_and_checkpoint_artifacts_server_only": True,
        })
        _safe_add_metadata_artifact(tracker, rebuild_path, "provenance/pretruth_rebuild_receipts.json")

        report, _receipts = evaluate_rebuilt_e0_outer(
            contract=f005_contract, config=config, source_receipt=source_receipt,
            prepared=prepared, output_directory=output,
        )
        report_path = output / "outer_evaluation_report.json"
        # Keep the complete class-level report server-only.  Only the
        # redacted aggregate report is uploaded to the external tracker.
        mlflow_report = mlflow_safe_outer_report(report)
        mlflow_report_path = output / "outer_evaluation_report_mlflow_safe.json"
        _write_json(mlflow_report_path, mlflow_report)
        _safe_add_metadata_artifact(
            tracker, mlflow_report_path, "outer_evaluation_report.json"
        )
        _log_metrics(tracker, report)
        tracker.write_report(mlflow_report, markdown=_compact_for_markdown(mlflow_report))
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.finish("FINISHED", strict=True)
        tracker.verify_remote_metadata()
        print(json.dumps({
            "status": "complete", "run_id": tracker.run_id, "output": str(output),
            "active_arms": list(active_spec["arm_ids"]),
            "metrics": report["metrics"], "deltas": report["deltas"],
            "selection_or_promotion_allowed": False,
            "local_model_transfer": False,
        }, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        failure = {
            "status": "failed", "stage": "E0_post_screen_outer_evaluation",
            "error_type": type(error).__name__,
            "error": tracker.redactor.text(str(error)),
            "selection_or_promotion_allowed": False,
            "raw_audio_uploaded": False, "embeddings_uploaded": False,
            "model_weights_uploaded": False, "optimizer_state_uploaded": False,
            "local_model_transfer": False,
        }
        failure_path = output / "failure.json"
        if not failure_path.exists():
            _write_json(failure_path, failure)
        try:
            _safe_add_metadata_artifact(tracker, failure_path, "failure.json")
            tracker.write_report(failure)
            tracker.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
