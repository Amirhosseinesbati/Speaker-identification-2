"""Validate or run the tracked, zero-step F008 shared-head energy preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confined(path: Path, prefix: str, *, must_exist: bool) -> Path:
    root = ROOT.resolve()
    unresolved = (root / path).absolute() if not Path(path).is_absolute() else Path(path).absolute()
    resolved = unresolved.resolve(strict=must_exist)
    base = (root / prefix).resolve()
    if unresolved.is_symlink() or not resolved.is_relative_to(base):
        raise ValueError(f"F008 path must stay below {prefix}")
    return resolved


def _binding(path: Path):
    from speaker_id.tracking import ExperimentBinding

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict) and isinstance(payload.get("binding"), dict),
             "F008 MLflow binding state is malformed")
    binding = ExperimentBinding(**payload["binding"])
    binding.validate()
    _require(binding.experiment_id == "1", "F008 requires MLflow experiment 1")
    return binding


def _source_receipt_summary(receipt: dict) -> dict:
    return {
        "schema_version": receipt["schema_version"],
        "source_run_directory": receipt["source_run_directory"],
        "experiment_state": receipt["experiment_state"],
        "folds": [
            {
                "outer_fold": row["outer_fold"],
                "shared_head_sha256": row["shared_head"]["checkpoint"]["sha256"],
                "control_tail_sha256": row["control_tail"]["checkpoint"]["sha256"],
            }
            for row in receipt["folds"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/campp_f008_unknown_oe.json"))
    parser.add_argument("--binding-state", type=Path,
                        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"))
    parser.add_argument("--execute", action="store_true",
                        help="Compute two fold-wise source-energy receipts; zero optimizer steps.")
    args = parser.parse_args()
    config_path = _confined(args.config, "configs/train", must_exist=True)

    from speaker_id.training.f008_config import config_signature, load_f008_config

    config = load_f008_config(config_path)
    signature = config_signature(config)
    if not args.execute:
        print(json.dumps({
            "status": "f008_config_validated_no_audio_no_cuda_no_mlflow",
            "config_signature": signature,
            "optimizer_steps": 0,
        }, ensure_ascii=False, indent=2), flush=True)
        return

    f005_config_path = _confined(Path(config["source_f005"]["config_path"]),
                                 "configs/train", must_exist=True)
    _require(_sha256_file(f005_config_path) == config["source_f005"]["config_sha256"],
             "F008 pinned F005 config bytes changed")
    binding_path = _confined(args.binding_state, "artifacts/infrastructure", must_exist=True)

    from speaker_id.training.f005_contract import load_f005_contract
    from speaker_id.training.f007_source import load_f005_source_receipt
    from speaker_id.training.f008_preflight import (
        authenticated_f005_source_contract,
        run_source_energy_preflight,
        validate_f005_control_selection,
        verify_preflight_receipt,
    )
    from speaker_id.tracking import DurableMLflowRun
    from speaker_id.tracking.snapshot import write_json

    # This validates the already pinned metadata/data contract without a repeat
    # raw-audio hash sweep. The GPU preflight itself reads only fit-role audio.
    current_f005_contract = load_f005_contract(f005_config_path, ROOT, verify_audio=False,
                                               verify_sources=False)
    source_root = Path(config["source_f005"]["run_dir"])
    f005_contract, source_contract_bridge = authenticated_f005_source_contract(
        current_f005_contract, source_root, ROOT, config["source_f005"],
    )
    source_receipt = load_f005_source_receipt(source_root)
    _require(source_receipt["experiment_state"]["experiment_signature"] == f005_contract["signature"],
             "F008 source F005 experiment signature differs from its pinned config")
    control_seals = validate_f005_control_selection(source_root, config["source_f005"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / config["output_root"] / f"F008_PREFLIGHT_{stamp}_{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        "f008_config": config,
        "f008_config_signature": signature,
        "f005_contract_signature": f005_contract["signature"],
        "current_reconstructed_f005_signature": current_f005_contract["signature"],
        "source_contract_bridge": source_contract_bridge,
        "f005_source": _source_receipt_summary(source_receipt),
        "control_selection_seals": control_seals,
        "execution": "source_energy_preflight_only",
        "optimizer_steps": 0,
        "mlflow_forbidden_payloads_uploaded": False,
    }
    tracker = DurableMLflowRun.prepare(
        project_root=ROOT,
        spool_dir=output / "tracking",
        binding=_binding(binding_path),
        run_name="F008-shared-head-energy-preflight",
        config=resolved,
        input_paths={
            "f008_config": config_path,
            "f005_config": f005_config_path,
            "launcher": Path(__file__),
            "f008_preflight": ROOT / "src/speaker_id/training/f008_preflight.py",
            "energy_margin_plan": ROOT / "src/speaker_id/training/f008_energy_margin_plan.py",
            "f008_protocol": ROOT / "src/speaker_id/training/f008_protocol.py",
            "open_set_oe": ROOT / "src/speaker_id/adaptation/open_set_oe.py",
        },
        run_kind="f008_shared_head_energy_preflight",
        training_started=False,
    )
    try:
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.verify_remote_metadata()
        bridge_path = output / "preflight" / "f005_source_contract_bridge.json"
        write_json(bridge_path, source_contract_bridge)
        tracker.add_artifact(bridge_path, "preflight/f005_source_contract_bridge.json")
        receipts: dict[str, dict] = {}
        for outer in config["fold_ids"]:
            receipt = run_source_energy_preflight(
                f005_contract, root=ROOT, f008_signature=signature,
                source_receipt=source_receipt, outer_fold=outer,
                energy_margin_config=config["energy_margin"],
                unknown_microbatch_pairs=config["training"]["microbatch_pairs"],
            )
            verify_preflight_receipt(receipt)
            receipt_path = output / "preflight" / f"fold_{outer}_energy_margin_receipt.json"
            write_json(receipt_path, receipt)
            tracker.add_artifact(receipt_path, f"preflight/{receipt_path.name}")
            plan = receipt["energy_margin_plan"]
            tracker.log_metrics({
                f"preflight/fold_{outer}/known_energy_mean": plan["known_diagnostics"]["mean"],
                f"preflight/fold_{outer}/unknown_energy_mean": plan["unknown_diagnostics"]["mean"],
                f"preflight/fold_{outer}/maximum_known_energy": plan["maximum_known_energy"],
                f"preflight/fold_{outer}/minimum_unknown_energy": plan["minimum_unknown_energy"],
                f"preflight/fold_{outer}/known_margin_satisfied_fraction": plan["known_diagnostics"]["boundary_satisfied_fraction"],
                f"preflight/fold_{outer}/unknown_margin_satisfied_fraction": plan["unknown_diagnostics"]["boundary_satisfied_fraction"],
                f"preflight/fold_{outer}/known_energy_views": float(receipt["known_energy_views"]),
                f"preflight/fold_{outer}/unknown_energy_views": float(receipt["unknown_energy_views"]),
                f"preflight/fold_{outer}/optimizer_steps": 0.0,
            }, step=outer, sync=True, strict=True)
            receipts[str(outer)] = receipt
        report = {
            "status": "complete",
            "run_kind": "f008_shared_head_energy_preflight",
            "optimizer_steps": 0,
            "f008_config_signature": signature,
            "f005_contract_signature": f005_contract["signature"],
            "source_contract_bridge": source_contract_bridge,
            "control_selection_seals": control_seals,
            "fold_receipts": receipts,
            "raw_audio_uploaded": False,
            "embeddings_uploaded": False,
            "model_weights_uploaded": False,
            "optimizer_state_uploaded": False,
        }
        write_json(output / "preflight_report.json", report)
        tracker.add_artifact(output / "preflight_report.json", "preflight/preflight_report.json")
        tracker.write_report(report, markdown=(
            "# F008 shared-head energy preflight\n\n"
            "Two authenticated F005 shared heads were evaluated only on their group-disjoint "
            "known/unknown encoder-fit pools. This run made zero optimizer steps and uploaded "
            "only scalar energy receipts, configuration, input hashes, and the source snapshot.\n"
        ))
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.finish("FINISHED", strict=True)
        tracker.verify_remote_metadata()
        print(json.dumps({
            "status": "complete", "run_id": tracker.run_id,
            "output": str(output), "optimizer_steps": 0,
            "fold_receipts": {fold: receipt["receipt_sha256"] for fold, receipt in receipts.items()},
        }, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        failure = {
            "status": "failed", "error_type": type(error).__name__,
            "error": tracker.redactor.text(str(error)), "optimizer_steps": 0,
            "raw_audio_uploaded": False, "embeddings_uploaded": False,
            "model_weights_uploaded": False, "optimizer_state_uploaded": False,
        }
        write_json(output / "failure.json", failure)
        try:
            tracker.add_artifact(output / "failure.json", "preflight/failure.json")
            tracker.write_report(failure)
            tracker.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
