"""Validate F005 or run its bounded no-step CUDA memory/backward probe.

Full head/tail execution is intentionally exposed through the reviewed worker
API only after the staged MLflow/scoring orchestrator is completed.  This CLI
cannot accidentally start the 600+500-step experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _confined(path: Path, prefix: str, *, must_exist: bool = False) -> Path:
    unresolved = (ROOT / path).absolute()
    resolved = unresolved.resolve(strict=must_exist)
    if (unresolved.is_symlink() or not resolved.is_relative_to((ROOT / prefix).resolve())):
        raise ValueError(f"F005 path must stay under {prefix}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_f005_consistency.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "artifacts/infrastructure/C002_preparation/mlflow_state.json")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--emit-complete-plan-hashes", action="store_true")
    parser.add_argument("--execute-probe", action="store_true",
                        help="Run one worst-case FP32 backward on RTX3090; zero optimizer steps")
    parser.add_argument("--outer-fold", type=int, default=0)
    args = parser.parse_args()

    args.config = _confined(args.config, "configs/train", must_exist=True)
    from speaker_id.training.f005_contract import load_f005_contract, validate_f005_config
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_f005_config(config)
    if not (args.validate_data or args.verify_audio or args.verify_sources
            or args.emit_complete_plan_hashes or args.execute_probe):
        print(json.dumps({"status": "f005_config_validated_no_data_no_cuda_no_training",
                          "config": str(args.config), "optimizer_steps": 0}, indent=2))
        return
    contract = load_f005_contract(args.config, ROOT, verify_audio=args.verify_audio or args.execute_probe,
                                  verify_sources=args.verify_sources or args.execute_probe)
    from speaker_id.training.f005_runner import execution_plan, probe_receipt
    dry = {"status": "f005_data_contract_validated_no_training", "signature": contract["signature"],
           "role_summary": contract["role_summary"], "probe": probe_receipt(contract),
           "execution_plan": execution_plan(contract, include_plan_hashes=args.emit_complete_plan_hashes)}
    if not args.execute_probe:
        print(json.dumps(dry, indent=2, ensure_ascii=False))
        return

    if args.outer_fold not in config["fold_ids"]:
        parser.error("--outer-fold is outside F005")
    from speaker_id.training.f005_contract import require_execution_environment
    require_execution_environment(contract)
    binding_path = _confined(args.binding, "artifacts/infrastructure", must_exist=True)
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding
    from speaker_id.tracking.snapshot import write_json
    binding = ExperimentBinding(**json.loads(binding_path.read_text(encoding="utf-8"))["binding"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts/infrastructure" / f"F005_probe_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    resolved = {"experiment": config, "signature": contract["signature"], "dry_contract": dry,
                "probe_only": True, "optimizer_steps": 0,
                "model_or_optimizer_mlflow_upload": False}
    tracker = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=output / "tracking", binding=binding,
        run_name=f"F005-backward-memory-probe-fold{args.outer_fold}", config=resolved,
        input_paths={"suite_config": args.config,
                     "advanced_model_config": ROOT / config["advanced_model_config"],
                     "readiness_config": ROOT / config["readiness_config"],
                     "launcher": Path(__file__)},
        run_kind="f005_backward_memory_probe", training_started=False,
    )
    try:
        tracker.flush(strict=True); tracker.verify_artifacts(); tracker.verify_remote_metadata()
        from speaker_id.training.f005_worker import probe_cuda
        report = probe_cuda(contract, ROOT, args.outer_fold)
        write_json(output / "probe_report.json", report)
        tracker.add_artifact(output / "probe_report.json")
        tracker.log_metrics({"probe/loss": report["loss"], "probe/gradient_norm": report["gradient_norm"],
                             "probe/peak_allocated_mb": report["peak_allocated_mb"],
                             "probe/optimizer_steps": 0}, sync=False)
        tracker.write_report(report, markdown=("# F005 bounded CUDA probe\n\nPinned advanced CAM++ 192D; "
            "one worst-case FP32 backward and zero optimizer steps. No weight or optimizer artifact was uploaded.\n"))
        tracker.flush(strict=True); tracker.verify_artifacts(); tracker.verify_remote_metadata()
        tracker.finish("FINISHED", strict=True); tracker.verify_remote_metadata()
        print(json.dumps({"status": report["status"], "run_id": tracker.run_id,
                          "report": str(output / "probe_report.json"), "optimizer_steps": 0}, indent=2))
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__,
                   "error": tracker.redactor.text(str(error)), "optimizer_steps": 0,
                   "model_or_optimizer_mlflow_upload": False}
        write_json(output / "failure.json", failure)
        try:
            tracker.write_report(failure); tracker.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
