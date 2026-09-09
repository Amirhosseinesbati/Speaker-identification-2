"""Validate F005, or execute it only with the explicit ``--execute`` gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _confined(path: Path, prefix: str, *, existing: bool = True) -> Path:
    base = (ROOT / prefix).resolve()
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve(strict=existing)
    if not resolved.is_relative_to(base) or unresolved.is_symlink():
        raise ValueError(f"F005 path must stay under {prefix}")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/train/campp_f005_consistency.json"),
    )
    parser.add_argument(
        "--binding", type=Path,
        default=Path("artifacts/infrastructure/C002_preparation/mlflow_state.json"),
    )
    execution_directory = parser.add_mutually_exclusive_group()
    execution_directory.add_argument("--resume-dir", type=Path)
    execution_directory.add_argument(
        "--managed-run-dir", type=Path,
        help=(
            "Stable run directory for a process supervisor: create it on the "
            "first invocation, resume it after interruption, and return the "
            "sealed result without retraining after completion"
        ),
    )
    parser.add_argument(
        "--verify-audio", action="store_true",
        help="Hash every installed audio file during validation",
    )
    parser.add_argument(
        "--verify-sources", action="store_true",
        help="Verify pinned advanced weights and historical C002b receipts",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Start/resume the full tracked CUDA experiment",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _confined(args.config, "configs/train")
    from speaker_id.training.f005_contract import load_f005_contract
    # A full execution cannot weaken either verification switch.  The default
    # invocation remains bounded validation and takes zero optimizer steps.
    contract = load_f005_contract(
        config_path, ROOT,
        verify_audio=args.execute or args.verify_audio,
        verify_sources=args.execute or args.verify_sources,
    )
    if not args.execute:
        if args.resume_dir is not None or args.managed_run_dir is not None:
            raise ValueError(
                "--resume-dir/--managed-run-dir are accepted only with explicit --execute"
            )
        from speaker_id.training.f005_runner import execution_plan, probe_receipt
        result = {
            "status": "validated_no_training",
            "experiment_signature": contract["signature"],
            "audio_hashes_checked": contract["readiness"]["summary"]["audio_hashes_checked"],
            "sources_verified": contract["source_verification"] is not None,
            "optimizer_steps": 0,
            "probe": probe_receipt(contract),
            "execution_plan": execution_plan(contract, include_plan_hashes=False),
            "full_execution_requires": "--execute",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    binding_path = _confined(args.binding, "artifacts/infrastructure")
    resume = None
    output = None
    if args.resume_dir is not None:
        resume = _confined(args.resume_dir, "artifacts/training/f005_consistency")
    elif args.managed_run_dir is not None:
        unresolved = (
            args.managed_run_dir if args.managed_run_dir.is_absolute()
            else ROOT / args.managed_run_dir
        )
        managed = _confined(
            args.managed_run_dir,
            "artifacts/training/f005_consistency",
            existing=unresolved.exists(),
        )
        if managed.exists():
            state_path = managed / "experiment_state.json"
            if state_path.is_file() and not state_path.is_symlink():
                resume = managed
            elif managed.is_dir() and not managed.is_symlink() and not any(managed.iterdir()):
                # A hard kill can occur in the tiny mkdir→initial-state window.
                # Removing this exact empty directory makes the same supervised
                # logical path safely creatable again.
                managed.rmdir()
                output = managed
            else:
                raise ValueError(
                    "Managed F005 directory exists without a valid experiment state"
                )
        else:
            output = managed
    from speaker_id.training.f005_experiment import execute_f005_experiment
    result = execute_f005_experiment(
        contract, ROOT, config_path, binding_path,
        resume_dir=resume, output_dir=output,
    )
    print(json.dumps({
        "status": result["status"], "resumed": result["resumed"],
        "output": result["output"], "parent_run_id": result.get("parent_run_id"),
        "promote_new_incumbent": result["report"]["result"].get("promote_new_incumbent"),
        "development_metric_goal_reached": result["report"]["result"].get(
            "development_metric_goal_reached"
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
