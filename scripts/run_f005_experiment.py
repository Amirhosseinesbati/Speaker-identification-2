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
    parser.add_argument("--resume-dir", type=Path)
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
        if args.resume_dir is not None:
            raise ValueError("--resume-dir is accepted only with explicit --execute")
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
    if args.resume_dir is not None:
        resume = _confined(args.resume_dir, "artifacts/training/f005_consistency")
    from speaker_id.training.f005_experiment import execute_f005_experiment
    result = execute_f005_experiment(
        contract, ROOT, config_path, binding_path, resume_dir=resume,
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
