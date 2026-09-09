"""Prepare and audit a five-fold fit-capacity protocol without training.

The historical split builder protects the original two-fold protocol by
requiring every known class in every validation fold.  F009 is a separate
protocol: every known class must remain in each training population, while a
class with only two eligible groups may be absent from some validation folds.
This script creates new folds and inner roles, records their hashes, and never
opens audio or starts CUDA.  Generated CSVs contain private labels and stay on
the training server; only the aggregate audit summary is eligible for MLflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  sort_keys=True, indent=2)
        stream.write("\n")
    return _sha256(path)


def _markdown(summary: dict[str, Any]) -> str:
    return (
        "# F009 five-fold fit-capacity audit\n\n"
        "This is a read-only data/split feasibility audit. No audio was opened, "
        "no CUDA was used, and no model or threshold was fitted.\n\n"
        f"- Folds: **{summary['actual_folds']}**\n"
        f"- Files: **{summary['file_count']}**\n"
        f"- Known classes: **{summary['known_classes']}**\n"
        f"- Unknown calibration fraction: **{summary['unknown_query_fraction']}**\n"
        "\nThe generated folds and role table contain private labels and remain "
        "server-only; MLflow receives aggregate counts and hashes only.\n"
    )


def _track(summary: dict[str, Any], output: Path, binding_path: Path) -> str:
    from speaker_id.tracking import DurableMLflowRun, ExperimentBinding

    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    binding = ExperimentBinding(**payload["binding"])
    binding.validate()
    if binding.experiment_id != "1":
        raise ValueError("F009 requires MLflow experiment 1")
    tracker = DurableMLflowRun.prepare(
        project_root=ROOT, spool_dir=output / "tracking", binding=binding,
        run_name="F009-fivefold-fit-capacity-audit",
        config={
            "stage": "F009_fit_capacity_audit",
            "actual_folds": summary["actual_folds"],
            "file_count": summary["file_count"],
            "known_classes": summary["known_classes"],
            "unknown_query_fraction": summary["unknown_query_fraction"],
            "require_validation_known_coverage": False,
            "training_started": False,
            "audio_opened": False,
            "raw_audio_uploaded": False,
            "embeddings_uploaded": False,
            "model_weights_uploaded": False,
            "optimizer_state_uploaded": False,
            "local_model_transfer": False,
        },
        input_paths={"launcher": Path(__file__)},
        run_kind="f009_fit_capacity_audit", training_started=False,
    )
    try:
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.verify_remote_metadata()
        tracker.add_artifact(output / "audit_summary.json", "audit/audit_summary.json")
        tracker.log_metrics({
            "audit/actual_folds": float(summary["actual_folds"]),
            "audit/file_count": float(summary["file_count"]),
            "audit/known_classes": float(summary["known_classes"]),
            "audit/role_rows": float(summary["role_rows"]),
            "audit/training_started": 0.0,
        }, sync=False)
        tracker.write_report(summary, markdown=_markdown(summary))
        tracker.flush(strict=True)
        tracker.verify_artifacts()
        tracker.finish("FINISHED", strict=True)
        tracker.verify_remote_metadata()
        return tracker.run_id
    except BaseException as error:
        try:
            tracker.write_report({"status": "failed", "error": tracker.redactor.text(str(error))})
            tracker.finish("FAILED", strict=False)
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--duplicate-pairs", type=Path,
                        default=ROOT / "reports/eda/duplicate_pairs.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/research/f009_fit_capacity/F009_DATA_AUDIT")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--unknown-query-fraction", type=float, default=0.5)
    parser.add_argument("--binding", type=Path,
                        default=ROOT / "artifacts/infrastructure/C002_preparation/mlflow_state.json")
    parser.add_argument("--track", action="store_true",
                        help="Register aggregate audit evidence in MLflow experiment 1")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    pairs_path = args.duplicate_pairs.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"F009 refuses to replace an existing output: {output}")
    if args.folds < 2:
        raise ValueError("F009 needs at least two folds")
    if not 0 < args.unknown_query_fraction < 1:
        raise ValueError("Unknown calibration fraction must be between zero and one")

    from speaker_id.data.calibration import build_calibration_roles
    from speaker_id.data.splits import construct_folds, truth

    manifest = _read_csv(manifest_path)
    if len(manifest) != 4529 or len({row.get("audio_file") for row in manifest}) != len(manifest):
        raise ValueError("F009 requires exactly 4529 unique manifest files")
    known = sorted({row.get("speaker_id") for row in manifest} - {"unknown"})
    if len(known) != 446:
        raise ValueError("F009 requires exactly 446 known classes")
    pairs = [
        (row["audio_file_a"], row["audio_file_b"])
        for row in _read_csv(pairs_path) if truth(row.get("verified"))
    ]
    folds, split_summary = construct_folds(
        manifest, pairs, requested_folds=args.folds, seed=args.seed,
        require_validation_known_coverage=False,
    )
    if split_summary.get("actual_folds") != args.folds:
        raise ValueError("F009 rare-class support cannot sustain the requested folds")
    roles, role_summary = build_calibration_roles(
        folds, seed=args.seed, unknown_query_fraction=args.unknown_query_fraction,
    )
    if role_summary.get("outer_fold_count") != args.folds:
        raise ValueError("F009 role builder returned an incomplete fold inventory")
    for outer in range(args.folds):
        selected = [row for row in roles if int(row["outer_fold"]) == outer]
        if len(selected) != len(manifest):
            raise ValueError("F009 requires one role row per source file and outer fold")
        fit_labels = {
            row["speaker_id"] for row in selected
            if truth(row["encoder_fit_allowed"]) and row["speaker_id"] != "unknown"
        }
        enrollment_labels = {
            row["speaker_id"] for row in selected
            if truth(row["enrollment_allowed"])
        }
        if fit_labels != set(known) or enrollment_labels != set(known):
            raise ValueError("F009 training/enrollment lost a known class")

    output.mkdir(parents=True, exist_ok=False)
    fold_path = output / "folds.csv"
    role_path = output / "calibration_roles.csv"
    label_path = output / "label_map.json"
    split_summary_path = output / "split_summary.json"
    role_summary_path = output / "role_summary.json"
    fold_sha = _write_csv(fold_path, folds)
    role_sha = _write_csv(role_path, roles)
    label_sha = _write_json(label_path, {"labels": ["unknown", *known], "unknown_index": 0})
    split_sha = _write_json(split_summary_path, split_summary)
    role_summary_sha = _write_json(role_summary_path, role_summary)
    summary: dict[str, Any] = {
        "schema_version": "f009-fit-capacity-audit-v1",
        "status": "complete_no_training",
        "protocol": "five_outer_folds_all_known_in_training_sparse_validation_allowed",
        "manifest_sha256": _sha256(manifest_path),
        "duplicate_pairs_sha256": _sha256(pairs_path),
        "verified_duplicate_pair_count": len(pairs),
        "file_count": len(manifest),
        "known_classes": len(known),
        "evaluation_classes": len(known) + 1,
        "actual_folds": split_summary["actual_folds"],
        "seed": args.seed,
        "unknown_query_fraction": args.unknown_query_fraction,
        "role_rows": len(roles),
        "training_started": False,
        "audio_opened": False,
        "raw_audio_uploaded": False,
        "embeddings_uploaded": False,
        "model_weights_uploaded": False,
        "optimizer_state_uploaded": False,
        "local_model_transfer": False,
        "split_summary": split_summary,
        "role_summary": role_summary,
        "generated_files": {
            "folds.csv": fold_sha,
            "calibration_roles.csv": role_sha,
            "label_map.json": label_sha,
            "split_summary.json": split_sha,
            "role_summary.json": role_summary_sha,
        },
    }
    summary_sha = _write_json(output / "audit_summary.json", summary)
    print(json.dumps({
        "status": summary["status"], "output": str(output),
        "audit_summary_sha256": summary_sha, "actual_folds": summary["actual_folds"],
        "file_count": summary["file_count"], "role_rows": summary["role_rows"],
        "training_started": False, "audio_opened": False,
        "mlflow_run_id": _track(summary, output, args.binding) if args.track else None,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
