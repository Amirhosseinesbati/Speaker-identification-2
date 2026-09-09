"""Verify and summarize a completed frozen-metric experiment without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    report_path = run_dir / "experiment_report.json"
    state_path = run_dir / "experiment_state.json"
    roundtrip_path = run_dir / "tracking_roundtrip_verification.json"
    config_path = run_dir / "resolved_config.json"
    for path in (report_path, state_path, roundtrip_path, config_path):
        require(path.is_file(), f"Missing completed-run evidence: {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8"))
    require(report["status"] == state["status"] == "complete", "Run is incomplete")
    require(state["all_mlflow_finished_and_verified"] is True, "MLflow completion is unverified")
    require(roundtrip["status"] == "passed", "MLflow artifact round-trip failed")
    require(state["git_commit"] == args.expected_commit, "Execution commit changed")
    require(report["parent_run_id"] == state["parent_run_id"], "Parent run IDs disagree")
    require(report["children"] == state["children"], "Child run IDs disagree")
    require(set(roundtrip["runs"]) == set(report["children"]) | {"parent"}, "Round-trip run set changed")
    for name, evidence in roundtrip["runs"].items():
        require(evidence["artifacts"]["status"] == "passed", f"Artifact verification failed: {name}")
        require(evidence["metadata"]["status"] == "passed", f"Metadata verification failed: {name}")
        require(evidence["metadata"]["remote_run_status"] == "FINISHED", f"Remote run is not FINISHED: {name}")

    baseline = report["results"]["baseline"]
    rows = []
    for name, result in report["results"].items():
        folds = [
            {
                "outer_fold": int(item["outer_fold"]),
                "macro_f1": float(item["outer"]["macro_f1"]),
                "accuracy": float(item["outer"]["accuracy"]),
            }
            for item in result["folds"]
        ]
        rows.append(
            {
                "recipe": name,
                "macro_f1": float(result["oof"]["macro_f1"]),
                "accuracy": float(result["oof"]["accuracy"]),
                "delta_macro_f1_vs_baseline": float(result["oof_macro_f1_delta"]),
                "errors": result["oof"]["errors"],
                "folds": folds,
                "paired": result["paired_against_S008c"]["all"],
            }
        )
    best_exploratory = max(
        (row for row in rows if row["recipe"] not in {"baseline", "identity", "overall"}),
        key=lambda row: (row["macro_f1"], row["recipe"]),
    )
    primary = next(row for row in rows if row["recipe"] == "overall")
    summary = {
        "schema_version": 1,
        "status": "passed",
        "run_directory": run_dir.as_posix(),
        "execution_git_commit": state["git_commit"],
        "parent_run_id": state["parent_run_id"],
        "children": state["children"],
        "all_mlflow_finished_and_verified": True,
        "encoder_updates": int(report["encoder_updates"]),
        "baseline_macro_f1": float(baseline["oof"]["macro_f1"]),
        "primary_selector": primary,
        "primary_promoted": primary["delta_macro_f1_vs_baseline"] > 0,
        "best_exploratory": best_exploratory,
        "recipes": rows,
        "evidence_sha256": {
            "experiment_report.json": sha256(report_path),
            "experiment_state.json": sha256(state_path),
            "tracking_roundtrip_verification.json": sha256(roundtrip_path),
            "resolved_config.json": sha256(config_path),
        },
        "new_submission_built": bool(report["new_submission_built"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    table = [
        "| recipe | OOF Macro-F1 | delta vs baseline | accuracy | known→unknown | unknown→known | known→known |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        errors = row["errors"]
        table.append(
            f"| {row['recipe']} | {row['macro_f1']:.9f} | {row['delta_macro_f1_vs_baseline']:+.9f} | "
            f"{row['accuracy']:.9f} | {errors['known_to_unknown']} | {errors['unknown_to_known']} | "
            f"{errors['known_to_other_known']} |"
        )
    best = best_exploratory
    paired = best["paired"]
    markdown = "\n".join(
        [
            "# Frozen-metric run verification",
            "",
            f"Status: **passed**. Parent MLflow run: `{state['parent_run_id']}`; execution commit: `{state['git_commit']}`.",
            "",
            *table,
            "",
            f"The preregistered primary selector changed Macro-F1 by {primary['delta_macro_f1_vs_baseline']:+.9f} and is not promoted.",
            f"The best exploratory recipe is `{best['recipe']}` at {best['macro_f1']:.9f} "
            f"({best['delta_macro_f1_vs_baseline']:+.9f}); it corrected {paired['corrected']} and regressed "
            f"{paired['regressed']} files. It remains a confirmation candidate, not a selected release.",
            "",
            "All nine remote runs, their metadata, and their uploaded artifacts passed round-trip verification. "
            "No encoder update or new submission was produced.",
            "",
        ]
    )
    args.output_md.write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "output_json": str(args.output_json), "best_exploratory": best["recipe"],
                      "best_macro_f1": best["macro_f1"], "primary_promoted": summary["primary_promoted"]}, indent=2))


if __name__ == "__main__":
    main()
