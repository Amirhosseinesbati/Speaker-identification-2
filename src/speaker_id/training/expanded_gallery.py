"""S005 schema and completed S004 control attestation for the shared runner."""
from __future__ import annotations

import json
from pathlib import Path
import re

from speaker_id.models.campp import file_sha256
from speaker_id.training.fusion_suite import project_path


RECIPES = tuple({"id": f"S005{letter}", "source": source, "method": "max_reference",
    "protocol": protocol, "unknown_weights": [0.0, .25, .5, .75, 1.0], "margin_weights": [0.0, .5]}
    for letter, source, protocol in (("a", "public", "fixed_inner_holdout"), ("b", "adapted", "fixed_inner_holdout"),
        ("c", "public", "original_heldout_queries_expanded_gallery"), ("d", "adapted", "original_heldout_queries_expanded_gallery")))


def validate_expanded_suite(suite: dict):
    from speaker_id.training.adapted_scoring import RECIPES as original_recipes, validate_adapted_suite
    if (suite.get("experiment_code") != "S005" or suite.get("recipes") != list(RECIPES)
            or suite.get("calibration_protocol") != "original_heldout_queries_expanded_gallery"):
        raise ValueError("S005 requires two exact S004 controls and two original-query expanded-gallery recipes")
    validate_adapted_suite({**suite, "experiment_code": "S004", "recipes": list(original_recipes),
                            "calibration_protocol": "fixed_inner_holdout"})
    control = suite.get("control")
    if (not isinstance(control, dict) or not isinstance(control.get("run"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", control.get("parent_run_id", ""))
            or set(control.get("recipes", {})) != {"public", "adapted"}):
        raise ValueError("S005 cannot run until the completed S004 output and parent are explicitly bound")
    for name, recipe in control["recipes"].items():
        if (recipe.get("id") != ("S004c" if name == "public" else "S004d")
                or not re.fullmatch(r"[a-f0-9]{32}", recipe.get("child_run_id", ""))):
            raise ValueError("Both completed S004c/d child identities must be explicit")


def verified_s004_controls(root: Path, suite: dict, source_proof: dict) -> tuple[dict, dict]:
    from speaker_id.training.adapted_scoring import RECIPES as original_recipes, validate_adapted_suite
    validate_expanded_suite(suite)
    specification = suite["control"]
    directory = project_path(root, specification["run"], "artifacts/training")
    def read(path):
        target = directory / path
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(directory):
            raise ValueError("S004 control artifact is not a confined regular file")
        return json.loads(target.read_text(encoding="utf-8"))
    state, report = read("experiment_state.json"), read("experiment_report.json")
    parent_state, parent_config = read("tracking/run_state.json"), read("tracking/artifacts/resolved_config.json")
    old_suite = parent_config["suite"]
    validate_adapted_suite(old_suite)
    if (state.get("status") != "complete" or report.get("status") != "complete"
            or state.get("parent_run_id") != specification["parent_run_id"]
            or report.get("parent_run_id") != specification["parent_run_id"]
            or parent_state.get("run_id") != specification["parent_run_id"] or parent_state.get("remote_status") != "FINISHED"
            or old_suite["sources"] != suite["sources"]
            or read("source_provenance.json") != source_proof
            or set(report.get("source_control_checks", {})) != {"public", "adapted"}
            or not all(value.get("exact_prediction_reproduction") is True for value in report["source_control_checks"].values())):
        raise ValueError("S004 completed source controls, source cache/checkpoint identity or parent differ")
    directories, proof = {}, {"parent_run_id": specification["parent_run_id"],
        "report_sha256": file_sha256(directory / "experiment_report.json"),
        "source_provenance_sha256": file_sha256(directory / "source_provenance.json"), "recipes": {}}
    for name, entry in specification["recipes"].items():
        expected = next(row for row in original_recipes if row["id"] == entry["id"])
        child = read(entry["id"] + "/tracking/run_state.json")
        captured = read(entry["id"] + "/tracking/artifacts/resolved_config.json")
        recipe_report = read(entry["id"] + "/experiment_report.json")
        if (child.get("run_id") != entry["child_run_id"] or child.get("remote_status") != "FINISHED"
                or child.get("tags", {}).get("mlflow.parentRunId") != specification["parent_run_id"]
                or captured.get("recipe") != expected or captured.get("suite") != old_suite
                or recipe_report.get("recipe") != expected
                or sorted(row["outer_fold"] for row in recipe_report["folds"]) != [0, 1]):
            raise ValueError("Completed S004 control child, recipe, grid or fold coverage differs")
        directories[name] = directory / entry["id"]
        proof["recipes"][name] = {**entry, "recipe": expected,
            "report_sha256": file_sha256(directory / entry["id"] / "experiment_report.json")}
    return directories, proof
