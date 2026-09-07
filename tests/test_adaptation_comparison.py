from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.training.adaptation_comparison import (
    DECISION_RULE, SELECTION_POLICY, SOURCE_CONFIGS, _remote_requests, _verify_remote_evidence,
    compare_results, comparison_recipes, known_ranking_diagnostics, paired_diagnostics, require_controls_before,
    validate_comparison_contracts, validate_comparison_suite, verified_historical_control,
    verify_control_fold, verify_final_schedule, verify_source_snapshot,
)
from speaker_id.training.runner import write_csv, write_json
from speaker_id.training.schedules import adaptation_checkpoint_state

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "configs/train/campp_finetune_fp32.json").is_file())


def fixture_suite(expanded=False):
    sources = {}
    for i, (name, config) in enumerate(SOURCE_CONFIGS.items()):
        sources[name] = {"config": config, "run": "artifacts/training/" + name + "_fixture",
            "parent_run_id": str(1 + i * 3) * 32, "signature": str(i + 1) * 64,
            "git_commit": str(i + 1) * 40, "export_manifest_paths": ["artifacts/exports/" + name + ".json"],
            "export_manifest_sha256": str(i + 1) * 64, "completed_steps": 600 if name == "F003" else 1100,
            "folds": {str(outer): {"child_run_id": str(2 + i * 3 + outer) * 32,
                "checkpoint_sha256": str(1 + i * 2 + outer) * 64} for outer in (0, 1)}}
    controls = {}
    for i, (arm, recipe) in enumerate((("fixed", "S004d"), ("expanded", "S005d"))):
        if arm == "expanded" and not expanded:
            continue
        controls[arm] = {"run": "artifacts/training/" + recipe[:4] + "_fixture",
            "parent_run_id": ("a" if i == 0 else "c") * 32, "child_run_id": ("b" if i == 0 else "d") * 32,
            "recipe_id": recipe, **{key: "f" * 64 for key in ("report_sha256", "recipe_report_sha256",
                                                            "source_provenance_sha256", "resolved_config_sha256")}}
    return {"schema_version": 1, "experiment_code": "S006", "run_name": "S006-fixture",
        "readiness_config": "configs/train/campp_coverage.json", "output_root": "artifacts/training",
        "calibration_protocol": "fixed_inner_holdout", "threshold_candidates": 201, "probability_temperature": .05,
        "expanded_arm": expanded, "sources": sources, "controls": controls, "recipes": comparison_recipes(expanded),
        "selection_policy": SELECTION_POLICY, "decision_rule": dict(DECISION_RULE)}


def fixture_contracts():
    roles = []
    for outer in (0, 1):
        for i, role in enumerate(("fit_enrollment", "query", "outer")):
            roles.append({"audio_file": f"{i}.wav", "outer_fold": outer, "group_id": f"g{i}", "role": role,
                "encoder_fit_allowed": i == 0, "enrollment_allowed": i == 0,
                "calibration_query": i == 1, "outer_evaluation_included": i == 2})
    contracts = {}
    for name, config in {**SOURCE_CONFIGS, "readiness": "configs/train/campp_coverage.json"}.items():
        contracts[name] = {"config": json.loads((ROOT / config).read_text(encoding="utf-8")),
            "model": {"weight": "same_public"}, "input_hashes": {"roles": "same"},
            "manifest": [], "folds": [], "roles": deepcopy(roles), "labels": ["unknown", "a"]}
    return contracts


class AdaptationComparisonTests(unittest.TestCase):
    def test_schema_freezes_both_arms_and_rejects_missing_or_swapped_source_identities(self):
        for enabled in (False, True):
            suite = fixture_suite(enabled)
            validate_comparison_suite(suite)
            for key, value in (("recipes", suite["recipes"][1:]), ("threshold_candidates", 501),
                               ("calibration_protocol", "leave_content_group_out"), ("decision_rule", {})):
                changed = deepcopy(suite)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_comparison_suite(changed)
            changed = deepcopy(suite)
            changed["sources"]["F004"]["completed_steps"] = 600
            with self.assertRaisesRegex(ValueError, "curriculum"):
                validate_comparison_suite(changed)
            changed = deepcopy(suite)
            changed["sources"]["F004"]["folds"] = deepcopy(changed["sources"]["F003"]["folds"])
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_comparison_suite(changed)
        changed = fixture_suite(True)
        del changed["controls"]["expanded"]
        with self.assertRaisesRegex(ValueError, "own completed"):
            validate_comparison_suite(changed)

    def test_pair_accepts_only_head_duration_difference_and_original_roles(self):
        suite, contracts = fixture_suite(), fixture_contracts()
        validate_comparison_contracts(suite, contracts)
        for mutate in (
            lambda c: c["F004"]["config"]["fit"].update(encoder_lr=2e-5),
            lambda c: c["F004"]["config"]["fit"].update(mixed_precision=True),
            lambda c: c["F004"]["config"]["fit"]["adaptation_schedule"].update(head_only_steps=100),
            lambda c: c["F004"]["input_hashes"].update(roles="changed"),
        ):
            changed = deepcopy(contracts)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_comparison_contracts(suite, changed)
        changed = deepcopy(contracts)
        for value in changed.values():
            value["roles"][1]["group_id"] = value["roles"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_comparison_contracts(suite, changed)

    def test_each_candidate_requires_every_exact_control(self):
        recipes = comparison_recipes(True)
        checks = {name: {"exact_prediction_reproduction": True} for name in ("F003_prototype", "F004_prototype", "S004d")}
        require_controls_before(recipes[3], checks)
        with self.assertRaisesRegex(ValueError, "All preregistered"):
            require_controls_before(recipes[5], checks)
        checks["S005d"] = {"exact_prediction_reproduction": True}
        require_controls_before(recipes[5], checks)
        for name in checks:
            changed = deepcopy(checks)
            changed[name]["exact_prediction_reproduction"] = False
            with self.assertRaises(ValueError):
                require_controls_before(recipes[5], changed)

    def test_final_checkpoint_schedule_rejects_wrong_step_fold_or_signature(self):
        suite, contracts = fixture_suite(), fixture_contracts()
        for name in SOURCE_CONFIGS:
            with tempfile.TemporaryDirectory() as temporary:
                directory, source = Path(temporary), deepcopy(suite["sources"][name])
                fit = contracts[name]["config"]["fit"]
                expected = adaptation_checkpoint_state(fit, source["completed_steps"])
                reports, payloads, inventory = [], {}, {}
                for outer in (0, 1):
                    relative = f"fold_{outer}/last.pt"
                    target = directory / relative
                    target.parent.mkdir()
                    target.write_bytes(str(outer).encode())
                    digest = file_sha256(target)
                    source["folds"][str(outer)]["checkpoint_sha256"] = digest
                    inventory[relative] = {"sha256": digest}
                    payloads[outer] = {"format_version": 2, "signature": source["signature"], "outer_fold": outer,
                        "completed_steps": source["completed_steps"], "schedule_state": expected,
                        "adaptation_schedule": fit["adaptation_schedule"]}
                    reports.append({"outer_fold": outer, "fit": {"schedule_state": expected,
                        "completed_steps": source["completed_steps"], "adaptation_schedule": fit["adaptation_schedule"],
                        "initial_step": 0, "epoch_selection": "fixed_steps_no_outer_selection"}})
                write_json(directory / "experiment_report.json", {"folds": reports})
                loader = lambda path: payloads[int(path.parent.name[-1])]
                proof = verify_final_schedule(directory, source, contracts[name], inventory, checkpoint_loader=loader)
                self.assertEqual(proof["0"]["head_only_completed_steps"], 100 if name == "F003" else 600)
                for key, bad in (("completed_steps", 599), ("outer_fold", 1), ("signature", "wrong")):
                    original = payloads[0][key]
                    payloads[0][key] = bad
                    with self.assertRaisesRegex(ValueError, "complete source-specific"):
                        verify_final_schedule(directory, source, contracts[name], inventory, checkpoint_loader=loader)
                    payloads[0][key] = original

    def test_source_snapshot_checks_actual_code_bytes_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "experiment_state.json", {"attempt": "original"})
            prefix = directory / "tracking/original/artifacts"
            prefix.mkdir(parents=True)
            payload = b"exact source\n"
            digest = hashlib.sha256(payload).hexdigest()
            source, original = {"git_commit": "a" * 40}, {"code_hashes": {"src/speaker_id/training/fit.py": digest}}
            archive = prefix / "source_snapshot.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("src/speaker_id/training/fit.py", payload)
            manifest = {"schema_version": 2, "archive_format": "zip", "src_dirty": False, "git_commit": source["git_commit"],
                "archive_sha256": file_sha256(archive), "file_count": 1,
                "files": [{"path": "src/speaker_id/training/fit.py", "bytes": len(payload), "sha256": digest}]}
            write_json(prefix / "source_manifest.json", manifest)
            write_json(prefix / "resolved_config.json", original)
            inventory = {"tracking/original/artifacts/" + name: {} for name in ("source_manifest.json", "source_snapshot.zip", "resolved_config.json")}
            self.assertTrue(verify_source_snapshot(directory, source, original, inventory)["captured_snapshot_preserved"])
            changed = deepcopy(original)
            changed["code_hashes"]["src/speaker_id/training/fit.py"] = "b" * 64
            write_json(prefix / "resolved_config.json", changed)
            with self.assertRaisesRegex(ValueError, "actual source snapshot"):
                verify_source_snapshot(directory, source, changed, inventory)
            self.assertEqual(file_sha256(archive), manifest["archive_sha256"])
            del inventory["tracking/original/artifacts/source_snapshot.zip"]
            with self.assertRaisesRegex(ValueError, "complete original"):
                verify_source_snapshot(directory, source, original, inventory)

    def test_exact_control_rejects_changed_calibration_even_when_predictions_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fold = directory / "fold_0"
            fold.mkdir()
            predictions = [{"audio_file": "one.wav", "speaker_id": "unknown"}]
            metrics = {"macro_f1": .8}
            calibration = {"unknown_weight": .5, "margin_weight": .5, "threshold": .2, "inner_macro_f1_447": .9}
            write_csv(fold / "predictions.csv", predictions)
            write_json(fold / "evaluation.json", {"outer": metrics})
            write_json(fold / "calibration.json", {"selected": calibration})
            self.assertTrue(verify_control_fold(directory, 0, predictions, metrics, calibration)["exact_calibration_reproduction"])
            with self.assertRaisesRegex(ValueError, "calibration"):
                verify_control_fold(directory, 0, predictions, metrics, {**calibration, "threshold": .21})
            with self.assertRaises(ValueError):
                verify_control_fold(directory, 0, predictions * 2, metrics, calibration)

    def historical_fixture(self, root):
        suite = fixture_suite()
        specification = suite["controls"]["fixed"]
        directory = root / specification["run"]
        old_suite = json.loads((ROOT / "configs/train/campp_adapted_scoring.json").read_text())
        old_suite["sources"]["adapted"] = deepcopy(suite["sources"]["F003"])
        recipe = next(row for row in old_suite["recipes"] if row["id"] == "S004d")
        proof = {"parent_run_id": suite["sources"]["F003"]["parent_run_id"], "folds": {"0": {"files": ["sha0"]}, "1": {"files": ["sha1"]}}}
        records = {
            "experiment_state.json": {"status": "complete", "parent_run_id": specification["parent_run_id"]},
            "experiment_report.json": {"status": "complete", "parent_run_id": specification["parent_run_id"],
                "source_control_checks": {key: {"exact_prediction_reproduction": True} for key in ("public", "adapted")}},
            "source_provenance.json": {"adapted": proof},
            "tracking/run_state.json": {"run_id": specification["parent_run_id"], "remote_status": "FINISHED"},
            "tracking/artifacts/resolved_config.json": {"suite": old_suite},
            "S004d/tracking/run_state.json": {"run_id": specification["child_run_id"], "remote_status": "FINISHED",
                "tags": {"mlflow.parentRunId": specification["parent_run_id"]}},
            "S004d/tracking/artifacts/resolved_config.json": {"suite": old_suite, "recipe": recipe},
            "S004d/experiment_report.json": {"recipe": recipe, "folds": [{"outer_fold": 0}, {"outer_fold": 1}]},
        }
        for relative, value in records.items():
            write_json(directory / relative, value)
        for relative, key in (("experiment_report.json", "report_sha256"), ("source_provenance.json", "source_provenance_sha256"),
                              ("tracking/artifacts/resolved_config.json", "resolved_config_sha256"),
                              ("S004d/experiment_report.json", "recipe_report_sha256")):
            specification[key] = file_sha256(directory / relative)
        return suite, directory, proof

    def test_historical_control_binds_source_cache_parent_child_grid_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, directory, proof = self.historical_fixture(root)
            _, verified = verified_historical_control(root, suite, "fixed", proof)
            self.assertTrue(verified["source_identity_exact"])
            changed = deepcopy(proof)
            changed["folds"]["0"] = proof["folds"]["1"]
            with self.assertRaisesRegex(ValueError, "source/cache/recipe"):
                verified_historical_control(root, suite, "fixed", changed)
            child = json.loads((directory / "S004d/tracking/run_state.json").read_text())
            child["remote_status"] = "RUNNING"
            write_json(directory / "S004d/tracking/run_state.json", child)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verified_historical_control(root, suite, "fixed", proof)

    def test_remote_verification_refuses_wrong_scope_status_or_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            evidence.write_bytes(b"original")
            binding = SimpleNamespace(experiment_id="1", project="project", scope_id="scope")
            remote = SimpleNamespace(info=SimpleNamespace(status="FINISHED", experiment_id="1"),
                data=SimpleNamespace(tags={"speaker_id.project": "project", "speaker_id.scope_id": "scope", "mlflow.parentRunId": "parent"}))
            class Client:
                def get_run(self, run):
                    return remote
                def download_artifacts(self, run, path, folder):
                    result = Path(folder) / "download.json"
                    shutil.copyfile(evidence, result)
                    return str(result)
            request = [("child", "parent", [("report.json", evidence)])]
            result = _verify_remote_evidence(Client(), binding, request, root / "good")
            self.assertEqual(result[0]["sha256"], file_sha256(evidence))
            remote.info.status = "RUNNING"
            with self.assertRaisesRegex(ValueError, "status or ownership"):
                _verify_remote_evidence(Client(), binding, request, root / "bad")
            remote.info.status = "FINISHED"
            remote.data.tags["speaker_id.scope_id"] = "another"
            with self.assertRaises(ValueError):
                _verify_remote_evidence(Client(), binding, request, root / "other")

    def test_remote_source_config_uses_tracking_bytes_not_different_json_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, suite = Path(temporary), fixture_suite()
            directories = {}
            for name in SOURCE_CONFIGS:
                directories[name] = root / name
                write_json(directories[name] / "experiment_state.json", {"attempt": "original"})
            controls = {"fixed": root / "S004_fixture/S004d"}
            requests = _remote_requests(suite, directories, controls)
            for request in (requests[0], requests[3]):
                self.assertEqual(request[2][0][1].parts[-4:], ("tracking", "original", "artifacts", "resolved_config.json"))
            self.assertTrue(any(path == "evaluation/calibration.json" for path, local in requests[1][2]))

    def test_paired_decision_and_quality_slices_include_zero_rows(self):
        old = {"oof": {"macro_f1": .90}, "folds": [{"outer_fold": 0, "outer": {"macro_f1": .91}}, {"outer_fold": 1, "outer": {"macro_f1": .89}}]}
        new = {"oof": {"macro_f1": .905}, "folds": [{"outer_fold": 0, "outer": {"macro_f1": .92}}, {"outer_fold": 1, "outer": {"macro_f1": .89}}]}
        self.assertTrue(compare_results(old, new)["engineering_gate_passed"])
        new["folds"][1]["outer"]["macro_f1"] = .884
        self.assertFalse(compare_results(old, new)["engineering_gate_passed"])
        labels = ["unknown"] + [f"known_{i}" for i in range(446)]
        rows = [{"audio_file": "zero.wav", "speaker_id": "unknown", "has_nonzero_signal": False, "duration_seconds": "3", "mono_rms_dbfs": "-inf"},
                {"audio_file": "known.wav", "speaker_id": "known_0", "has_nonzero_signal": True, "duration_seconds": "35", "mono_rms_dbfs": "-55"}]
        before = [{"audio_file": "zero.wav", "speaker_id": "unknown"}, {"audio_file": "known.wav", "speaker_id": "unknown"}]
        after = [{"audio_file": row["audio_file"], "speaker_id": row["speaker_id"]} for row in rows]
        result = paired_diagnostics(rows, before, after, labels)
        self.assertEqual(result["zero_signal"]["files"], 1)
        self.assertEqual(result["all"]["corrected"], 1)
        self.assertEqual(result["nonzero_below_minus_35_dbfs"]["corrected"], 1)
        self.assertEqual(result["nonzero_below_minus_50_dbfs"]["corrected"], 1)

    def test_known_ranking_distinguishes_rejection_from_representation(self):
        labels = ["unknown"] + [f"known_{i}" for i in range(446)]
        rows = [{"audio_file": str(i), "speaker_id": "unknown" if i == 5 else "known_0"} for i in range(6)]
        scores, probabilities = np.zeros((6, 446)), np.zeros((6, 447))
        for i, rank in enumerate((0, 1, 1, 0, 1, 1)):
            scores[i, rank] = 1
        for i, prediction in enumerate((0, 0, 2, 1, 0, 2)):
            probabilities[i, prediction] = 1
        report = known_ranking_diagnostics(rows, scores, probabilities, np.asarray([True] * 4 + [False, True]), labels)
        self.assertEqual(report["nonzero_known_files"], 4)
        self.assertEqual(report["rank1_correct"], 2)
        self.assertEqual(report["rejected_correct_rank"], 1)
        self.assertEqual(report["rejected_wrong_rank"], 1)
        self.assertEqual(report["accepted_wrong"], 1)
        self.assertEqual(report["accepted_correct"], 1)


if __name__ == "__main__":
    unittest.main()
