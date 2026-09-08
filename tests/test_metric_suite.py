"""S013 prerequisite safety, explicit score scopes and portable metric evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.postprocessing import metric_suite as suite
from speaker_id.postprocessing.metric_scoring import _fit_score_case
from speaker_id.postprocessing.frozen_metric import transform


class MetricSuiteTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/postprocessing/campp_s013.json").read_text(encoding="utf-8"))
        self.config["s012_prerequisite"] = {key: None for key in suite.PREREQUISITE_KEYS}

    def test_pending_pins_reject_execution_before_environment_sources_or_tracking(self):
        suite.validate_config(self.config)
        with self.assertRaisesRegex(ValueError, "execution is blocked"):
            suite.validate_config(self.config, allow_pending=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "suite.json").write_text(json.dumps(self.config), encoding="utf-8")
            (root / "binding.json").write_text("{}", encoding="utf-8")
            with patch.object(suite, "_execution_environment") as environment, \
                    patch.object(suite, "load_sources") as sources, \
                    patch.object(suite.DurableMLflowRun, "prepare") as tracking, \
                    self.assertRaisesRegex(ValueError, "execution is blocked"):
                suite._execute_suite(root, Path("suite.json"), Path("binding.json"))
            environment.assert_not_called()
            sources.assert_not_called()
            tracking.assert_not_called()

    def test_fixed_grid_backend_and_complete_prerequisite_requirements(self):
        edits = [lambda c: c.update(advanced_weight=.75),
            lambda c: c["metric_specs"][2].update(power=1.),
            lambda c: c.update(matching_device="cuda"),
            lambda c: c["promotion"].update(reference="historical_calibration_fit_score"),
            lambda c: c["baseline_numerical_policy"].update(threshold_atol=.01),
            lambda c: c["s012_prerequisite"].update(parent_run_id="a" * 32)]
        for edit in edits:
            changed = deepcopy(self.config)
            edit(changed)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                suite.validate_config(changed)
        self.assertEqual(len(suite.RECIPES), 8)
        self.assertEqual(len(set(suite.RECIPES)), 8)
        self.assertEqual(suite.RECIPES[0], "baseline")
        self.assertEqual(suite.RECIPES[-1], "overall")

    def test_completed_s012_pins_bind_both_audit_and_report_identity(self):
        config = deepcopy(self.config)
        prior = {"run_path": "artifacts/training/S012_20260908T150000Z_1234abcd", "parent_run_id": "a" * 32,
            "git_commit": "b" * 40, "verification_path": "artifacts/infrastructure/S012_verification/S012_20260908T150000Z_1234abcd/verification.json",
            "verification_sha256": "c" * 64, "report_sha256": "d" * 64}
        config["s012_prerequisite"] = prior
        suite.validate_config(config, allow_pending=False)
        children = {f"family{i}": str(i) * 32 for i in range(6)}
        audit = {"status": "verified", "parent_run_id": prior["parent_run_id"], "git_commit": prior["git_commit"],
            "run_path": prior["run_path"], "children": children, "results": {"baseline": {"macro_f1": suite.BASELINE_MACRO_F1}}}
        report = {"status": "complete", "parent_run_id": prior["parent_run_id"], "source_binding": {"execution_git_commit": prior["git_commit"]},
            "children": children, "encoder_updates": 0, "source_arrays_unchanged": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path, report_path = root / prior["verification_path"], root / prior["run_path"] / "experiment_report.json"
            for path, value in ((audit_path, audit), (report_path, report)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            prior["verification_sha256"], prior["report_sha256"] = suite.file_sha256(audit_path), suite.file_sha256(report_path)
            with patch.object(suite, "_historical_sources", return_value={"paths": {}}):
                result = suite._checked_sources(root, config)
                self.assertEqual(result["paths"]["s012_report"], report_path.resolve())
            audit["parent_run_id"] = "e" * 32
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            prior["verification_sha256"] = suite.file_sha256(audit_path)
            with patch.object(suite, "_historical_sources", return_value={"paths": {}}), \
                    self.assertRaisesRegex(ValueError, "independently preserved"):
                suite._checked_sources(root, config)

    def test_historical_fallback_never_claims_identity_geometry_score_as_its_own(self):
        baseline = {"calibration": {"threshold": .2}, "inner_macro_f1_447": .96755}
        search = {"candidate_summary": [{"calibration": {"meta_macro_f1_447": .94,
            "meta_fold_macro_f1_447": [.93, .94, .95]}}]}
        for recipe, result in (("baseline", baseline), ("overall", {"baseline_retained": True})):
            scope = suite._score_scope(recipe, result, baseline, search)
            self.assertTrue(scope["historical_baseline_retained"])
            self.assertIsNone(scope["selected_geometry_meta_macro_f1_447"])
            self.assertEqual(scope["identity_control_meta_macro_f1_447"], .94)
            self.assertEqual(scope["historical_full_pool_baseline_calibration_macro_f1_447"], .96755)
        selected = {"calibration": {"meta_macro_f1_447": .955, "meta_fold_macro_f1_447": [.95, .95, .96]}}
        scope = suite._score_scope("centering_only", selected, baseline, search)
        self.assertEqual(scope["selected_geometry_meta_macro_f1_447"], .955)
        self.assertFalse(scope["historical_baseline_retained"])

    def test_npz_json_payload_roundtrip_reconstructs_scores_without_embedding_duplicates(self):
        rng = np.random.default_rng(931)
        values = rng.normal(size=(7, 4)).astype(np.float32)
        targets = np.asarray([1, 1, 2, 2, 0, -1, -1])
        groups = np.asarray([f"g{i}" for i in range(7)])
        spec = suite.METRIC_SPECS[-1]
        case = _fit_score_case(values, targets, groups, np.arange(5), np.asarray([5, 6]), spec, 2)
        case["meta_fold"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            receipt = suite._save_cases(directory, {spec["id"]: [case]}, nested=True)
            self.assertEqual(receipt["case_count"], 1)
            prefix = spec["id"] + "__meta0__"
            with zipfile.ZipFile(directory / "metric_metadata.zip") as archive:
                metadata = json.loads(archive.read(spec["id"] + "__meta0.json"))
            with np.load(directory / "metric_cases.npz", allow_pickle=False) as arrays:
                self.assertNotIn(prefix + "transformed_references", arrays.files)
                self.assertNotIn(prefix + "transformed_queries", arrays.files)
                matrix = suite._restore_symmetric_matrix(arrays[prefix + "matrix_upper"], metadata["matrix_dimension"])
                payload = {"metadata": metadata["payload_metadata"], "mean": arrays[prefix + "mean"], "matrix": matrix}
                self.assertEqual(matrix.tobytes(), case["payload"]["matrix"].tobytes())
                self.assertEqual(len(arrays[prefix + "matrix_upper"]), 10)
                np.testing.assert_array_equal(transform(payload, values[5:]), case["transformed_queries"])
                np.testing.assert_array_equal(arrays[prefix + "known_scores"], case["known_scores"])
                np.testing.assert_array_equal(arrays[prefix + "unknown_similarity"], case["unknown_similarity"])
                self.assertEqual(payload["mean"].dtype, np.float64)
                self.assertEqual(payload["matrix"].dtype, np.float64)
            self.assertEqual(metadata["reconstructable_transformed_arrays"]["transformed_queries"]["shape"], [2, 4])
        changed = deepcopy(case)
        changed["payload"]["matrix"][0, 0] += .1
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "fitted receipt"):
            suite._save_cases(Path(temporary), {spec["id"]: [changed]}, nested=True)


if __name__ == "__main__":
    unittest.main()
