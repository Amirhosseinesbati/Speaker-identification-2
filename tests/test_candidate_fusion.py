from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.training import candidate_fusion as fusion
from speaker_id.training.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def complete_control():
    return {"exact_prediction_reproduction": True, "exact_pooled_metrics": True,
        "folds": {str(outer): {key: True for key in ("exact_prediction_reproduction", "exact_metrics_reproduction", "exact_calibration_reproduction")}
                  for outer in (0, 1)}}


def fixture_suite():
    return {"schema_version": 1, "experiment_code": "S008", "run_name": "S008-campp-public512-advanced192-paired-reference",
        "readiness_config": "configs/train/campp_coverage.json", "output_root": "artifacts/training", "source_public": deepcopy(fusion.SOURCE),
        "source_advanced": {"run": "artifacts/training/S007_fixture", "parent_run_id": "a" * 32, "source_signature": "b" * 64,
            "git_commit": "c" * 40, "export_manifest_paths": ["artifacts/exports/S007_fixture.json"], "export_manifest_sha256": "d" * 64,
            "children": {"S007a": "1" * 32, "S007b": "2" * 32}},
        "alphas": list(fusion.ALPHAS), "alpha_tie_order": list(fusion.TIE_ORDER),
        "unknown_weights": list(fusion.UNKNOWN_WEIGHTS), "margin_weights": list(fusion.MARGIN_WEIGHTS),
        "threshold_candidates": 201, "probability_temperature": .05,
        "selection_policy": fusion.SELECTION_POLICY, "execution_policy": fusion.EXECUTION_POLICY}


class OuterLabelForbidden(dict):
    def __getitem__(self, key):
        if key == "speaker_id":
            raise AssertionError("Outer label was accessed")
        return super().__getitem__(key)


class CandidateFusionTests(unittest.TestCase):
    def test_five_alphas_exact_tie_order_and_completed_sources_are_mandatory(self):
        suite = fixture_suite()
        fusion.validate_fusion_suite(suite)
        for key, value in (("alphas", [0.0, .5, 1.0]), ("alpha_tie_order", list(reversed(fusion.TIE_ORDER))),
                           ("selection_policy", "choose best outer"), ("threshold_candidates", 501)):
            changed = deepcopy(suite)
            changed[key] = value
            with self.assertRaises(ValueError):
                fusion.validate_fusion_suite(changed)
        changed = deepcopy(suite)
        changed["source_advanced"]["source_signature"] = None
        with self.assertRaises(ValueError):
            fusion.validate_fusion_suite(changed)

    def test_unequal_dimensions_preserve_same_reference_cosine_and_zero_rows(self):
        public = np.zeros((3, 512), dtype=np.float32)
        advanced = np.zeros((3, 192), dtype=np.float32)
        public[0, 0], public[1, 1] = 1, 1
        advanced[0, 0], advanced[1, 0] = 1, 1
        valid = np.asarray([True, True, False])
        mixed = fusion.weighted_encoder_pair(public, advanced, valid, .25)
        self.assertEqual(mixed.shape, (3, 704))
        self.assertEqual(mixed.dtype, np.float32)
        self.assertFalse(mixed[2].any())
        self.assertAlmostEqual(float(mixed[0] @ mixed[1]), .25, places=6)
        np.testing.assert_allclose(np.linalg.norm(mixed[:2], axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            fusion.weighted_encoder_pair(public, advanced[:2], valid, .25)
        advanced[2, 0] = 1
        with self.assertRaisesRegex(ValueError, "invalid zeros"):
            fusion.weighted_encoder_pair(public, advanced, valid, .25)

    def test_endpoints_return_original_objects_without_any_concatenation_or_rescoring(self):
        endpoints = {"public": {"sentinel": 512}, "advanced": {"sentinel": 192}}
        with patch.object(fusion, "weighted_encoder_pair", side_effect=AssertionError("No endpoint concatenation")), patch.object(fusion, "crossfit_scores", side_effect=AssertionError("No endpoint recomputation")):
            self.assertIs(fusion.scores_for_alpha(None, None, None, None, None, None, 0.0, endpoints), endpoints["public"])
            self.assertIs(fusion.scores_for_alpha(None, None, None, None, None, None, 1.0, endpoints), endpoints["advanced"])

    def test_interior_fp32_arithmetic_has_frozen_rounding(self):
        # Unit 3-4-5 vectors expose FP64-intermediate promotion in NumPy 2.
        # These IEEE754 binary32 results pin the inference/scoring contract.
        public, advanced = np.zeros((1, 512), dtype=np.float32), np.zeros((1, 192), dtype=np.float32)
        public[0, :2], advanced[0, :2] = [.6, .8], [.8, .6]
        mixed = fusion.weighted_encoder_pair(public, advanced, np.asarray([True]), .25)
        self.assertEqual(mixed[0, [0, 1, 512, 513]].view(np.uint32).tolist(),
                         [1057293697, 1060199596, 1053609165, 1050253722])

    def test_inner_selector_rejects_outer_fields_and_prefers_endpoint_on_exact_tie(self):
        known = np.asarray([[.9, .1], [.1, .9], [.2, .2]], dtype=np.float32)
        unknown = np.asarray([.1, .1, .9], dtype=np.float32)
        candidates = {alpha: {"known": known, "unknown": unknown} for alpha in fusion.ALPHAS}
        selected, curves = fusion.select_inner_alpha(candidates, np.asarray([1, 2, 0]), classes=3)
        self.assertEqual(selected["advanced_weight"], 0.0)
        self.assertEqual(set(curves), {str(alpha) for alpha in fusion.ALPHAS})
        candidates[.25] = {**candidates[.25], "outer": known}
        with self.assertRaisesRegex(ValueError, "never outer"):
            fusion.select_inner_alpha(candidates, np.asarray([1, 2, 0]), classes=3)

    def test_mixed_scores_exclude_whole_query_group_and_never_read_outer_labels(self):
        rows = [("a1", "A", "ga1", 1, 0), ("a1-copy", "A", "ga1", 1, 0), ("a2", "A", "ga2", 1, 1),
                ("b1", "B", "gb", 1, 2), ("u1", "unknown", "gu1", 1, 0), ("u2", "unknown", "gu2", 1, 1),
                ("outer", "PRIVATE", "go", 0, 0)]
        manifest = [(OuterLabelForbidden if fold == 0 else dict)(audio_file=name, speaker_id=label) for name, label, group, fold, pos in rows]
        folds = [{"audio_file": name, "group_id": group, "fold": fold, "train_eligible": True} for name, label, group, fold, pos in rows]
        public, advanced = np.zeros((len(rows), 512), dtype=np.float32), np.zeros((len(rows), 192), dtype=np.float32)
        for i, row in enumerate(rows):
            public[i, row[-1]] = advanced[i, row[-1]] = 1
        valid = np.ones(len(rows), dtype=bool)
        endpoints = {"public": fusion.crossfit_scores(public, valid, manifest, folds, 0, "max_reference", 2),
                     "advanced": fusion.crossfit_scores(advanced, valid, manifest, folds, 0, "max_reference", 2)}
        mixed = fusion.scores_for_alpha(public, advanced, valid, manifest, folds, 0, .5, endpoints, classes=2)
        for index in (0, 1):
            position = mixed["calibration_indices"].tolist().index(index)
            self.assertEqual(float(mixed["inner_known_scores"][position, 0]), 0.0)
        self.assertFalse(mixed["provenance"]["outer_labels_accessed"])
        self.assertNotIn(6, mixed["provenance"]["reference_indices"])

    def test_both_complete_endpoint_controls_gate_mixed_evaluation(self):
        checks = {"public": complete_control(), "advanced": complete_control()}
        fusion.require_endpoint_controls(checks)
        for side in checks:
            altered = deepcopy(checks)
            altered[side]["folds"]["1"]["exact_calibration_reproduction"] = False
            with self.assertRaises(ValueError):
                fusion.require_endpoint_controls(altered)

    def test_child_source_readback_uses_attested_parent_bytes_after_export_deduplication(self):
        directory = Path("synthetic_source")
        requests = fusion._candidate_remote_requests(directory, fixture_suite()["source_advanced"])
        for name, request in zip(("S007a", "S007b"), requests[1:]):
            files = dict(request[2])
            self.assertEqual(files["resolved_config.json"], directory / name / "tracking/artifacts/resolved_config.json")
            self.assertEqual(files["source_manifest.json"], directory / "tracking/artifacts/source_manifest.json")
            self.assertEqual(files["source_snapshot.zip"], directory / "tracking/artifacts/source_snapshot.zip")

    def historical_fixture(self, root):
        source = fixture_suite()["source_advanced"]
        directory = root / source["run"]
        original_suite = json.loads((ROOT / "configs/train/campp_advanced_scoring.json").read_text())
        critical = ["src/speaker_id/candidates/campp_advanced.py", "src/speaker_id/models/campp.py",
                    "src/speaker_id/models/vendor/campplus/DTDNN.py", "src/speaker_id/training/candidate_comparison.py", "scripts/score_candidate.py"]
        code = {}
        for name in critical:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original " + name, encoding="utf-8")
            code[name] = file_sha256(path)
        (root / "src/speaker_id/training/new_S008_evaluator.py").write_text("new unrelated source")
        parent_artifacts = directory / "tracking/artifacts"
        (parent_artifacts / "candidate_model").mkdir(parents=True)
        weights = parent_artifacts / "candidate_model/campplus_cn_en_common.pt"
        weights.write_bytes(b"synthetic immutable public weights")
        model = {"weights_sha256": file_sha256(weights), "weights_bytes": weights.stat().st_size,
                 "weights_path": "artifacts/models/campp_advanced/campplus_cn_en_common.pt"}
        write_json(parent_artifacts / "candidate_model/model_config.json", model)
        contract = {"input_hashes": {"manifest": "fixed-data", "model_config": "public512"}, "labels": ["unknown", "A", "B"]}
        identity = {"schema_version": 1, "encoder_kind": "public_frozen_campp_advanced_192", "model": model,
            "embedding_dim": 192, "weights_sha256": model["weights_sha256"], "model_config_sha256": file_sha256(parent_artifacts / "candidate_model/model_config.json"),
            "data_input_hashes": {"manifest": "fixed-data"}, "labels": contract["labels"], "source_code_hashes": code,
            "encoder_updates": 0, "inference": {"seconds": 180.0, "maximum_windows": 1}}
        identity["signature"] = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        source["source_signature"] = identity["signature"]
        original = {"suite": original_suite, "candidate_identity": identity, "data_readiness_contract": {}}
        write_json(directory / "resolved_config.json", original)
        write_json(parent_artifacts / "resolved_config.json", original)
        write_json(directory / "candidate_identity.json", identity)
        write_json(directory / "experiment_state.json", {"status": "complete", "parent_run_id": source["parent_run_id"]})
        write_json(directory / "experiment_report.json", {"status": "complete", "parent_run_id": source["parent_run_id"],
            "encoder_updates": 0, "source_control_checks": complete_control()})
        write_json(directory / "tracking/run_state.json", {"run_id": source["parent_run_id"], "remote_status": "FINISHED",
            "tags": {"mlflow.source.git.commit": source["git_commit"]}})
        files = []
        archive_path = parent_artifacts / "source_snapshot.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, digest in code.items():
                if name.startswith("src/"):
                    payload = (root / name).read_bytes()
                    archive.writestr(name, payload)
                    files.append({"path": name, "sha256": digest, "bytes": len(payload)})
        write_json(parent_artifacts / "source_manifest.json", {"schema_version": 2, "archive_format": "zip", "src_dirty": False,
            "git_commit": source["git_commit"], "archive_sha256": file_sha256(archive_path), "file_count": len(files), "files": files})
        (parent_artifacts / "input_configs").mkdir()
        (parent_artifacts / "input_configs/launcher.py").write_bytes((root / "scripts/score_candidate.py").read_bytes())
        write_json(directory / "candidate_cache_manifest.json", {"identity": identity, "weights_sha256_before": identity["weights_sha256"],
            "weights_sha256_after": identity["weights_sha256"], "encoder_state_sha256_before": "e" * 64, "encoder_state_sha256_after": "e" * 64})
        for recipe in original_suite["recipes"]:
            name = recipe["id"]
            write_json(directory / name / "tracking/run_state.json", {"run_id": source["children"][name], "remote_status": "FINISHED",
                "tags": {"mlflow.parentRunId": source["parent_run_id"], "mlflow.source.git.commit": source["git_commit"]}})
            write_json(directory / name / "tracking/artifacts/resolved_config.json", {**original, "recipe": recipe})
            write_json(directory / name / "experiment_report.json", {"recipe": recipe, "folds": [{"outer_fold": 0}, {"outer_fold": 1}]})
        inventory = {path.relative_to(directory).as_posix(): {} for path in directory.rglob("*") if path.is_file()}
        return directory, source, contract, inventory

    def test_old_all_source_identity_survives_new_evaluator_but_not_critical_source_change(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(fusion, "validate_advanced_config"):
            root = Path(temporary)
            directory, source, contract, inventory = self.historical_fixture(root)
            identity, proof = fusion.validate_historical_candidate(directory, source, contract, inventory, root)
            self.assertEqual(identity["signature"], source["source_signature"])
            self.assertTrue(proof["historical_all_src_identity_preserved"])
            (root / "src/speaker_id/models/campp.py").write_text("changed feature extraction")
            with self.assertRaisesRegex(ValueError, "Feature-critical"):
                fusion.validate_historical_candidate(directory, source, contract, inventory, root)

    def test_historical_cli_capture_and_completed_child_must_match(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(fusion, "validate_advanced_config"):
            root = Path(temporary)
            directory, source, contract, inventory = self.historical_fixture(root)
            cli = directory / "tracking/artifacts/input_configs/launcher.py"
            original = cli.read_bytes()
            cli.write_bytes(b"different archived launcher")
            with self.assertRaisesRegex(ValueError, "separately archived"):
                fusion.validate_historical_candidate(directory, source, contract, inventory, root)
            cli.write_bytes(original)
            child_state = directory / "S007b/tracking/run_state.json"
            state = json.loads(child_state.read_text())
            state["remote_status"] = "RUNNING"
            write_json(child_state, state)
            with self.assertRaisesRegex(ValueError, "child identity"):
                fusion.validate_historical_candidate(directory, source, contract, inventory, root)


if __name__ == "__main__":
    unittest.main()
