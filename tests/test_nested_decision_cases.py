"""Synthetic checks for group-isolated nonlinear decision calibration."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speaker_id.postprocessing.nested_cases import (
    _prepare_case, make_meta_plan, make_nested_cases,
)
from speaker_id.training.reference_scoring import gate_scores


class LabelDenied(dict):
    def __getitem__(self, key):
        if key == "speaker_id":
            raise AssertionError("Original outer label was read")
        return super().__getitem__(key)


def fixture(*, singleton=False):
    rng = np.random.default_rng(991)
    labels = ["unknown", "a", "b", "c"]
    manifest, folds, vectors = [], [], []
    def add(label, group, fold):
        name = f"sample_{len(manifest):03}.wav"
        manifest.append({"audio_file": name, "speaker_id": label, "duration_seconds": "20",
                         "mono_rms_dbfs": "-25", "has_nonzero_signal": "true"})
        folds.append({"audio_file": name, "speaker_id": label, "group_id": group,
                      "fold": str(fold), "train_eligible": "true", "evaluation_included": "true"})
        value = rng.normal(0, .015, 704)
        if label != "unknown":
            position = labels.index(label)
            value[position] += 1
            value[512 + position] += 1
        vectors.append(value)
    for label in labels[1:]:
        for group in range(1 if singleton and label == "c" else 3):
            add(label, f"{label}_{group}", 1)
    for group in range(9):
        add("unknown", f"u_{group}", 1)
    # Duplicate content remains indivisible even with two file rows.
    add("unknown", "u_0", 1)
    vectors[-1] = vectors[-10].copy()
    train_count = len(manifest)
    for label in labels:
        add(label, f"outer_{label}", 0)
        manifest[-1] = LabelDenied(manifest[-1])
        folds[-1] = LabelDenied(folds[-1])
    matrix = np.asarray(vectors, dtype=np.float32)
    public, advanced = matrix[:, :512].copy(), matrix[:, 512:].copy()
    public /= np.linalg.norm(public, axis=1, keepdims=True)
    advanced /= np.linalg.norm(advanced, axis=1, keepdims=True)
    return public, advanced, np.ones(len(manifest), dtype=bool), manifest, folds, labels, train_count


def features(prepared, baseline, scope):
    if "inner_truth" in prepared:
        raise AssertionError("The feature builder received direct query truth")
    scores = baseline["scores"]
    indices = scores["calibration_indices" if scope == "inner" else "outer_indices"]
    known, unknown = scores[scope + "_known_scores"], scores[scope + "_unknown_similarity"]
    matrix = np.zeros((len(indices), 28), dtype=np.float64)
    matrix[:, :known.shape[1]] = known
    matrix[:, 3] = unknown
    matrix[:, 4:6] = prepared["quality"][indices]
    calibration = baseline["calibration"]
    margin = gate_scores(known, unknown, calibration["unknown_weight"], calibration["margin_weight"]) - calibration["threshold"]
    return {"features": matrix, "feature_names": [f"feature_{i}" for i in range(28)],
            "guess": known.argmax(axis=1).astype(np.int64) + 1, "margin": margin,
            "valid": np.ones(len(indices), dtype=bool), "indices": indices.copy(),
            "known_scores": known, "unknown_similarity": unknown}


class NestedDecisionCasesTests(unittest.TestCase):
    def test_plan_is_deterministic_group_disjoint_and_keeps_all_classes(self):
        _, _, valid, manifest, folds, labels, count = fixture()
        first = make_meta_plan(valid, manifest, folds, labels, 0)
        second = make_meta_plan(valid, manifest, folds, labels, 0)
        np.testing.assert_array_equal(first["assignments"], second["assignments"])
        groups = first["groups"]
        for case in first["cases"]:
            train, validation = case["reference_global_indices"], case["validation_global_indices"]
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertTrue(np.all(train < count))
            self.assertEqual({manifest[int(i)]["speaker_id"] for i in train}, set(labels))
        duplicate = np.flatnonzero(groups[first["query_global_indices"]] == "u_0")
        self.assertEqual(len(duplicate), 2)
        self.assertEqual(len(set(first["assignments"][duplicate])), 1)

    def test_original_outer_labels_never_read_and_query_coverage_exact(self):
        public, advanced, valid, manifest, folds, labels, count = fixture()
        result = make_nested_cases(public, advanced, valid, manifest, folds, labels, 0, features)
        query = result["query_global_indices"]
        self.assertEqual(len(query), count)
        seen = np.concatenate([case["validation_global_indices"] for case in result["cases"]])
        np.testing.assert_array_equal(np.sort(seen), query)
        for case in result["cases"]:
            self.assertTrue(case["provenance"]["fit_reference_groups_disjoint_from_validation"])
            self.assertTrue(np.all(case["fit_global_indices"] < count))
            self.assertTrue(np.all(case["validation_global_indices"] < count))
            self.assertEqual(case["fit"]["features"].shape[1], 28)
            self.assertFalse(set(case["fit"]["groups"]) & set(case["validation"]["groups"]))

    def test_known_singleton_group_is_reference_only(self):
        public, advanced, valid, manifest, folds, labels, _ = fixture(singleton=True)
        result = make_nested_cases(public, advanced, valid, manifest, folds, labels, 0, features)
        singleton = next(i for i, row in enumerate(manifest) if row["audio_file"] == "sample_006.wav")
        self.assertNotIn(singleton, result["query_global_indices"])
        for case in result["cases"]:
            self.assertIn(singleton, case["reference_global_indices"])
            self.assertNotIn(singleton, case["fit_global_indices"])
            self.assertIn("c", case["provenance"]["known_singleton_classes_in_meta_training"])

    def test_validation_embedding_cannot_change_fit_gallery_baseline_or_features(self):
        public, advanced, valid, manifest, folds, labels, _ = fixture()
        plan = make_meta_plan(valid, manifest, folds, labels, 0)
        entry = plan["cases"][0]
        original = _prepare_case(public, advanced, valid, manifest, labels, plan["groups"], entry, features)
        changed_public, changed_advanced = public.copy(), advanced.copy()
        rows = entry["validation_global_indices"]
        changed_public[rows] *= -1
        changed_advanced[rows] *= -1
        altered = _prepare_case(changed_public, changed_advanced, valid, manifest, labels, plan["groups"], entry, features)
        np.testing.assert_array_equal(original["fit"]["features"], altered["fit"]["features"])
        np.testing.assert_array_equal(original["fit"]["known_scores"], altered["fit"]["known_scores"])
        self.assertEqual(original["baseline"]["calibration"], altered["baseline"]["calibration"])
        self.assertEqual(original["baseline"]["baseline_alpha_candidates"], altered["baseline"]["baseline_alpha_candidates"])
        self.assertFalse(np.array_equal(original["validation"]["features"], altered["validation"]["features"]))

    def test_fixed_plan_validation_labels_only_change_evaluation_truth(self):
        public, advanced, valid, manifest, folds, labels, _ = fixture()
        plan = make_meta_plan(valid, manifest, folds, labels, 0)
        entry = plan["cases"][0]
        original = _prepare_case(public, advanced, valid, manifest, labels, plan["groups"], entry, features)
        changed = list(manifest)
        for i in entry["validation_global_indices"]:
            changed[int(i)] = dict(manifest[int(i)])
            changed[int(i)]["speaker_id"] = "b" if manifest[int(i)]["speaker_id"] != "b" else "a"
        altered = _prepare_case(public, advanced, valid, changed, labels, plan["groups"], entry, features)
        np.testing.assert_array_equal(original["fit"]["features"], altered["fit"]["features"])
        np.testing.assert_array_equal(original["validation"]["features"], altered["validation"]["features"])
        self.assertEqual(original["baseline"]["calibration"], altered["baseline"]["calibration"])
        np.testing.assert_array_equal(original["baseline"]["probabilities"], altered["baseline"]["probabilities"])
        self.assertFalse(np.array_equal(original["validation"]["truth"], altered["validation"]["truth"]))

    def test_overlap_is_rejected_before_features(self):
        public, advanced, valid, manifest, folds, labels, _ = fixture()
        plan = make_meta_plan(valid, manifest, folds, labels, 0)
        entry = dict(plan["cases"][0])
        entry["reference_global_indices"] = np.r_[entry["reference_global_indices"], entry["validation_global_indices"][:1]]
        with self.assertRaisesRegex(ValueError, "Validation group leaked"):
            _prepare_case(public, advanced, valid, manifest, labels, plan["groups"], entry, features)

    def test_feature_builder_cannot_smuggle_truth_or_wrong_rows(self):
        public, advanced, valid, manifest, folds, labels, _ = fixture()
        def contaminated(prepared, baseline, scope):
            result = features(prepared, baseline, scope)
            result["truth"] = np.zeros(len(result["indices"]), dtype=np.int64)
            return result
        with self.assertRaisesRegex(ValueError, "must not receive or emit query truth"):
            make_nested_cases(public, advanced, valid, manifest, folds, labels, 0, contaminated)


if __name__ == "__main__":
    unittest.main()
