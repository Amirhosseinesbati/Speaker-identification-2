import unittest

from speaker_id.data.calibration import build_calibration_roles


class CalibrationTests(unittest.TestCase):
    def samples(self):
        rows = []
        for fold in range(2):
            for label, count in (("alice", 3), ("bob", 1), ("unknown", 4)):
                for group in range(count):
                    name = f"{fold}-{label}-{group}"
                    rows.append({"audio_file": name, "speaker_id": label, "group_id": name,
                                 "fold": fold, "train_eligible": True, "evaluation_included": True,
                                 "exclusion_reasons": ""})
            rows.append({**rows[-1], "audio_file": f"duplicate-{fold}"})
            for label in ("alice", "unknown"):
                rows.append({"audio_file": f"zero-{fold}-{label}", "speaker_id": label,
                             "group_id": f"zero-{fold}", "fold": fold, "train_eligible": False,
                             "evaluation_included": True, "exclusion_reasons": "unusable_signal"})
        return rows

    def test_queries_fit_and_outer_validation_are_group_disjoint(self):
        sources = self.samples()
        roles, summary = build_calibration_roles(sources)
        self.assertEqual(len(roles), 2 * len(sources))
        for outer in range(2):
            rows = [r for r in roles if r["outer_fold"] == outer]
            self.assertEqual(len(rows), len(sources))
            fit = {r["group_id"] for r in rows if r["encoder_fit_allowed"]}
            query = {r["group_id"] for r in rows if r["calibration_query"]}
            valid = {r["group_id"] for r in rows if r["outer_evaluation_included"]}
            self.assertFalse(fit & query or query & valid or fit & valid)
            enrollment = [r for r in rows if r["enrollment_allowed"]]
            self.assertEqual({r["speaker_id"] for r in enrollment}, {"alice", "bob"})
            self.assertTrue(all(r["encoder_fit_allowed"] and not r["calibration_query"] for r in enrollment))
            self.assertEqual(summary["folds"][outer]["known_singleton_enrollment_only"], ["bob"])
            self.assertEqual(summary["folds"][outer]["known_calibration_query_classes"], 1)
            self.assertEqual(summary["folds"][outer]["unknown_calibration_query_groups"], 2)
            by_group = {}
            for row in rows:
                self.assertEqual(by_group.setdefault(row["group_id"], row["role"]), row["role"])
        for source in sources:
            assigned = [r for r in roles if r["audio_file"] == source["audio_file"]]
            self.assertEqual(sum(r["outer_evaluation_included"] for r in assigned), 1)

    def test_invalid_files_remain_outer_scoreable_and_never_inner_queries(self):
        roles, _ = build_calibration_roles(self.samples())
        for row in roles:
            if not row["source_train_eligible"]:
                self.assertFalse(row["encoder_fit_allowed"] or row["enrollment_allowed"] or row["calibration_query"])
                self.assertIn(row["role"], {"outer_validation", "training_excluded"})

    def test_order_independence_and_no_mutation(self):
        sources = self.samples()
        before = [dict(r) for r in sources]
        expected = build_calibration_roles(sources)
        self.assertEqual(expected, build_calibration_roles(list(reversed(sources))))
        self.assertEqual(sources, before)

    def test_invalid_outer_groups_and_missing_enrollment_are_rejected(self):
        sources = self.samples()
        cross_fold = [dict(r) for r in sources]
        cross_fold[-3]["fold"] = 1 - cross_fold[-3]["fold"]
        bad_label = [dict(r) for r in sources]
        bad_label[-3]["speaker_id"] = "alice"
        missing = [r for r in sources if not (r["speaker_id"] == "bob" and r["fold"] == 1)]
        for rows in (cross_fold, bad_label, missing, sources + [sources[0]]):
            with self.assertRaises(ValueError):
                build_calibration_roles(rows)


if __name__ == "__main__":
    unittest.main()
