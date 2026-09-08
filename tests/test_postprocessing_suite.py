"""Synthetic source-identity, baseline-tolerance and output-contract checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.postprocessing import suite


ROOT = Path(__file__).resolve().parents[1]


def configured_suite():
    return json.loads((ROOT / 'configs/postprocessing/campp_s011.json').read_text(encoding='utf-8'))


def output_fixture():
    labels = ['unknown'] + [f'speaker-{i:03}' for i in range(1, 447)]
    manifest = [{'audio_file': f'input-{i}', 'speaker_id': labels[label]}
                for i, label in enumerate((1, 3, 2, 0))]
    probabilities = np.zeros((3, 447), dtype=np.float64)
    probabilities[np.arange(3), [2, 1, 0]] = 1
    prepared = {'scores_by_alpha': {0.: {'outer_indices': np.asarray([2, 0, 3]),
                                         'outer_valid': np.asarray([True, True, False])}}}
    return {'labels': labels, 'manifest': manifest}, prepared, {'probabilities': probabilities}


def synthetic_source_files(root, *, audit_update=None, package_update=None, report_update=None):
    package = {'selection': {'family': 'public_advanced'}}
    audit = {'status': 'verified', 'parent_run_id': suite.SOURCE_PARENT,
             'children': {'S008c': suite.SOURCE_CHILD}, 'git_commit': suite.SOURCE_COMMIT,
             'all_four_mlflow_finished': True}
    report = {'oof': {'macro_f1': suite.BASELINE_MACRO_F1}}
    if audit_update:
        audit.update(audit_update)
    if package_update:
        package.update(package_update)
    if report_update:
        report.update(report_update)
    constants = {}
    for relative, data, constant in ((suite.PACKAGE_PATH, package, 'PACKAGE_SHA'),
                                     (suite.AUDIT_PATH, audit, 'AUDIT_SHA'),
                                     (suite.SOURCE_RUN + '/S008c/experiment_report.json', report, 'SOURCE_REPORT_SHA')):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')
        constants[constant] = hashlib.sha256(path.read_bytes()).hexdigest()
    return constants


class PostprocessingSuiteTests(unittest.TestCase):
    def test_committed_configuration_pins_sources_device_candidates_and_tolerances(self):
        original = configured_suite()
        suite.validate_config(original)
        changes = [('source_package_config', 'configs/package/campp_old.json'),
                   ('source_package_config_sha256', '0' * 64),
                   ('source_verification', 'artifacts/infrastructure/wrong.json'),
                   ('source_verification_sha256', '0' * 64), ('experiment_code', 'S999'),
                   ('device', 'cpu'), ('cpu_threads', True), ('cpu_threads', 0),
                   ('cpu_threads', 17), ('probability_temperature', .1),
                   ('candidates', original['candidates'][:-1])]
        for key, value in changes:
            with self.subTest(field=key, value=value):
                changed = deepcopy(original)
                changed[key] = value
                with self.assertRaises(ValueError):
                    suite.validate_config(changed)
        for key, value in [('threshold_atol', 1e-5), ('probability_atol', 1e-4),
                           ('relative_tolerance', 1e-5), ('rationale', '')]:
            with self.subTest(numerical_policy=key):
                changed = deepcopy(original)
                changed['baseline_numerical_policy'][key] = value
                with self.assertRaisesRegex(ValueError, 'no automatic relaxation'):
                    suite.validate_config(changed)
        changed = deepcopy(original)
        del changed['baseline_numerical_policy']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            suite.validate_config(changed)

    def test_only_threshold_float_drift_is_tolerated_in_baseline_calibration(self):
        expected = {'threshold': .25, 'unknown_weight': .5, 'margin_weight': 0.,
                    'inner_macro_f1_447': .95}
        observed = {**expected, 'threshold': .25 + .9e-6}
        self.assertAlmostEqual(suite._compare_calibration(observed, expected), .9e-6, places=14)
        for key, value in [('threshold', .25 + 1.1e-6), ('threshold', np.nan),
                           ('threshold', np.inf), ('unknown_weight', .500000001),
                           ('margin_weight', .000000001), ('inner_macro_f1_447', .950000001)]:
            with self.subTest(field=key, value=value):
                with self.assertRaises(ValueError):
                    suite._compare_calibration({**expected, key: value}, expected)
        with self.assertRaises(ValueError):
            suite._compare_calibration({**expected, 'unregistered_field': 1}, expected)

    def test_alpha_search_verifies_every_curve_not_just_selected_setting(self):
        calibration = {'threshold': .25, 'unknown_weight': .5, 'margin_weight': 0.,
                       'inner_macro_f1_447': .95}
        expected = {str(alpha): {'advanced_weight': alpha, 'selected': dict(calibration),
                                 'curve': [dict(calibration), {**calibration, 'threshold': .4}]}
                    for alpha in (0., .25, .5, .75, 1.)}
        result = suite._compare_alpha_search(deepcopy(expected), expected)
        self.assertEqual(result['curve_rows_verified'], 10)
        observed = deepcopy(expected)
        observed['0.25']['curve'][1]['threshold'] += .8e-6
        result = suite._compare_alpha_search(observed, expected)
        self.assertLessEqual(result['curve_threshold_max_abs_difference'], 1e-6)
        for mutation in ('missing_alpha', 'changed_identity', 'missing_curve', 'changed_unselected_f1'):
            with self.subTest(mutation=mutation):
                observed = deepcopy(expected)
                if mutation == 'missing_alpha':
                    del observed['0.25']
                elif mutation == 'changed_identity':
                    observed['0.25']['advanced_weight'] = .5
                elif mutation == 'missing_curve':
                    observed['0.25']['curve'].pop()
                else:
                    observed['0.25']['curve'][1]['inner_macro_f1_447'] += 1e-10
                with self.assertRaises(ValueError):
                    suite._compare_alpha_search(observed, expected)

    def test_source_semantic_identity_checked_even_after_hash_validation(self):
        cases = [({}, {}, {}),
                 ({'parent_run_id': 'a' * 32}, {}, {}),
                 ({'children': {'S008c': 'a' * 32}}, {}, {}),
                 ({'git_commit': 'a' * 40}, {}, {}),
                 ({'all_four_mlflow_finished': False}, {}, {}),
                 ({'status': 'pending'}, {}, {}),
                 ({}, {'selection': {'family': 'adapted_advanced'}}, {}),
                 ({}, {}, {'oof': {'macro_f1': .96755}})]
        for i, (audit, package, report) in enumerate(cases):
            with self.subTest(case=i), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                constants = synthetic_source_files(root, audit_update=audit, package_update=package,
                                                   report_update=report)
                with patch.multiple(suite, **constants):
                    if i == 0:
                        result = suite._source_checks(root, configured_suite())
                        self.assertEqual(result['audit']['parent_run_id'], suite.SOURCE_PARENT)
                    else:
                        with self.assertRaises(ValueError):
                            suite._source_checks(root, configured_suite())

    def test_changed_source_bytes_are_rejected_without_interpreting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constants = synthetic_source_files(root)
            (root / suite.PACKAGE_PATH).write_text('unexpected bytes', encoding='utf-8')
            with patch.multiple(suite, **constants), self.assertRaisesRegex(ValueError, 'Pinned source'):
                suite._source_checks(root, configured_suite())

    def test_result_predictions_keep_source_outer_order_and_all_447_classes(self):
        contract, prepared, result = output_fixture()
        predictions, metrics = suite._result_predictions(contract, prepared, result)
        self.assertEqual([row['audio_file'] for row in predictions], ['input-2', 'input-0', 'input-3'])
        self.assertEqual(metrics['row_count'], 3)
        self.assertEqual(metrics['class_count'], 447)
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertEqual(predictions[-1]['speaker_id'], 'unknown')

    def test_result_predictions_reject_invalid_probability_shape_values_and_fallback(self):
        for mutation in ('missing_row', 'missing_class', 'nan', 'negative', 'not_normalized', 'invalid_known'):
            with self.subTest(mutation=mutation):
                contract, prepared, result = output_fixture()
                values = result['probabilities']
                if mutation == 'missing_row':
                    result['probabilities'] = values[:-1]
                elif mutation == 'missing_class':
                    result['probabilities'] = values[:, :-1]
                elif mutation == 'nan':
                    values[0, 0] = np.nan
                elif mutation == 'negative':
                    values[0, 0] = -1e-6
                    values[0, 2] += 1e-6
                elif mutation == 'not_normalized':
                    values[0, 2] = .9
                else:
                    values[2, 0] = 1-1e-6
                    values[2, 1] = 1e-6
                with self.assertRaises(ValueError):
                    suite._result_predictions(contract, prepared, result)

    def test_result_predictions_reject_duplicate_source_rows_and_nonfixed_label_map(self):
        contract, prepared, result = output_fixture()
        prepared['scores_by_alpha'][0.]['outer_indices'][1] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate audio_file'):
            suite._result_predictions(contract, prepared, result)
        contract, prepared, result = output_fixture()
        contract['labels'][-1] = contract['labels'][-2]
        with self.assertRaisesRegex(ValueError, 'fixed 447-class'):
            suite._result_predictions(contract, prepared, result)


if __name__ == '__main__':
    unittest.main()
