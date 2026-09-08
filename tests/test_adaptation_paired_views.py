"""Synthetic crop alignment and sampler invariants; no real audio or Torch."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.adaptation.paired_views import paired_waveform_views


class PairedViewsTests(unittest.TestCase):
    def pair(self, signal, **overrides):
        arguments = {"rng": np.random.default_rng(81), "short_seconds": .8,
                     "long_seconds": 2., "sample_rate": 10}
        arguments.update(overrides)
        return paired_waveform_views(signal, **arguments)

    def test_long_teacher_and_nested_student_match_recorded_real_coordinates(self):
        signal = np.arange(117, dtype=np.float32)
        expected_rng = np.random.default_rng(81)
        teacher_start = int(expected_rng.integers(0, 98))
        offset = int(expected_rng.integers(0, 13))
        student, teacher, info = self.pair(signal)
        self.assertEqual(info['teacher'], {'start_sample': teacher_start, 'stop_sample': teacher_start + 20,
                                          'real_samples': 20, 'padding_samples': 0})
        self.assertEqual(info['student'], {'start_sample': teacher_start + offset,
            'stop_sample': teacher_start + offset + 8, 'real_samples': 8, 'offset_in_teacher': offset,
            'padding_samples': 0})
        np.testing.assert_array_equal(teacher, signal[teacher_start:teacher_start + 20])
        np.testing.assert_array_equal(student, teacher[offset:offset + 8])
        self.assertTrue(info['consistency_mask'])

    def test_short_inputs_and_exact_boundaries_are_retained_without_padding(self):
        for size in (1, 7, 8, 9, 19, 20, 21):
            with self.subTest(size=size):
                signal = np.arange(size, dtype=np.float32)
                student, teacher, info = self.pair(signal)
                self.assertEqual(len(teacher), min(size, 20))
                self.assertEqual(len(student), min(size, 8))
                self.assertEqual(info['consistency_mask'], size > 8)
                self.assertEqual(info['teacher']['padding_samples'], 0)
                self.assertEqual(info['student']['padding_samples'], 0)
                if size <= 20:
                    self.assertEqual(teacher.tobytes(), signal.tobytes())
                if size <= 8:
                    self.assertEqual(student.tobytes(), signal.tobytes())

    def test_whole_signal_copies_do_not_alias_input_or_each_other(self):
        signal = np.asarray([0., -0., .5, -.25], dtype=np.float32)
        before = signal.tobytes()
        student, teacher, info = self.pair(signal)
        self.assertEqual(student.tobytes(), before)
        self.assertEqual(teacher.tobytes(), before)
        self.assertFalse(info['consistency_mask'])
        self.assertFalse(np.shares_memory(student, teacher))
        self.assertFalse(np.shares_memory(student, signal))
        self.assertFalse(np.shares_memory(teacher, signal))
        student[0] = 1.
        self.assertEqual(teacher.tobytes(), before)
        teacher[1] = 2.
        self.assertEqual(signal.tobytes(), before)

    def test_strided_readonly_input_keeps_values_and_returns_contiguous_copies(self):
        signal = np.arange(100, dtype=np.float32)[::2]
        signal.flags.writeable = False
        before = signal.tobytes()
        student, teacher, info = self.pair(signal)
        for name, view in (('student', student), ('teacher', teacher)):
            self.assertTrue(view.flags.c_contiguous and view.flags.writeable)
            self.assertEqual(view.dtype, np.float32)
            self.assertEqual(view.tobytes(), signal[info[name]['start_sample']:info[name]['stop_sample']].tobytes())
        self.assertEqual(signal.tobytes(), before)

    def test_seed_and_restored_generator_state_reproduce_views_and_metadata(self):
        signal = np.arange(117, dtype=np.float32)
        rng = np.random.default_rng(902)
        state = deepcopy(rng.bit_generator.state)
        first = self.pair(signal, rng=rng)
        rng.bit_generator.state = state
        second = self.pair(signal, rng=rng)
        seeded = self.pair(signal, rng=np.random.default_rng(902))
        for compared in (second, seeded):
            self.assertEqual(first[0].tobytes(), compared[0].tobytes())
            self.assertEqual(first[1].tobytes(), compared[1].tobytes())
            self.assertEqual(first[2], compared[2])

    def test_global_rng_functions_are_never_used(self):
        signal = np.arange(50, dtype=np.float32)
        rng = np.random.default_rng(81)
        with (patch('numpy.random.default_rng', side_effect=AssertionError('implicit RNG')),
             patch('numpy.random.randint', side_effect=AssertionError('global RNG')),
             patch('numpy.random.random', side_effect=AssertionError('global RNG'))):
            paired_waveform_views(signal, rng=rng, short_seconds=.8, long_seconds=2., sample_rate=10)

    def test_uniform_ranges_include_both_teacher_and_nested_student_endpoints(self):
        signal = np.arange(11, dtype=np.float32)
        teacher_starts, student_offsets = set(), set()
        for seed in range(100):
            _, _, info = self.pair(signal, rng=np.random.default_rng(seed), short_seconds=.3, long_seconds=.5)
            teacher_starts.add(info['teacher']['start_sample'])
            student_offsets.add(info['student']['offset_in_teacher'])
        self.assertEqual(teacher_starts, set(range(7)))
        self.assertEqual(student_offsets, {0, 1, 2})

    def test_zero_waveform_is_not_filtered_or_mistaken_for_role_eligibility(self):
        student, teacher, info = self.pair(np.zeros(11, dtype=np.float32))
        self.assertEqual((len(student), len(teacher)), (8, 11))
        self.assertTrue(info['consistency_mask'])  # Length-only; caller owns signal/role checks.
        self.assertFalse(student.any() or teacher.any())

    def test_rounding_and_metadata_are_explicit_and_json_safe(self):
        _, _, info = self.pair(np.arange(11, dtype=np.float32), short_seconds=.25, long_seconds=.75)
        self.assertEqual(info['requested_short_samples'], 2)
        self.assertEqual(info['requested_long_samples'], 8)
        self.assertEqual(info['sample_rounding'], 'python_round_nearest_ties_even')
        self.assertEqual(json.loads(json.dumps(info, allow_nan=False)), info)
        self.assertEqual(set(info), {'schema_version', 'algorithm', 'sample_rate', 'input_samples',
            'requested_short_seconds', 'requested_long_seconds', 'requested_short_samples', 'requested_long_samples',
            'sample_rounding', 'rng_bit_generator', 'student', 'teacher', 'consistency_mask'})

    def test_invalid_duration_and_rate_arguments_leave_explicit_rng_untouched(self):
        invalid = [
            {'short_seconds': value} for value in (True, np.bool_(True), 0, -.5, np.nan, np.inf, '0.8', .01)
        ] + [
            {'long_seconds': value} for value in (False, 0, -1, np.nan, np.inf, '2', .8, .7, 1e100)
        ] + [{'short_seconds': .11, 'long_seconds': .14}] + [
            {'sample_rate': value} for value in (True, 0, -10, 10., '10', 10**1000)
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                rng = np.random.default_rng(71)
                state = deepcopy(rng.bit_generator.state)
                with self.assertRaises(ValueError):
                    self.pair(np.arange(50, dtype=np.float32), rng=rng, **kwargs)
                self.assertEqual(rng.bit_generator.state, state)

    def test_invalid_waveforms_leave_explicit_rng_untouched(self):
        for signal in ([1.], np.ones(3, dtype=np.float64), np.ones((2, 3), dtype=np.float32),
                       np.ones((), dtype=np.float32), np.asarray([], dtype=np.float32),
                       np.asarray([np.nan], dtype=np.float32), np.asarray([np.inf], dtype=np.float32),
                       np.ma.array([1., 2.], dtype=np.float32, mask=[False, True])):
            with self.subTest(signal=repr(signal)):
                rng = np.random.default_rng(71)
                state = deepcopy(rng.bit_generator.state)
                with self.assertRaises(ValueError): self.pair(signal, rng=rng)
                self.assertEqual(rng.bit_generator.state, state)

    def test_implicit_and_legacy_rngs_are_refused(self):
        for rng in (None, 17, np.random.RandomState(81)):
            with self.subTest(rng=type(rng).__name__):
                with self.assertRaises(ValueError): self.pair(np.arange(20, dtype=np.float32), rng=rng)


if __name__ == '__main__':
    unittest.main()
