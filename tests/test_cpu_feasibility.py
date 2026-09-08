"""Synthetic metadata/resource tests; no audio, model, network or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from speaker_id.infrastructure import cpu_feasibility as cpu


def manifest():
    rows = [{'has_nonzero_signal':True,'duration_seconds':i+2,'mono_rms_dbfs':-20-i} for i in range(12)]
    rows[1].update(has_nonzero_signal=False, duration_seconds=3, mono_rms_dbfs='-inf')
    rows[2]['duration_seconds'] = .2
    rows[4]['duration_seconds'] = 250
    rows[8]['mono_rms_dbfs'] = -120
    return rows


def capacity():
    return {'quota_cores':15.36,'affinity_cpus':128,'memory_headroom_bytes':80*2**30,'disk_free_bytes':90*2**30}


class CPUFeasibilityTests(unittest.TestCase):
    def test_exact_config_and_no_alternate_device_or_gain_search(self):
        root = Path(__file__).resolve().parents[1]
        cpu.validate_pilot_config(json.loads((root/'configs/infra/campp_cpu_pilot.json').read_text()))
        for key, value in [('device','cuda'),('threads',128),('max_elapsed_seconds',6000),('schema_version',True),
                ('instance_id',50079024),('gain_policy',{}),('calibration',True)]:
            changed = deepcopy(cpu.FIXED); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): cpu.validate_pilot_config(changed)

    def test_deterministic_metadata_extrema_and_deduplication(self):
        self.assertEqual(cpu.pilot_indices(manifest()), [0,1,2,4,6,8,11])
        tiny = manifest()[:2]
        self.assertEqual(cpu.pilot_indices(tiny),[0,1])

    def test_labels_roles_and_errors_do_not_participate(self):
        rows = manifest(); original = cpu.pilot_indices(rows)
        for i,row in enumerate(rows):
            row.update(speaker_id=object(),outer_fold=1-i%2,predicted=object(),error=object(),role=object())
        self.assertEqual(cpu.pilot_indices(rows),original)

    def test_reject_missing_zero_nonfinite_or_invalid_metadata(self):
        for field,value in [('duration_seconds',float('nan')),('duration_seconds',0),('mono_rms_dbfs',float('inf'))]:
            rows = manifest(); rows[0][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError): cpu.pilot_indices(rows)
        with self.assertRaises(ValueError): cpu.pilot_indices([row for row in manifest() if row['has_nonzero_signal']])

    def test_quota_not_visible_cpu_count_controls_resource_gate(self):
        cpu.validate_capacity(capacity(),cpu.FIXED)
        for key,value in [('quota_cores',3.99),('affinity_cpus',3),('memory_headroom_bytes',2**30),
                ('disk_free_bytes',2**30),('quota_cores',float('nan'))]:
            row=capacity(); row[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): cpu.validate_capacity(row,cpu.FIXED)

    def test_execution_on_local_windows_is_rejected_before_process_or_network(self):
        with patch.object(cpu.platform,'system',return_value='Windows'), patch.object(cpu.subprocess,'check_output') as process:
            with self.assertRaises(ValueError): cpu.server_cpu_gate(Path.cwd(),cpu.FIXED)
            process.assert_not_called()

    def test_soft_deadline_allows_case_then_rejects_boundary_and_overrun(self):
        cpu.check_deadline(10,600,now=609.999)
        for now in (610,901):
            with self.assertRaises(TimeoutError): cpu.check_deadline(10,600,now=now)

    def test_hash_failure_does_not_hide_other_hashes_or_original_failure(self):
        def hasher(value):
            if value is None: raise FileNotFoundError('missing retained file')
            return value
        hashes,errors=cpu.best_effort_hashes({'public':'a'*64,'advanced':None},hasher)
        self.assertEqual(hashes,{'public':'a'*64})
        self.assertEqual(errors,{'advanced':'FileNotFoundError'})

    def test_effective_seconds_pad_cap_and_zero_cost(self):
        rows=manifest()
        self.assertEqual(cpu.effective_seconds(rows[2]),1)
        self.assertEqual(cpu.effective_seconds(rows[4]),180)
        self.assertEqual(cpu.effective_seconds(rows[1]),0)

    def test_estimates_are_two_frontend_hypotheses_not_automatic_full_execution(self):
        records=[]
        for frontend in ('identity','gain'):
            records += [{'frontend':frontend,'valid':True,'effective_audio_seconds':10.,'elapsed_seconds':2.},
                        {'frontend':frontend,'valid':False,'effective_audio_seconds':0.,'elapsed_seconds':.1}]
        value=cpu.estimate_runtime(manifest(),records,capacity())
        self.assertFalse(value['automatic_full_run_authorized'])
        self.assertFalse(value['parallel_scaling_measured'])
        self.assertAlmostEqual(value['hypothetical_three_worker_seconds'],value['conservative_two_frontend_sequential_seconds']/2)
        low=capacity();low['quota_cores']=4
        self.assertIsNone(cpu.estimate_runtime(manifest(),records,low)['hypothetical_three_worker_seconds'])
        low=capacity();low['affinity_cpus']=4
        self.assertIsNone(cpu.estimate_runtime(manifest(),records,low)['hypothetical_three_worker_seconds'])
        records[0]['elapsed_seconds']=float('nan')
        with self.assertRaises(ValueError):cpu.estimate_runtime(manifest(),records,capacity())


if __name__ == '__main__':
    unittest.main()
