"""CPU-only synthetic embedding/gradient contracts; no models or optimizers."""
from copy import deepcopy
import json
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from speaker_id.adaptation.consistency import normalized_cosine_alignment


@unittest.skipIf(torch is None, "CPU Torch is available in the separate release QA environment")
class ConsistencyTests(unittest.TestCase):
    def call(self, student, teacher, eligible=None, short=None, long=None, dimension=2):
        batch = student.shape[0]
        eligible = torch.ones(batch, dtype=torch.bool) if eligible is None else eligible
        short = torch.full((batch,), 10, dtype=torch.int64) if short is None else short
        long = torch.full((batch,), 20, dtype=torch.int64) if long is None else long
        return normalized_cosine_alignment(student, teacher, eligible,
            student_real_samples=short, teacher_real_samples=long, embedding_dim=dimension)

    def test_exact_aligned_orthogonal_and_opposite_mean(self):
        student = torch.tensor([[2.,0.],[0.,3.],[-4.,0.]], requires_grad=True)
        teacher = torch.tensor([[7.,0.],[8.,0.],[9.,0.]], requires_grad=True)
        loss, info = self.call(student, teacher)
        self.assertEqual(loss.dtype, torch.float32); self.assertEqual(loss.item(), 1.0)
        self.assertEqual(info['effective_count'], 3); self.assertEqual(info['mean_cosine'], 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all()); self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad[1], torch.tensor([-1.0 / 9.0, 0.0]))

    def test_true_lengths_override_mask_and_mask_does_not_expand(self):
        student = torch.tensor([[1.,0.],[1.,0.],[1.,0.]], requires_grad=True)
        teacher = torch.tensor([[0.,1.],[-1.,0.],[1.,0.]], requires_grad=True)
        loss, info = self.call(student, teacher, torch.tensor([True,True,False]),
            torch.tensor([10,10,10]), torch.tensor([20,10,20]))
        self.assertEqual(loss.item(), 1.0); self.assertEqual(info['effective_count'], 1)
        self.assertEqual(info['equal_real_length_count'], 1)
        loss.backward(); self.assertTrue(torch.equal(student.grad[1:], torch.zeros((2,2))))
        self.assertIsNone(teacher.grad)

    def test_false_mask_huge_values_returns_exact_differentiable_zero(self):
        student = torch.full((3,192), torch.finfo(torch.float32).max, requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        loss, info = self.call(student, teacher, torch.zeros(3,dtype=torch.bool), dimension=192)
        self.assertEqual(loss.item(), 0.0); self.assertTrue(loss.requires_grad)
        loss.backward(); self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
        self.assertIsNone(teacher.grad); self.assertEqual(info['effective_count'], 0)
        self.assertIsNone(info['mean_cosine'])

    def test_empty_batch_remains_connected_without_nan(self):
        student = torch.empty((0,192), requires_grad=True)
        teacher = torch.empty((0,192), requires_grad=True)
        loss, info = self.call(student, teacher, dimension=192)
        loss.backward(); self.assertEqual(loss.item(), 0.0)
        self.assertEqual(student.grad.shape, (0,192)); self.assertIsNone(teacher.grad)
        self.assertEqual(info['effective_fraction'], 0.0)

    def test_maximum_and_tiny_nonzero_scales_preserve_directions(self):
        for value in (torch.finfo(torch.float32).max, 1e-30, torch.finfo(torch.float32).tiny):
            student = torch.tensor([[value,0.]], requires_grad=True)
            teacher = torch.tensor([[0.,value]], requires_grad=True)
            loss, _ = self.call(student, teacher); self.assertEqual(loss.item(), 1.0)
            loss.backward(); self.assertTrue(torch.isfinite(student.grad).all()); self.assertIsNone(teacher.grad)
        # The forward direction of even a subnormal is meaningful; its exact
        # nonzero derivative need not be representable in FP32, so no false
        # finite-gradient guarantee is made for that extreme input.
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
        loss, _ = self.call(torch.stack([tiny, tiny])[None], torch.stack([tiny, -tiny])[None])
        self.assertEqual(loss.item(), 1.0)

    def test_teacher_inference_mode_tensor_and_requires_grad_are_safe(self):
        student = torch.tensor([[1.,2.]], requires_grad=True)
        with torch.inference_mode():
            teacher = torch.tensor([[2.,1.]])
        loss, _ = self.call(student, teacher); loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        student = torch.tensor([[1.,2.]], requires_grad=True)
        teacher = (student * 2).clone(); teacher.retain_grad()
        loss, _ = self.call(student, teacher); loss.backward()
        self.assertIsNone(teacher.grad)

    def test_inputs_unmodified_and_diagnostics_plain_detached_values(self):
        student = torch.tensor([[1.,2.],[4.,3.]], requires_grad=True)
        teacher = torch.tensor([[2.,1.],[1.,4.]], requires_grad=True)
        mask = torch.tensor([True,False]); short = torch.tensor([3,4]); long = torch.tensor([6,4])
        inputs = [student,teacher,mask,short,long]; copies = [v.detach().clone() for v in inputs]
        loss, info = self.call(student,teacher,mask,short,long); snapshot = deepcopy(info)
        json.dumps(info, allow_nan=False); loss.backward()
        self.assertEqual(info,snapshot)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs,copies)))
        self.assertTrue(all(type(v) in (int,float,bool,str,type(None)) for v in info.values()))

    def test_default_dimension_is_192_and_override_strict(self):
        s = torch.ones((1,2)); t = s.clone(); mask = torch.ones(1,dtype=torch.bool); count = torch.ones(1,dtype=torch.int64)
        with self.assertRaises(ValueError): normalized_cosine_alignment(s,t,mask,student_real_samples=count,teacher_real_samples=count)
        for bad in (True,2.0,0,-1):
            with self.assertRaises(ValueError): self.call(s,t,dimension=bad)
        with self.assertRaises(ValueError): self.call(s,s)

    def test_invalid_dtype_shape_device_mask_and_real_counts_rejected(self):
        s = torch.ones((2,2)); t = s.clone()
        bad_cases = [dict(eligible=torch.ones(2,dtype=torch.int64)),dict(eligible=torch.ones((2,1),dtype=torch.bool)),
            dict(eligible=torch.ones(2,dtype=torch.bool,device='meta')),
            dict(short=torch.tensor([1.,1.])),dict(short=torch.tensor([0,1])),dict(short=torch.tensor([-1,1])),
            dict(short=torch.tensor([3,1]),long=torch.tensor([2,2])),dict(short=torch.ones((2,1),dtype=torch.int64)),
            dict(long=torch.ones(2,dtype=torch.int64,device='meta'))]
        for kwargs in bad_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.call(s,t,**kwargs)
        for a,b in ((s.double(),t),(s,t.half()),(s,t[:1]),(s,torch.ones((2,2),device='meta'))):
            with self.assertRaises(ValueError): self.call(a,b)

    def test_nonfinite_in_excluded_rows_and_eligible_zeros_rejected(self):
        s = torch.ones((2,2)); t = s.clone()
        for bad in (float('nan'),float('inf'),-float('inf')):
            changed = t.clone(); changed[1,0] = bad
            with self.assertRaises(ValueError): self.call(s,changed,torch.tensor([True,False]))
        for a,b in ((torch.zeros((2,2)),t),(s,torch.zeros((2,2)))):
            with self.assertRaises(ValueError): self.call(a,b)
        loss,_ = self.call(torch.zeros((2,2),requires_grad=True),torch.zeros((2,2)),torch.zeros(2,dtype=torch.bool))
        self.assertEqual(loss.item(),0.0)

    def test_cpu_autocast_does_not_change_loss_precision(self):
        student = torch.tensor([[1.,2.]],requires_grad=True); teacher = torch.tensor([[2.,1.]])
        expected,_ = self.call(student,teacher)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            observed,_ = self.call(student,teacher)
        self.assertEqual(observed.dtype,torch.float32); self.assertEqual(observed.item(),expected.item())


if __name__ == '__main__':
    unittest.main()
