"""A data-agnostic, stop-gradient short/long embedding alignment objective.

No model, crop generation, optimizer, classification loss or schedule lives here.
Real sample counts must come from the paired-view metadata, before any padding.
Tensor lengths alone cannot authenticate the original waveform or its fit role.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def normalized_cosine_alignment(
    student: "torch.Tensor",
    teacher: "torch.Tensor",
    eligible: "torch.Tensor",
    *,
    student_real_samples: "torch.Tensor",
    teacher_real_samples: "torch.Tensor",
    embedding_dim: int = 192,
) -> tuple["torch.Tensor", dict]:
    """Return mean ``1-cosine`` on eligible pairs with strictly longer real views.

    Embeddings are dense finite FP32 ``[batch, embedding_dim]`` tensors. The mask
    is bool ``[batch]``; real sample counts are positive int64 ``[batch]`` on the
    same device. Nested views require teacher counts >= student counts. Equal
    counts are retained by the caller for classification, but contribute no
    alignment. The supplied mask can only exclude additional rows.

    Teacher targets are detached and copied internally, including targets made
    under inference mode. Student and teacher must be distinct tensor objects.
    Max-absolute scaling followed by FP32 L2 avoids overflow/underflow in the
    forward norm without an epsilon that would change tiny nonzero directions.
    The exact derivative for an extremely tiny vector can still exceed FP32;
    an eventual training loop must check its gradients independently.

    Empty effective eligibility returns the sum of an empty student selection:
    exact zero, with a student autograd connection when autograd is enabled.
    No loss coefficient or mean over the entire batch is applied. Diagnostics
    contain only detached Python values and cannot retain an autograd graph.
    """
    import torch

    if type(embedding_dim) is not int or embedding_dim <= 0:
        raise ValueError("embedding_dim must be an explicit positive integer")
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError("Student and teacher embeddings must be tensors")
    if student is teacher:
        raise ValueError("Student and teacher must be distinct tensor objects")
    if (student.layout != torch.strided or teacher.layout != torch.strided
            or student.dtype != torch.float32 or teacher.dtype != torch.float32
            or student.ndim != 2 or teacher.shape != student.shape
            or student.shape[1] != embedding_dim):
        raise ValueError("Embeddings must be matching dense FP32 [batch, embedding_dim] tensors")
    if student.device != teacher.device or student.device.type not in ("cpu", "cuda"):
        raise ValueError("Embedding devices must match and contain real CPU or CUDA data")
    batch = student.shape[0]
    for name, tensor, dtype in (
        ("eligible", eligible, torch.bool),
        ("student_real_samples", student_real_samples, torch.int64),
        ("teacher_real_samples", teacher_real_samples, torch.int64),
    ):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.dtype != dtype or tensor.shape != (batch,) or tensor.device != student.device):
            raise ValueError(f"{name} must have its exact dtype, [batch] shape and embedding device")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("All embeddings, including excluded rows, must be finite")
    if bool((student_real_samples <= 0).any()) or bool((teacher_real_samples <= 0).any()):
        raise ValueError("Real sample counts must be positive; padding is not real evidence")
    if bool((teacher_real_samples < student_real_samples).any()):
        raise ValueError("A nested teacher view cannot contain fewer real samples than its student")

    longer = teacher_real_samples > student_real_samples
    effective = eligible & longer
    count = int(effective.sum().item())
    diagnostics = {
        "batch_size": batch,
        "requested_eligible_count": int(eligible.sum().item()),
        "genuinely_longer_count": int(longer.sum().item()),
        "equal_real_length_count": int((~longer).sum().item()),
        "effective_count": count,
        "effective_fraction": count / batch if batch else 0.0,
        "excluded_count": batch - count,
        "mean_cosine": None,
        "minimum_cosine": None,
        "maximum_cosine": None,
        "alignment_loss": 0.0,
        "teacher_stop_gradient": True,
        "reduction": "mean_over_effective_pairs",
        "eligibility_basis": "caller_mask_and_strictly_longer_real_sample_count",
    }
    selected_student = student[effective]
    if count == 0:
        return selected_student.sum(), diagnostics

    # This explicit copy also converts an inference-mode target into an ordinary
    # tensor that autograd may safely save while differentiating the student.
    selected_teacher = teacher.detach()[effective].clone()
    with torch.autocast(device_type=student.device.type, enabled=False):
        def unit_vectors(values):
            scale = values.abs().amax(dim=1, keepdim=True).detach()
            if bool((scale == 0).any()):
                raise ValueError("An eligible embedding is exactly zero; its direction is undefined")
            scaled = values / scale
            norm = scaled.square().sum(dim=1, keepdim=True).sqrt()
            if not bool(torch.isfinite(norm).all()) or bool((norm <= 0).any()):
                raise ValueError("FP32 normalization produced an invalid norm")
            return scaled / norm

        cosines = (unit_vectors(selected_student) * unit_vectors(selected_teacher)).sum(dim=1).clamp(-1.0, 1.0)
        loss = (1.0 - cosines).mean()
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise ValueError("Alignment loss must be finite FP32")
    detached = cosines.detach()
    diagnostics.update(
        mean_cosine=float(detached.mean().item()),
        minimum_cosine=float(detached.min().item()),
        maximum_cosine=float(detached.max().item()),
        alignment_loss=float(loss.detach().item()),
    )
    return loss, diagnostics
