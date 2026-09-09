"""Small open-set outlier-exposure objectives for an existing AAM classifier.

The functions deliberately accept an embedding tensor and an AAM head's
``weight`` tensor rather than importing or modifying :class:`AAMHead`.  They
therefore keep the known-speaker AAM objective intact: F008 can compute
pre-margin, unscaled cosine logits for unknown examples and apply exactly one
explicit auxiliary objective to them.

All public tensor inputs are dense finite FP32 matrices on one CPU or CUDA
device.  The validation is intentionally stricter than a convenience loss:
silently accepting a mixed-precision, malformed, or non-cosine input would
make an open-set experiment hard to reproduce.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_COSINE_ROUNDING_TOLERANCE = 1.0e-5


def raw_cosine_logits(
    embeddings: "torch.Tensor",
    head_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Return pre-margin, unscaled cosine logits against AAM head weights.

    ``head_weight`` is normally ``aam_head.weight`` with shape
    ``[known_classes, embedding_dim]``.  No AAM angular margin, scale, target,
    unknown class, or classifier copy is involved.  Row-wise max-absolute
    scaling avoids norm overflow before the ordinary L2 normalization; it does
    not change a nonzero vector's direction or the mathematical cosine.

    Empty batches, zero vectors, nonfinite values, non-FP32 values, device
    mismatches, sparse/meta tensors, and incompatible shapes are rejected.
    The result stays FP32 even when the caller has autocast enabled.
    """
    import torch

    _validate_embedding_and_head_weight(embeddings, head_weight)
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        normalized_embeddings = _stable_l2_normalize_rows(embeddings, name="embeddings")
        normalized_weight = _stable_l2_normalize_rows(head_weight, name="head_weight")
        logits = normalized_embeddings @ normalized_weight.transpose(0, 1)
        # FP32 dot products can exceed the mathematical interval by a few ulps.
        # The clamp preserves the intended cosine domain for subsequent losses.
        logits = logits.clamp(min=-1.0, max=1.0)
    _require_finite_tensor("raw cosine logits", logits)
    return logits


def energy_margin_oe_loss(
    cosine_logits: "torch.Tensor",
    *,
    energy_temperature: Real,
    minimum_energy: Real,
    softplus_temperature: Real = 1.0,
) -> tuple["torch.Tensor", dict[str, float | int | str]]:
    """Penalize unknown rows whose free energy is below a declared margin.

    For each unknown row, this computes

    ``energy = -T * logsumexp(cosine_logits / T)``

    and returns the mean smooth penalty

    ``softplus((minimum_energy - energy) / softplus_temperature)``.

    Minimizing it drives unknown examples away from *all* known AAM weight
    vectors without treating diverse unknown speakers as one additional class.
    The companion diagnostics are detached JSON-safe values for MLflow logs.
    """
    import torch
    from torch.nn import functional as F

    _validate_cosine_logits(cosine_logits)
    temperature = _positive_real("energy_temperature", energy_temperature)
    margin = _finite_real("minimum_energy", minimum_energy)
    softness = _positive_real("softplus_temperature", softplus_temperature)

    with torch.autocast(device_type=cosine_logits.device.type, enabled=False):
        energy = -temperature * torch.logsumexp(cosine_logits / temperature, dim=1)
        loss = F.softplus((margin - energy) / softness).mean()
    _require_finite_tensor("open-set energy", energy)
    _require_finite_scalar("energy-margin OE loss", loss)

    detached_energy = energy.detach()
    return loss, {
        "objective": "energy_margin_oe",
        "batch_size": int(cosine_logits.shape[0]),
        "class_count": int(cosine_logits.shape[1]),
        "energy_temperature": temperature,
        "minimum_energy": margin,
        "softplus_temperature": softness,
        "mean_energy": float(detached_energy.mean().cpu()),
        "minimum_observed_energy": float(detached_energy.min().cpu()),
        "maximum_observed_energy": float(detached_energy.max().cpu()),
        "margin_satisfied_fraction": float((detached_energy >= margin).float().mean().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def energy_separation_oe_loss(
    known_cosine_logits: "torch.Tensor",
    unknown_cosine_logits: "torch.Tensor",
    *,
    energy_temperature: Real,
    maximum_known_energy: Real,
    minimum_unknown_energy: Real,
    softplus_temperature: Real = 1.0,
) -> tuple["torch.Tensor", dict[str, float | int | str]]:
    """Keep known energy low while pushing unknown energy above a fixed gap.

    The two inputs are pre-margin, unscaled cosine logits against the same AAM
    head.  They may contain different numbers of rows, so known and unknown
    data loaders can use independent batch sizes; both must still expose the
    same known-class columns on the same device.  For
    ``energy = -T * logsumexp(logits / T)``, the objective is

    ``0.5 * [mean(softplus((known_energy - max_known) / S))``
    ``       + mean(softplus((min_unknown - unknown_energy) / S))]``.

    ``minimum_unknown_energy`` must be strictly greater than
    ``maximum_known_energy``.  This makes the separation gap explicit and
    prevents a configuration from silently requesting overlapping acceptance
    regions.  Equal weighting of the two batch means prevents one stream's
    batch size from changing the intended objective scale.
    """
    import torch
    from torch.nn import functional as F

    _validate_energy_separation_logits(known_cosine_logits, unknown_cosine_logits)
    temperature = _positive_real("energy_temperature", energy_temperature)
    maximum_known = _finite_real("maximum_known_energy", maximum_known_energy)
    minimum_unknown = _finite_real("minimum_unknown_energy", minimum_unknown_energy)
    if minimum_unknown <= maximum_known:
        raise ValueError(
            "minimum_unknown_energy must be strictly greater than maximum_known_energy"
        )
    softness = _positive_real("softplus_temperature", softplus_temperature)

    with torch.autocast(device_type=known_cosine_logits.device.type, enabled=False):
        known_energy = -temperature * torch.logsumexp(known_cosine_logits / temperature, dim=1)
        unknown_energy = -temperature * torch.logsumexp(
            unknown_cosine_logits / temperature, dim=1
        )
        known_penalty = F.softplus((known_energy - maximum_known) / softness).mean()
        unknown_penalty = F.softplus((minimum_unknown - unknown_energy) / softness).mean()
        loss = 0.5 * (known_penalty + unknown_penalty)
    _require_finite_tensor("known open-set energy", known_energy)
    _require_finite_tensor("unknown open-set energy", unknown_energy)
    _require_finite_scalar("known energy-separation penalty", known_penalty)
    _require_finite_scalar("unknown energy-separation penalty", unknown_penalty)
    _require_finite_scalar("energy-separation OE loss", loss)

    detached_known = known_energy.detach()
    detached_unknown = unknown_energy.detach()
    return loss, {
        "objective": "energy_separation_oe",
        "known_batch_size": int(known_cosine_logits.shape[0]),
        "unknown_batch_size": int(unknown_cosine_logits.shape[0]),
        "class_count": int(known_cosine_logits.shape[1]),
        "energy_temperature": temperature,
        "maximum_known_energy": maximum_known,
        "minimum_unknown_energy": minimum_unknown,
        "declared_energy_gap": minimum_unknown - maximum_known,
        "softplus_temperature": softness,
        "mean_known_energy": float(detached_known.mean().cpu()),
        "mean_unknown_energy": float(detached_unknown.mean().cpu()),
        "known_margin_satisfied_fraction": float(
            (detached_known <= maximum_known).float().mean().cpu()
        ),
        "unknown_margin_satisfied_fraction": float(
            (detached_unknown >= minimum_unknown).float().mean().cpu()
        ),
        "known_penalty": float(known_penalty.detach().cpu()),
        "unknown_penalty": float(unknown_penalty.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def uniform_oe_loss(
    cosine_logits: "torch.Tensor",
    *,
    temperature: Real,
) -> tuple["torch.Tensor", dict[str, float | int | str]]:
    """Return cross-entropy from the uniform known-class distribution.

    This is the standard uniform outlier-exposure control:
    ``-mean_c log softmax(cosine_logits / temperature)_c``.  Its additive
    ``log(class_count)`` constant is retained so the logged loss is a genuine
    cross-entropy; ``mean_uniform_kl`` reports the corresponding KL term.
    It encourages uncertainty over known speakers but never creates an
    artificial shared unknown identity.
    """
    import torch

    _validate_cosine_logits(cosine_logits)
    resolved_temperature = _positive_real("temperature", temperature)

    with torch.autocast(device_type=cosine_logits.device.type, enabled=False):
        log_probabilities = torch.log_softmax(cosine_logits / resolved_temperature, dim=1)
        loss = -log_probabilities.mean()
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(dim=1)
    _require_finite_tensor("uniform-OE log probabilities", log_probabilities)
    _require_finite_scalar("uniform OE loss", loss)
    _require_finite_tensor("uniform-OE entropy", entropy)

    classes = int(cosine_logits.shape[1])
    detached_probabilities = probabilities.detach()
    detached_entropy = entropy.detach()
    return loss, {
        "objective": "uniform_oe",
        "batch_size": int(cosine_logits.shape[0]),
        "class_count": classes,
        "temperature": resolved_temperature,
        "mean_max_probability": float(detached_probabilities.max(dim=1).values.mean().cpu()),
        "mean_entropy": float(detached_entropy.mean().cpu()),
        "mean_uniform_kl": float((loss.detach() - math.log(classes)).cpu()),
        "loss": float(loss.detach().cpu()),
    }


def _validate_embedding_and_head_weight(
    embeddings: "torch.Tensor",
    head_weight: "torch.Tensor",
) -> None:
    import torch

    _validate_dense_fp32_matrix("embeddings", embeddings)
    _validate_dense_fp32_matrix("head_weight", head_weight)
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must contain at least one unknown row")
    if head_weight.shape[0] == 0:
        raise ValueError("head_weight must contain at least one known class")
    if embeddings.shape[1] != head_weight.shape[1]:
        raise ValueError("embeddings and head_weight must have the same embedding dimension")
    if embeddings.device != head_weight.device:
        raise ValueError("embeddings and head_weight must be on the same device")
    _require_nonzero_rows("embeddings", embeddings)
    _require_nonzero_rows("head_weight", head_weight)


def _validate_cosine_logits(cosine_logits: "torch.Tensor") -> None:
    _validate_dense_fp32_matrix("cosine_logits", cosine_logits)
    if cosine_logits.shape[0] == 0:
        raise ValueError("cosine_logits must contain at least one unknown row")
    if cosine_logits.shape[1] < 2:
        raise ValueError("cosine_logits must contain at least two known classes")
    detached = cosine_logits.detach()
    minimum = float(detached.min().cpu())
    maximum = float(detached.max().cpu())
    if minimum < -1.0 - _COSINE_ROUNDING_TOLERANCE or maximum > 1.0 + _COSINE_ROUNDING_TOLERANCE:
        raise ValueError("cosine_logits must be bounded by the cosine interval [-1, 1]")


def _validate_energy_separation_logits(
    known_cosine_logits: "torch.Tensor",
    unknown_cosine_logits: "torch.Tensor",
) -> None:
    _validate_cosine_logits(known_cosine_logits)
    _validate_cosine_logits(unknown_cosine_logits)
    if known_cosine_logits.shape[1] != unknown_cosine_logits.shape[1]:
        raise ValueError("known and unknown cosine logits must have the same class count")
    if known_cosine_logits.device != unknown_cosine_logits.device:
        raise ValueError("known and unknown cosine logits must be on the same device")


def _validate_dense_fp32_matrix(name: str, tensor: object) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if tensor.layout != torch.strided or tensor.is_meta:
        raise ValueError(f"{name} must be a materialized dense strided tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be on a real CPU or CUDA device")
    if tensor.ndim != 2 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must have shape [rows, nonzero_columns]")
    _require_finite_tensor(name, tensor)


def _require_nonzero_rows(name: str, values: "torch.Tensor") -> None:
    # Max-absolute scaling is overflow-safe for any finite FP32 magnitude.  The
    # detached scale intentionally only validates the input; the cosine still
    # differentiates through the live values and both head and encoder receive
    # gradients from an OE loss.
    maximum_absolute = values.detach().abs().amax(dim=1)
    if bool((maximum_absolute <= 0.0).any().item()):
        raise ValueError(f"{name} must not contain a zero vector")


def _stable_l2_normalize_rows(values: "torch.Tensor", *, name: str) -> "torch.Tensor":
    import torch

    _require_nonzero_rows(name, values)
    maximum_absolute = values.detach().abs().amax(dim=1, keepdim=True)
    scaled = values / maximum_absolute
    norm = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
    # A finite, nonzero scaled vector has a finite positive norm; retain this
    # check so an unexpected backend failure cannot become a silent bad loss.
    _require_finite_tensor(f"{name} scaled norm", norm)
    if bool((norm <= 0.0).any().item()):
        raise ValueError(f"{name} normalization norm must be positive")
    normalized = scaled / norm
    _require_finite_tensor(f"normalized {name}", normalized)
    return normalized


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite real number")
    return resolved


def _positive_real(name: str, value: Real) -> float:
    resolved = _finite_real(name, value)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return resolved


def _require_finite_tensor(name: str, tensor: "torch.Tensor") -> None:
    import torch

    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise FloatingPointError(f"{name} must be finite")


def _require_finite_scalar(name: str, value: "torch.Tensor") -> None:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    _require_finite_tensor(name, value)
