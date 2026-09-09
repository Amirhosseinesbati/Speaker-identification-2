"""Small, explicit L2-SP primitives for encoder-only adaptation.

The anchor intentionally contains only trainable encoder parameters selected at
construction time.  It never serializes source weights in its receipt, and it
does not know about an optimizer, data loader, classifier, or training loop.
"""
from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class L2SPAnchor:
    """Immutable metadata plus frozen reference tensors for selected parameters.

    ``references`` is a read-only mapping, while its tensors are deliberately
    kept on the encoder's original device so normal training does not repeatedly
    copy a whole anchor.  Build the anchor after moving the encoder to its
    training device.  The ``receipt`` property is JSON-safe and contains no
    parameter values.
    """

    references: Mapping[str, "torch.Tensor"]
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    excluded_names: frozenset[str]
    anchor_byte_count: int
    anchor_bytes_sha256: str

    @property
    def receipt(self) -> dict[str, object]:
        """Return a fresh deterministic, non-secret anchor identity receipt."""
        return {
            "schema": "l2sp-anchor-v1",
            "parameter_count": len(self.names),
            "parameter_names": list(self.names),
            "parameter_shapes": {
                name: list(shape) for name, shape in zip(self.names, self.shapes, strict=True)
            },
            "parameter_dtypes": {
                name: dtype for name, dtype in zip(self.names, self.dtypes, strict=True)
            },
            "anchor_byte_count": self.anchor_byte_count,
            "anchor_bytes_sha256": self.anchor_bytes_sha256,
        }


def build_l2sp_anchor(
    encoder: object,
    *,
    exclude_names: AbstractSet[str] | None = None,
) -> L2SPAnchor:
    """Freeze an exact anchor of trainable encoder parameters.

    ``exclude_names`` is an exact set of encoder parameter names.  Unknown names
    are rejected so a spelling error cannot silently leave a parameter anchored.
    Anchor bytes are concatenated in lexicographic name order for the SHA-256
    receipt; the receipt itself exposes only names, shapes, dtypes, counts, and
    that digest.
    """
    import torch

    excluded = _canonical_exclusions(exclude_names)
    named = _named_parameter_map(encoder)
    unknown_exclusions = sorted(excluded.difference(named))
    if unknown_exclusions:
        raise ValueError(f"L2-SP excluded parameter names are absent: {unknown_exclusions}")

    selected = [
        (name, parameter)
        for name, parameter in named.items()
        if parameter.requires_grad and name not in excluded
    ]
    if not selected:
        raise ValueError("L2-SP anchor needs at least one non-excluded trainable encoder parameter")

    references: dict[str, torch.Tensor] = {}
    digest = hashlib.sha256()
    byte_count = 0
    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for name, parameter in selected:
        _validate_floating_strided_parameter(name, parameter, purpose="anchor")
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise FloatingPointError(f"L2-SP cannot anchor nonfinite parameter: {name}")
        # Do not create an inference tensor if the caller happens to be inside
        # inference_mode; the reference must be usable in a later autograd loss.
        with torch.inference_mode(False):
            reference = parameter.detach().clone(memory_format=torch.preserve_format)
        reference.requires_grad_(False)
        payload = _raw_tensor_bytes(reference)
        digest.update(payload)
        byte_count += len(payload)
        names.append(name)
        shapes.append(tuple(int(value) for value in reference.shape))
        dtypes.append(_dtype_name(reference.dtype))
        references[name] = reference

    return L2SPAnchor(
        references=MappingProxyType(references),
        names=tuple(names),
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
        excluded_names=excluded,
        anchor_byte_count=byte_count,
        anchor_bytes_sha256=digest.hexdigest(),
    )


def l2sp_penalty(encoder: object, anchor: L2SPAnchor) -> "torch.Tensor":
    """Return exactly ``0.5 * sum((parameter - anchor)**2)`` over the anchor.

    The sum is built once, with one squared-distance reduction per anchored
    parameter.  A changed parameter inventory, missing name, shape/dtype change,
    or nonfinite distance is an error rather than a silent partial penalty.
    """
    import torch

    live = _anchored_live_parameters(encoder, anchor)
    squared_distance_sum: torch.Tensor | None = None
    for index, (name, parameter) in enumerate(live):
        reference = _validated_reference(anchor, index, name, parameter)
        if reference.device != parameter.device:
            reference = reference.to(device=parameter.device)
        distance = (parameter - reference).square().sum()
        squared_distance_sum = distance if squared_distance_sum is None else squared_distance_sum + distance

    if squared_distance_sum is None:  # Defensive: construction rejects empty anchors.
        raise ValueError("L2-SP anchor has no trainable parameters")
    penalty = squared_distance_sum * 0.5
    if not bool(torch.isfinite(penalty.detach()).item()):
        raise FloatingPointError("L2-SP distance is nonfinite")
    return penalty


def l2sp_gradient_norm_ratio(
    task_loss: "torch.Tensor",
    sp_loss: "torch.Tensor",
    encoder: object,
    anchor: L2SPAnchor,
    *,
    lambda_sp: float,
) -> dict[str, float | int | str | None]:
    """Measure task-versus-``lambda_sp * L2-SP`` gradients without updating state.

    This uses :func:`torch.autograd.grad`, never ``backward()``, ``zero_grad()``,
    or an optimizer.  It leaves ``parameter.grad`` untouched and retains both
    graphs so the caller can subsequently backpropagate its combined objective.
    ``sp_loss`` must be the already-computed :func:`l2sp_penalty`; this helper
    never recomputes the squared-distance objective.
    """
    import torch

    if isinstance(lambda_sp, bool):
        raise TypeError("lambda_sp must be a finite nonnegative real number")
    try:
        coefficient = float(lambda_sp)
    except (TypeError, ValueError) as error:
        raise TypeError("lambda_sp must be a finite nonnegative real number") from error
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("lambda_sp must be a finite nonnegative real number")

    _validate_scalar_loss("task_loss", task_loss)
    _validate_scalar_loss("sp_loss", sp_loss)
    live = _anchored_live_parameters(encoder, anchor)
    parameters = tuple(parameter for _, parameter in live)
    try:
        task_gradients = torch.autograd.grad(
            task_loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
        sp_gradients = torch.autograd.grad(
            sp_loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
    except RuntimeError as error:
        raise ValueError("L2-SP gradient ratio needs live, unconsumed autograd graphs") from error

    task_norm, task_active = _gradient_norm(task_gradients, label="task")
    sp_norm, sp_active = _gradient_norm(sp_gradients, label="L2-SP")
    weighted_sp_norm = coefficient * sp_norm
    if not math.isfinite(weighted_sp_norm):
        raise FloatingPointError("Weighted L2-SP gradient norm is nonfinite")
    if weighted_sp_norm == 0.0:
        ratio: float | None = None
        status = "weighted_l2sp_gradient_zero"
    else:
        ratio = task_norm / weighted_sp_norm
        if not math.isfinite(ratio):
            raise FloatingPointError("Task-to-weighted-L2-SP gradient ratio is nonfinite")
        status = "finite"
    return {
        "task_gradient_norm": task_norm,
        "l2sp_gradient_norm": sp_norm,
        "weighted_l2sp_gradient_norm": weighted_sp_norm,
        "task_to_weighted_l2sp_ratio": ratio,
        "ratio_status": status,
        "anchored_parameter_count": len(parameters),
        "task_active_parameter_count": task_active,
        "l2sp_active_parameter_count": sp_active,
        "lambda_sp": coefficient,
    }


def _canonical_exclusions(exclude_names: AbstractSet[str] | None) -> frozenset[str]:
    if exclude_names is None:
        return frozenset()
    if isinstance(exclude_names, (str, bytes)) or not isinstance(exclude_names, AbstractSet):
        raise TypeError("exclude_names must be a set of nonempty parameter names")
    values = frozenset(exclude_names)
    if any(not isinstance(name, str) or not name for name in values):
        raise TypeError("exclude_names must be a set of nonempty parameter names")
    return values


def _named_parameter_map(encoder: object) -> dict[str, "torch.Tensor"]:
    import torch

    method = getattr(encoder, "named_parameters", None)
    if not callable(method):
        raise TypeError("encoder must provide named_parameters()")
    try:
        entries = list(method())
    except Exception as error:
        raise ValueError("encoder.named_parameters() could not be read") from error
    result: dict[str, torch.Tensor] = {}
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("encoder.named_parameters() yielded an invalid entry")
        name, parameter = entry
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("encoder.named_parameters() must yield unique nonempty names")
        if not isinstance(parameter, torch.Tensor):
            raise ValueError(f"Encoder parameter {name!r} is not a tensor")
        result[name] = parameter
    return dict(sorted(result.items()))


def _validate_floating_strided_parameter(name: str, parameter: "torch.Tensor", *, purpose: str) -> None:
    import torch

    if parameter.layout != torch.strided or not torch.is_floating_point(parameter):
        raise ValueError(f"L2-SP {purpose} parameter {name!r} must be a dense real floating tensor")
    if parameter.is_meta:
        raise ValueError(f"L2-SP {purpose} parameter {name!r} cannot be a meta tensor")


def _raw_tensor_bytes(tensor: "torch.Tensor") -> bytes:
    """Return canonical raw value bytes without converting the tensor's dtype."""
    import torch

    if tensor.layout != torch.strided or tensor.is_meta:
        raise ValueError("L2-SP anchor bytes require a materialized dense tensor")
    values = tensor.detach().contiguous().view(torch.uint8).cpu()
    return values.numpy().tobytes()


def _dtype_name(dtype: object) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


def _anchored_live_parameters(encoder: object, anchor: L2SPAnchor) -> tuple[tuple[str, "torch.Tensor"], ...]:
    if not isinstance(anchor, L2SPAnchor):
        raise TypeError("anchor must be an L2SPAnchor")
    if (len(anchor.names) == 0 or len(anchor.names) != len(anchor.shapes)
            or len(anchor.names) != len(anchor.dtypes)
            or tuple(sorted(anchor.names)) != anchor.names
            or set(anchor.references) != set(anchor.names)):
        raise ValueError("L2-SP anchor metadata is malformed")

    named = _named_parameter_map(encoder)
    missing_exclusions = sorted(anchor.excluded_names.difference(named))
    if missing_exclusions:
        raise ValueError(f"L2-SP excluded parameter names are now absent: {missing_exclusions}")
    current = {
        name for name, parameter in named.items()
        if parameter.requires_grad and name not in anchor.excluded_names
    }
    expected = set(anchor.names)
    missing = sorted(expected.difference(current))
    unexpected = sorted(current.difference(expected))
    if missing:
        absent = [name for name in missing if name not in named]
        if absent:
            raise ValueError(f"L2-SP missing anchored parameter names: {absent}")
        raise ValueError(f"L2-SP anchored parameters are no longer trainable: {missing}")
    if unexpected:
        raise ValueError(f"L2-SP has unexpected unanchored trainable parameters: {unexpected}")
    return tuple((name, named[name]) for name in anchor.names)


def _validated_reference(
    anchor: L2SPAnchor,
    index: int,
    name: str,
    parameter: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    _validate_floating_strided_parameter(name, parameter, purpose="live")
    reference = anchor.references[name]
    if not isinstance(reference, torch.Tensor):
        raise ValueError(f"L2-SP anchor reference for {name!r} is not a tensor")
    shape = tuple(int(value) for value in parameter.shape)
    reference_shape = tuple(int(value) for value in reference.shape)
    if shape != anchor.shapes[index] or reference_shape != anchor.shapes[index]:
        raise ValueError(f"L2-SP parameter shape changed for {name!r}")
    if parameter.dtype != reference.dtype or _dtype_name(reference.dtype) != anchor.dtypes[index]:
        raise ValueError(f"L2-SP parameter dtype changed for {name!r}")
    if reference.layout != torch.strided or reference.requires_grad:
        raise ValueError(f"L2-SP anchor reference for {name!r} is malformed")
    if reference.is_meta:
        raise ValueError(f"L2-SP anchor reference for {name!r} cannot be a meta tensor")
    return reference


def _validate_scalar_loss(name: str, loss: "torch.Tensor") -> None:
    import torch

    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ValueError(f"{name} must be a differentiable scalar tensor")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError(f"{name} is nonfinite")


def _gradient_norm(
    gradients: tuple["torch.Tensor | None", ...],
    *,
    label: str,
) -> tuple[float, int]:
    import torch

    squared = 0.0
    active = 0
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise FloatingPointError(f"{label} gradient is nonfinite")
        value = gradient.detach().to(dtype=torch.float64).square().sum()
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError(f"{label} gradient norm is nonfinite")
        squared += float(value.item())
        active += 1
    return math.sqrt(squared), active
