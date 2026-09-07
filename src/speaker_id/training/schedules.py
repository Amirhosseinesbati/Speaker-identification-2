"""Pure, deterministic adaptation schedules indexed by committed optimizer steps."""
from __future__ import annotations

import math


def validate_adaptation_schedule(fit: dict) -> dict | None:
    if "adaptation_schedule" not in fit:
        return None
    schedule = fit["adaptation_schedule"]
    required = {"scheme", "head_only_steps", "margin_ramp_tail_steps",
                "encoder_lr_warmup_tail_steps", "head_lr_warmup_start_factor"}
    if not isinstance(schedule, dict) or set(schedule) != required:
        raise ValueError("Adaptation schedule requires exactly the supported explicit fields")
    if schedule["scheme"] != "head_warmup_tail_v1":
        raise ValueError("Unsupported adaptation schedule scheme")
    for name in ("epochs", "steps_per_epoch"):
        if type(fit[name]) is not int or fit[name] <= 0:
            raise ValueError("Tail epochs and steps per epoch must be positive integers")
    tail_steps = fit["epochs"] * fit["steps_per_epoch"]
    for name in ("head_only_steps", "margin_ramp_tail_steps", "encoder_lr_warmup_tail_steps"):
        if type(schedule[name]) is not int or schedule[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # Each ramp has at least two endpoints and leaves a nonempty cosine decay.
    if not 2 <= schedule["margin_ramp_tail_steps"] <= tail_steps:
        raise ValueError("Margin ramp must span between two and all tail steps")
    if not 1 <= schedule["encoder_lr_warmup_tail_steps"] < tail_steps:
        raise ValueError("Encoder warmup must leave at least one tail decay step")
    factor = schedule["head_lr_warmup_start_factor"]
    if (type(factor) not in (int, float) or not math.isfinite(factor)
            or not 0 < factor <= 1):
        raise ValueError("Head LR warmup start factor must be finite and in (0, 1]")
    return schedule


def adaptation_total_steps(fit: dict) -> int:
    schedule = validate_adaptation_schedule(fit)
    return fit["epochs"] * fit["steps_per_epoch"] + (schedule["head_only_steps"] if schedule else 0)


def adaptation_step(fit: dict, step: int) -> dict:
    """Return settings USED at a zero-based update; never advance mutable state.

    Head-only LR grows linearly from start_factor*peak to peak. Tail head LR
    follows cosine decay. Tail margin starts at zero and reaches its target on
    the last ramp step. Encoder LR reaches its peak on the last warmup step,
    then cosine-decays to zero on the final tail update.
    """
    schedule = validate_adaptation_schedule(fit)
    if schedule is None:
        raise ValueError("Legacy F001 uses its original PyTorch scheduler")
    head_steps = schedule["head_only_steps"]
    tail_steps = fit["epochs"] * fit["steps_per_epoch"]
    if type(step) is not int or not 0 <= step < head_steps + tail_steps:
        raise ValueError("Schedule step is outside the planned updates")
    if step < head_steps:
        progress = step / (head_steps - 1) if head_steps > 1 else 1.0
        factor = schedule["head_lr_warmup_start_factor"]
        return {"phase": "head_only", "tail_step": None, "margin": 0.0,
                "encoder_lr": 0.0,
                "head_lr": fit["head_lr"] * (factor + (1 - factor) * progress)}
    tail_step = step - head_steps
    warmup = schedule["encoder_lr_warmup_tail_steps"]
    if tail_step < warmup:
        encoder_factor = (tail_step + 1) / warmup
    else:
        progress = (tail_step - (warmup - 1)) / (tail_steps - warmup)
        encoder_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return {"phase": "tail", "tail_step": tail_step,
            "margin": fit["margin"] * min(1.0, tail_step / (schedule["margin_ramp_tail_steps"] - 1)),
            "encoder_lr": fit["encoder_lr"] * encoder_factor,
            "head_lr": fit["head_lr"] * 0.5 * (1 + math.cos(math.pi * tail_step / (tail_steps - 1)))}


def adaptation_checkpoint_state(fit: dict, completed_steps: int) -> dict:
    schedule = validate_adaptation_schedule(fit)
    if schedule is None:
        raise ValueError("Legacy checkpoints do not have an adaptation phase")
    total = adaptation_total_steps(fit)
    if type(completed_steps) is not int or not 0 <= completed_steps <= total:
        raise ValueError("Checkpoint committed-step count is outside the planned updates")
    head_completed = min(completed_steps, schedule["head_only_steps"])
    return {"scheme": schedule["scheme"], "completed_steps": completed_steps,
            "head_only_completed_steps": head_completed,
            "tail_completed_steps": completed_steps - head_completed,
            "next_phase": "complete" if completed_steps == total else adaptation_step(fit, completed_steps)["phase"]}
