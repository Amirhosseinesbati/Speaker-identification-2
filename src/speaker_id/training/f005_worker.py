"""CUDA worker for one F005 arm; checkpoints never enter MLflow.

The worker trains only the pinned advanced CAM++ 192D endpoint.  The frozen
public 512D endpoint is absent from this module and is used later by scoring.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from speaker_id.adaptation.paired_views import paired_waveform_views
from speaker_id.training.f005_contract import ADVANCED_DIMENSION, ADVANCED_WEIGHTS_SHA256
from speaker_id.training.f005_runner import (
    arm_identity, checkpoint_metadata, plan_range_sha256, shared_head_checkpoint_metadata,
    shared_head_identity, training_step_plan, validate_resume_payload, validate_shared_head_payload,
)


def _raw_h_mse_alignment(student, teacher, eligible, student_real_samples, teacher_real_samples):
    """MSE on final pre-L2 CAM++ 192D h with a dynamic detached long target.

    Detachment applies only to this consistency term.  The same long-view h is
    consumed without detachment by long-view AAM, so that branch still updates
    the encoder.
    """
    import torch

    if (not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor)
            or student is teacher or student.dtype != torch.float32 or teacher.dtype != torch.float32
            or student.shape != teacher.shape or student.ndim != 2
            or student.shape[1] != ADVANCED_DIMENSION or student.device != teacher.device):
        raise ValueError("F005 raw-h MSE requires distinct matching FP32 [batch,192] tensors")
    batch = student.shape[0]
    if (not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool
            or eligible.shape != (batch,) or eligible.device != student.device):
        raise ValueError("F005 raw-h MSE eligibility must be bool [batch] on the embedding device")
    for value in (student_real_samples, teacher_real_samples):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                or value.shape != (batch,) or value.device != student.device):
            raise ValueError("F005 raw-h MSE real lengths must be int64 [batch]")
    if (not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all())
            or bool((student_real_samples <= 0).any())
            or bool((teacher_real_samples < student_real_samples).any())):
        raise ValueError("F005 raw-h MSE received nonfinite embeddings or invalid real lengths")
    effective = eligible & (teacher_real_samples > student_real_samples)
    if not bool(effective.any()):
        return student[effective].sum()
    target = teacher.detach()[effective].clone()
    loss = (student[effective] - target).square().mean()
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("F005 raw-h MSE became nonfinite")
    return loss


def consistency_ramp_scale(fit: dict, step: int) -> float:
    """Return the preregistered coefficient scale for one zero-based update.

    The first tail update (global step 600) has no consistency contribution;
    tail step 99 (global step 699) reaches the full configured coefficients.
    """
    from speaker_id.training.schedules import adaptation_step

    ramp_steps = fit.get("consistency_ramp_tail_steps")
    tail_steps = fit.get("epochs", 0) * fit.get("steps_per_epoch", 0)
    if (type(ramp_steps) is not int or not 2 <= ramp_steps <= tail_steps):
        raise ValueError("F005 consistency ramp must span between two and all tail steps")
    scheduled = adaptation_step(fit, step)
    if scheduled["phase"] == "head_only":
        return 0.0
    return min(1.0, scheduled["tail_step"] / (ramp_steps - 1))


def f005_objective(short_h, long_h, targets, eligible, student_real_samples,
                   teacher_real_samples, head, arm: dict, *, consistency_active: bool,
                   consistency_scale: float = 1.0,
                   normalization_pairs: int | None = None,
                   normalization_eligible_pairs: int | None = None):
    """Dual AAM plus optional stop-gradient cosine and raw-h consistency."""
    import torch
    from torch.nn import functional as F
    from speaker_id.adaptation.consistency import normalized_cosine_alignment

    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64
            or targets.shape != (short_h.shape[0],) or targets.device != short_h.device
            or (targets.numel() and (bool((targets < 0).any()) or bool((targets >= 446).any())))):
        raise ValueError("F005 objective requires known targets 0..445")
    if arm.get("id") == "control":
        if arm != {"id": "control", "kind": "dual_aam_control", "cosine_gamma": 0.0, "raw_h_mse_lambda": 0.0}:
            raise ValueError("F005 control coefficients changed")
    elif (arm.get("kind") != "dual_aam_consistency" or arm.get("cosine_gamma") != 0.5
          or arm.get("raw_h_mse_lambda") not in (0.0, 0.1, 0.5)):
        raise ValueError("F005 treatment coefficients changed")
    if (type(consistency_scale) not in (int, float) or not np.isfinite(consistency_scale)
            or not 0.0 <= consistency_scale <= 1.0):
        raise ValueError("F005 consistency scale must be finite and in [0,1]")
    short_logits, long_logits = head(short_h, targets), head(long_h, targets)
    local_pairs = int(targets.numel())
    normalization_pairs = local_pairs if normalization_pairs is None else normalization_pairs
    local_effective = int((eligible & (teacher_real_samples > student_real_samples)).sum().item())
    normalization_eligible_pairs = (local_effective if normalization_eligible_pairs is None
                                    else normalization_eligible_pairs)
    if (type(normalization_pairs) is not int or normalization_pairs < local_pairs
            or type(normalization_eligible_pairs) is not int
            or normalization_eligible_pairs < local_effective):
        raise ValueError("F005 objective normalization counts do not cover this microbatch")
    short_ce_sum = F.cross_entropy(short_logits, targets, reduction="sum")
    long_ce_sum = F.cross_entropy(long_logits, targets, reduction="sum")
    classification = (short_ce_sum + long_ce_sum) / (2 * normalization_pairs)
    cosine, diagnostics = normalized_cosine_alignment(
        short_h, long_h, eligible, student_real_samples=student_real_samples,
        teacher_real_samples=teacher_real_samples, embedding_dim=ADVANCED_DIMENSION,
    )
    mse = _raw_h_mse_alignment(short_h, long_h, eligible, student_real_samples, teacher_real_samples)
    applied_scale = float(consistency_scale) if consistency_active else 0.0
    base_gamma = float(arm["cosine_gamma"])
    base_coefficient = float(arm["raw_h_mse_lambda"])
    gamma = base_gamma * applied_scale
    coefficient = base_coefficient * applied_scale
    denominator = max(1, normalization_eligible_pairs)
    cosine_contribution = gamma * cosine * (local_effective / denominator)
    mse_contribution = coefficient * mse * (local_effective / denominator)
    loss = classification + cosine_contribution + mse_contribution
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("F005 combined objective became nonfinite")
    return loss, {
        "loss": float(loss.detach().cpu()),
        "classification": float(classification.detach().cpu()),
        "short_aam_sum": float(short_ce_sum.detach().cpu()),
        "long_aam_sum": float(long_ce_sum.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "raw_h_mse": float(mse.detach().cpu()),
        "cosine_contribution": float(cosine_contribution.detach().cpu()),
        "raw_h_mse_contribution": float(mse_contribution.detach().cpu()),
        "cosine_gamma_used": gamma,
        "raw_h_mse_lambda_used": coefficient,
        "consistency_scale": applied_scale,
        "base_cosine_gamma": base_gamma,
        "base_raw_h_mse_lambda": base_coefficient,
        "effective_cosine_gamma": gamma,
        "effective_raw_h_mse_lambda": coefficient,
        "consistency_active": bool(consistency_active),
        "effective_consistency_pairs": diagnostics["effective_count"],
        "normalization_pairs": normalization_pairs,
        "normalization_views": 2 * normalization_pairs,
        "normalization_eligible_pairs": normalization_eligible_pairs,
        "short_correct": int((short_logits.argmax(1) == targets).sum().detach().cpu()),
        "long_correct": int((long_logits.argmax(1) == targets).sum().detach().cpu()),
        "rows": int(targets.numel()),
    }


def _padded_view(values: np.ndarray, requested: int) -> np.ndarray:
    if values.ndim != 1 or values.dtype != np.float32 or not 0 < len(values) <= requested:
        raise ValueError("F005 paired view is malformed")
    if len(values) < requested:
        values = np.pad(values, (0, requested - len(values)))
    return np.ascontiguousarray(values, dtype=np.float32)


class _WaveformCache:
    """Bounded per-worker LRU of immutable decoded/resampled waveforms."""
    def __init__(self, maximum_bytes: int):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("F005 waveform cache needs a positive byte limit")
        self.maximum_bytes, self.bytes = maximum_bytes, 0
        self.values = OrderedDict()

    def get(self, name):
        value = self.values.pop(name, None)
        if value is not None:
            self.values[name] = value
        return value

    def put(self, name, value):
        if value.nbytes > self.maximum_bytes:
            return
        while self.values and self.bytes + value.nbytes > self.maximum_bytes:
            _, removed = self.values.popitem(last=False)
            self.bytes -= removed.nbytes
        self.values[name] = value
        self.bytes += value.nbytes

    def __len__(self):
        return len(self.values)


def _microbatch(contract: dict, plan: list[dict], root: Path, waveform_cache: dict | None = None,
                io_stats: dict | None = None):
    """Decode one plan identically for every arm and return padded FBanks."""
    import torch
    from speaker_id.models.campp import make_fbank, read_mono

    views = contract["config"]["views"]
    short_samples = round(views["short_seconds"] * views["sample_rate"])
    long_samples = round(views["long_seconds"] * views["sample_rate"])
    data = root / contract["readiness"]["config"]["data_dir"]
    short_features, long_features, real_short, real_long, eligible, targets = [], [], [], [], [], []
    for item in plan:
        name = item["audio_file"]
        signal = None if waveform_cache is None else waveform_cache.get(name)
        if signal is None:
            started = time.monotonic()
            signal = read_mono(data / name, sample_rate=views["sample_rate"])
            signal.setflags(write=False)
            if waveform_cache is not None:
                waveform_cache.put(name, signal)
            if io_stats is not None:
                io_stats["decode_seconds"] += time.monotonic() - started
                io_stats["decode_misses"] += 1
        elif io_stats is not None:
            io_stats["cache_hits"] += 1
        short, long, metadata = paired_waveform_views(
            signal, rng=np.random.default_rng(item["crop_seed"]),
            short_seconds=views["short_seconds"], long_seconds=views["long_seconds"],
            sample_rate=views["sample_rate"],
        )
        short_features.append(make_fbank(_padded_view(short, short_samples)))
        long_features.append(make_fbank(_padded_view(long, long_samples)))
        real_short.append(metadata["student"]["real_samples"])
        real_long.append(metadata["teacher"]["real_samples"])
        eligible.append(metadata["consistency_mask"])
        targets.append(item["target"])
    return {
        "short": torch.stack(short_features), "long": torch.stack(long_features),
        "student_real_samples": torch.tensor(real_short, dtype=torch.int64),
        "teacher_real_samples": torch.tensor(real_long, dtype=torch.int64),
        "eligible": torch.tensor(eligible, dtype=torch.bool),
        "targets": torch.tensor(targets, dtype=torch.int64),
    }


def state_dict_sha256(state: dict) -> str:
    """Hash exact named tensor bytes without torch serialization metadata."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("F005 model state must be a plain named tensor mapping")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_checkpoint(payload: dict, path: Path) -> None:
    import torch
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("F005 partial checkpoint requires explicit forensic handling")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_advanced_trainable(contract: dict, root: Path, device: str):
    """Authoritative advanced192 loader; public512 cannot enter this worker."""
    from speaker_id.candidates.campp_advanced import load_advanced
    from speaker_id.training.fit import set_trainable_tail

    model = contract["advanced_model"]
    if model["embedding_dim"] != 192 or model["weights_sha256"] != ADVANCED_WEIGHTS_SHA256:
        raise ValueError("F005 worker refuses any model other than pinned advanced CAM++ 192D")
    encoder = load_advanced(model, root, device)
    details = set_trainable_tail(encoder, contract["config"]["fit"]["trainable_prefixes"])
    return encoder, details


def _require_worker_execution_evidence(contract: dict) -> dict:
    """Fail closed before Torch/model construction when server evidence is absent."""
    if contract.get("source_verification") is None:
        raise ValueError("F005 source weights and C002b receipts must be verified before training")
    try:
        audio_hashes_checked = contract["readiness"]["summary"]["audio_hashes_checked"]
        expected_instance = str(contract["readiness"]["config"]["expected_vast_instance_id"])
    except (KeyError, TypeError) as error:
        raise ValueError("F005 worker requires the authoritative readiness receipt") from error
    if audio_hashes_checked is not True:
        raise ValueError("F005 worker requires readiness audio_hashes_checked=True")
    if os.environ.get("VAST_INSTANCE_ID") != expected_instance:
        raise RuntimeError("F005 worker requires the authorized Vast instance marker")
    return {"audio_hashes_checked": True, "vast_instance_id": expected_instance}


def _components(contract: dict, root: Path, outer: int):
    _require_worker_execution_evidence(contract)
    config, fit = contract["config"], contract["config"]["fit"]
    execution = config["execution"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["cublas_workspace_config"]:
        raise RuntimeError("F005 deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG before launch")
    if any(os.environ.get(key) != value for key, value in execution["thread_environment"].items()):
        raise RuntimeError("F005 numerical thread environment differs from the launcher contract")
    import torch
    from speaker_id.training.fit import AAMHead

    if (not torch.cuda.is_available() or config["device"] != "cuda"
            or execution["gpu_name_contains"] not in torch.cuda.get_device_name(0)):
        raise RuntimeError("F005 fitting requires the authorized RTX 3090 CUDA worker")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    encoder, trainable = _load_advanced_trainable(contract, root, "cuda")
    pairing = shared_head_identity(contract, outer, plan_sha256="0" * 64)["pairing_signature"]
    initialization_seed = int(pairing[:16], 16) % (2 ** 63 - 1)
    torch.manual_seed(initialization_seed); torch.cuda.manual_seed_all(initialization_seed)
    head = AAMHead(embedding_dim=ADVANCED_DIMENSION, classes=446,
                   margin=fit["margin"], scale=fit["scale"]).to("cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": fit["encoder_lr"]},
        {"params": head.parameters(), "lr": fit["head_lr"]},
    ], weight_decay=fit["weight_decay"])
    return encoder, head, optimizer, trainable, initialization_seed


def _trim_history(path: Path, completed_steps: int) -> None:
    if path.exists():
        rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                if int(json.loads(line)["step"]) <= completed_steps]
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _run_updates(contract, root, outer, arm, encoder, head, optimizer, start_step, stop_step,
                 history_path, tracker, save_checkpoint):
    import torch
    from speaker_id.training.fit import freeze_batchnorm
    from speaker_id.training.schedules import adaptation_step

    fit = contract["config"]["fit"]
    micro = fit["microbatch_pairs"]
    waveform_cache = _WaveformCache(fit["waveform_cache_max_bytes"])
    io_stats = {"decode_seconds": 0.0, "decode_misses": 0, "cache_hits": 0}
    started = time.monotonic()
    for step in range(start_step, stop_step):
        scheduled, plan = adaptation_step(fit, step), training_step_plan(contract, outer, step)
        phase = scheduled["phase"]
        consistency_scale = consistency_ramp_scale(fit, step)
        optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] = scheduled["encoder_lr"], scheduled["head_lr"]
        head.margin = scheduled["margin"]
        if phase == "head_only": encoder.eval()
        else: encoder.train(); freeze_batchnorm(encoder)
        head.train(); optimizer.zero_grad(set_to_none=True)
        cpu_batches = [_microbatch(contract, plan[start:start + micro], root, waveform_cache, io_stats)
                       for start in range(0, len(plan), micro)]
        effective = sum(int((batch["eligible"] & (batch["teacher_real_samples"] > batch["student_real_samples"])).sum())
                        for batch in cpu_batches)
        aggregate = defaultdict(float)
        for cpu_batch in cpu_batches:
            batch = {key: value.to("cuda") for key, value in cpu_batch.items()}
            if phase == "head_only":
                with torch.no_grad(): short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
            else:
                short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
            if (short_h.shape != (len(batch["targets"]), ADVANCED_DIMENSION)
                    or long_h.shape != short_h.shape or short_h.dtype != torch.float32 or long_h.dtype != torch.float32):
                raise ValueError("F005 advanced encoder output is not FP32 [batch,192]")
            loss, diagnostics = f005_objective(
                short_h, long_h, batch["targets"], batch["eligible"], batch["student_real_samples"],
                batch["teacher_real_samples"], head, arm, consistency_active=phase == "tail",
                consistency_scale=consistency_scale,
                normalization_pairs=fit["batch_pairs"], normalization_eligible_pairs=effective,
            )
            loss.backward()
            for key in ("loss", "classification", "short_aam_sum", "long_aam_sum",
                        "cosine_contribution", "raw_h_mse_contribution", "short_correct",
                        "long_correct", "effective_consistency_pairs"):
                aggregate[key] += diagnostics[key]
            aggregate["cosine_weighted"] += diagnostics["cosine"] * diagnostics["effective_consistency_pairs"]
            aggregate["mse_weighted"] += diagnostics["raw_h_mse"] * diagnostics["effective_consistency_pairs"]
        parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, fit["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"F005 nonfinite gradient at step {step}; optimizer not advanced")
        optimizer.step()
        count = float(fit["batch_pairs"])
        metrics = {
            "fit/loss": aggregate["loss"], "fit/classification": aggregate["classification"],
            "fit/short_aam": aggregate["short_aam_sum"] / count,
            "fit/long_aam": aggregate["long_aam_sum"] / count,
            "fit/cosine": aggregate["cosine_weighted"] / max(1, effective),
            "fit/raw_h_mse": aggregate["mse_weighted"] / max(1, effective),
            "fit/cosine_contribution": aggregate["cosine_contribution"],
            "fit/raw_h_mse_contribution": aggregate["raw_h_mse_contribution"],
            "fit/short_accuracy": aggregate["short_correct"] / count,
            "fit/long_accuracy": aggregate["long_correct"] / count,
            "fit/effective_consistency_fraction": effective / count,
            "fit/gradient_norm": float(gradient_norm.detach().cpu()),
            "fit/encoder_lr": float(scheduled["encoder_lr"]), "fit/head_lr": float(scheduled["head_lr"]),
            "fit/margin": float(scheduled["margin"]), "fit/phase_head_only": float(phase == "head_only"),
            "fit/consistency_scale": consistency_scale,
            "fit/effective_cosine_gamma": float(arm["cosine_gamma"]) * consistency_scale,
            "fit/effective_raw_h_mse_lambda": float(arm["raw_h_mse_lambda"]) * consistency_scale,
            "fit/gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "fit/io_decode_seconds": io_stats["decode_seconds"],
            "fit/io_cache_hits": io_stats["cache_hits"], "fit/io_decode_misses": io_stats["decode_misses"],
            "fit/elapsed_seconds": time.monotonic() - started,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"step": step + 1, "phase": phase, **metrics}, allow_nan=False) + "\n")
        if tracker is not None:
            tracker.log_metrics(metrics, step=step + 1, sync=(step + 1) % 10 == 0, strict=False)
        if (step + 1) % fit["checkpoint_every_steps"] == 0 or step + 1 == stop_step:
            save_checkpoint(step + 1)
    return {"elapsed_seconds": time.monotonic() - started, "waveforms_cached": len(waveform_cache),
            "waveform_cache_bytes": waveform_cache.bytes,
            "waveform_cache_max_bytes": waveform_cache.maximum_bytes, **io_stats}


def fit_shared_head(contract: dict, root: Path, outer: int, output: Path, tracker=None,
                    *, resume: bool = False) -> dict:
    """Train the 600 arm-independent head steps once for one outer fold."""
    import torch
    from speaker_id.training.schedules import adaptation_checkpoint_state

    fit, output = contract["config"]["fit"], Path(output)
    output.mkdir(parents=True, exist_ok=True)
    head_steps = fit["adaptation_schedule"]["head_only_steps"]
    plan_hash = plan_range_sha256(contract, outer, 0, head_steps)
    identity = shared_head_identity(contract, outer, plan_sha256=plan_hash)
    checkpoint, history = output / "shared_head.pt", output / "fit_history.jsonl"
    encoder, head, optimizer, trainable, seed = _components(contract, root, outer)
    initial_encoder, initial_head = state_dict_sha256(encoder.state_dict()), state_dict_sha256(head.state_dict())
    start = 0
    if checkpoint.exists():
        if not resume: raise FileExistsError("Existing F005 shared-head checkpoint requires explicit resume")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = validate_shared_head_payload(payload, identity, fit); start = metadata["completed_steps"]
        encoder.load_state_dict(payload["encoder"], strict=True); head.load_state_dict(payload["head"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); torch.set_rng_state(payload["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([item.cpu() for item in payload["cuda_rng"]]); _trim_history(history, start)
    elif resume: raise FileNotFoundError("F005 shared-head resume requested without checkpoint")
    control = contract["config"]["arms"][0]
    def save(completed):
        _atomic_checkpoint({"metadata": shared_head_checkpoint_metadata(identity, fit, completed),
            "encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint)
    runtime = _run_updates(contract, root, outer, control, encoder, head, optimizer, start, head_steps,
                           history, tracker, save)
    report = {"status": "complete", "stage": "shared_head", "outer_fold": outer,
        "unit_signature": identity["signature"], "pairing_signature": identity["pairing_signature"],
        "head_plan_sha256": plan_hash, "initialization_seed": seed,
        "initial_encoder_sha256": initial_encoder, "initial_head_sha256": initial_head,
        "completed_steps": head_steps, "schedule_state": adaptation_checkpoint_state(fit, head_steps),
        "fork_source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fork_source_byte_identical_for_all_arms": True, "checkpoint_mlflow_uploaded": False,
        "checkpoint_local_transfer": False, **runtime, **trainable}
    if tracker is not None: tracker.add_artifact(history, "training/fit_history.jsonl")
    return {"report": report, "checkpoint": checkpoint}


def fit_tail_arm(contract: dict, root: Path, outer: int, arm_id: str, shared_head_checkpoint: Path,
                 output: Path, tracker=None, *, resume: bool = False) -> dict:
    """Fork the exact completed shared head and train one 500-step tail arm."""
    import torch
    from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_total_steps

    fit, output = contract["config"]["fit"], Path(output)
    output.mkdir(parents=True, exist_ok=True)
    head_steps, total = fit["adaptation_schedule"]["head_only_steps"], adaptation_total_steps(fit)
    shared_head_checkpoint = Path(shared_head_checkpoint)
    shared_sha = hashlib.sha256(shared_head_checkpoint.read_bytes()).hexdigest()
    head_identity = shared_head_identity(contract, outer,
        plan_sha256=plan_range_sha256(contract, outer, 0, head_steps))
    shared_payload = torch.load(shared_head_checkpoint, map_location="cpu", weights_only=True)
    shared_metadata = validate_shared_head_payload(shared_payload, head_identity, fit)
    if shared_metadata["completed_steps"] != head_steps or shared_metadata["byte_identical_fork_source"] is not True:
        raise ValueError("F005 tail requires the completed shared-head fork source")
    tail_plan = plan_range_sha256(contract, outer, head_steps, total)
    identity = arm_identity(contract, outer, arm_id, shared_head_checkpoint_sha256=shared_sha,
                            tail_plan_sha256=tail_plan)
    checkpoint, history = output / "last.pt", output / "fit_history.jsonl"
    encoder, head, optimizer, trainable, seed = _components(contract, root, outer)
    start = head_steps
    if checkpoint.exists():
        if not resume: raise FileExistsError("Existing F005 tail checkpoint requires explicit resume")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = validate_resume_payload(payload, identity, fit); start = metadata["completed_steps"]
    elif resume: raise FileNotFoundError("F005 tail resume requested without checkpoint")
    else: payload = shared_payload
    encoder.load_state_dict(payload["encoder"], strict=True); head.load_state_dict(payload["head"], strict=True)
    optimizer.load_state_dict(payload["optimizer"]); torch.set_rng_state(payload["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in payload["cuda_rng"]]); _trim_history(history, start)
    fork_encoder, fork_head = state_dict_sha256(encoder.state_dict()), state_dict_sha256(head.state_dict())
    def save(completed):
        _atomic_checkpoint({"metadata": checkpoint_metadata(identity, fit, completed),
            "encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint)
    runtime = _run_updates(contract, root, outer, identity["arm"], encoder, head, optimizer, start, total,
                           history, tracker, save)
    encoder.eval()
    report = {"status": "complete", "stage": "tail", "outer_fold": outer, "arm": identity["arm"],
        "arm_signature": identity["signature"], "pairing_signature": identity["pairing_signature"],
        "tail_plan_sha256": tail_plan, "shared_head_checkpoint_sha256": shared_sha,
        "fork_encoder_state_sha256": fork_encoder, "fork_head_state_sha256": fork_head,
        "initialization_seed": seed, "completed_steps": total, "initial_step": start,
        "schedule_state": adaptation_checkpoint_state(fit, total), "trainable_endpoint": "advanced_campp_192d",
        "frozen_public_endpoint_touched": False, "checkpoint_mlflow_uploaded": False,
        "checkpoint_local_transfer": False, **runtime, **trainable}
    if tracker is not None: tracker.add_artifact(history, "training/fit_history.jsonl")
    return {"report": report, "checkpoint": checkpoint}


def probe_cuda(contract: dict, root: Path, outer: int = 0) -> dict:
    """One worst-case FP32 backward, zero optimizer steps and no checkpoint."""
    import torch
    from speaker_id.training.fit import freeze_batchnorm

    if contract.get("source_verification") is None:
        raise ValueError("F005 probe requires verified source bytes")
    identity = arm_identity(contract, outer, "treatment_mse05",
                            shared_head_checkpoint_sha256="0" * 64, tail_plan_sha256="0" * 64)
    encoder, head, _, trainable, seed = _components(contract, root, outer)
    before_encoder, before_head = state_dict_sha256(encoder.state_dict()), state_dict_sha256(head.state_dict())
    encoder.train(); freeze_batchnorm(encoder); head.train()
    probe_step = 699
    scale = consistency_ramp_scale(contract["config"]["fit"], probe_step)
    plan = training_step_plan(contract, outer, probe_step)[:contract["config"]["fit"]["microbatch_pairs"]]
    batch = {key: value.to("cuda") for key, value in _microbatch(contract, plan, root).items()}
    torch.cuda.reset_peak_memory_stats()
    short_h, long_h = encoder(batch["short"]), encoder(batch["long"])
    loss, diagnostics = f005_objective(
        short_h, long_h, batch["targets"], batch["eligible"], batch["student_real_samples"],
        batch["teacher_real_samples"], head, identity["arm"], consistency_active=True,
        consistency_scale=scale,
    )
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(
        [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters()), 5.0
    )
    if not bool(torch.isfinite(gradient)):
        raise FloatingPointError("F005 probe produced a nonfinite gradient")
    after_encoder, after_head = state_dict_sha256(encoder.state_dict()), state_dict_sha256(head.state_dict())
    if before_encoder != after_encoder or before_head != after_head:
        raise RuntimeError("F005 no-step probe unexpectedly changed model parameters")
    return {
        "status": "completed_backward_probe_no_optimizer_step",
        "outer_fold": outer, "arm_id": "treatment_mse05", "probe_step": probe_step,
        "embedding_dim": short_h.shape[1],
        "advanced_weights_sha256": ADVANCED_WEIGHTS_SHA256,
        "microbatch_pairs": len(plan), "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient.detach().cpu()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "model_state_unchanged": True, "optimizer_steps": 0,
        "public512_loaded": False, "model_or_optimizer_uploaded_to_mlflow": False,
        "objective": diagnostics, **trainable,
    }
