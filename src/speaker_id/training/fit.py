"""Known-only AAM fine-tuning; called exclusively by the explicit execution gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from speaker_id.models.campp import crop_waveform, make_fbank, read_mono
from speaker_id.training.schedules import (adaptation_checkpoint_state, adaptation_step,
                                          adaptation_total_steps, validate_adaptation_schedule)


class AAMHead(nn.Module):
    def __init__(self, embedding_dim: int = 512, classes: int = 446,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        if classes != 446 or not 0 <= margin < math.pi / 2 or scale <= 0:
            raise ValueError("AAM head requires 446 known identities and a valid margin/scale")
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings, targets):
        if targets.numel() and (targets.min() < 0 or targets.max() >= 446):
            raise ValueError("AAM targets must be known identities indexed 0..445")
        cosine = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1 - cosine.square(), min=1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi - self.margin), phi,
                          cosine - math.sin(math.pi - self.margin) * self.margin)
        one_hot = F.one_hot(targets, num_classes=446).to(cosine.dtype)
        return self.scale * (one_hot * phi + (1 - one_hot) * cosine)


def set_trainable_tail(encoder, prefixes: list[str]) -> dict:
    if not prefixes:
        raise ValueError("Fine-tuning needs an explicit nonempty set of parameter prefixes")
    matched = {prefix: 0 for prefix in prefixes}
    for name, parameter in encoder.named_parameters():
        selected = [prefix for prefix in prefixes if name == prefix or name.startswith(prefix + ".")]
        parameter.requires_grad_(bool(selected))
        for prefix in selected:
            matched[prefix] += parameter.numel()
    if not all(matched.values()):
        raise ValueError(f"Unmatched trainable parameter prefix: {matched}")
    return {"prefix_parameter_counts": matched,
            "trainable_encoder_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
            "total_encoder_parameters": sum(p.numel() for p in encoder.parameters())}


def freeze_batchnorm(encoder):
    for module in encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def atomic_checkpoint(payload: dict, path: Path):
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_encoder(encoder, roles: list[dict], labels: list[str], root: Path,
                config: dict, signature: str, output: Path, tracker,
                *, resume: bool = False) -> dict:
    from speaker_id.data.splits import truth
    fit = config["fit"]
    seed = int(config["seed"])
    outer = int(roles[0]["outer_fold"])
    torch.manual_seed(seed + outer)
    torch.cuda.manual_seed_all(seed + outer)
    torch.backends.cudnn.benchmark = False
    # Deterministic warnings remain visible if CUDA lacks a deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    details = set_trainable_tail(encoder, fit["trainable_prefixes"])
    head = AAMHead(margin=fit["margin"], scale=fit["scale"]).to(config["device"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": fit["encoder_lr"]},
        {"params": head.parameters(), "lr": fit["head_lr"]}], weight_decay=fit["weight_decay"])
    adaptation = validate_adaptation_schedule(fit)
    total_steps = adaptation_total_steps(fit)
    # Preserve the original F001 scheduler and update order when not opted in.
    scheduler = (None if adaptation else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0))
    scaler = torch.amp.GradScaler("cuda", enabled=fit["mixed_precision"])
    checkpoint = output / "last.pt"
    start_step = 0
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=config["device"], weights_only=True)
        if state["signature"] != signature or state["outer_fold"] != outer:
            raise ValueError("Resume checkpoint belongs to a different recipe or fold")
        start_step = state["completed_steps"]
        if type(start_step) is not int or not 0 <= start_step <= total_steps:
            raise ValueError("Checkpoint committed-step count is outside this recipe")
        if adaptation and (state.get("adaptation_schedule") != adaptation
                           or state.get("schedule_state") != adaptation_checkpoint_state(fit, start_step)):
            raise ValueError("Checkpoint adaptation phase or schedule differs from its committed step")
        encoder.load_state_dict(state["encoder"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
    elif checkpoint.exists():
        raise FileExistsError("Existing checkpoint requires explicit --resume")
    label_to_index = {label: index for index, label in enumerate(labels[1:])}
    by_label = {label: [] for label in labels[1:]}
    for row in roles:
        if row["speaker_id"] != "unknown" and truth(row["encoder_fit_allowed"]):
            by_label[row["speaker_id"]].append(row)
    if not all(by_label.values()):
        raise ValueError("Some known identities have no permitted fit samples")
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "fit_history.jsonl"
    if resume and history_path.exists():
        # Keep failed-attempt evidence, while final history contains one committed
        # trajectory through the checkpoint boundary rather than duplicate steps.
        original_history = history_path.read_text(encoding="utf-8")
        archive = output / ("fit_history_before_resume_" + str(time.time_ns()) + ".jsonl")
        archive.write_text(original_history, encoding="utf-8")
        lines = [line for line in original_history.splitlines() if int(json.loads(line)["step"]) <= start_step]
        history_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tracker.add_artifact(archive, "training/resume_evidence/" + archive.name)
    started = time.monotonic()
    encoder.train()
    freeze_batchnorm(encoder)
    head.train()
    def save(completed_steps):
        payload = {"format_version": 2 if adaptation else 1, "signature": signature, "outer_fold": outer,
                   "completed_steps": completed_steps, "encoder": encoder.state_dict(),
                   "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict() if scheduler is not None else None,
                   "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all()}
        if adaptation:
            payload.update(adaptation_schedule=dict(adaptation),
                           schedule_state=adaptation_checkpoint_state(fit, completed_steps))
        atomic_checkpoint(payload, checkpoint)
    current_phase = None
    for step in range(start_step, total_steps):
        phase = "tail"
        if adaptation:
            scheduled = adaptation_step(fit, step)
            phase = scheduled["phase"]
            optimizer.param_groups[0]["lr"] = scheduled["encoder_lr"]
            optimizer.param_groups[1]["lr"] = scheduled["head_lr"]
            head.margin = scheduled["margin"]
            if phase != current_phase:
                if phase == "head_only":
                    encoder.eval()
                else:
                    encoder.train()
                    freeze_batchnorm(encoder)
                current_phase = phase
        # Reproducible UUID-balanced sampling, independent of global NumPy state.
        rng = np.random.default_rng(seed + outer * 10_000_000 + step)
        selected_labels = rng.choice(labels[1:], fit["batch_size"], replace=fit["batch_size"] > 446)
        batch, targets = [], []
        for label in selected_labels:
            row = by_label[str(label)][int(rng.integers(len(by_label[str(label)])))]
            signal = read_mono(root / config["data_dir"] / row["audio_file"])
            crop = crop_waveform(signal, fit["crop_seconds"], position=float(rng.random()),
                                 minimum_seconds=fit["crop_seconds"])
            batch.append(make_fbank(crop))
            targets.append(label_to_index[str(label)])
        features = torch.stack(batch).to(config["device"])
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=config["device"])
        optimizer.zero_grad(set_to_none=True)
        if phase == "head_only":
            # Eval alone does not stop gradients; keep both requirements explicit.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fit["mixed_precision"]):
                embeddings = encoder(features)
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fit["mixed_precision"]):
                embeddings = encoder(features)
        logits = head(embeddings, targets_tensor)
        loss = F.cross_entropy(logits, targets_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite AAM loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), fit["gradient_clip_norm"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}; optimizer was not advanced")
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        metrics = {"fit/loss_aam": float(loss.detach().cpu()),
                   "fit/head_accuracy": float((logits.argmax(1) == targets_tensor).float().mean().detach().cpu()),
                   "fit/gradient_norm": float(gradient_norm.detach().cpu()),
                   "fit/encoder_lr": float(optimizer.param_groups[0]["lr"]),
                   "fit/head_lr": float(optimizer.param_groups[1]["lr"]),
                   "fit/gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                   "fit/elapsed_seconds": time.monotonic() - started}
        if adaptation:
            committed = adaptation_checkpoint_state(fit, step + 1)
            metrics.update({"fit/margin": float(head.margin), "fit/phase_head_only": float(phase == "head_only"),
                            "fit/committed_steps": step + 1,
                            "fit/head_only_committed_steps": committed["head_only_completed_steps"],
                            "fit/tail_committed_steps": committed["tail_completed_steps"]})
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step + 1, **({"phase": phase} if adaptation else {}), **metrics}) + "\n")
        tracker.log_metrics(metrics, step=step + 1, sync=(step + 1) % 10 == 0, strict=False)
        if adaptation and ((step + 1) % 10 == 0 or step + 1 == total_steps):
            print(json.dumps({"stage": "fit", "outer_fold": outer, "phase": phase,
                              "completed_steps": step + 1, "total_steps": total_steps,
                              "tail_completed_steps": committed["tail_completed_steps"],
                              "margin": head.margin, "encoder_lr": metrics["fit/encoder_lr"],
                              "head_lr": metrics["fit/head_lr"], "loss_aam": metrics["fit/loss_aam"]}), flush=True)
        if (step + 1) % fit["checkpoint_every_steps"] == 0 or step + 1 == total_steps:
            save(step + 1)
    encoder.eval()
    # Model and optimizer artifacts are server-only for controlled research
    # runs.  Historical recipes keep the legacy default for compatibility;
    # newer configs explicitly set this retention gate to false.
    retention = config.get("retention", {})
    if retention.get("mlflow_upload_model_artifacts", True):
        tracker.add_artifact(checkpoint, "checkpoints/last.pt")
        tracker.add_artifact(history_path, "training/fit_history.jsonl")
    adaptation_report = ({"adaptation_schedule": dict(adaptation),
                          "schedule_state": adaptation_checkpoint_state(fit, total_steps),
                          "final_update_settings": adaptation_step(fit, total_steps - 1),
                          "learning_rate_log_semantics": "Rates actually used for the reported update",
                          "head_only_encoder_policy": "eval and no_grad; no encoder optimizer update"}
                         if adaptation else {})
    return {**details, **adaptation_report, "completed_steps": total_steps, "initial_step": start_step,
            "epoch_selection": "fixed_steps_no_outer_selection",
            "elapsed_seconds": time.monotonic() - started, "augmentation": fit["augmentation"]}
