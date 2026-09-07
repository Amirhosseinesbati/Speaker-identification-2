"""Reproduce F002's first tail backward in three precisions, without any optimizer.

Default mode validates configuration and deterministic batch planning only.
--execute is restricted to the authorized RTX 3090 and never updates weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = "artifacts/training/F002_20260907T153529Z_4675ff03"
SIGNATURE = "bede1f908396b619aee52a14e7f845e3749abba66a1c5dec59d1d9ca79dda398"
PARENT_RUN_ID = "1303b2c6d7e245d3ba2742734bde0942"
CHECKPOINT_SHA256 = "6e54ff3390547a951dbd965bb88f78e34cbba7a27d53ac79c8be43ccf98ab2c8"
STEP = 100


def confined(path: Path, directory: str, *, must_exist: bool = False) -> Path:
    path = (ROOT / path).absolute()
    # Resolve each existing ancestor so a symlink cannot redirect reads/writes.
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to((ROOT / directory).resolve()) or path.is_symlink():
        raise ValueError("Diagnostic path escapes its permitted project directory")
    return resolved


def batch_plan(config: dict, roles: list[dict], labels: list[str]) -> list[dict]:
    import numpy as np
    from speaker_id.data.splits import truth
    from speaker_id.training.schedules import adaptation_step
    if (config.get("experiment_code") != "F002" or config.get("mode") != "fine_tune"
            or config["fit"]["adaptation_schedule"]["head_only_steps"] != STEP
            or adaptation_step(config["fit"], STEP)["phase"] != "tail"):
        raise ValueError("This bounded diagnostic targets F002's first tail step only")
    by_label = {label: [] for label in labels[1:]}
    for row in roles:
        if int(row["outer_fold"]) == 0 and row["speaker_id"] != "unknown" and truth(row["encoder_fit_allowed"]):
            by_label[row["speaker_id"]].append(row)
    if not all(by_label.values()):
        raise ValueError("Every known label needs a permitted fold0 fit example")
    target_index = {label: index for index, label in enumerate(labels[1:])}
    rng = np.random.default_rng(int(config["seed"]) + STEP)  # outer fold0
    selected = rng.choice(labels[1:], config["fit"]["batch_size"], replace=config["fit"]["batch_size"] > 446)
    result = []
    for label in selected:
        row = by_label[str(label)][int(rng.integers(len(by_label[str(label)])))]
        result.append({"audio_file": row["audio_file"], "speaker_id": str(label),
                       "target": target_index[str(label)], "crop_position": float(rng.random())})
    return result


def gradient_summary(named_parameters, scale: float) -> dict:
    import torch
    present = [(name, p.grad) for name, p in named_parameters if p.grad is not None]
    missing = [name for name, p in named_parameters if p.requires_grad and p.grad is None]
    scaled_bad = [name for name, grad in present if not torch.isfinite(grad).all()]
    # Match scaler unscaling without constructing or calling an optimizer.
    inverse = torch.tensor(scale, dtype=torch.float64, device="cuda").reciprocal().float()
    for _, grad in present:
        grad.mul_(inverse)
    bad = [{"name": name, "nonfinite_elements": int((~torch.isfinite(grad)).sum().item())}
           for name, grad in present if not torch.isfinite(grad).all()]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad, 2) for _, grad in present]), 2)
    return {"parameters_with_gradients": len(present), "missing_trainable_gradient_names": missing,
            "scaled_nonfinite_gradient_names": scaled_bad, "nonfinite_gradients": bad,
            "nonfinite_gradient_elements": sum(row["nonfinite_elements"] for row in bad),
            "all_gradients_finite": not bad and not missing,
            "gradient_norm_finite": bool(torch.isfinite(norm)),
            "gradient_norm": float(norm.item()) if torch.isfinite(norm) else None}


def execute_probe(args, contract: dict, plan: list[dict]) -> dict:
    if os.environ.get("VAST_INSTANCE_ID") != "50079023":
        raise RuntimeError("Backward diagnostics require authorized Vast instance50079023")
    import torch
    from speaker_id.models.campp import crop_waveform, file_sha256, load_campp, make_fbank, read_mono
    from speaker_id.training.fit import AAMHead, freeze_batchnorm, set_trainable_tail
    from speaker_id.training.schedules import adaptation_checkpoint_state, adaptation_step
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Backward diagnostics require the authorized RTX 3090")
    source = confined(args.source_run, "artifacts/training", must_exist=True)
    checkpoint = confined(source / "fold_0/last.pt", "artifacts/training", must_exist=True)
    original = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    experiment_state = json.loads((source / "experiment_state.json").read_text(encoding="utf-8"))
    recomputed = hashlib.sha256(json.dumps({"config": original["experiment"], "model": original["model"],
        "input_hashes": original["input_hashes"], "code_hashes": original["code_hashes"]}, sort_keys=True).encode()).hexdigest()
    if (recomputed != SIGNATURE or original["signature"] != SIGNATURE
            or experiment_state.get("parent_run_id") != PARENT_RUN_ID
            or experiment_state.get("status") != "failed"
            or original["experiment"] != contract["config"] or original["model"] != contract["model"]
            or original["input_hashes"] != contract["input_hashes"]):
        raise ValueError("Diagnostic source does not match the preserved failed F002 run")
    for name, digest in contract["code_hashes"].items():
        if (name.startswith("src/speaker_id/models/") or name in
                {"src/speaker_id/training/fit.py", "src/speaker_id/training/schedules.py"}):
            if original["code_hashes"].get(name) != digest:
                raise ValueError("The sampler, frontend, model or schedule changed since the failure")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("The diagnostic requires the separately verified preserved checkpoint bytes")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fit = contract["config"]["fit"]
    if (state.get("format_version") != 2 or state.get("signature") != SIGNATURE
            or state.get("outer_fold") != 0 or state.get("completed_steps") != STEP
            or state.get("scheduler") is not None or state.get("adaptation_schedule") != fit["adaptation_schedule"]
            or state.get("schedule_state") != adaptation_checkpoint_state(fit, STEP)):
        raise ValueError("Expected the unmodified format2 head-complete checkpoint at step100")
    scale = float(state["scaler"]["scale"])
    if not math.isfinite(scale) or scale <= 0 or not 0 < args.lower_scale < scale:
        raise ValueError("Lower diagnostic scale must be positive and below the saved AMP scale")
    torch.set_num_threads(contract["config"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = {row["audio_file"]: row for row in contract["manifest"]}
    features = []
    for item in plan:
        audio = confined(ROOT / contract["config"]["data_dir"] / item["audio_file"], "data/raw", must_exist=True)
        if file_sha256(audio) != manifest[item["audio_file"]]["input_sha256"]:
            raise ValueError("A diagnostic batch audio file failed its source hash")
        item["audio_sha256"] = manifest[item["audio_file"]]["input_sha256"]
        crop = crop_waveform(read_mono(audio), fit["crop_seconds"], position=item["crop_position"],
                             minimum_seconds=fit["crop_seconds"])
        features.append(make_fbank(crop))
    cpu_batch = torch.stack(features)
    batch_hash = hashlib.sha256(cpu_batch.contiguous().numpy().tobytes()).hexdigest()
    batch = cpu_batch.to("cuda")
    targets = torch.tensor([item["target"] for item in plan], dtype=torch.long, device="cuda")
    scheduled = adaptation_step(fit, STEP)
    report = {"status": "completed_backward_diagnostic", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "diagnostic_backward_only": True, "optimizer_steps": 0,
              "source_run_id": PARENT_RUN_ID, "source_signature": SIGNATURE, "outer_fold": 0,
              "checkpoint_completed_steps": STEP, "next_zero_based_step": STEP,
              "checkpoint_sha256": checkpoint_hash, "resolved_config_sha256": file_sha256(source / "resolved_config.json"),
              "source_git_commit": "79759c0291c1d349fc2186551b4e188cfd061eeb",
              "public_model_sha256": contract["model"]["weights_sha256"],
              "diagnostic_script_sha256": file_sha256(Path(__file__)), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "scheduled_first_tail_settings": scheduled,
              "batch_shape": list(batch.shape), "batch_fbank_sha256": batch_hash, "batch_plan": plan, "cases": []}
    for name, amp, loss_scale in (("amp_checkpoint_scale", True, scale),
                                  ("amp_lower_scale", True, args.lower_scale), ("fp32", False, 1.0)):
        encoder = head = embeddings = logits = loss = None
        started = time.monotonic()
        outcome = {"case": name, "autocast_fp16": amp, "loss_scale": loss_scale, "backward_calls": 0}
        try:
            encoder = load_campp(contract["model"], ROOT, "cuda")
            encoder.load_state_dict(state["encoder"], strict=True)
            set_trainable_tail(encoder, fit["trainable_prefixes"])
            encoder.train()
            freeze_batchnorm(encoder)
            head = AAMHead(margin=scheduled["margin"], scale=fit["scale"]).to("cuda")
            head.load_state_dict(state["head"], strict=True)
            head.train()
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                embeddings = encoder(batch)
            logits = head(embeddings, targets)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            outcome.update(loss_finite=bool(torch.isfinite(loss)),
                           loss=float(loss.item()) if torch.isfinite(loss) else None)
            if outcome["loss_finite"]:
                (loss * loss_scale).backward()
                outcome["backward_calls"] = 1
                outcome.update(gradient_summary(list(encoder.named_parameters()) +
                    [("head." + n, p) for n, p in head.named_parameters()], loss_scale))
            outcome["model_state_unchanged"] = all(torch.equal(value.detach().cpu(), state["encoder"][key])
                for key, value in encoder.state_dict().items()) and all(torch.equal(value.detach().cpu(), state["head"][key])
                for key, value in head.state_dict().items())
            if not outcome["model_state_unchanged"]:
                raise RuntimeError("A diagnostic forward/backward unexpectedly changed model state")
        except Exception as error:
            outcome.update(error_type=type(error).__name__, error=str(error))
        finally:
            torch.cuda.synchronize()
            outcome.update(peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                           elapsed_seconds=time.monotonic() - started)
            report["cases"].append(outcome)
            encoder = head = embeddings = logits = loss = None
            torch.cuda.empty_cache()
    report["checkpoint_sha256_after"] = file_sha256(checkpoint)
    if report["checkpoint_sha256_after"] != checkpoint_hash:
        raise RuntimeError("Preserved checkpoint changed during the read-only diagnostic")
    report["checkpoint_unchanged"] = True
    report["backward_calls"] = sum(case["backward_calls"] for case in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_finetune_warmup.json")
    parser.add_argument("--source-run", type=Path, default=ROOT / SOURCE)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/infrastructure/F002_precision_diagnostic.json")
    parser.add_argument("--lower-scale", type=float, default=1024.)
    parser.add_argument("--execute", action="store_true", help="Allow three backward-only CUDA diagnostics; zero optimizer steps")
    args = parser.parse_args()
    if not math.isfinite(args.lower_scale) or args.lower_scale <= 0:
        parser.error("--lower-scale must be positive and finite")
    args.config = confined(args.config, "configs/train", must_exist=True)
    args.source_run = confined(args.source_run, "artifacts/training")
    args.report = confined(args.report, "artifacts/infrastructure")
    from speaker_id.training.contracts import load_contract
    contract = load_contract(args.config, ROOT)
    plan = batch_plan(contract["config"], contract["roles"], contract["labels"])
    if not args.execute:
        print(json.dumps({"status": "validated_no_backward_started", "optimizer_steps": 0, "backward_calls": 0,
                          "batch_files": len(plan), "next_zero_based_step": STEP,
                          "source_checkpoint_verified": False, "source_run": str(args.source_run)}))
        return
    report = execute_probe(args, contract, plan)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report), "optimizer_steps": 0,
                      "cases": report["cases"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
