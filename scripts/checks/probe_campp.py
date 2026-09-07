"""Forward-only CAM++ preflight: no backward, optimizer or calibration fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model/campp.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/campp_baseline.json")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/infrastructure/campp_probe.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchaudio
    from speaker_id.models.campp import crop_waveform, extract_embedding, file_sha256, load_campp, make_fbank, read_mono
    from speaker_id.training.fit import AAMHead, freeze_batchnorm, set_trainable_tail
    from speaker_id.training.contracts import load_contract
    from speaker_id.tracking.snapshot import git_provenance
    torch.set_num_threads(args.threads)
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    contract = load_contract(args.config, ROOT)
    if contract["model"] != config:
        raise ValueError("Probe model differs from the selected experiment model contract")
    started = time.monotonic()
    encoder = load_campp(config, ROOT, args.device)
    vector, info = extract_embedding(encoder, args.audio, device=args.device, seconds=3.0, maximum_windows=1)
    if not info["nonzero_signal"] or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 1e-5:
        raise ValueError("Probe must use nonzero real audio and produce a unit 512D embedding")
    fit_config = contract["config"]["fit"]
    tail = set_trainable_tail(encoder, fit_config["trainable_prefixes"])
    encoder.train()
    freeze_batchnorm(encoder)
    features = make_fbank(crop_waveform(read_mono(args.audio), 3.0, minimum_seconds=3.0))
    batch = features.unsqueeze(0).repeat(2, 1, 1).to(args.device)
    output = encoder(batch)
    head = AAMHead().to(args.device)
    logits = head(output, torch.tensor([0, 445], device=args.device))
    if output.shape != (2, 512) or logits.shape != (2, 446) or not logits.isfinite().all() or not logits.requires_grad:
        raise ValueError("Trainable forward graph/head contract failed")
    report = {"status": "passed_forward_only", "training_started": False, "optimizer_steps": 0,
              "backward_calls": 0, "model_weight_sha256": config["weights_sha256"],
              "audio_file": args.audio.name, "audio_sha256": file_sha256(args.audio),
              "embedding_shape": list(vector.shape), "embedding_norm": float(np.linalg.norm(vector)),
              "fit_forward_logits_shape": list(logits.shape), "gradient_graph_constructed": True,
              "trainable_parameters": tail, "device": args.device,
              "torch": torch.__version__, "torchaudio": torchaudio.__version__,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "elapsed_seconds": time.monotonic() - started, **info}
    report["model_config_sha256"] = file_sha256(args.model_config)
    report["input_hashes"] = contract["input_hashes"]
    report["code_hashes"] = contract["code_hashes"]
    report["contract_signature"] = contract["signature"]
    report.update(git_provenance(ROOT))
    report["source_sha256"] = {path.relative_to(ROOT).as_posix(): file_sha256(path)
                              for path in sorted((ROOT / "src").rglob("*.py"))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
