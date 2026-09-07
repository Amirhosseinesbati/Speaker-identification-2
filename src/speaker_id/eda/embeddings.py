"""Frozen public ECAPA probes for descriptive EDA; never fits competition labels."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
REPOSITORY = "speechbrain/spkrec-ecapa-voxceleb"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe_windows(frames: int, rate: int, seconds: float = 6., maximum: int = 3):
    """Up to three equally spaced, disjoint windows; no artificial repetition."""
    length = min(frames, round(rate * seconds))
    if length <= 0:
        return []
    count = min(maximum, max(1, frames // length))
    return [(int(start), length) for start in np.linspace(0, frames - length, count)]


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def gain_probe(x: np.ndarray) -> tuple[np.ndarray, float]:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x), initial=0))
    factor = max(1., min(100., .1 / max(rms, 1e-12), .95 / max(peak, 1e-12)))
    return x * factor, 20 * math.log10(factor)


def load_encoder(model_dir: Path, device: str, threads: int):
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization

    torch.set_num_threads(threads)
    torch.manual_seed(20260907)
    features = Fbank(n_mels=80).to(device).eval()
    normalize = InputNormalization(norm_type="sentence", std_norm=False).to(device).eval()
    encoder = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                         kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                         attention_channels=128, lin_neurons=192).to(device).eval()
    encoder.load_state_dict(torch.load(model_dir / "embedding_model.ckpt", map_location="cpu", weights_only=True), strict=True)
    encoder.requires_grad_(False)
    features.requires_grad_(False)
    normalize.requires_grad_(False)

    def encode(samples):
        # Each file's windows have equal length. Short clips are zero-padded only to 1s.
        array = np.stack([np.pad(x, (0, max(0, 16000 - len(x)))) for x in samples])
        with torch.inference_mode():
            wav = torch.from_numpy(array.astype(np.float32)).to(device)
            lengths = torch.ones(len(wav), device=device)
            result = encoder(normalize(features(wav), lengths)).squeeze(1).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite encoder output")
        return unit(result).astype(np.float32)

    return encode


def run(manifest: Path, data_dir: Path, model_dir: Path, cache: Path, report_dir: Path,
        device="cuda", threads=2, limit=None):
    import torch
    started = time.monotonic()
    frame = pd.read_csv(manifest)
    if limit is not None:
        frame = frame.head(limit)
    cache.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    version_names = ("numpy", "scipy", "soundfile", "torch", "torchaudio", "speechbrain")
    versions = {name: importlib.metadata.version(name) for name in version_names}
    model_hashes = {name: digest(model_dir / name) for name in ("embedding_model.ckpt", "hyperparams.yaml")}
    config = {"manifest_sha256": digest(manifest), "model_sha256": model_hashes,
              "repository": REPOSITORY, "revision": REVISION, "code_sha256": digest(Path(__file__)),
              "versions": versions, "seconds": 6, "maximum_windows": 3, "minimum_padding_seconds": 1,
              "input_policy": "original strongest channel, resample_poly to 16k, no VAD/no gain in primary view",
              "gain_probe_policy": "RMS<-50dBFS only; target -20dBFS, gain<=40dB, peak<=0.95, no attenuation",
              "device": device, "threads": threads}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    encode = load_encoder(model_dir, device, threads)
    rows, segments, gains, names, vectors = [], [], [], [], []
    resumed = 0
    for i, row in enumerate(frame.itertuples(index=False)):
        target = cache / (Path(row.audio_file).stem + ".json")
        result = None
        if target.exists():
            old = json.loads(target.read_text())
            if old.get("signature") == signature and old.get("input_sha256") == row.input_sha256:
                result = old
                resumed += 1
        if result is None:
            result = {"signature": signature, "audio_file": row.audio_file,
                      "input_sha256": row.input_sha256, "status": "skipped_zero", "segments": [], "gain": []}
            if row.status == "ok" and row.has_nonzero_signal:
                path = data_dir / row.audio_file
                if digest(path) != row.input_sha256:
                    raise ValueError(f"Raw bytes changed: {row.audio_file}")
                with sf.SoundFile(path) as sound:
                    windows = probe_windows(len(sound), sound.samplerate)
                    samples = []
                    specs = []
                    for start, length in windows:
                        sound.seek(start)
                        x = sound.read(length, dtype="float32", always_2d=True)[:, int(row.analysis_channel_index)]
                        if sound.samplerate != 16000:
                            gcd = math.gcd(sound.samplerate, 16000)
                            x = resample_poly(x, 16000 // gcd, sound.samplerate // gcd).astype(np.float32)
                        samples.append(x)
                        specs.append({"start_seconds": start / sound.samplerate,
                                      "seconds": length / sound.samplerate,
                                      "nonzero_samples": int(np.count_nonzero(x)),
                                      "rms_dbfs": 20 * np.log10(max(float(np.sqrt(np.mean(x.astype(float)**2))), 1e-12))})
                embedded = encode(samples)
                result["status"] = "ok"
                result["segments"] = [dict(spec, embedding=e.tolist()) for spec, e in zip(specs, embedded)]
                if row.max_channel_rms_dbfs < -50:
                    probes = [gain_probe(x) for x in samples]
                    amplified = encode([p[0] for p in probes])
                    result["gain"] = [{"gain_db": p[1], "embedding": e.tolist()} for p, e in zip(probes, amplified)]
            target.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        if result["status"] == "ok":
            e = np.array([s["embedding"] for s in result["segments"]], dtype=np.float32)
            pooled = unit(e.mean(axis=0))
            sim = e @ e.T
            within = sim[np.triu_indices(len(e), 1)]
            names.append(row.audio_file)
            vectors.append(pooled)
            item = {"audio_file": row.audio_file, "speaker_id": row.speaker_id, "status": "ok",
                    "segment_count": len(e), "probe_seconds": sum(s["seconds"] for s in result["segments"]),
                    "minimum_segment_cosine": float(within.min()) if len(within) else None,
                    "mean_segment_cosine": float(within.mean()) if len(within) else None,
                    "silent_probe_count": sum(s["nonzero_samples"] == 0 for s in result["segments"])}
            for k, s in enumerate(result["segments"]):
                segments.append({"audio_file": row.audio_file, "segment": k, **{a:b for a,b in s.items() if a != "embedding"}})
            if result["gain"]:
                g = unit(np.array([p["embedding"] for p in result["gain"]]).mean(axis=0))
                gains.append({"audio_file": row.audio_file, "speaker_id": row.speaker_id,
                              "original_to_gain_cosine": float(pooled @ g),
                              "mean_gain_db": float(np.mean([p["gain_db"] for p in result["gain"]]))})
        else:
            item = {"audio_file": row.audio_file, "speaker_id": row.speaker_id,
                    "status": result["status"], "segment_count": 0, "probe_seconds": 0}
        rows.append(item)
        if (i + 1) % 100 == 0 or i + 1 == len(frame):
            print(json.dumps({"completed": i + 1, "total": len(frame), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    pd.DataFrame(rows).to_csv(report_dir / "embedding_files.csv", index=False)
    pd.DataFrame(segments).to_csv(report_dir / "embedding_segments.csv", index=False)
    pd.DataFrame(gains).to_csv(report_dir / "embedding_gain.csv", index=False)
    np.savez_compressed(cache / "embeddings.npz", audio_file=np.array(names), embeddings=np.array(vectors, dtype=np.float32))
    summary = {**config, "status": "complete" if limit is None else "limited", "signature": signature,
               "source_files": len(frame), "embedded_files": len(vectors), "skipped_zero_files": sum(r["status"]=="skipped_zero" for r in rows),
               "segments": len(segments), "probe_seconds": sum(s["seconds"] for s in segments),
               "gain_comparator_files": len(gains), "cached_files": resumed,
               "elapsed_seconds": time.monotonic()-started, "python": platform.python_version(),
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "artifact_sha256": {n: digest(report_dir/n) for n in ("embedding_files.csv", "embedding_segments.csv", "embedding_gain.csv")},
               "embedding_npz_sha256": digest(cache/"embeddings.npz"),
               "limitations": ["Frozen public encoder only; no competition fitting, calibration or performance estimate",
                               "At most 18 seconds per file; unseen sections are not semantically reviewed",
                               "Speaker similarity does not prove recording-session overlap or true identity",
                               "Silent/very quiet windows and short padding can create uninformative embeddings",
                               "Public model provenance is cached repository revision plus SHA256; no old competition checkpoint used"]}
    (report_dir/"embedding_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary
