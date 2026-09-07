"""Forward-only QA of a standalone release, from an unrelated working directory.

The subprocess sees only release code, fresh model caches, and selected audio.
Python network connection and DNS entry points are blocked before any imports.
This is an offline smoke test, not the organizer's unavailable evaluator image.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]

WORKER = r'''
import json, pathlib, runpy, socket, sys, time
started = time.monotonic()
attempts = []
def blocked(*args, **kwargs):
    attempts.append("network entry point called")
    raise RuntimeError("Network is disabled for standalone release QA")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
root, inputs, out, result, vectors = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root))
import numpy as np
import torch, torchaudio, soundfile, scipy
torch.set_num_threads(4)
from speaker_id.inference.runtime import verify_payload
from speaker_id.inference.scoring import score_embeddings
from speaker_id.models.campp import load_campp, extract_embedding
verify_payload(root)
model_config = json.loads((root / "assets/model_config.json").read_text())
calibration = json.loads((root / "assets/calibration.json").read_text())
with np.load(root / "assets/gallery.npz", allow_pickle=False) as saved:
    gallery = {key: saved[key] for key in saved.files}
encoder = load_campp(model_config, root, "cpu")
names, embeddings, valid = [], [], []
for path in sorted(inputs.iterdir()):
    if not path.is_file():
        continue
    vector, info = extract_embedding(encoder, path, device="cpu", seconds=180.0, maximum_windows=1)
    names.append(path.name)
    embeddings.append(vector)
    valid.append(info["nonzero_signal"])
prob = score_embeddings(np.asarray(embeddings), np.asarray(valid), gallery, calibration)
assert prob.shape == (len(names), 447)
assert np.isfinite(prob).all() and (prob >= 0).all()
assert np.allclose(prob.sum(axis=1), 1, atol=1e-12)
assert (prob[~np.asarray(valid)].argmax(axis=1) == 0).all()
np.savez(vectors, audio_file=np.asarray(names), embedding=np.asarray(embeddings), probabilities=prob)
del encoder
sys.argv = [str(root / "submission.py"), "--data-dir", str(inputs), "--predictions-file-path", str(out)]
try:
    runpy.run_path(str(root / "submission.py"), run_name="__main__")
except SystemExit as error:
    if error.code not in (None, 0):
        raise
for forbidden in ("mlflow", "speaker_id.training", "speaker_id.tracking", "huggingface_hub", "modelscope"):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules), forbidden
assert not attempts, attempts
result.write_text(json.dumps({"status": "passed", "network_attempts": len(attempts),
    "network_block": "Python socket connect/connect_ex/create_connection/getaddrinfo",
    "device": "cpu", "optimizer_steps": 0, "backward_calls": 0,
    "torch": torch.__version__, "torchaudio": torchaudio.__version__,
    "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__,
    "python": sys.version, "files": len(names), "elapsed_seconds": time.monotonic()-started}, indent=2))
'''


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_examples(rows):
    nonzero = [r for r in rows if r["has_nonzero_signal"].lower() == "true"]
    selections = [
        ("zero", next(r for r in rows if r["has_nonzero_signal"].lower() == "false")),
        ("shortest_nonzero", min(nonzero, key=lambda r: float(r["duration_seconds"]))),
        ("longest", max(nonzero, key=lambda r: float(r["duration_seconds"]))),
        ("lowest_rms_nonzero", min(nonzero, key=lambda r: float(r["mono_rms_dbfs"]))),
        ("known", next(r for r in nonzero if r["speaker_id"] != "unknown" and float(r["duration_seconds"]) > 30)),
        ("unknown", next(r for r in nonzero if r["speaker_id"] == "unknown" and float(r["duration_seconds"]) > 30)),
    ]
    for description, predicate in [
        ("real_mp3", lambda r: r["detected_format"] in {"MP3", "MPEG"}),
        ("container_extension_mismatch", lambda r: r["extension_matches_container"].lower() == "false"),
        ("resampled", lambda r: int(r["sample_rate_hz"]) != 16000),
    ]:
        match = next((r for r in nonzero if predicate(r)), None)
        if match is not None:
            selections.append((description, match))
    unique = {}
    for description, row in selections:
        unique.setdefault(row["audio_file"], {"row": row, "cases": []})["cases"].append(description)
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "artifacts/training/B002_20260907T150724Z_9b11fe4b/frozen_embedding_cache")
    args = parser.parse_args()
    release = args.release_dir.resolve()
    if not (release / "submission.py").is_file():
        raise ValueError("Expected an extracted standalone release")
    run_dir = ROOT / "artifacts/qa" / ("offline_" + uuid.uuid4().hex[:12])
    inputs, cwd = run_dir / "inputs", run_dir / "unrelated_cwd"
    inputs.mkdir(parents=True)
    cwd.mkdir()
    with (ROOT / "data/processed/eda_v1/audio_manifest.csv").open(newline="", encoding="utf-8") as handle:
        examples = select_examples(list(csv.DictReader(handle)))
    for item in examples:
        row = item["row"]
        source = ROOT / "data/raw" / row["audio_file"]
        if sha256(source) != row["input_sha256"]:
            raise ValueError("Audio does not match the immutable EDA manifest")
        shutil.copyfile(source, inputs / source.name)
    # Explicit environment allowlist keeps project tracking credentials absent.
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT", "LD_LIBRARY_PATH"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                "HF_HOME": str(run_dir / "empty_hf"), "TORCH_HOME": str(run_dir / "empty_torch"),
                "XDG_CACHE_HOME": str(run_dir / "empty_xdg"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
        Path(env[name]).mkdir()
    output = cwd / "new_parent" / "predictions.csv"
    evidence, vectors = run_dir / "worker_report.json", run_dir / "vectors.npz"
    completed = subprocess.run([sys.executable, "-c", WORKER, str(release), str(inputs), str(output),
                                str(evidence), str(vectors)], cwd=cwd, env=env, capture_output=True,
                               text=True, timeout=600)
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Standalone offline QA failed; inspect {run_dir / 'stderr.log'}")
    import numpy as np
    labels = json.loads((release / "assets/labels.json").read_text(encoding="utf-8"))
    labels = labels["labels"] if isinstance(labels, dict) else labels
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["audio_file", "speaker_id"]:
            raise ValueError("Submission CSV columns differ from the competition contract")
        predictions = list(reader)
    names = [r["audio_file"] for r in predictions]
    if len(names) != len(set(names)) or set(names) != {item["row"]["audio_file"] for item in examples}:
        raise ValueError("Submission CSV does not cover every input exactly once")
    if any(row["speaker_id"] not in labels for row in predictions):
        raise ValueError("Submission emitted an invalid label")
    with np.load(vectors, allow_pickle=False) as saved:
        expected = {name: labels[int(index)] for name, index in zip(saved["audio_file"], saved["probabilities"].argmax(axis=1))}
        if any(row["speaker_id"] != expected[row["audio_file"]] for row in predictions):
            raise ValueError("CLI and probability argmax disagree")
        parity, cached_vectors = [], []
        for name, vector in zip(saved["audio_file"], saved["embedding"]):
            cache = args.source_cache / (Path(str(name)).stem + ".npz")
            with np.load(cache, allow_pickle=False) as cached:
                reference = cached["embedding"]
            difference = float(np.max(np.abs(vector - reference)))
            cosine = float(vector.astype(np.float64) @ reference.astype(np.float64)) if np.any(reference) else None
            # CPU and CUDA convolution backends need not be bit-identical.
            # Keep a conservative numerical guard AND require identical actual
            # decisions below; production CUDA is checked independently.
            if difference > .002 or (cosine is not None and cosine < .9999):
                raise ValueError(f"CPU release extraction differs from B002 cache: {name}: {difference}")
            parity.append({"audio_file": str(name), "maximum_absolute_difference": difference, "cosine": cosine})
            cached_vectors.append(reference.copy())
        spec = importlib.util.spec_from_file_location("_portable_scoring_qa", release / "speaker_id/inference/scoring.py")
        scoring = importlib.util.module_from_spec(spec)
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(scoring)
        finally:
            sys.dont_write_bytecode = previous_bytecode
        with np.load(release / "assets/gallery.npz", allow_pickle=False) as arrays:
            gallery = {name: arrays[name] for name in arrays.files}
        calibration = json.loads((release / "assets/calibration.json").read_text())
        cached_vectors = np.asarray(cached_vectors)
        cached_probabilities = scoring.score_embeddings(cached_vectors, np.any(cached_vectors, axis=1), gallery, calibration)
        if not np.array_equal(cached_probabilities.argmax(axis=1), saved["probabilities"].argmax(axis=1)):
            raise ValueError("CPU audio inference and CUDA cached embeddings produce different speaker decisions")
        probability_drift = float(np.max(np.abs(cached_probabilities - saved["probabilities"])))
    report = {**json.loads(evidence.read_text()), "verified_at": datetime.now(timezone.utc).isoformat(),
              "release_dir": str(release), "qa_directory": str(run_dir),
              "submission_sha256": sha256(release / "submission.py"), "csv_sha256": sha256(output),
              "cli_probability_argmax_parity": True, "input_coverage_exact": True,
              "cpu_vs_cuda_cache_decisions_exact": True,
              "cpu_vs_cuda_cache_max_probability_difference": probability_drift,
              "cross_device_tolerance": {"maximum_absolute_embedding_difference": .002, "minimum_embedding_cosine": .9999,
                                         "speaker_decisions_must_match": True, "bit_identical_embeddings_required": False},
              "cases": [{"audio_file": item["row"]["audio_file"], "cases": item["cases"]} for item in examples],
              "embedding_parity_with_server_B002": parity,
              "limitations": ["Host Python dependencies; organizer evaluator image is not supplied.",
                              "Network blocking covers Python socket APIs, not an OS network namespace.",
                              "CPU and CUDA numerical drift is recorded; matched decisions apply to these examples, not every future input.",
                              "Representative forward/CSV checks, not a leaderboard score or held-out quality estimate."]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "files": report["files"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
