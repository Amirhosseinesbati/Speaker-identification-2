"""Integrity-checked, standalone audio-to-CSV inference without online services."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

import numpy as np

from .scoring import score_embeddings, validate_calibration, validate_gallery


class IntegrityError(ValueError):
    """The portable payload is missing, malformed, or changed."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _relative_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise IntegrityError("Manifest paths must be canonical relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != name:
        raise IntegrityError("Manifest contains a noncanonical or escaping path")
    return path


def verify_payload(root: Path, *, allowed_output: Path | None = None) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or _is_link(manifest_path):
        raise IntegrityError("A regular manifest.json is required beside submission.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if (manifest.get("schema_version") != 1 or not isinstance(files, dict) or not files
            or not isinstance(manifest.get("release_id"), str) or not manifest["release_id"]):
        raise IntegrityError("Invalid release manifest schema")
    source = "speaker_id" if (root / "speaker_id").is_dir() else "src/speaker_id"
    required = {"submission.py", "assets/model_config.json", "assets/campplus_voxceleb.bin",
                "assets/gallery.npz", "assets/labels.json", "assets/calibration.json",
                source + "/inference/runtime.py", source + "/inference/scoring.py", source + "/models/campp.py"}
    if not required.issubset(files):
        raise IntegrityError("Release manifest omits a required runtime or model asset")
    seen = set()
    for name, record in files.items():
        relative = _relative_path(name)
        if name == "manifest.json" or name.casefold() in seen:
            raise IntegrityError("Duplicate, case-colliding, or self-referential manifest entry")
        seen.add(name.casefold())
        if (not isinstance(record, dict) or type(record.get("bytes")) is not int or record["bytes"] < 0
                or not isinstance(record.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", record["sha256"])):
            raise IntegrityError("Invalid file integrity record")
        path = root.joinpath(*relative.parts)
        if any(_is_link(parent) for parent in [path, *path.parents] if parent != root.parent):
            raise IntegrityError("Symlinks and junctions are forbidden in a portable release")
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise IntegrityError(f"Missing or changed payload file: {name}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise IntegrityError(f"Payload checksum mismatch: {name}")
    actual = set()
    for path in root.rglob("*"):
        if _is_link(path):
            raise IntegrityError("Symlinks and junctions are forbidden in a portable release")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    expected = set(files) | {"manifest.json"}
    if allowed_output is not None:
        output = Path(allowed_output).resolve()
        if output.is_relative_to(root) and output.is_file():
            # Permit only this explicitly requested output on repeated CLI runs.
            expected.add(output.relative_to(root).as_posix())
    if actual != expected:
        raise IntegrityError("Unlisted files are present in the portable payload")
    return manifest


def _labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if (not isinstance(labels, list) or len(labels) != 447 or len(set(labels)) != 447
            or labels[0] != "unknown" or payload.get("unknown_index") != 0
            or labels[1:] != sorted(labels[1:])):
        raise IntegrityError("Expected unknown followed by the sorted 446 known labels")
    try:
        if any(str(uuid.UUID(label)) != label for label in labels[1:]):
            raise ValueError("Noncanonical known label")
    except (ValueError, TypeError, AttributeError) as error:
        raise IntegrityError("Known labels must be canonical UUID strings") from error
    return labels


def _load_encoder(config: dict, root: Path, device: str | None):
    import torch
    from speaker_id.models.campp import load_campp
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if chosen not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu or cuda")
    torch.set_num_threads(4)
    return load_campp(config, root, chosen).float().eval(), chosen


def _extract(encoder, path: Path, *, device: str):
    from speaker_id.models.campp import extract_embedding
    return extract_embedding(encoder, path, device=device, seconds=180.0, maximum_windows=1)


def run_submission(root: Path, data_dir: Path, predictions_file_path: Path, *, device: str | None = None) -> dict:
    root = Path(root).resolve()
    output = Path(predictions_file_path).resolve()
    manifest = verify_payload(root, allowed_output=output)
    labels = _labels(root / "assets/labels.json")
    calibration = validate_calibration(json.loads((root / "assets/calibration.json").read_text()), require_inference=True)
    with np.load(root / "assets/gallery.npz", allow_pickle=False) as arrays:
        gallery = validate_gallery({name: arrays[name].copy() for name in arrays.files})
    config = json.loads((root / "assets/model_config.json").read_text())
    if config.get("weights_path") != "assets/campplus_voxceleb.bin":
        raise IntegrityError("The model must use the bundled public CAM++ weights")
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError("--data-dir must be an existing audio directory")
    extensions = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".m4a", ".aif", ".aiff", ".au", ".snd", ".wma"}
    inputs = sorted((path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions), key=lambda path: path.name)
    if any(_is_link(path) for path in inputs):
        raise ValueError("Input audio symlinks are not supported")
    if output in {path.resolve() for path in inputs} or output == data_dir:
        raise ValueError("The predictions output cannot overwrite input audio")
    if output.is_relative_to(root) and output.relative_to(root).as_posix() in set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("The predictions output cannot overwrite a portable model asset")
    encoder, chosen = _load_encoder(config, root, device)
    rows, errors = [], []
    for path in inputs:
        try:
            embedding, info = _extract(encoder, path, device=chosen)
        except Exception as error:
            # Decoder/file errors are recoverable; model/GPU/frontend failures are fatal.
            import soundfile as sf
            recoverable = isinstance(error, (OSError, sf.LibsndfileError)) or (
                isinstance(error, ValueError) and str(error).startswith("Empty or nonfinite waveform:"))
            if not recoverable:
                raise
            embedding, info = np.zeros(512, dtype=np.float32), {"nonzero_signal": False}
            errors.append({"audio_file": path.name, "error_type": type(error).__name__})
            print(f"Audio decoding failed for {path.name}; emitting unknown ({type(error).__name__}).", file=sys.stderr)
        probabilities = score_embeddings(np.asarray(embedding)[None, :], np.asarray([bool(info["nonzero_signal"])]), gallery, calibration)
        rows.append({"audio_file": path.name, "speaker_id": labels[int(probabilities[0].argmax())]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_file", "speaker_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"release_id": manifest["release_id"], "files": len(rows), "device": chosen,
            "audio_decode_failures": errors, "predictions_file_path": str(output)}
