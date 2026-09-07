"""Readiness checks for the server runtime; no optimizer or training is run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.data import confined_path, sha256_file, write_json_atomic


def command(arguments: list[str], *, cwd: Path, timeout: int = 60) -> dict:
    try:
        result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=Path("artifacts/infrastructure/runtime_readiness.json"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpu", default=None,
                        help="GPU name substring, matched ignoring spaces/case, e.g. RTX3090")
    parser.add_argument("--min-free-disk-gb", type=float, default=10)
    parser.add_argument("--min-gpu-memory-gb", type=float, default=20)
    parser.add_argument("--require-import", action="append", default=[])
    parser.add_argument("--leaderboard-packages", type=Path,
                        default=Path("Competition-Guide/leaderbordpakage.txt"))
    parser.add_argument("--skip-audio-check", action="store_true",
                        help="Only for bootstrap diagnostics before transferring data")
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    report = confined_path(workspace, args.report)
    if not report.is_relative_to(workspace / "artifacts/infrastructure"):
        parser.error("Report must be under artifacts/infrastructure")
    errors = []
    result = {"status": "failed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "workspace": str(workspace),
              "python": {"version": platform.python_version(), "executable": sys.executable},
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "packages": {}, "imports": {}, "checks": {}}
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required for parity with the leaderboard")
    disk = shutil.disk_usage(workspace)
    result["disk"] = {"total_bytes": disk.total, "free_bytes": disk.free,
                      "minimum_free_gb": args.min_free_disk_gb}
    if disk.free < args.min_free_disk_gb * 1024**3:
        errors.append("Insufficient free disk space for the requested readiness threshold")
    memory_file = Path("/proc/meminfo")
    if memory_file.exists():
        result["host_memory_kib"] = {line.split(":", 1)[0]: int(line.split()[1])
                                     for line in memory_file.read_text().splitlines()
                                     if line.startswith(("MemTotal:", "MemAvailable:"))}
    modules = {}
    for name in dict.fromkeys(["numpy", "scipy", "soundfile", "torch", "torchaudio", "mlflow",
                              *args.require_import]):
        try:
            modules[name] = importlib.import_module(name)
            result["imports"][name] = "passed"
        except Exception as error:
            result["imports"][name] = {"status": "failed", "error": str(error)}
            errors.append(f"Required import failed: {name}")
    for distribution in sorted(importlib.metadata.distributions(),
                               key=lambda item: (item.metadata.get("Name") or "").lower()):
        name = distribution.metadata.get("Name")
        if name:
            result["packages"][name] = distribution.version
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.utils import canonicalize_name
        contract_file = confined_path(workspace, args.leaderboard_packages)
        contract = tomllib.loads(contract_file.read_text(encoding="utf-8-sig"))
        constraints = {canonicalize_name(requirement.name): requirement.specifier
                       for line in contract["project"]["dependencies"]
                       for requirement in [Requirement(line)]}
        comparisons = []
        for name in ("numpy", "scipy", "soundfile", "torch", "torchaudio", "mlflow"):
            actual_distribution = name
            try:
                installed = importlib.metadata.version(actual_distribution)
            except importlib.metadata.PackageNotFoundError:
                if name != "mlflow":
                    raise
                # The official lightweight client exports the same mlflow module.
                actual_distribution = "mlflow-skinny"
                installed = importlib.metadata.version(actual_distribution)
            specifier = constraints[canonicalize_name(name)]
            passed = Version(installed) in specifier
            comparisons.append({"guide_distribution": name, "actual_distribution": actual_distribution,
                                "installed": installed,
                                "leaderboard_constraint": str(specifier), "passed": passed})
            if not passed:
                errors.append(f"Core runtime version outside leaderboard range: {name} {installed} {specifier}")
        if Version(importlib.metadata.version("torch")).base_version != Version(
                importlib.metadata.version("torchaudio")).base_version:
            errors.append("Torch and torchaudio base versions differ")
        result["checks"]["leaderboard_core_versions"] = {
            "source_sha256": sha256_file(contract_file), "comparisons": comparisons,
            "scope": "core imported runtime packages; source contains ranges, not an exact server freeze"}
    except Exception as error:
        errors.append(f"Leaderboard core version validation failed: {error}")
    # pip check checks installed dependency metadata, not just successful imports.
    result["checks"]["pip_check"] = command([sys.executable, "-m", "pip", "check"], cwd=workspace)
    if result["checks"]["pip_check"]["returncode"] != 0:
        errors.append("python -m pip check failed (pip must exist in the runtime)")
    result["nvidia_smi"] = command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
                                     "--format=csv,noheader,nounits"], cwd=workspace, timeout=15)
    torch = modules.get("torch")
    if torch is not None:
        gpu = {"available": bool(torch.cuda.is_available()), "torch_cuda_version": torch.version.cuda,
               "torch_cudnn_version": torch.backends.cudnn.version(), "devices": []}
        result["cuda"] = gpu
        if gpu["available"]:
            try:
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    gpu["devices"].append({"index": index, "name": props.name,
                                           "total_memory_bytes": props.total_memory,
                                           "compute_capability": [props.major, props.minor]})
                # A tiny arithmetic check allocates no model and has no gradient.
                with torch.inference_mode():
                    probe = torch.ones((16, 16), device="cuda")
                    gpu["arithmetic_check_passed"] = bool(torch.all((probe @ probe) == 16).item())
                torch.cuda.synchronize()
                del probe
                torch.cuda.empty_cache()
                if not gpu["arithmetic_check_passed"]:
                    errors.append("CUDA arithmetic probe failed")
            except Exception as error:
                errors.append(f"CUDA initialization/arithmetic failed: {error}")
        elif args.require_cuda:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
        if args.expected_gpu:
            expected = "".join(args.expected_gpu.lower().split())
            if not any(expected in "".join(device["name"].lower().split()) for device in gpu["devices"]):
                errors.append(f"Required GPU name not found: {args.expected_gpu}")
        if args.require_cuda and not any(device["total_memory_bytes"] >= args.min_gpu_memory_gb * 1024**3
                                         for device in gpu["devices"]):
            errors.append("Required GPU memory capacity not available")
    if not args.skip_audio_check:
        try:
            manifest = workspace / "data/processed/eda_v1/audio_manifest.csv"
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            selected = {}
            for row in rows:
                key = (row["detected_format"], row["sample_rate_hz"], row["channels"])
                if key not in selected:
                    selected[key] = row
            probes = []
            sf = modules["soundfile"]
            np = modules["numpy"]
            for row in selected.values():
                source = confined_path(workspace, Path("data/raw") / row["audio_file"])
                with sf.SoundFile(source) as stream:
                    header = {"format": stream.format, "sample_rate_hz": stream.samplerate,
                              "channels": stream.channels, "frames": stream.frames}
                    samples = stream.read(min(stream.frames, stream.samplerate * 3),
                                          dtype="float32", always_2d=True)
                if (header["format"] != row["detected_format"]
                        or header["sample_rate_hz"] != int(row["sample_rate_hz"])
                        or header["frames"] != int(row["decoded_frames"])
                        or header["channels"] != int(row["channels"])
                        or not np.isfinite(samples).all() or samples.shape[0] == 0):
                    raise ValueError(f"SoundFile format/decode probe failed: {row['audio_file']}")
                probes.append({"audio_file": row["audio_file"], **header,
                               "probe_frames_read": len(samples)})
            result["checks"]["audio_decode"] = {"status": "passed", "probes": probes,
                                                 "libsndfile_version": sf.__libsndfile_version__,
                                                 "manifest_sha256": sha256_file(manifest),
                                                 "scope": "one file per observed container/rate/channel combination"}
        except Exception as error:
            errors.append(f"Audio decode readiness failed: {error}")
            result["checks"]["audio_decode"] = {"status": "failed", "error": str(error)}
    else:
        result["checks"]["audio_decode"] = {"status": "skipped", "reason": "bootstrap before data transfer"}
    commit = command(["git", "rev-parse", "HEAD"], cwd=workspace)
    result["git_commit"] = commit.get("stdout") if commit["returncode"] == 0 else None
    status = command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=workspace)
    result["git_worktree_clean"] = status["returncode"] == 0 and not status.get("stdout")
    result["git_cleanliness_scope"] = "tracked and untracked files; Git-ignored artifacts and secrets excluded"
    if commit["returncode"] != 0 or not result["git_worktree_clean"]:
        errors.append("A committed Git worktree with no changes or untracked files is required")
    result["errors"] = errors
    result["status"] = "passed" if not errors else "failed"
    write_json_atomic(report, result)
    print(json.dumps({"status": result["status"], "report": str(report), "errors": errors,
                      "training_started": False}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
