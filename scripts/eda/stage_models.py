"""Stage only public EDA model assets with hashes recorded by the executed audits.

This does not download models, install packages, read prior competition checkpoints,
or modify the provenance of prior executions. --verify-only works from the staged
registry without access to the original project or Hugging Face cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def stage_file(source, destination, expected, purpose):
    relative(destination)  # All write targets must remain in the current workspace.
    if digest(source) != expected:
        raise ValueError(f"Source hash differs from executed audit: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Refusing to overwrite a different staged asset: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".staging")
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        if digest(temporary) != expected:
            raise ValueError(f"Staged temporary file hash differs: {temporary}")
        os.replace(temporary, destination)
    if digest(destination) != expected:
        raise ValueError(f"Final staged hash differs: {destination}")
    return {"path": relative(destination), "sha256": expected, "bytes": destination.stat().st_size,
            "purpose": purpose, "verified": True}


def verify(registry):
    result = {"status": "passed", "verified_files": 0, "verified_bytes": 0}
    for model in registry["models"].values():
        for asset in model["assets"]:
            path = ROOT / asset["path"]
            relative(path)
            if not path.is_file() or digest(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError(f"Missing or changed staged model asset: {path}")
            result["verified_files"] += 1
            result["verified_bytes"] += asset["bytes"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecapa-source", type=Path,
                        default=Path.home() / ".cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286")
    parser.add_argument("--extra-site-packages", type=Path,
                        help="Read package metadata only; never installs or imports this environment")
    parser.add_argument("--registry", type=Path, default=ROOT / "reports/eda/model_assets.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    relative(args.registry)
    if args.verify_only:
        print(json.dumps(verify(json.loads(args.registry.read_text(encoding="utf-8"))), indent=2))
        return
    semantic_path = ROOT / "reports/eda/semantic_summary.json"
    embedding_path = ROOT / "reports/eda/embedding_summary.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    embedding = json.loads(embedding_path.read_text(encoding="utf-8"))
    if set(embedding["model_sha256"]) != {"embedding_model.ckpt", "hyperparams.yaml"}:
        raise ValueError("Unexpected ECAPA assets; review the public-asset allowlist")
    models = {"ecapa": {"repository": embedding["repository"], "revision": embedding["revision"],
                        "execution_source_directory": str(args.ecapa_source.resolve()),
                        "directory": "artifacts/models/ecapa", "assets": []}}
    for name, expected in embedding["model_sha256"].items():
        models["ecapa"]["assets"].append(stage_file(args.ecapa_source / name, ROOT / "artifacts/models/ecapa" / name,
                                                     expected, "public_encoder_weights" if name.endswith(".ckpt") else "public_model_configuration"))
    for key, destination_name in (("ast", "ast"), ("whisper", "whisper_base")):
        info = semantic["models"][key]
        source = Path(info["local_directory"])
        model = {"repository": info["repository"],
                 "revisions_from_cached_download_metadata": info["revision_from_huggingface_download_metadata"],
                 "execution_source_directory": str(source), "directory": f"artifacts/models/{destination_name}", "assets": []}
        for name, expected in info["files_sha256"].items():
            # Summary supplies an explicit allowlist; no recursive model/checkpoint discovery.
            if Path(name).name != name or not name.endswith((".json", ".txt", ".safetensors")):
                raise ValueError(f"Unexpected model asset name: {name}")
            model["assets"].append(stage_file(source / name, ROOT / model["directory"] / name,
                                              expected, "public_weights" if name.endswith(".safetensors") else "public_tokenizer_or_configuration"))
        for name, expected in info["download_metadata_sha256"].items():
            if not name.startswith(".cache/huggingface/download/") or not name.endswith(".metadata") or ".." in Path(name).parts:
                raise ValueError(f"Unexpected provenance metadata name: {name}")
            model["assets"].append(stage_file(source / name, ROOT / model["directory"] / name,
                                              expected, "public_download_provenance_metadata"))
        models[destination_name] = model
    inventory = None
    if args.extra_site_packages:
        distributions = importlib.metadata.distributions(path=[str(args.extra_site_packages.resolve())])
        packages = sorted(({"name": dist.metadata["Name"], "version": dist.version}
                           for dist in distributions), key=lambda item: item["name"].lower())
        inventory = {"kind": "installed_distribution_metadata_inventory_not_dependency_lock",
                     "source_site_packages": str(args.extra_site_packages.resolve()),
                     "python_executable_used_for_audits": str(Path(sys.executable).resolve()),
                     "python_version": platform.python_version(),
                     "package_count": len(packages), "packages": packages,
                     "limitations": ["Read-only package metadata inventory; no credential files or direct URLs recorded.",
                                     "Includes unused packages from the external environment; not a minimal requirements list.",
                                     "No wheel hashes, resolver replay, clean-environment installation, or Linux compatibility verification."]}
        inventory_path = ROOT / "reports/eda/model_runtime_inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        inventory = {"path": relative(inventory_path), "sha256": digest(inventory_path), "package_count": len(packages)}
    registry = {"version": "public-eda-model-assets-v1", "models": models,
                "source_execution_reports": {relative(p): digest(p) for p in (semantic_path, embedding_path)},
                "execution_provenance_was_rewritten": False,
                "model_training_performed": False, "network_or_download_used": False,
                "runtime_inventory": inventory,
                "execution_versions": {"embedding": embedding["versions"], "semantic": semantic["runtime"]},
                "environment_status": "Research audit runtime only; not the leaderboard environment and not submission smoke-tested.",
                "known_leaderboard_range_conflicts": {"numpy": {"used": "2.4.6", "guide": ">=2,<2.3.0"},
                                                      "scipy": {"used": "1.18.0", "guide": "<1.16"},
                                                      "soundfile": {"used": "0.14.0", "guide": ">=0.13.1,<0.14.0"}},
                "portable_reverification_command": ".venv/Scripts/python.exe scripts/eda/stage_models.py --verify-only",
                "notes": ["Staged assets are ignored local artifacts; copy these directories explicitly when transferring the project.",
                          "Only the three public pretrained models used in EDA are included; no competition-finetuned checkpoint is staged.",
                          "Original execution paths in the EDA reports remain truthful historical provenance.",
                          "Same file hashes do not certify a new environment; clean runtime replay and leaderboard validation remain separate tasks."]}
    registry["verification"] = verify(registry)
    args.registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(registry["verification"], indent=2))


if __name__ == "__main__":
    main()
