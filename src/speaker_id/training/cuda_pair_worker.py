"""One fresh CUDA worker for C002 identity/gain cache pairs.

The worker owns no scoring decisions.  It refuses CPU execution, C001 paths,
existing cache files and partial output, so an interrupted run remains forensic
evidence rather than an implicit resume point.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from speaker_id.audio.gain import IDENTITY_POLICY
from speaker_id.data.splits import truth
from speaker_id.infrastructure.data import confined_path
from speaker_id.models.campp import file_sha256
from speaker_id.tracking.snapshot import write_json
from speaker_id.training.gain_suite import require, verify_gain_cache


NAMES = ("public", "advanced")
FRONTENDS = ("identity", "gain")
DIMS = {"public": 512, "advanced": 192}


def check_budget(started, maximum, *, now=None):
    if (time.monotonic() if now is None else now) - started >= maximum:
        raise TimeoutError("C002 CUDA extraction soft budget exceeded; partial files are preserved and cannot resume")


def _same_array(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def publish_npz(path, vectors, valid, row, signature):
    """Publish a cache record atomically without replacing a completed file."""
    temporary = path.with_suffix(".partial")
    require(not path.exists() and not temporary.exists(), "Fresh C002 CUDA cache collision")
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            **vectors,
            valid=bool(valid),
            signature=signature,
            audio_file=row["audio_file"],
            audio_sha256=row["input_sha256"],
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "audio_file": row["audio_file"],
        "audio_sha256": row["input_sha256"],
        "cache_file": path.name,
        "bytes": path.stat().st_size,
        "cache_sha256": file_sha256(path),
        "valid": bool(valid),
    }


def validate_pair(vectors, info, valid, policy):
    require(
        set(vectors) == set(DIMS)
        and type(info["nonzero_signal"]) is bool
        and info["nonzero_signal"] == valid
        and info["gain"]["policy"] == policy["name"],
        "C002 frontend validity or policy changed",
    )
    for name, dimension in DIMS.items():
        vector = vectors[name]
        require(
            isinstance(vector, np.ndarray)
            and vector.shape == (dimension,)
            and vector.dtype == np.float32
            and np.isfinite(vector).all()
            and (
                np.isclose(np.linalg.norm(vector), 1, atol=1e-5)
                if valid
                else vector.tobytes() == np.zeros(dimension, dtype=np.float32).tobytes()
            ),
            "Malformed fresh CUDA vector",
        )
    require(
        type(info["gain"]["applied"]) is bool
        and np.isfinite(info["gain"]["gain"])
        and info["gain"]["gain"] >= 1
        and info["gain"]["applied"] == (info["gain"]["gain"] != 1),
        "Invalid C002 gain application flag",
    )
    require(
        (valid and policy["name"] != "identity_v1") or info["gain"]["gain"] == 1,
        "Identity and zero-signal policies cannot change waveform gain",
    )


def require_noop_equal(identity, gain, gain_info):
    if not gain_info["gain"]["applied"]:
        require(
            all(_same_array(identity[name], gain[name]) for name in NAMES),
            "No-op gain changed a fresh CUDA embedding",
        )


def historical_diagnostic(fresh, historical, valid):
    """Record numerical drift only; never turn it into a historical parity gate."""
    result = {}
    for name in NAMES:
        observed = fresh[name].astype(np.float64)
        prior = historical[name].astype(np.float64)
        require(observed.shape == prior.shape and np.isfinite(prior).all(), "Historical diagnostic values are malformed")
        result[name] = {
            "max_abs": float(np.max(np.abs(observed - prior))),
            "l2": float(np.linalg.norm(observed - prior)),
            "cosine": (
                float(np.dot(observed, prior) / (np.linalg.norm(observed) * np.linalg.norm(prior)))
                if valid
                else None
            ),
        }
    return result


def summarize_historical_diagnostics(rows):
    return {
        name: {
            "files": len(rows),
            "nonzero_files": sum(row[name]["cosine"] is not None for row in rows),
            "maximum_absolute_difference": max(row[name]["max_abs"] for row in rows),
            "maximum_l2_difference": max(row[name]["l2"] for row in rows),
            "minimum_nonzero_cosine": min(
                (row[name]["cosine"] for row in rows if row[name]["cosine"] is not None), default=None
            ),
            "diagnostic_only_no_acceptance_threshold": True,
            "historical_embedding_parity_required": False,
        }
        for name in NAMES
    }


def best_effort_hashes(items, hasher):
    hashes, errors = {}, {}
    for name, value in items.items():
        try:
            hashes[name] = hasher(value)
        except Exception as error:  # pragma: no cover - only failure preservation.
            errors[name] = type(error).__name__
    return hashes, errors


def _require_fresh_c002_layout(root, output):
    from speaker_id.training.cuda_pair_contract import require_c002_path

    root = Path(root).resolve()
    output = require_c002_path(root, output, require_existing=True)
    require(output.name.startswith("C002_"), "C002 output directory must have a fresh C002 identifier")
    for child in (
        "identity_embedding_cache",
        "gain_embedding_cache",
        "pair_receipts",
        "identity_cache_identity.json",
        "gain_cache_identity.json",
        "identity_cache_manifest.json",
        "gain_cache_manifest.json",
        "paired_execution_report.json",
        "paired_extraction_progress.json",
        "paired_extraction_failure.json",
    ):
        require(not (output / child).exists(), "C002 caches/evidence are fresh-only; existing or C001 caches cannot be reused")
    return root, output


def extract_cuda_pair_caches(root, suite, contract, sources, output, control, backend, *, progress_callback):
    """Extract C002b/C002c from fresh CUDA inference; do not upload embedding files."""
    from speaker_id.training.cuda_pair_contract import (
        build_cuda_identity,
        require_c002_path,
        validate_cuda_backend,
        validate_cuda_gain_config,
    )
    from speaker_id.training.candidate_comparison import encoder_state_sha256

    validate_cuda_gain_config(suite)
    validate_cuda_backend(backend, suite)
    root, output = _require_fresh_c002_layout(root, output)
    settings = suite["execution"]
    require(settings["device"] == "cuda" and settings["no_cpu_fallback"] is True and settings["no_resume"] is True,
            "C002 execution must remain CUDA-only and fresh-only")
    require(
        all(control.get(key) for key in (
            "exact_prediction_reproduction",
            "exact_pooled_metrics",
            "exact_inner_alpha_curves",
            "exact_probability_and_support_arrays",
        )),
        "C002a historical control must complete before fresh CUDA extraction",
    )
    require(set(sources["assets"]) == set(NAMES), "Missing C002 dual-encoder assets")
    manifest = contract["manifest"]
    require(
        len(manifest) > 0
        and sources["valid"].shape == (len(manifest),)
        and all(sources["vectors"][name].shape == (len(manifest), DIMS[name]) for name in NAMES),
        "C002 source rows/dimensions differ",
    )
    names = [row["audio_file"] for row in manifest]
    require(
        len(set(names)) == len(names)
        and len({Path(name).stem for name in names}) == len(names)
        and all(Path(name).name == name and "/" not in name and "\\" not in name and ":" not in name for name in names),
        "Audio/cache filenames must remain unique and flat",
    )
    caches = {name: require_c002_path(root, output / f"{name}_embedding_cache") for name in FRONTENDS}
    receipts_dir = require_c002_path(root, output / "pair_receipts")
    identities, records, encoders, paths = {}, {name: [] for name in FRONTENDS}, {}, {}
    before, weights_before, drift = {}, {}, []
    started = time.monotonic()
    try:
        identities = {name: build_cuda_identity(root, suite, contract, sources, name, backend) for name in FRONTENDS}
        require(identities["identity"]["signature"] != identities["gain"]["signature"],
                "C002 frontends require independent identities")
        for name in FRONTENDS:
            write_json(output / f"{name}_cache_identity.json", identities[name])
            caches[name].mkdir(exist_ok=False)
        receipts_dir.mkdir(exist_ok=False)
        progress_callback("identities", {"identities": identities})

        paths = {
            name: require_c002_path(root, confined_path(root, asset["config"]["weights_path"]), require_existing=True)
            for name, asset in sources["assets"].items()
        }
        require(all("c001" not in path.as_posix().lower() for path in paths.values()),
                "C002 must not use a C001 model/cache path")
        weights_before = {name: file_sha256(path) for name, path in paths.items()}
        require(
            all(
                weights_before[name]
                == sources["assets"][name]["source_record"]["weights_sha256"]
                == identities["identity"]["model_sources"][name]["weights_sha256"]
                == identities["gain"]["model_sources"][name]["weights_sha256"]
                for name in NAMES
            ),
            "Frozen source weights changed",
        )

        import torch
        from speaker_id.candidates.campp_advanced import load_advanced
        from speaker_id.candidates.gain_frontend import extract_gain_pair
        from speaker_id.models.campp import load_campp

        require(torch.cuda.is_available(), "C002 CUDA disappeared; CPU fallback is prohibited")
        require(torch.get_default_dtype() == torch.float32, "C002 requires FP32 default tensors")
        require(torch.cuda.get_device_name(torch.cuda.current_device()) == backend["device_name"],
                "C002 active GPU changed after preflight")
        encoders["public"] = load_campp(sources["assets"]["public"]["config"], root, "cuda")
        encoders["advanced"] = load_advanced(sources["assets"]["advanced"]["config"], root, "cuda")
        for encoder in encoders.values():
            encoder.requires_grad_(False).eval()
            require(not any(module.training for module in encoder.modules()), "C002 encoders must stay in eval mode")
            require(
                all(
                    tensor.device.type == "cuda"
                    and (not tensor.is_floating_point() or tensor.dtype == torch.float32)
                    for tensor in (*encoder.parameters(), *encoder.buffers())
                ),
                "C002 models/buffers must remain CUDA FP32; CPU fallback is prohibited",
            )
        before = {name: encoder_state_sha256(model) for name, model in encoders.items()}
        noops = 0
        with torch.inference_mode():
            for index, row in enumerate(manifest):
                check_budget(started, settings["maximum_extraction_seconds"])
                raw = confined_path(root, Path(contract["config"]["data_dir"]) / row["audio_file"])
                expected_valid = truth(row["has_nonzero_signal"])
                require(expected_valid == bool(sources["valid"][index]), "Original zero-signal mask changed")
                values, infos, pair_records = {}, {}, {}
                for frontend, policy in (("identity", IDENTITY_POLICY), ("gain", suite["gain_policy"])):
                    require(raw.is_file() and file_sha256(raw) == row["input_sha256"],
                            "Raw input changed before C002 extraction")
                    tick = time.monotonic()
                    values[frontend], infos[frontend] = extract_gain_pair(
                        encoders["public"], encoders["advanced"], raw,
                        device="cuda", policy=policy, **contract["config"]["inference"]
                    )
                    duration = time.monotonic() - tick
                    validate_pair(values[frontend], infos[frontend], expected_valid, policy)
                    require(file_sha256(raw) == row["input_sha256"], "Raw input changed during C002 extraction")
                    if frontend == "gain":
                        require_noop_equal(values["identity"], values["gain"], infos["gain"])
                        noops += int(not infos["gain"]["gain"]["applied"])
                    record = publish_npz(
                        caches[frontend] / f"{Path(row['audio_file']).stem}.npz",
                        values[frontend], expected_valid, row, identities[frontend]["signature"],
                    )
                    record.update(frontend_diagnostics=infos[frontend], elapsed_seconds=duration)
                    records[frontend].append(record)
                    pair_records[frontend] = record
                # New CUDA identity is characterized against S008c but never gated by exact parity.
                drift.append(historical_diagnostic(
                    values["identity"], {name: sources["vectors"][name][index] for name in NAMES}, expected_valid
                ))
                receipt_path = receipts_dir / f"{index:05d}.json"
                require(not receipt_path.exists(), "C002 pair receipt collision")
                write_json(receipt_path, {
                    "index": index,
                    "records": pair_records,
                    "fresh_cuda_identity_vs_s008c_diagnostic": drift[-1],
                    "historical_embedding_parity_required": False,
                })
                progress = {
                    "status": "extracting",
                    "completed_pairs": index + 1,
                    "total": len(manifest),
                    "elapsed_seconds": time.monotonic() - started,
                    "signatures": {name: identity["signature"] for name, identity in identities.items()},
                    "last_pair_receipt": receipt_path.relative_to(output).as_posix(),
                    "last_pair_receipt_sha256": file_sha256(receipt_path),
                    "no_op_gain_pairs": noops,
                    "encoder_updates": 0,
                    "historical_embedding_parity_required": False,
                }
                write_json(output / "paired_extraction_progress.json", progress)
                if (index + 1) % settings["progress_every_pairs"] == 0 or index + 1 == len(manifest):
                    progress_callback("progress", progress)
                    print(json.dumps({"stage": "cuda_paired_extraction", **progress}), flush=True)
                check_budget(started, settings["maximum_extraction_seconds"])

        after = {name: encoder_state_sha256(model) for name, model in encoders.items()}
        weights_after = {name: file_sha256(path) for name, path in paths.items()}
        require(before == after and weights_before == weights_after, "Frozen CUDA tensors or weight files changed")
        receipts = {
            frontend: {
                "schema_version": 1,
                "identity": identities[frontend],
                "file_count": len(manifest),
                "files": records[frontend],
                "encoder_state_sha256_before": before,
                "encoder_state_sha256_after": after,
                "weight_file_sha256_before": weights_before,
                "weight_file_sha256_after": weights_after,
                "elapsed_seconds": time.monotonic() - started,
                "encoder_updates": 0,
            }
            for frontend in FRONTENDS
        }
        vectors, masks = {}, {}
        for frontend in FRONTENDS:
            write_json(output / f"{frontend}_cache_manifest.json", receipts[frontend])
            vectors[frontend], masks[frontend] = verify_gain_cache(
                caches[frontend], identities[frontend], manifest, receipts[frontend]
            )
            require(_same_array(masks[frontend], sources["valid"]), "Completed C002 cache changed original validity")
        require(_same_array(masks["identity"], masks["gain"]), "C002 frontend eligibility differs")
        report = {
            "status": "complete",
            "completed_pairs": len(manifest),
            "elapsed_seconds": time.monotonic() - started,
            "backend": backend,
            "no_op_gain_pairs": noops,
            "no_op_bitwise_parity_verified": True,
            "encoder_state_sha256_before": before,
            "encoder_state_sha256_after": after,
            "weight_file_sha256_before": weights_before,
            "weight_file_sha256_after": weights_after,
            "fresh_cuda_identity_vs_s008c": summarize_historical_diagnostics(drift),
            "historical_embedding_parity_required": False,
            "encoder_updates": 0,
            "raw_audio_sha_before_and_after_each_frontend": True,
            "cache_manifest_sha256": {
                name: file_sha256(output / f"{name}_cache_manifest.json") for name in FRONTENDS
            },
            "deadline_scope": "eight-hour soft budget through complete pair extraction; no implicit retry or resume",
            "recognition_scoring_or_calibration": False,
            "embedding_artifacts_uploaded": False,
        }
        write_json(output / "paired_execution_report.json", report)
        progress_callback("complete", {"identities": identities, "receipts": receipts, "execution_report": report})
        return {
            "vectors": vectors,
            "valid": masks["identity"],
            "identities": identities,
            "receipts": receipts,
            "execution_report": report,
        }
    except Exception as error:
        state_after, state_errors = best_effort_hashes(encoders, encoder_state_sha256)
        weights_after, weight_errors = best_effort_hashes(paths, file_sha256)
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "completed_frontend_files": {name: len(value) for name, value in records.items()},
            "elapsed_seconds": time.monotonic() - started,
            "encoder_updates": 0,
            "no_implicit_retry_or_resume": True,
            "historical_embedding_parity_required": False,
            "identities": identities,
            "encoder_state_sha256_before": before,
            "encoder_state_sha256_after": state_after,
            "weight_file_sha256_before": weights_before,
            "weight_file_sha256_after": weights_after,
            "state_hash_errors": state_errors,
            "weight_hash_errors": weight_errors,
            "files_preserved": True,
        }
        write_json(output / "paired_extraction_failure.json", failure)
        try:
            progress_callback("failure", failure)
        except Exception as callback_error:  # Preserve original failure and its on-disk evidence.
            failure["failure_callback_error_type"] = type(callback_error).__name__
            write_json(output / "paired_extraction_failure.json", failure)
        raise
