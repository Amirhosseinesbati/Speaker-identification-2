"""Audit exact and acoustically verified duplicate audio without modifying raw data."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from speaker_id.eda.duplicates import (
    FingerprintConfig, acoustic_components, cache_signature, exact_groups,
    find_candidates, load_waveform, spectral_landmarks, verify_alignment,
)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    config = FingerprintConfig(max_candidates=args.max_candidates)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["audio_file"])
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row["file_sha256"] = row.get("input_sha256", row.get("file_sha256", ""))
        row["decoded_sha256"] = row.get("pcm_sha256", row.get("decoded_sha256", ""))
        row["_no_signal"] = row.get("has_nonzero_signal", "").lower() == "false"
    groups = exact_groups(rows)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(rows, config)
    cache_path = args.cache_dir / "duplicate_fingerprints.npz"
    metadata_path = args.cache_dir / "duplicate_fingerprints.json"
    used_cache = False
    errors = []
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(cache_path) as cached:
                hashes, files, anchors = cached["hashes"], cached["files"], cached["anchors"]
            errors = metadata.get("errors", [])
            used_cache = True
    if not used_cache:
        hash_chunks, file_chunks, anchor_chunks = [], [], []
        file_metadata = []
        for index, row in enumerate(rows):
            if index % 200 == 0:
                print(f"fingerprints {index}/{len(rows)} elapsed={time.monotonic() - started:.1f}s", flush=True)
            if row.get("status", "ok") != "ok" or row["_no_signal"]:
                file_metadata.append({"audio_file": row["audio_file"], "landmarks": 0, "skipped": "decode_error_or_no_signal"})
                continue
            try:
                waveform = load_waveform(args.data_dir / row["audio_file"], config)
                file_hashes, file_anchors = spectral_landmarks(waveform, config)
                hash_chunks.append(file_hashes)
                anchor_chunks.append(file_anchors)
                file_chunks.append(np.full(len(file_hashes), index, dtype=np.uint16 if len(rows) < 65536 else np.uint32))
                file_metadata.append({"audio_file": row["audio_file"], "landmarks": len(file_hashes), "fingerprint_seconds": len(waveform) / config.sample_rate})
            except Exception as error:
                errors.append({"audio_file": row["audio_file"], "error": f"{type(error).__name__}: {error}"})
        hashes = np.concatenate(hash_chunks) if hash_chunks else np.empty(0, dtype=np.uint32)
        anchors = np.concatenate(anchor_chunks) if anchor_chunks else np.empty(0, dtype=np.uint32)
        files = np.concatenate(file_chunks) if file_chunks else np.empty(0, dtype=np.uint16)
        np.savez_compressed(cache_path, hashes=hashes, files=files, anchors=anchors)
        metadata = {"signature": signature, "config": asdict(config), "files": file_metadata, "errors": errors}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"building landmark candidates from {len(hashes)} landmarks elapsed={time.monotonic() - started:.1f}s", flush=True)
    candidates, candidate_stats = find_candidates(hashes, files, anchors, config)
    verified_pairs = []
    # At most two full waveforms are retained at once during candidate verification.
    verification_errors = []
    for number, candidate in enumerate(candidates):
        a, b = rows[candidate["file_index_a"]], rows[candidate["file_index_b"]]
        candidate.update({"audio_file_a": a["audio_file"], "audio_file_b": b["audio_file"], "speaker_id_a": a["speaker_id"], "speaker_id_b": b["speaker_id"], "label_conflict": a["speaker_id"] != b["speaker_id"]})
        try:
            waveform_a = load_waveform(args.data_dir / a["audio_file"], config)
            waveform_b = load_waveform(args.data_dir / b["audio_file"], config)
            candidate.update(verify_alignment(waveform_a, waveform_b, candidate["candidate_offset_seconds"], config, candidate["vote_start_seconds_a"], candidate["vote_end_seconds_a"]))
        except Exception as error:
            candidate.update({"verified": False, "verification_reason": "decode_error"})
            verification_errors.append({"audio_file_a": a["audio_file"], "audio_file_b": b["audio_file"], "error": str(error)})
        if candidate["verified"]:
            verified_pairs.append((a["audio_file"], b["audio_file"]))
        if number % 100 == 0:
            print(f"verified candidates {number + 1}/{len(candidates)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    edges = list(verified_pairs)
    for group in groups:
        if group["use_for_split_grouping"]:
            names = group["audio_files"]
            edges.extend((names[0], name) for name in names[1:])
    row_by_name = {row["audio_file"]: row for row in rows}
    for index, names in enumerate(acoustic_components(list(row_by_name), edges)):
        labels = sorted({row_by_name[name]["speaker_id"] for name in names})
        groups.append({"group_id": f"acoustic_{index:05d}", "kind": "verified_acoustic_component", "hash": "", "no_signal": False, "use_for_split_grouping": True, "label_conflict": len(labels) > 1, "audio_files": names, "speaker_ids": labels})
    zero_landmark_files = [item["audio_file"] for item in metadata["files"] if item["landmarks"] == 0 and "skipped" not in item]
    zero_landmark_rows = [row_by_name[name] for name in zero_landmark_files]
    zero_landmark_durations = [float(row.get("duration_seconds") or 0) for row in zero_landmark_rows]
    zero_landmark_rms = np.asarray([float(row.get("max_channel_rms_dbfs") or "nan") for row in zero_landmark_rows])
    finite_rms = zero_landmark_rms[np.isfinite(zero_landmark_rms)]
    rms_summary = {"finite_count": len(finite_rms)}
    if len(finite_rms):
        rms_summary.update(dict(zip(("minimum", "p05", "median", "p95", "maximum"), map(float, np.quantile(finite_rms, [0, 0.05, 0.5, 0.95, 1])))))
    verified_nonexact_pairs = [
        (first, second) for first, second in verified_pairs
        if row_by_name[first]["file_sha256"] != row_by_name[second]["file_sha256"]
        and row_by_name[first]["decoded_sha256"] != row_by_name[second]["decoded_sha256"]
    ]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pair_columns = ["audio_file_a", "audio_file_b", "speaker_id_a", "speaker_id_b", "label_conflict", "landmark_votes", "vote_span_seconds", "vote_start_seconds_a", "vote_end_seconds_a", "candidate_offset_seconds", "verified", "verification_reason", "overlap_seconds", "verification_region_start_seconds_a", "verification_region_end_seconds_a", "verified_offset_seconds", "verification_correlations", "verification_offsets_seconds"]
    write_csv(args.report_dir / "duplicate_pairs.csv", candidates, pair_columns)
    memberships = []
    for group in groups:
        for name in group["audio_files"]:
            memberships.append({**{key: group[key] for key in ("group_id", "kind", "hash", "no_signal", "use_for_split_grouping", "label_conflict")}, "group_size": len(group["audio_files"]), "audio_file": name, "speaker_id": row_by_name[name]["speaker_id"]})
    write_csv(args.report_dir / "duplicate_groups.csv", memberships, ["group_id", "kind", "hash", "group_size", "no_signal", "use_for_split_grouping", "label_conflict", "audio_file", "speaker_id"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "algorithm_version": 1,
        "scope": "bounded exact-hash and spectral-landmark acoustic duplication audit",
        "input_manifest": str(args.manifest.resolve()), "file_count": len(rows),
        "input_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "config": asdict(config), "cache_signature": signature, "cache_used": used_cache,
        "elapsed_seconds": time.monotonic() - started, "candidate_search": candidate_stats,
        "fingerprint_coverage": {
            "analyzed_signal_files": sum("skipped" not in item for item in metadata["files"]),
            "indexed_files_with_landmarks": sum(item["landmarks"] > 0 for item in metadata["files"]),
            "analyzed_signal_seconds_at_4khz": sum(item.get("fingerprint_seconds", 0) for item in metadata["files"]),
            "indexed_signal_seconds_at_4khz": sum(item.get("fingerprint_seconds", 0) for item in metadata["files"] if item["landmarks"] > 0),
            "signal_files_without_landmarks_count": len(zero_landmark_files),
            "signal_files_without_landmarks": zero_landmark_files,
            "zero_landmark_duration_under_1s_count": sum(value < 1 for value in zero_landmark_durations),
            "zero_landmark_duration_at_least_1s_count": sum(value >= 1 for value in zero_landmark_durations),
            "zero_landmark_known_count": sum(row["speaker_id"] != "unknown" for row in zero_landmark_rows),
            "zero_landmark_unknown_count": sum(row["speaker_id"] == "unknown" for row in zero_landmark_rows),
            "zero_landmark_max_channel_rms_dbfs_summary": rms_summary,
            "zero_landmark_rms_below_minus50_dbfs_count": int(np.sum(zero_landmark_rms < -50)),
        },
        "fingerprint_errors": errors, "verification_errors": verification_errors,
        "no_signal_files": sum(row["_no_signal"] for row in rows),
        "exact_file_groups": sum(group["kind"] == "file_sha256" for group in groups),
        "exact_decoded_groups": sum(group["kind"] == "decoded_sha256" for group in groups),
        "no_signal_exact_file_groups": sum(group["kind"] == "file_sha256" and group["no_signal"] for group in groups),
        "signal_exact_file_groups": sum(group["kind"] == "file_sha256" and not group["no_signal"] for group in groups),
        "verified_acoustic_pairs": len(verified_pairs),
        "verified_nonexact_acoustic_pairs": len(verified_nonexact_pairs),
        "verified_acoustic_components": sum(group["kind"] == "verified_acoustic_component" for group in groups),
        "verified_cross_label_pairs": sum(bool(row["verified"] and row["label_conflict"]) for row in candidates),
        "unverified_candidates": sum(not row["verified"] for row in candidates),
        "groups": groups,
        "limitations": [
            "No result establishes exhaustive source/session grouping or speaker identity.",
            "Landmarks use the strongest channel resampled to 4 kHz; pitch/tempo shifts, heavy processing and different channel content can be missed.",
            "Nonzero recordings shorter than 1 second or without sufficiently strong spectral peaks can yield zero landmarks; these are listed in fingerprint_coverage.",
            "The absolute 1e-5 spectral/amplitude floor limits sensitivity to very low-level audio. Moderate gain robustness in tests does not establish invariance at all amplitude scales.",
            "Frequent hashes, overlaps below 4 seconds and candidates beyond the recorded cap are excluded from acoustic confirmation.",
            "Verification requires at least two consistent waveform windows at >=0.985 absolute Pearson correlation; positive matches support shared audio, negative matches are inconclusive.",
            "Transitive acoustic components should stay within a split; no-signal exact conflicts are explicitly excluded from ordinary components.",
            "File and decoded hash groups overlap by design and must not be summed as independent groups.",
        ],
    }
    (args.report_dir / "duplicate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console_summary = {key: value for key, value in summary.items() if key not in {"groups", "limitations", "config", "fingerprint_coverage"}}
    console_summary["fingerprint_coverage"] = {key: value for key, value in summary["fingerprint_coverage"].items() if key != "signal_files_without_landmarks"}
    print(json.dumps(console_summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/eda")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/eda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=1000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
