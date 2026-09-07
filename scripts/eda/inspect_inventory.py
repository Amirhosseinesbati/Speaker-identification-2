"""Reproduce the preliminary data audit without decoding full audio signals.

Run from the project root with Python 3.12+ (or use the absolute script path):
    python scripts/eda/inspect_inventory.py

Only the JSON output is written. Raw inputs are opened read-only. Archive
inspection reads its central directory; it does not decompress or verify CRCs.
"""

from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import struct
import wave
import zipfile


def count_distribution(counts: collections.Counter) -> dict[str, int]:
    """Map files per known speaker to number of speakers."""
    return {
        str(count): speakers
        for count, speakers in sorted(collections.Counter(counts.values()).items())
    }


def duration_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "total_hours": sum(values) / 3600,
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
        "under_1_millisecond_count": sum(value < 0.001 for value in values),
        "under_1_second_count": sum(value < 1 for value in values),
        "under_5_seconds_count": sum(value < 5 for value in values),
    }


def inspect(data_dir: Path, archive_path: Path) -> dict:
    files = sorted(path for path in data_dir.iterdir() if path.is_file())
    audio_files = [path for path in files if path.suffix.lower() == ".mp3"]
    labels_path = data_dir / "labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if columns is None or len(columns) != 2 or set(columns) != {"speaker_id", "audio_file"}:
        raise ValueError(f"Unexpected labels.csv columns: {columns!r}")

    labels_by_file: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        labels_by_file[row["audio_file"]].append(row["speaker_id"])
    audio_names = {path.name for path in audio_files}
    label_names = set(labels_by_file)
    label_counts = collections.Counter(row["speaker_id"] for row in rows)
    known_counts = collections.Counter(
        {speaker: count for speaker, count in label_counts.items() if speaker != "unknown"}
    )
    pair_counts = collections.Counter(
        (row["speaker_id"], row["audio_file"]) for row in rows
    )

    formats = collections.Counter()
    all_durations: list[float] = []
    durations_by_label: dict[str, list[float]] = collections.defaultdict(list)
    non_wave_files = []
    wave_header_errors = []
    riff_size_mismatches = []
    standard_data_size_mismatches = []
    standard_data_size_checked_count = 0
    one_frame_files = []
    one_frame_hashes: dict[str, list[str]] = collections.defaultdict(list)

    for path in audio_files:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(96)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            has_mpeg_sync = any(
                header[index] == 0xFF and header[index + 1] & 0xE0 == 0xE0
                for index in range(len(header) - 1)
            )
            non_wave_files.append(
                {
                    "audio_file": path.name,
                    "speaker_ids": labels_by_file.get(path.name, []),
                    "bytes": size,
                    "starts_with_id3": header[:3] == b"ID3",
                    "mpeg_frame_sync_in_first_96_bytes": has_mpeg_sync,
                    "detected_format": (
                        "MP3 (ID3 and MPEG frame header)"
                        if header[:3] == b"ID3" and has_mpeg_sync
                        else "unidentified; requires decoder inspection"
                    ),
                    "duration_seconds": None,
                    "header_hex_first_96_bytes": header.hex(),
                }
            )
            continue
        try:
            with wave.open(str(path), "rb") as stream:
                channels = stream.getnchannels()
                sample_rate = stream.getframerate()
                sample_width = stream.getsampwidth()
                frames = stream.getnframes()
                compression = stream.getcomptype()
        except (wave.Error, EOFError) as error:
            wave_header_errors.append({"audio_file": path.name, "error": str(error)})
            continue
        formats[("RIFF/WAVE", compression, channels, sample_rate, sample_width * 8)] += 1
        if struct.unpack("<I", header[4:8])[0] + 8 != size:
            riff_size_mismatches.append(path.name)
        if (
            header[12:16] == b"fmt "
            and struct.unpack("<I", header[16:20])[0] == 16
            and header[36:40] == b"data"
        ):
            standard_data_size_checked_count += 1
            if struct.unpack("<I", header[40:44])[0] + 44 != size:
                standard_data_size_mismatches.append(path.name)
        duration = frames / sample_rate
        all_durations.append(duration)
        file_labels = labels_by_file.get(path.name, [])
        label_group = (
            "unknown" if file_labels == ["unknown"] else
            "known" if len(file_labels) == 1 and file_labels[0] else "ambiguous_or_unlabeled"
        )
        durations_by_label[label_group].append(duration)
        if frames == 1:
            # Bound payload hashing to the known tiny cases, not ordinary audio.
            if size > 1024:
                raise ValueError(f"One-frame file unexpectedly exceeds 1 KiB: {path.name}")
            content = path.read_bytes()
            one_frame_files.append(path.name)
            one_frame_hashes[hashlib.sha256(content).hexdigest()].append(path.name)

    one_frame_names = set(one_frame_files)
    after_counts = collections.Counter(
        row["speaker_id"] for row in rows
        if row["speaker_id"] != "unknown" and row["audio_file"] not in one_frame_names
    )
    hash_groups = []
    for digest, names in sorted(one_frame_hashes.items()):
        first_content = (data_dir / names[0]).read_bytes()
        hash_groups.append(
            {
                "sha256": digest,
                "file_count": len(names),
                "bytes_per_file": len(first_content),
                "pcm_data_is_one_zero_stereo_frame": (
                    len(first_content) == 48
                    and first_content[36:44] == b"data\x04\x00\x00\x00"
                    and first_content[44:] == b"\x00\x00\x00\x00"
                ),
                "label_counts": dict(sorted(collections.Counter(
                    label for name in names for label in labels_by_file.get(name, [])
                ).items())),
                "audio_files": names,
            }
        )

    with zipfile.ZipFile(archive_path) as archive:
        archive_entries = archive.infolist()
    archived_files = [entry for entry in archive_entries if not entry.is_dir()]
    archive_counts = collections.Counter(entry.filename for entry in archived_files)
    archived_sizes = {entry.filename: entry.file_size for entry in archived_files}
    disk_sizes = {
        path.relative_to(data_dir.parent).as_posix(): path.stat().st_size
        for path in files
    }
    shared_names = archived_sizes.keys() & disk_sizes.keys()

    return {
        "audit_kind": "initial_inventory",
        "full_eda_completed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {"data_directory": "data/raw", "archive": "data/raw.zip"},
        "scope": {
            "performed": [
                "Directory filenames and sizes; complete CSV schema and label inventory",
                "Audio magic bytes and WAVE headers; header-derived WAVE durations",
                "SHA256 and payload check only for one-frame files of at most 1 KiB",
                "Archive central-directory comparison against extracted filenames and sizes",
            ],
            "not_performed": [
                "Full audio decoding or whole-dataset payload/CRC integrity verification",
                "Genuine MP3 decoding or duration measurement",
                "Hashes/fingerprints of ordinary audio and near-duplicate detection",
                "Stereo channel analysis, silence, clipping, SNR, listening or source-leakage audit",
                "Training, MLflow connection, remote-server connection or dependency installation",
            ],
            "limitations": [
                "Consistent WAVE headers do not establish full waveform validity or usable speech",
                "Archive filename/size agreement does not establish byte-for-byte equality",
                "Unknown is an observed label value, not a verified single speaker identity",
                "Count after excluding one-frame cases is a diagnostic; no raw inputs were changed",
                "The genuine MP3 is excluded from every duration summary",
            ],
        },
        "directory": {
            "file_count": len(files),
            "audio_file_count": len(audio_files),
            "total_bytes_including_csv": sum(path.stat().st_size for path in files),
            "extension_counts": dict(sorted(collections.Counter(path.suffix for path in files).items())),
            "audio_minimum_bytes": min(path.stat().st_size for path in audio_files),
            "audio_maximum_bytes": max(path.stat().st_size for path in audio_files),
            "audio_under_1_kib_count": sum(path.stat().st_size < 1024 for path in audio_files),
        },
        "labels": {
            "columns": columns,
            "row_count": len(rows),
            "unique_labeled_audio_count": len(label_names),
            "unique_label_value_count": len(label_counts),
            "known_speaker_count": len(known_counts),
            "known_row_count": sum(known_counts.values()),
            "unknown_row_count": label_counts["unknown"],
            "unknown_row_fraction": label_counts["unknown"] / len(rows),
            "missing_audio_files": sorted(label_names - audio_names),
            "unlabeled_audio_files": sorted(audio_names - label_names),
            "duplicate_label_row_groups": sum(count > 1 for count in pair_counts.values()),
            "duplicate_audio_filename_groups": sum(len(value) > 1 for value in labels_by_file.values()),
            "empty_speaker_count": sum(not row["speaker_id"] for row in rows),
            "empty_audio_filename_count": sum(not row["audio_file"] for row in rows),
            "known_files_per_speaker_distribution_before_one_frame_exclusion": count_distribution(known_counts),
            "known_files_per_speaker_distribution_after_one_frame_exclusion": count_distribution(after_counts),
            "known_speakers_after_one_frame_exclusion": len(after_counts),
            "known_rows_after_one_frame_exclusion": sum(after_counts.values()),
        },
        "audio_headers": {
            "wave_format_distribution": [
                {"container": key[0], "wave_compression_type": key[1], "channels": key[2],
                 "sample_rate_hz": key[3], "bits_per_sample": key[4], "count": count}
                for key, count in sorted(formats.items())
            ],
            "non_wave_files": non_wave_files,
            "wave_header_errors": wave_header_errors,
            "riff_declared_size_mismatches": riff_size_mismatches,
            "standard_wave_data_size_checked_count": standard_data_size_checked_count,
            "standard_wave_data_size_mismatches": standard_data_size_mismatches,
            "wave_header_duration_summary_excluding_mp3": duration_summary(all_durations),
            "wave_header_duration_by_label_group_excluding_mp3": {
                key: duration_summary(value) for key, value in sorted(durations_by_label.items())
            },
        },
        "one_frame_cases": {
            "file_count": len(one_frame_files),
            "description": "One PCM stereo frame lasting 62.5 microseconds at 16 kHz; not zero-length files",
            "full_dataset_deduplication_performed": False,
            "hash_groups": hash_groups,
        },
        "archive_comparison": {
            "scope": "Central-directory filenames and uncompressed sizes only; no extraction, content hash or CRC verification",
            "archive_bytes": archive_path.stat().st_size,
            "central_directory_entry_count_including_directories": len(archive_entries),
            "central_directory_file_count": len(archived_files),
            "total_uncompressed_file_bytes": sum(entry.file_size for entry in archived_files),
            "duplicate_member_filename_groups": sum(count > 1 for count in archive_counts.values()),
            "archive_members_missing_on_disk": sorted(archived_sizes.keys() - disk_sizes.keys()),
            "disk_files_absent_from_archive": sorted(disk_sizes.keys() - archived_sizes.keys()),
            "size_mismatched_members": sorted(
                name for name in shared_names if archived_sizes[name] != disk_sizes[name]
            ),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data/raw")
    parser.add_argument("--archive", type=Path, default=project_root / "data/raw.zip")
    parser.add_argument("--output", type=Path, default=project_root / "reports/eda/initial_inventory.json")
    args = parser.parse_args()
    result = inspect(args.data_dir.resolve(), args.archive.resolve())
    result["input_paths"] = {
        "data_directory": str(args.data_dir.resolve()),
        "archive": str(args.archive.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    print(f"Audio files: {result['directory']['audio_file_count']}; known speakers: {result['labels']['known_speaker_count']}; one-frame files: {result['one_frame_cases']['file_count']}")


if __name__ == "__main__":
    main()
