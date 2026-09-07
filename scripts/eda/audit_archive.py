"""Verify anomalous extracted audio against the original archive payload.

This is a targeted integrity investigation, not a full-archive certification.
ZIP entries are streamed to hashes and never extracted into another directory.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def run():
    manifest = ROOT / "data/processed/eda_v1/audio_manifest.csv"
    archive_path = ROOT / "data/raw.zip"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        reasons = []
        if row.get("has_nonzero_signal") == "False":
            reasons.append("all_zero_signal")
        if row.get("max_channel_rms_dbfs") and float(row["max_channel_rms_dbfs"]) < -50:
            reasons.append("rms_below_minus50_dbfs")
        if row.get("vad_mode3_speech_fraction") and float(row["vad_mode3_speech_fraction"]) < .1:
            reasons.append("vad_mode3_below_10percent")
        if reasons:
            selected.append((row, reasons))
    results = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {p.filename: p for p in archive.infolist()}
        selected.sort(key=lambda item: members["raw/" + item[0]["audio_file"]].header_offset)
        for row, reasons in selected:
            member = members["raw/" + row["audio_file"]]
            digest = hashlib.sha256()
            result = {"audio_file": row["audio_file"], "speaker_id": row["speaker_id"],
                      "selection_reasons": "|".join(reasons), "archive_crc32": f"{member.CRC:08x}",
                      "archive_crc_verified": False, "matches_extracted_sha256": False, "error": ""}
            try:
                with archive.open(member) as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                result["archive_crc_verified"] = True
                result["archive_member_sha256"] = digest.hexdigest()
                result["matches_extracted_sha256"] = digest.hexdigest() == row["input_sha256"]
            except (OSError, zipfile.BadZipFile, RuntimeError) as error:
                result["error"] = f"{type(error).__name__}: {error}"
            results.append(result)
    report = ROOT / "reports/eda"
    columns = ["audio_file", "speaker_id", "selection_reasons", "archive_crc32", "archive_crc_verified", "archive_member_sha256", "matches_extracted_sha256", "error"]
    with (report / "archive_anomalies.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["audio_file"]))
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Targeted ZIP payload SHA256 and CRC verification for all-zero, RMS<-50dBFS or VAD3<0.1 files",
               "full_archive_verified": False, "selected_files": len(results),
               "crc_verified_files": sum(r["archive_crc_verified"] for r in results),
               "matches_extracted_files": sum(r["matches_extracted_sha256"] for r in results),
               "errors": [r for r in results if r["error"] or not r["matches_extracted_sha256"]],
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "archive_size_bytes": archive_path.stat().st_size,
               "interpretation": "Matching hashes establish these observed anomalies already exist in the supplied archive; they do not establish the intended source audio was correct"}
    (report / "archive_anomalies_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
