"""Cross-check final EDA artifacts against the original labels and each other."""
import csv
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    reports = ROOT / "reports/eda"
    processed = ROOT / "data/processed/eda_v1"
    labels = read_csv(ROOT / "data/raw/labels.csv")
    manifest_path = processed / "audio_manifest.csv"
    manifest = read_csv(manifest_path)
    folds = read_csv(processed / "folds.csv")
    signal = json.loads((reports / "signal_summary.json").read_text())
    split = json.loads((reports / "split_summary.json").read_text())
    stats = json.loads((reports / "report_stats.json").read_text())
    label_map = json.loads((processed / "label_map.json").read_text())["labels"]
    original = {r["audio_file"]: r["speaker_id"] for r in labels}
    assert len(original) == len(labels), "Duplicate source label rows"
    for table in (manifest, folds):
        assert len(table) == len(original)
        assert {r["audio_file"]: r["speaker_id"] for r in table} == original
    assert len(label_map) == 447 and label_map[0] == "unknown"
    assert set(label_map) == set(original.values())
    for label in label_map[1:]:
        uuid.UUID(label)
    assert signal["coverage"]["limit_applied"] is None
    assert signal["coverage"]["failures"] == 0
    assert signal["coverage"]["file_sha256_count"] == len(original)
    assert signal["coverage"]["pcm_sha256_count"] == len(original)
    assert split["manifest_sha256"] == digest(manifest_path) == stats["manifest_sha256"]
    assert split["folds_sha256"] == digest(processed / "folds.csv")
    assert split["duplicate_pairs_sha256"] == digest(reports / "duplicate_pairs.csv")
    assert stats["signal_summary_sha256"] == digest(reports / "signal_summary.json")
    assert stats["duplicate_summary_sha256"] == digest(reports / "duplicate_summary.json")
    assert stats["report_builder_sha256"] == digest(ROOT / "src/speaker_id/eda/report.py"), "Stale report renderer"
    for name, expected in stats["supplementary_summary_sha256"].items():
        assert digest(reports / name) == expected, f"Stale report supplement: {name}"
    by_name = {r["audio_file"]: r for r in folds}
    for field in ("input_sha256", "pcm_sha256"):
        assignment = {}
        for row in manifest:
            fold = by_name[row["audio_file"]]["fold"]
            assert assignment.setdefault(row[field], fold) == fold, "Exact duplicate crossed fold boundary"
    for row in read_csv(reports / "duplicate_pairs.csv"):
        if row["verified"] == "True":
            assert by_name[row["audio_file_a"]]["fold"] == by_name[row["audio_file_b"]]["fold"]
    for row in manifest:
        if row["usable_for_training"] == "False":
            assert by_name[row["audio_file"]]["train_eligible"] == "False"
        assert by_name[row["audio_file"]]["evaluation_included"] == "True"
    known = set(label_map) - {"unknown"}
    for fold in {r["fold"] for r in folds}:
        train = [r for r in folds if r["fold"] != fold and r["train_eligible"] == "True"]
        valid = [r for r in folds if r["fold"] == fold]
        assert known <= {r["speaker_id"] for r in train}
        assert known <= {r["speaker_id"] for r in valid}
        assert not ({r["group_id"] for r in train} & {r["group_id"] for r in valid})
    for relative, expected in signal["code_sha256"].items():
        assert digest(ROOT / relative) == expected, "Scanner source changed after artifact generation"
    archive = json.loads((reports / "archive_anomalies_summary.json").read_text())
    assert archive["selected_files"] == archive["crc_verified_files"] == archive["matches_extracted_files"]
    assert archive["manifest_sha256"] == digest(manifest_path)
    result = {"status": "passed", "source_label_rows": len(labels), "class_count": len(label_map),
              "fold_count": split["actual_folds"], "all_rows_scoreable": True,
              "exact_and_verified_duplicates_crossing_folds": 0,
              "scanner_code_hashes_match": True,
              "manifest_sha256": digest(manifest_path),
              "checks": "Label mapping, coverage, artifact hashes, raw-label conservation, known-class enrollment, duplicate grouping and targeted archive integrity"}
    (reports / "artifact_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
