"""Build provisional grouped folds after signal and duplication audits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.data.splits import truth, write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--pairs", type=Path, default=ROOT / "reports/eda/duplicate_pairs.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/eda_v1")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/eda/split_summary.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    with args.pairs.open(encoding="utf-8", newline="") as stream:
        pairs = [(r["audio_file_a"], r["audio_file_b"]) for r in csv.DictReader(stream) if truth(r["verified"])]
    summary = write_splits(args.manifest, pairs, args.output_dir, args.summary, args.folds, args.seed)
    summary["duplicate_pairs_sha256"] = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "actual_folds", "file_count", "group_count", "minimum_eligible_known_groups")}, indent=2))
    if summary["status"] == "infeasible":
        print("No folds.csv generated: content groups cannot support validation for every known class; see split_summary.json and split_support.csv")


if __name__ == "__main__":
    main()
