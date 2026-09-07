"""Build the final Persian computational EDA report after all essential audits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.eda.final_report import build_final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stats = build_final_report(args.root, args.output_dir)
    print(json.dumps({key: stats[key] for key in ("status", "source_files", "embedded_files", "human_listening_completed", "optional_summaries_included")}, ensure_ascii=False, indent=2))
