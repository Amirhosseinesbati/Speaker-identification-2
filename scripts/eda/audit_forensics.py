"""Read-only deeper PCM forensics on suspicious files and matched controls."""

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.eda.forensics import run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/eda/forensics.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/eda/forensics_summary.json")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports/eda/figures")
    args = parser.parse_args()
    run(args.manifest, args.data_dir, args.output, args.summary, args.figures_dir)
