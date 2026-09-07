"""Check bounded embedding candidate relationships using actual waveforms."""

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.eda.embedding_overlap import run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/eda_v1/audio_manifest.csv")
    parser.add_argument("--candidates", type=Path, default=ROOT / "reports/eda/embedding_candidate_pairs.csv")
    parser.add_argument("--neighbors", type=Path, default=ROOT / "reports/eda/embedding_neighbors.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/eda/embedding_overlap_pairs.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/eda/embedding_overlap_summary.json")
    args = parser.parse_args()
    run(args.manifest, args.candidates, args.neighbors, args.data_dir, args.output, args.summary)
