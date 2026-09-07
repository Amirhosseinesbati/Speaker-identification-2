"""Compute descriptive embedding geometry and fold distribution checks."""
import os
from pathlib import Path
import sys
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from speaker_id.eda.geometry import run
if __name__ == "__main__":
    run(ROOT/"data/processed/eda_v1/audio_manifest.csv", ROOT/"data/processed/eda_v1/folds.csv",
        ROOT/"artifacts/eda/ecapa", ROOT/"reports/eda")
