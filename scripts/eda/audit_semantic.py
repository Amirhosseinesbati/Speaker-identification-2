"""Run bounded offline AST and Whisper EDA; never trains or uploads audio."""
from pathlib import Path
import os
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from speaker_id.eda.semantic import main

if __name__ == "__main__":
    main()
