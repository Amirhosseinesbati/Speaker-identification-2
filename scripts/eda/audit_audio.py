"""Thin entrypoint; keeps CPU numerical-library threads bounded per worker."""

import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from speaker_id.eda.audit import main  # noqa: E402

if __name__ == "__main__":
    main()
