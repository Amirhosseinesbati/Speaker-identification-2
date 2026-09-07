"""Combine measured quality and model review candidates without changing labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from speaker_id.eda.triage import run
if __name__ == "__main__":
    run(ROOT/"data/processed/eda_v1/audio_manifest.csv", ROOT/"reports/eda")
