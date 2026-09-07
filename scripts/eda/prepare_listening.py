"""Export a small review queue; no automatic semantic or speaker labels."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speaker_id.eda.review_samples import prepare_review

if __name__ == "__main__":
    print(json.dumps(prepare_review(ROOT / "data/processed/eda_v1/audio_manifest.csv",
                                  ROOT / "data/raw", ROOT / "reports/eda/audio_samples"), indent=2))
