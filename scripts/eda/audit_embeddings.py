"""Extract fixed pretrained speaker features for EDA, without model fitting."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extra-site-packages", type=Path, help="Optional read-only existing model runtime; versions recorded")
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--manifest", type=Path, default=ROOT/"data/processed/eda_v1/audio_manifest.csv")
parser.add_argument("--data-dir", type=Path, default=ROOT/"data/raw")
parser.add_argument("--cache", type=Path, default=ROOT/"artifacts/eda/ecapa")
parser.add_argument("--report-dir", type=Path, default=ROOT/"reports/eda")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
if args.limit and (args.report_dir == ROOT/"reports/eda" or args.cache == ROOT/"artifacts/eda/ecapa"):
    parser.error("Limited runs require separate --report-dir and --cache")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = str(args.threads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
if args.extra_site_packages:
    sys.path.insert(0, str(args.extra_site_packages.resolve()))
sys.path.insert(0, str(ROOT/"src"))
from speaker_id.eda.embeddings import run
run(args.manifest, args.data_dir, args.model_dir, args.cache, args.report_dir, args.device, args.threads, args.limit)
