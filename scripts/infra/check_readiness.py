"""Combine measured server evidence without starting a training run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.infrastructure.readiness import DEFAULT_EVIDENCE, check_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/train/campp_baseline.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/infrastructure/readiness.json"))
    for name, default in DEFAULT_EVIDENCE.items():
        parser.add_argument(f"--{name}-report", type=Path, default=Path(default))
    arguments = parser.parse_args()
    result = check_readiness(arguments.workspace, arguments.config,
                             evidence_paths={key: getattr(arguments, f"{key}_report") for key in DEFAULT_EVIDENCE},
                             report_path=arguments.report)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
