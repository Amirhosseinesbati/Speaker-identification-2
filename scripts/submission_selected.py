"""Run the selected CAM++ portable speaker-identification release offline."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src' if (ROOT / 'src/speaker_id').is_dir() else ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--predictions-file-path', type=Path, required=True)
    args = parser.parse_args(argv)
    from speaker_id.inference.selected_runtime import run_submission
    result = run_submission(ROOT, args.data_dir, args.predictions_file_path)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
