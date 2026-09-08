"""Validate P002 configuration; fitting/building requires an explicit server flag."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from speaker_id.packaging.selected_sources import validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, default=Path('artifacts/infrastructure/mlflow_state.json'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    validate_config(config)
    if not args.execute:
        print(json.dumps({'status': 'schema_valid_only', 'recipe': config['selection']['recipe_id'], 'executed': False}))
        return
    from speaker_id.packaging.selected import execute_build
    print(json.dumps(execute_build(ROOT, args.config, args.binding)))


if __name__ == '__main__':
    main()
