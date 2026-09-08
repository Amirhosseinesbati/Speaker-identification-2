"""Validate S009 by default; completed verified S008/S007/S006 evidence is required."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    from speaker_id.training.adapted_candidate_fusion import load_contracts, execute
    from speaker_id.training.fusion_suite import project_path
    path = project_path(ROOT, args.config, 'configs/train')
    suite = json.loads(path.read_text(encoding='utf-8'))
    contracts = load_contracts(ROOT, suite, verify_audio=args.execute)
    if not args.execute:
        print(json.dumps({'status': 'validated_no_experiment_started', 'experiment': 'S009',
            'completed_sources_verified': False, 'execution_policy': suite['execution_policy']}))
        return
    print(json.dumps(execute(ROOT, path, suite, contracts, ROOT / 'artifacts/infrastructure/mlflow_state.json'), indent=2))


if __name__ == '__main__':
    main()
