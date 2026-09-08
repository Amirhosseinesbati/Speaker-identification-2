"""Validate the fixed S010 hypothesis; GPU work requires --execute."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from speaker_id.training.gain_suite import load_gain_inputs, execute_gain_suite
    from speaker_id.training.fusion_suite import project_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    path = project_path(ROOT, args.config, 'configs/train')
    suite = json.loads(path.read_text(encoding='utf-8'))
    contract, source_config, _ = load_gain_inputs(ROOT, suite)
    if not args.execute:
        print(json.dumps({'status': 'validated_no_experiment_started', 'experiment': 'S010',
            'candidate_frontends': 1, 'gain_policy': suite['gain_policy'], 'encoder_training': False}))
        return
    print(json.dumps(execute_gain_suite(ROOT, path, suite, contract, source_config,
        ROOT / 'artifacts/infrastructure/mlflow_state.json'), indent=2))


if __name__ == '__main__':
    main()
