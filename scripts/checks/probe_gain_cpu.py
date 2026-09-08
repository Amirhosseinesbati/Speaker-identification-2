"""Validate CP001 metadata; only --execute-pilot enables same-server CPU profiling."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from speaker_id.infrastructure.cpu_feasibility import validate_pilot_config, pilot_indices, execute_pilot
    from speaker_id.training.gain_suite import load_gain_inputs
    from speaker_id.packaging.selected_sources import project_file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/infra/campp_cpu_pilot.json')
    parser.add_argument('--execute-pilot', action='store_true')
    args = parser.parse_args()
    path = project_file(ROOT, args.config)
    pilot = json.loads(path.read_text(encoding='utf-8')); validate_pilot_config(pilot)
    suite = json.loads(project_file(ROOT, pilot['source_suite']).read_text(encoding='utf-8'))
    contract, source_config, _ = load_gain_inputs(ROOT, suite)
    if not args.execute_pilot:
        print(json.dumps({'status':'validated_metadata_only','experiment':'CP001','device':'cpu',
            'metadata_indices':pilot_indices(contract['manifest']),'model_execution':False,'new_mlflow_run':False}))
        return
    print(json.dumps(execute_pilot(ROOT,path,pilot,contract,source_config),indent=2))


if __name__ == '__main__':
    main()
