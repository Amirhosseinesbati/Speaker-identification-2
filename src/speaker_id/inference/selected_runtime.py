"""Selected CAM++ release runtime; local assets only, no training or tracking."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import uuid

import numpy as np

from .scoring import score_embeddings, validate_calibration, validate_gallery
from .selected_policy import (ADAPTED_PROTOCOL, FROZEN_PROTOCOL, ARCHITECTURE_512, F004_SOURCE, INFERENCE, POLICIES, ROLES_SHA256,
    is_digest, paired_embedding, require, state_dict_sha256, validate_adapted_config, validate_policy, validate_public_config)


class IntegrityError(ValueError):
    """A selected-model package is incomplete, inconsistent or changed."""


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _read(path):
    def invalid(value):
        raise IntegrityError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def _no_links(path):
    if any(_link(parent) for parent in (path, *path.parents)):
        raise IntegrityError('Linked paths are forbidden for package, input and output files')


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise IntegrityError('Payload paths must be relative POSIX filenames')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name or any(part.startswith('.') for part in path.parts):
        raise IntegrityError('Unsafe or noncanonical payload path')
    return path


def _file(root, name):
    path = root.joinpath(*_relative(name).parts)
    _no_links(path)
    if not path.resolve().is_relative_to(root) or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise IntegrityError('A confined regular payload file is required: ' + name)
    return path


def _source_files(prefix, components):
    names = {'__init__.py', 'inference/__init__.py', 'inference/selected_runtime.py', 'inference/selected_policy.py',
        'inference/scoring.py', 'models/__init__.py', 'models/campp.py', 'models/vendor/__init__.py',
        'models/vendor/campplus/__init__.py', 'models/vendor/campplus/DTDNN.py', 'models/vendor/campplus/layers.py',
        'models/vendor/campplus/LICENSE', 'models/vendor/campplus/PROVENANCE.json'}
    if 'advanced' in components:
        names |= {'candidates/__init__.py', 'candidates/campp_advanced.py'}
    return {prefix + '/' + name for name in names}


def _provenance(provenance, manifest, policy, models, root):
    require(isinstance(provenance, dict) and provenance.get('schema_version') == 1
        and provenance.get('release_id') == manifest['release_id'] and provenance.get('policy') == policy,
        'Provenance release/policy identity differs')
    components = set(models)
    require(set(provenance.get('model_config_sha256', {})) == components
            and set(provenance.get('model_sources', {})) == components, 'Provenance model coverage differs')
    for key, config in models.items():
        require(provenance['model_config_sha256'][key] == _sha(root / policy['model_configs'][key]), 'Model config provenance SHA differs')
        source = provenance['model_sources'][key]
        if key == 'adapted':
            expected = {**F004_SOURCE, 'weights_sha256': config['weights_sha256'], 'encoder_state_sha256': config['encoder_state_sha256']}
            require(source == expected, 'Adapted checkpoint/tensor provenance differs from fixed F004 fold0')
        else:
            expected_kind = 'public_voxceleb_512' if key == 'public' else 'advanced_public_192'
            require(isinstance(source, dict) and source.get('encoder_kind') == expected_kind
                and source.get('weights_sha256') == config['weights_sha256'] and type(source.get('encoder_updates')) is int
                and source['encoder_updates'] == 0 and is_digest(source.get('source_parent_run_id'), 32)
                and is_digest(source.get('source_signature')) and is_digest(source.get('source_git_commit'), 40),
                'Frozen public source provenance is incomplete or mismatched')
    selection = provenance.get('selection')
    require(isinstance(selection, dict) and all(isinstance(selection.get(key), str) and selection[key].strip()
        for key in ('experiment_code', 'recipe_id')) and is_digest(selection.get('parent_run_id'), 32)
        and is_digest(selection.get('report_sha256')), 'Completed selection receipt is required')
    families = {
        'public_only': ('S002', 'S002f', FROZEN_PROTOCOL, {'public_voxceleb_512'}),
        'adapted_only': ('S006', 'S006f', ADAPTED_PROTOCOL, {'adapted_f004_fold0_512'}),
        'advanced_only': ('S007', 'S007b', FROZEN_PROTOCOL, {'advanced_public_192'}),
        'public_advanced': ('S008', 'S008c', FROZEN_PROTOCOL, {'public_voxceleb_512', 'advanced_public_192', 'paired_public512_advanced192_704'}),
        'adapted_advanced': ('S009', 'S009d', ADAPTED_PROTOCOL, {'adapted_f004_fold0_512', 'advanced_public_192', 'paired_f004_advanced_704'}),
    }
    require(isinstance(selection.get('family'), str) and selection['family'] in families, 'Unknown selected procedure family')
    experiment, recipe, protocol, kinds = families[selection['family']]
    require(selection['experiment_code'] == experiment and selection['recipe_id'] == recipe
            and policy['calibration_protocol'] == protocol and policy['kind'] in kinds, 'Model pruning changed the selected procedure or calibration family')
    expected = {'protocol': policy['calibration_protocol'], 'source_files': 4529, 'known_reference_files': 2217,
        'unknown_reference_files': 2223, 'zero_signal_files': 89,
        'calibration_query_files': 999 if protocol == ADAPTED_PROTOCOL else 4440, 'roles_sha256': ROLES_SHA256,
        'encoder_fit_queries_used': 0, 'whole_query_group_excluded': True, 'metric_scope': 'fitted_calibration_not_oof'}
    calibration = provenance.get('calibration')
    require(isinstance(calibration, dict) and all(type(calibration.get(key)) is type(value)
            and calibration[key] == value for key, value in expected.items()), 'Final calibration scope/support is incompatible with selected encoders')
    if protocol == ADAPTED_PROTOCOL:
        require(provenance.get('procedure_history') == {'adaptation_source': F004_SOURCE, 'roles_sha256': ROLES_SHA256,
                'final_encoder_excludes_adapted': 'adapted' not in components}, 'Original Q0 procedure/adaptation history must survive endpoint pruning')


def verify_payload(root, *, allowed_output=None):
    root = Path(root).absolute()
    _no_links(root)
    root = root.resolve()
    manifest = _read(_file(root, 'manifest.json'))
    require(isinstance(manifest, dict) and type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
        and isinstance(manifest.get('release_id'), str) and manifest['release_id'] and isinstance(manifest.get('files'), dict)
        and manifest['files'], 'Invalid selected-model manifest')
    files, seen = manifest['files'], set()
    for name, record in files.items():
        _relative(name)
        require(name != 'manifest.json' and name.casefold() not in seen, 'Duplicate/case-colliding/self-referencing payload entry')
        seen.add(name.casefold())
        require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'} and type(record['bytes']) is int
                and record['bytes'] >= 0 and is_digest(record['sha256']), 'Invalid integrity record')
        path = _file(root, name)
        require(path.stat().st_size == record['bytes'] and _sha(path) == record['sha256'], 'Payload digest mismatch: ' + name)
    actual = set()
    for path in root.rglob('*'):
        if _link(path):
            raise IntegrityError('Linked file or directory in payload')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    expected = set(files) | {'manifest.json'}
    if allowed_output is not None:
        output = Path(allowed_output).absolute()
        _no_links(output)
        output = output.resolve()
        if output.is_relative_to(root) and output.is_file():
            expected.add(output.relative_to(root).as_posix())
    require(actual == expected, 'Missing or unlisted payload files')
    policy = validate_policy(_read(_file(root, 'assets/policy.json')))
    components = set(policy['model_configs'])
    flat, nested = (root / 'speaker_id').is_dir(), (root / 'src/speaker_id').is_dir()
    require(flat != nested, 'Exactly one portable source layout is required')
    source = 'speaker_id' if flat else 'src/speaker_id'
    required = {'submission.py', 'assets/policy.json', 'assets/provenance.json', 'assets/gallery.npz',
                'assets/labels.json', 'assets/calibration.json'} | set(policy['model_configs'].values()) | _source_files(source, components)
    models = {}
    for key, relative in policy['model_configs'].items():
        config = _read(_file(root, relative))
        if key == 'adapted':
            validate_adapted_config(config)
        elif key == 'public':
            validate_public_config(config)
        else:
            from speaker_id.candidates.campp_advanced import validate_advanced_config
            validate_advanced_config(config)
            require(config['weights_path'] == 'artifacts/models/campp_advanced/campplus_cn_en_common.pt', 'Advanced weight path must remain unchanged')
        weights = _file(root, config['weights_path'])
        require(config['weights_path'] in files and files[config['weights_path']]['sha256'] == config['weights_sha256'], 'Model SHA differs from payload manifest')
        if 'weights_bytes' in config:
            require(weights.stat().st_size == config['weights_bytes'], 'Model size differs from its contract')
        required.add(config['weights_path'])
        models[key] = config
    notices = {name for name in files if name.startswith('notices/') and Path(name).suffix.lower() in {'.txt', '.md', '.json'}}
    require(required <= set(files) <= required | notices | {'README.md'}, 'Unexpected source/model assets or missing runtime files')
    provenance = _read(root / 'assets/provenance.json')
    require(manifest.get('provenance') == {'policy_sha256': _sha(root / 'assets/policy.json'),
            'provenance_sha256': _sha(root / 'assets/provenance.json')}, 'Manifest does not bind selected policy/provenance bytes')
    if 'encoder_updates' in manifest:
        require(type(manifest['encoder_updates']) is int and manifest['encoder_updates'] == (500 if 'adapted' in components else 0),
                'Ambiguous/misleading manifest encoder-update history')
    _provenance(provenance, manifest, policy, models, root)
    return {'manifest': manifest, 'policy': policy, 'provenance': provenance, 'models': models}


def _labels(path):
    payload = _read(path)
    labels = payload.get('labels')
    require(isinstance(labels, list) and len(labels) == 447 and len(set(labels)) == 447 and labels[0] == 'unknown'
            and payload.get('unknown_index') == 0 and labels[1:] == sorted(labels[1:]), 'Expected unknown plus446sorted known UUID labels')
    try:
        require(all(str(uuid.UUID(value)) == value for value in labels[1:]), 'Noncanonical known UUID')
    except (TypeError, AttributeError) as error:
        raise IntegrityError('Known labels must be canonical UUID strings') from error
    return labels


def load_adapted_encoder(config, root, device='cpu'):
    validate_adapted_config(config)
    root = Path(root).resolve()
    path = _file(root, config['weights_path'])
    require(path.stat().st_size == config['weights_bytes'] and _sha(path) == config['weights_sha256'], 'Exported adapted encoder SHA/size mismatch')
    import torch
    from speaker_id.models.vendor.campplus.DTDNN import CAMPPlus
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state and all(isinstance(key, str) and isinstance(value, torch.Tensor)
            and torch.isfinite(value).all() for key, value in state.items()), 'Expected a finite plain encoder tensor dictionary, never a training checkpoint')
    for key, value in state.items():
        require(value.dtype == torch.float32 or (key.endswith('num_batches_tracked') and value.dtype == torch.int64 and value.ndim == 0),
                'Adapted tensors must remain FP32 with integer BatchNorm counters')
    require(state_dict_sha256(state) == config['encoder_state_sha256'], 'Serialized encoder tensors differ from captured tensor digest')
    encoder = CAMPPlus(**ARCHITECTURE_512)
    encoder.load_state_dict(state, strict=True)
    require(state_dict_sha256(encoder.state_dict()) == config['encoder_state_sha256'], 'Loaded encoder differs from exported tensors')
    encoder.to(device=device, dtype=torch.float32)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _load_models(models, root, device):
    import torch
    chosen = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    require(chosen in {'cpu', 'cuda'}, 'Inference device must be cpu or cuda')
    torch.set_num_threads(4)
    encoders = {}
    for key, config in models.items():
        if key == 'adapted':
            encoder = load_adapted_encoder(config, root, chosen)
        elif key == 'advanced':
            from speaker_id.candidates.campp_advanced import load_advanced
            encoder = load_advanced(config, root, chosen)
        else:
            from speaker_id.models.campp import load_campp
            encoder = load_campp(config, root, chosen)
            encoder.float().requires_grad_(False).eval()
        require(not encoder.training and not any(parameter.requires_grad for parameter in encoder.parameters()), 'Portable encoders must stay frozen in eval mode')
        encoders[key] = encoder
    return encoders, chosen


def _extract(encoders, policy, path, device):
    values, info = {}, {}
    for key, encoder in encoders.items():
        if key == 'advanced':
            from speaker_id.candidates.campp_advanced import extract_advanced_embedding
            vector, details = extract_advanced_embedding(encoder, path, device=device, **INFERENCE)
            dimension = 192
        else:
            from speaker_id.models.campp import extract_embedding
            vector, details = extract_embedding(encoder, path, device=device, **INFERENCE)
            dimension = 512
        vector = np.asarray(vector)
        require(vector.dtype == np.float32 and vector.shape == (dimension,) and np.isfinite(vector).all()
                and type(details.get('nonzero_signal')) in (bool, np.bool_), 'Invalid extracted component vector/validity')
        valid = bool(details['nonzero_signal'])
        require(np.isclose(np.linalg.norm(vector), 1, atol=1e-5) if valid else not vector.any(), 'Extracted unit/zero semantics changed')
        values[key], info[key] = vector, details
    flags = {bool(item['nonzero_signal']) for item in info.values()}
    require(len(flags) == 1, 'Paired encoders disagree on audio validity')
    valid = flags.pop()
    if len(values) == 1:
        # Endpoint bytes are returned directly, with no concatenation/extra norm.
        return next(iter(values.values())), {'nonzero_signal': valid}
    require(len({item['seconds'] for item in info.values()}) == 1, 'Paired encoders decoded different audio lengths')
    left = 'adapted' if 'adapted' in values else 'public'
    return paired_embedding(values[left], values['advanced'], valid, policy['advanced_weight']), {'nonzero_signal': valid}


def run_submission(root, data_dir, predictions_file_path, *, device=None):
    root = Path(root).absolute()
    output, data_dir = Path(predictions_file_path).absolute(), Path(data_dir).absolute()
    _no_links(output)
    _no_links(data_dir)
    root, output, data_dir = root.resolve(), output.resolve(), data_dir.resolve()
    payload = verify_payload(root, allowed_output=output)
    policy = payload['policy']
    labels = _labels(root / 'assets/labels.json')
    calibration = validate_calibration(_read(root / 'assets/calibration.json'), require_inference=True)
    require(calibration['unknown_weight'] in (0, .25, .5, .75, 1) and calibration['margin_weight'] in (0, .5), 'Calibrated coefficients left the preregistered grid')
    with np.load(root / 'assets/gallery.npz', allow_pickle=False) as saved:
        gallery = validate_gallery({name: saved[name].copy() for name in saved.files}, embedding_dim=policy['embedding_dim'])
    require(len(gallery['known_embeddings']) == 2217 and len(gallery['unknown_embeddings']) == 2223, 'Global reference counts changed')
    require(data_dir.is_dir(), '--data-dir must be an existing audio directory')
    extensions = {'.mp3', '.wav', '.flac', '.ogg', '.opus', '.aac', '.m4a', '.aif', '.aiff', '.au', '.snd', '.wma'}
    inputs = []
    for path in data_dir.iterdir():
        if path.suffix.lower() in extensions:
            _no_links(path)
            if path.is_file():
                inputs.append(path)
    inputs.sort(key=lambda path: path.name)
    require(output != data_dir and output not in {path.resolve() for path in inputs}, 'Output cannot overwrite input audio')
    require(not output.exists() or output.is_file(), 'Output must be a regular CSV file path')
    require(not output.is_relative_to(root) or output.relative_to(root).as_posix() not in set(payload['manifest']['files']) | {'manifest.json'}, 'Output cannot overwrite a model asset')
    encoders, chosen = _load_models(payload['models'], root, device)
    results, errors = [], []
    for path in inputs:
        try:
            embedding, info = _extract(encoders, policy, path, chosen)
        except Exception as error:
            import soundfile as sf
            recoverable = isinstance(error, (OSError, sf.LibsndfileError)) or (isinstance(error, ValueError) and str(error).startswith('Empty or nonfinite waveform:'))
            if not recoverable:
                raise
            embedding = np.zeros(policy['embedding_dim'], dtype=np.float32)
            info = {'nonzero_signal': False}
            errors.append({'audio_file': path.name, 'error_type': type(error).__name__})
            print(f'Audio decoding failed for {path.name}; emitting unknown ({type(error).__name__}).', file=sys.stderr)
        probabilities = score_embeddings(embedding[None, :], np.asarray([bool(info['nonzero_signal'])]), gallery, calibration)
        results.append({'audio_file': path.name, 'speaker_id': labels[int(probabilities[0].argmax())]})
    output.parent.mkdir(parents=True, exist_ok=True)
    _no_links(output)
    temporary = output.with_name(output.name + '.tmp_' + uuid.uuid4().hex[:8])
    try:
        with temporary.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['audio_file', 'speaker_id'])
            writer.writeheader()
            writer.writerows(results)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {'release_id': payload['manifest']['release_id'], 'policy_kind': policy['kind'], 'embedding_dim': policy['embedding_dim'],
            'files': len(results), 'device': chosen, 'audio_decode_failures': errors, 'predictions_file_path': str(output)}
