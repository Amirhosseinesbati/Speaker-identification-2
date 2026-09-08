"""Explicit portable CAM++ policies and exact float32 paired embeddings."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np


INFERENCE = {'seconds': 180.0, 'maximum_windows': 1}
FROZEN_PROTOCOL = 'frozen_all_training_group_excluded'
ADAPTED_PROTOCOL = 'original_q0_group_excluded_global_gallery'
ROLES_SHA256 = '882b7535da2f72060c1bf7aaf4549eb3052f621ad0efd88f23f6a6aad28e3fa9'
MODEL_PATHS = {'adapted': 'assets/adapted_model_config.json', 'public': 'assets/public_model_config.json',
               'advanced': 'assets/advanced_model_config.json'}
POLICIES = {
    'adapted_f004_fold0_512': (512, ('adapted',), ADAPTED_PROTOCOL),
    'public_voxceleb_512': (512, ('public',), FROZEN_PROTOCOL),
    'advanced_public_192': (192, ('advanced',), FROZEN_PROTOCOL),
    'paired_f004_advanced_704': (704, ('adapted', 'advanced'), ADAPTED_PROTOCOL),
    'paired_public512_advanced192_704': (704, ('public', 'advanced'), FROZEN_PROTOCOL),
}
F004_SOURCE = {
    'experiment_code': 'F004', 'parent_run_id': 'a87bd6a800724f9f9963d9ee5b780bc1',
    'child_run_id': 'e854a866477a444ebb674637a19990a1', 'outer_fold': 0,
    'checkpoint_sha256': 'e05801fc9a240a05735d6113f99ed7afc75c9edbc1bcafab2c93ded4cb4a2eec',
    'source_git_commit': 'ce13ca81705f40b96d110dd12b1a1fc2aed81298',
    'completed_steps': 1100, 'head_only_steps': 600, 'encoder_tail_steps': 500,
}
ARCHITECTURE_512 = {'feat_dim': 80, 'embedding_size': 512, 'growth_rate': 32, 'bn_size': 4,
                    'init_channels': 128, 'config_str': 'batchnorm-relu', 'memory_efficient': False}
PUBLIC_WEIGHT_SHA = '5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(exact(value[key], item) for key, item in expected.items())
    return value == expected


def is_digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{' + str(length) + '}', value) is not None


def validate_policy(policy):
    require(isinstance(policy, dict) and set(policy) == {'schema_version', 'kind', 'embedding_dim', 'advanced_weight',
            'inference', 'model_configs', 'calibration_protocol'} and type(policy['schema_version']) is int
            and policy['schema_version'] == 1 and policy['kind'] in POLICIES, 'Unsupported explicit release policy')
    dimension, components, protocol = POLICIES[policy['kind']]
    allowed = (.25, .5, .75) if len(components) == 2 else (1.0,) if components == ('advanced',) else (0.0,)
    require(type(policy['advanced_weight']) is float and policy['advanced_weight'] in allowed
            and type(policy['embedding_dim']) is int and policy['embedding_dim'] == dimension
            and policy['model_configs'] == {key: MODEL_PATHS[key] for key in components}
            and policy['calibration_protocol'] in ((FROZEN_PROTOCOL, ADAPTED_PROTOCOL) if components == ('advanced',) else (protocol,))
            and exact(policy['inference'], INFERENCE),
            'Release kind, components, dimension, alpha or inference policy disagree')
    return policy


def validate_adapted_config(config):
    from speaker_id.models.campp import EXPECTED_FRONTEND
    fixed = {'schema_version': 1, 'encoder_kind': 'adapted_f004_fold0', 'architecture': 'CAMPPlus',
             'embedding_dim': 512, 'sample_rate': 16000, 'fbank_bins': 80,
             'frontend': EXPECTED_FRONTEND, 'architecture_kwargs': ARCHITECTURE_512,
             'inference': INFERENCE, 'source': F004_SOURCE, 'weights_path': 'assets/f004_fold0_encoder.pt'}
    require(isinstance(config, dict) and set(config) == set(fixed) | {'weights_bytes', 'weights_sha256', 'encoder_state_sha256'},
            'Adapted release requires the complete encoder-only configuration')
    require(all(exact(config[key], value) for key, value in fixed.items())
            and type(config['weights_bytes']) is int and config['weights_bytes'] > 0
            and is_digest(config['weights_sha256']) and is_digest(config['encoder_state_sha256']),
            'Adapted checkpoint/fold/schedule/architecture or exported-state identity changed')
    return config


def validate_public_config(config):
    from speaker_id.models.campp import EXPECTED_FRONTEND, validate_model_config
    expected = {'architecture': 'CAMPPlus', 'embedding_dim': 512, 'sample_rate': 16000, 'fbank_bins': 80,
        'weights_path': 'assets/campplus_voxceleb.bin', 'weights_sha256': PUBLIC_WEIGHT_SHA,
        'public_model': 'iic/speech_campplus_sv_en_voxceleb_16k', 'public_revision': 'v1.0.2',
        'source_commit': '065629c313eaf1a01c65c640c46d77e61e9607b4', 'license': 'Apache-2.0', 'frontend': EXPECTED_FRONTEND}
    require(exact(config, expected), 'Public512 portable configuration differs from the pinned public source')
    validate_model_config(config)
    return config


def state_dict_sha256(state):
    """Serialization-independent tensor digest matching the captured encoder hash."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def paired_embedding(left512, advanced192, valid, alpha):
    """Exactly weighted_encoder_pair arithmetic for one query; no final renorm."""
    require(type(valid) in (bool, np.bool_) and type(alpha) is float and alpha in (.25, .5, .75), 'Invalid paired query policy')
    views = []
    for value, dimension in ((left512, 512), (advanced192, 192)):
        vector = np.asarray(value)
        require(vector.dtype == np.float32 and vector.shape == (dimension,) and np.isfinite(vector).all(), 'Invalid paired source vector')
        values = vector[None, :]
        mask = np.asarray([valid], dtype=bool)
        norms = np.linalg.norm(values, axis=1)
        require(not np.any(values[~mask]) and np.allclose(norms[mask], 1, atol=1e-5), 'Pair sources must preserve original unit/zero semantics')
        normalized = np.zeros_like(values)
        normalized[mask] = values[mask] / norms[mask, None]
        views.append(normalized)
    left_weight = np.float32(np.sqrt(np.float32(1.0 - alpha)))
    advanced_weight = np.float32(np.sqrt(np.float32(alpha)))
    return np.concatenate((left_weight * views[0], advanced_weight * views[1]), axis=1).astype(np.float32)[0]
