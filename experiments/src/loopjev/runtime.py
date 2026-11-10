"""Configuration, runtime settings and provenance shared by execution entry points."""
import json
import os
import random

import numpy as np
import torch

from loopjev.io import digest
from loopjev.paths import ROOT


def load_config(path):
    config = json.loads((ROOT / path).read_text())
    from loopjev.input_protocol import validate_config
    validate_config(config)
    if config['family'] not in ('ouro', 'qwen3') or config['supervision'] not in ('uniform', 'last', 'single'):
        raise ValueError('Unsupported model/supervision')
    if config['family'] == 'qwen3' and (config['loops'] != 1 or config['supervision'] != 'single'):
        raise ValueError('Dense baseline must execute once')
    positive = ('epochs', 'loops', 'batch_records', 'candidate_batch', 'padded_token_budget',
        'log_every', 'checkpoint_every', 'lora_rank')
    if any(type(config[key]) is not int or config[key] < 1 for key in positive):
        raise ValueError('Training budgets must be positive integers')
    for key, default in [('record_microbatch', 1), ('microbatch_token_budget', 4096), ('keep_checkpoints', 3)]:
        value = config.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if any(not np.isfinite(config[key]) or config[key] <= 0 for key in ('head_lr', 'lora_lr', 'gradient_clip')):
        raise ValueError('Invalid optimizer configuration')
    if not 0 <= config['warmup_ratio'] < 1 or not np.isfinite(config['brier_weight']) or config['brier_weight'] < 0:
        raise ValueError('Invalid warmup or loss weight')
    return config


def configure_runtime(config):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(config['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    torch.set_num_threads(4)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])


def source_hashes():
    sources = [ROOT / 'experiments/run.py', *(ROOT / 'experiments/src/loopjev').rglob('*.py')]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(sources)}


def model_provenance(config):
    root = ROOT / config['model']
    paths = sorted(root.glob('*.safetensors')) + sorted(root.glob('*.py')) + [root / 'config.json']
    if not any(path.suffix == '.safetensors' for path in paths):
        raise FileNotFoundError('Local safetensors weights required')
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}
