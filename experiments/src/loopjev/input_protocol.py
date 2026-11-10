"""Identity and output isolation for the NOUL dual-input experiment."""
from pathlib import Path
from loopjev.paths import ROOT
INPUT_PROTOCOL = 'noul-dual-v1'
def validate_lock(lock):
    if lock.get('input_protocol') != INPUT_PROTOCOL:
        raise ValueError('Wrong data lock: this runtime requires noul-dual-v1')
def validate_config(config):
    if config.get('input_protocol') != INPUT_PROTOCOL:
        raise ValueError('Refusing legacy/single-input config in noul-dual-v1 runtime')
    expected = f"loopjev_noul_dual_v1_uniform_loop{config['loops']}_seed{config['seed']}"
    if config.get('experiment_id') != expected:
        raise ValueError('Experiment identity must name dual input, loops and seed')
    if config['family'] != 'ouro' or config['supervision'] != 'uniform':
        raise ValueError('This experiment is Ouro uniform Stage I')
def validate_checkpoint_manifest(manifest, lock):
    validate_lock(lock)
    validate_config(manifest['config'])
    protocol = manifest['protocol']
    if protocol.get('input_protocol') != INPUT_PROTOCOL:
        raise ValueError('Refusing checkpoint from a different NOUL input protocol')
    if protocol.get('experiment_id') != manifest['config']['experiment_id']:
        raise ValueError('Checkpoint experiment identity mismatch')
def validate_output(path, config):
    validate_config(config)
    path = Path(path).resolve()
    allowed = (ROOT/'experiments/outputs').resolve()
    if not path.is_relative_to(allowed) or config['experiment_id'] not in path.parts:
        raise ValueError('Output must stay in this runtime/experiments/outputs and contain experiment_id')
    return path
def identity(config):
    validate_config(config)
    return {'input_protocol': INPUT_PROTOCOL, 'experiment_id': config['experiment_id']}
