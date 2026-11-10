"""Immutable checkpoints, atomic latest links and protected weight loading."""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch

from loopjev.data.records import load_lock
from loopjev.io import digest, write_json
from loopjev.paths import ROOT


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    manifest: dict
    sha256: str


@contextmanager
def checkpoint_lock(output, exclusive):
    with (output / '.checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def saved_directories(output):
    directory = output / 'checkpoints'
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir()
        if path.is_dir() and re.fullmatch(r'update-\d{8}', path.name))


def publish_latest(output, destination):
    latest, pending = output / 'checkpoint', output / 'checkpoint.link.pending'
    if latest.exists() and not latest.is_symlink():
        raise ValueError('Use a new output condition for the checkpoint retention layout')
    pending.unlink(missing_ok=True)
    pending.symlink_to(destination.relative_to(output), target_is_directory=True)
    pending.replace(latest)


def verify_checkpoint(path, lock):
    content = (path / 'manifest.json').read_bytes()
    manifest = json.loads(content)
    from loopjev.input_protocol import validate_checkpoint_manifest
    validate_checkpoint_manifest(manifest, lock)
    protocol = manifest['protocol']
    if protocol['data_lock_sha256'] != digest(ROOT / protocol['data_lock']):
        raise ValueError('Checkpoint data lock changed')
    if protocol['dataset_revision'] != lock['dataset_revision']:
        raise ValueError('Checkpoint dataset mismatch')
    if lock != load_lock(protocol['data_lock']):
        raise ValueError('Evaluation data allocation differs from training')
    for name, expected in protocol['model_files_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Base model differs from training: {name}')
    for name, expected in manifest['files_sha256'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    return Checkpoint(path, manifest, hashlib.sha256(content).hexdigest())


@contextmanager
def checkpoint_reader(path, lock, recover=False):
    """Pin identity and prevent pruning until the caller finishes loading weights.

    Resume also finds a complete directory left by an interrupted latest-link
    update. A corrupt checkpoint raises an error; it is never silently skipped.
    """
    output = path.parent.parent if path.parent.name == 'checkpoints' else path.parent
    with checkpoint_lock(output, exclusive=recover):
        candidates = saved_directories(output) if recover else []
        selected = candidates[-1] if candidates else path
        checkpoint = verify_checkpoint(selected.resolve(strict=True), lock)
        yield checkpoint
        # Only publish after the caller also accepts its config/code and loads it.
        if recover and candidates:
            publish_latest(output, checkpoint.path)


def checkpoint_save(model, optimizer, output, config, protocol, updates, stats):
    retention = config.get('keep_checkpoints', 3)
    if retention < 1:
        raise ValueError('At least one checkpoint must be retained')
    with checkpoint_lock(output, exclusive=True):
        directory = output / 'checkpoints'
        directory.mkdir(exist_ok=True)
        name = f'update-{updates:08d}'
        destination, temporary = directory / name, directory / (name + '.pending')
        if destination.exists():
            raise FileExistsError(f'{destination}: resume from the latest complete checkpoint')
        if temporary.exists():
            shutil.rmtree(temporary)
        model.save(temporary)
        torch.save({'optimizer': optimizer.state_dict(), 'completed_updates': updates, 'statistics': stats,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary / 'training.pt')
        write_json(temporary / 'manifest.json', {'config': config, 'protocol': protocol, 'completed_updates': updates,
            'files_sha256': {str(path.relative_to(temporary)): digest(path) for path in temporary.rglob('*') if path.is_file()}})
        temporary.rename(destination)
        publish_latest(output, destination)
        for obsolete in saved_directories(output)[:-retention]:
            shutil.rmtree(obsolete)
