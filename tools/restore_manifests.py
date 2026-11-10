"""Restore byte-identical frozen membership metadata; no datasets or GPU needed."""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import json


def restore(root):
    root = Path(root).resolve()
    lock = json.loads((root / 'experiments/protocols/data.json').read_bytes())
    relative = PurePosixPath(lock['manifest_root'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid manifest root')
    directory = root.joinpath(*relative.parts)
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Manifest directory leaves project root')
    pending = []
    for role, expected in lock['manifest_sha256'].items():
        if not role.isidentifier():
            raise ValueError('Invalid role name')
        path = directory / (role + '.json')
        if not path.resolve().is_relative_to(root):
            raise ValueError('Manifest file leaves project root')
        data = gzip.decompress(path.with_suffix('.json.gz').read_bytes())
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Compressed manifest checksum mismatch: ' + role)
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError('Refusing to replace differing manifest: ' + str(path))
        else:
            pending.append((path, data))
    for path, data in pending:
        with path.open('xb') as stream:
            stream.write(data)
    return {'verified_roles': len(lock['manifest_sha256']), 'restored_files': len(pending)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(restore(parser.parse_args().root)))
