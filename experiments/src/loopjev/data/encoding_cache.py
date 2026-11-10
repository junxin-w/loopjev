import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from loopjev.io import digest, write_json


def encoding_key(records, encoder, lock):
    signature = hashlib.sha256()
    context = {'version': 1, 'family': encoder.family, 'limit': encoder.limit,
        'renderer': lock['renderer_sha256'], 'tokenizers': lock['tokenizers_sha256'],
        'transformers_version': version('transformers'), 'tokenizers_version': version('tokenizers'),
        'encoder_source': digest(Path(__file__).with_name('records.py'))}
    signature.update(json.dumps(context, sort_keys=True).encode())
    for row in records:
        signature.update(json.dumps([row[key] for key in ('id', 'kind', 'state', 'question', 'options')],
            ensure_ascii=False, separators=(',', ':')).encode())
        signature.update(b'\n')
    return signature.hexdigest()


class EncodedRecords:
    def __init__(self, directory, key):
        self.directory = Path(directory)
        metadata = json.loads((self.directory / 'manifest.json').read_text())
        if metadata['key'] != key:
            raise ValueError('Token cache identity mismatch')
        for name, expected in metadata['files_sha256'].items():
            if digest(self.directory / name) != expected:
                raise ValueError(f'Token cache checksum mismatch: {name}')
        self.metadata = metadata
        self.tokens = np.memmap(self.directory / 'tokens.bin', dtype='<u4', mode='r')
        self.candidate_offsets = np.load(self.directory / 'candidate_offsets.npy', mmap_mode='r')
        self.record_offsets = np.load(self.directory / 'record_offsets.npy', mmap_mode='r')
        if (len(self.tokens) != metadata['tokens'] or len(self.record_offsets) != metadata['records'] + 1
                or self.candidate_offsets[-1] != len(self.tokens)
                or self.record_offsets[-1] != len(self.candidate_offsets) - 1):
            raise ValueError('Invalid token cache offsets')

    def __len__(self):
        return self.metadata['records']

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        first, last = self.record_offsets[index:index + 2]
        return [self.tokens[self.candidate_offsets[member]:self.candidate_offsets[member + 1]].tolist()
            for member in range(first, last)]


def cached_encodings(records, encoder, lock, root, chunk_records=128):
    key = encoding_key(records, encoder, lock)
    root = Path(root) / encoder.family
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    with (root / (key + '.lock')).open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if not directory.exists():
            temporary = Path(tempfile.mkdtemp(prefix=key + '.pending-', dir=root))
            try:
                candidate_offsets, record_offsets = [0], [0]
                with (temporary / 'tokens.bin').open('wb') as stream:
                    for start in range(0, len(records), chunk_records):
                        encoded = encoder.encode_many(records[start:start + chunk_records])
                        for candidates in encoded:
                            for sequence in candidates:
                                np.asarray(sequence, dtype='<u4').tofile(stream)
                                candidate_offsets.append(candidate_offsets[-1] + len(sequence))
                            record_offsets.append(record_offsets[-1] + len(candidates))
                        if start == 0 or start + chunk_records >= len(records) or start // chunk_records % 100 == 0:
                            print(f'Token cache: {min(start + chunk_records, len(records))}/{len(records)} records', flush=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                np.save(temporary / 'candidate_offsets.npy', np.asarray(candidate_offsets, dtype='<i8'))
                np.save(temporary / 'record_offsets.npy', np.asarray(record_offsets, dtype='<i8'))
                write_json(temporary / 'manifest.json', {'key': key, 'records': len(records),
                    'candidates': len(candidate_offsets) - 1, 'tokens': candidate_offsets[-1],
                    'files_sha256': {path.name: digest(path) for path in temporary.iterdir() if path.is_file()}})
                temporary.rename(directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return EncodedRecords(directory, key)
