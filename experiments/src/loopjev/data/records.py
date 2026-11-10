import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import pyarrow.parquet as parquet

from loopjev.data.audit import KINDS, identity
from loopjev.io import digest
from loopjev.paths import ROOT


DEVELOPMENT_ROLES = ('stage1', 'risk_train', 'risk_val', 'policy_fit', 'validation')


def validate_record(row):
    """Check immutable label semantics once, before tensors reach the GPU."""
    target = row['target']
    if row['kind'] not in KINDS or len(target) < 2 or len(target) != len(row['options']):
        raise ValueError('Invalid task or target shape')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in target) or not math.isclose(sum(target), 1., abs_tol=1e-8):
        raise ValueError('Invalid probability target')
    if row['kind'] == 'noul' and row['options'] != ['no', 'yes']:
        raise ValueError('Noul options must be [no, yes]')
    if row['kind'] == 'score':
        levels = row['metadata'].get('score_values', [])
        if (len(levels) != len(target) or any(not math.isfinite(value) for value in levels)
                or any(left >= right for left, right in zip(levels, levels[1:]))):
            raise ValueError('Score levels are missing or unordered')


def load_lock(path='experiments/protocols/data.json'):
    path = ROOT / path
    lock = json.loads(path.read_text())
    from loopjev.input_protocol import validate_lock
    validate_lock(lock)
    if not lock['frozen']:
        raise ValueError('Data protocol has not been approved')
    return lock


def read_split(split, lock):
    if split not in ('train', 'validation'):
        raise ValueError('This development framework reads Train/Validation only')
    records = []
    paths = sorted(name for name in lock['data_sha256'] if Path(name).name.startswith(split + '-'))
    if not paths:
        raise ValueError(f'Missing dataset split: {split}')
    for name in paths:
        path = ROOT / name
        if digest(path) != lock['data_sha256'][name]['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {name}')
        records.extend(json.loads(value) for value in parquet.read_table(path, columns=['record_json'])['record_json'].to_pylist())
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Duplicate record ID')
    for row in records:
        validate_record(row)
    return records


def load_records(role, lock):
    if role not in DEVELOPMENT_ROLES:
        raise ValueError(f'Unsupported development role: {role}')
    records = read_split('validation' if role == 'validation' else 'train', lock)
    if role == 'validation':
        return sorted(records, key=lambda row: row['id'])
    path = ROOT / lock['manifest_root'] / (role + '.json')
    if digest(path) != lock['manifest_sha256'][role]:
        raise ValueError(f'Manifest checksum mismatch: {role}')
    manifest = json.loads(path.read_text())
    lookup = {row['id']: row for row in records}
    result = []
    for member in manifest:
        row = lookup[member['id']]
        if any(row[key] != member[key] for key in ('group_id', 'kind', 'source')) or identity(row) != member['input_sha256']:
            raise ValueError('Manifest metadata or candidate input mismatch')
        result.append(row)
    expected = lock['roles'][role]
    if len(result) != expected['records'] or len({row['group_id'] for row in result}) != expected['groups']:
        raise ValueError('Frozen role counts differ')
    return result


def verify_roles(lock):
    groups = {}
    memberships = {}
    for role in lock['manifest_sha256']:
        path = ROOT / lock['manifest_root'] / (role + '.json')
        if digest(path) != lock['manifest_sha256'][role]:
            raise ValueError(f'Manifest checksum mismatch: {role}')
        records = json.loads(path.read_text())
        memberships[role] = {row['id'] for row in records}
        groups[role] = {row['group_id'] for row in records}
        if len(memberships[role]) != lock['roles'][role]['records'] or len(groups[role]) != lock['roles'][role]['groups']:
            raise ValueError('Role count mismatch')
    if not memberships['risk_train'] <= memberships['stage1']:
        raise ValueError('Risk-train is no longer contained in Stage I')
    for position, left in enumerate(groups):
        for right in list(groups)[position + 1:]:
            if {left, right} != {'stage1', 'risk_train'} and groups[left] & groups[right]:
                raise ValueError(f'Forbidden Group overlap: {left}, {right}')
    return {role: {'records': len(memberships[role]), 'groups': len(groups[role])} for role in groups}


def smoke_records(records, per_kind=1):
    if per_kind < 1:
        raise ValueError('Smoke records per kind must be positive')
    selected = []
    for kind in KINDS:
        members = sorted((row for row in records if row['kind'] == kind), key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())
        if len(members) < per_kind:
            raise ValueError(f'Insufficient smoke coverage: {kind}')
        selected.extend(members[:per_kind])
    return selected


def renderer(lock):
    path = ROOT / 'repos/Open-Jev/jev/api.py'
    if digest(path) != lock['renderer_sha256']:
        raise ValueError('Official renderer changed after length audit')
    sys.path.insert(0, str(ROOT / 'repos/Open-Jev'))
    return importlib.import_module('jev.api').candidate_prompts


class RecordEncoder:
    def __init__(self, tokenizer, family, lock):
        from loopjev.input_protocol import validate_lock
        validate_lock(lock)
        self.tokenizer = tokenizer
        self.family = family
        self.limit = lock['max_input_tokens']
        self.render = renderer(lock)

    def __call__(self, row):
        return self.encode_many([row])[0]

    def encode_many(self, records):
        prompts, counts = [], []
        for row in records:
            members = self.render(row)
            expected = len(row['options'])
            if len(members) != expected:
                raise ValueError(f'Encoding count violation at {row["id"]}')
            if self.family == 'ouro':
                members = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for prompt in members]
            prompts.extend(members)
            counts.append(len(members))
        sequences = self.tokenizer(prompts, add_special_tokens=False, truncation=False)['input_ids']
        result, start = [], 0
        for row, count in zip(records, counts, strict=True):
            candidates = sequences[start:start + count]
            if len(candidates) != count or any(not sequence or len(sequence) > self.limit for sequence in candidates):
                raise ValueError(f'Encoding length violation at {row["id"]}; truncation is forbidden')
            result.append(candidates)
            start += count
        return result
