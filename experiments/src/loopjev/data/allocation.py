import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loopjev.data.audit import KINDS, SEED, apportion, distribution, historical_inventory, identity, load_release, ordered_key
from loopjev.io import digest, write_json


def choose_groups(groups, count, preferred):
    strata = defaultdict(list)
    for group, records in groups.items():
        key = (tuple(sorted({row['source'] for row in records})), tuple(sorted({row['kind'] for row in records})))
        strata[key].append(group)
    quotas = apportion({key: len(value) for key, value in strata.items()}, count)
    selected = set()
    for key, members in strata.items():
        ranked = sorted(members, key=lambda group: (group not in preferred, ordered_key(group)))
        selected.update(ranked[:quotas[key]])
    if len(selected) != count:
        raise ValueError('Group selection count mismatch')
    return selected


def token_report(records, indexes, cache):
    members = [indexes[row['id']] for row in records]
    result = {}
    for model in ['Ouro_1_4B', 'Qwen3_1_7B_Base']:
        tokens = cache[model + '_sum'][members]
        result[model] = {'tokens': int(tokens.sum()), 'encoded_inputs': int(cache[model + '_inputs'][members].sum()),
            'tokens_per_record': float(tokens.mean()) if len(tokens) else None,
            'by_kind': {kind: int(sum(value for value, row in zip(tokens, records) if row['kind'] == kind)) for kind in KINDS}}
    return result


def role_manifest(records, historical):
    return [{'id': row['id'], 'group_id': row['group_id'], 'source': row['source'], 'kind': row['kind'],
        'historical_group': row['group_id'] in historical, 'input_sha256': identity(row)}
        for row in sorted(records, key=lambda item: item['id'])]


def allocate_data(data_root, audit_root, output, max_input_tokens=2048):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    source_metrics = json.loads((audit_root / 'results/metrics.json').read_text())
    source_protocol = json.loads((audit_root / 'results/protocol.json').read_text())
    source_verification = {}
    for name, expected in source_protocol['sources_sha256'].items():
        path = Path(name)
        checked_path = path
        if (not path.is_file() or digest(path) != expected) and path.name in ('run.py', 'audit.py'):
            checked_path = audit_root / 'sources' / path.name
        if digest(checked_path) != expected:
            raise ValueError(f'Audit source changed: {name}')
        source_verification[name] = {'audited_sha256': expected, 'verified_path': str(checked_path),
            'current_sha256': digest(path) if path.is_file() else None}
    cache_path = Path(source_metrics['token_cache']['path'])
    if not cache_path.exists() and cache_path.as_posix().startswith(('artifacts/main/data_audit_v2/', 'experiments/outputs/data_audit_v2/')):
        cache_path = audit_root / 'cache' / cache_path.name
    if digest(cache_path) != source_metrics['token_cache']['sha256']:
        raise ValueError('Token cache changed')
    splits, provenance, revision = load_release(data_root)
    if provenance != source_protocol['data_sha256']:
        raise ValueError('Dataset changed since audit')
    historical, inventory = historical_inventory()
    with np.load(cache_path, allow_pickle=False) as stream:
        cache = {key: stream[key] for key in stream.files}
    indexes = {value: index for index, value in enumerate(cache['id'])}
    accepted = {value for index, value in enumerate(cache['id'])
        if cache['Ouro_1_4B_max'][index] <= max_input_tokens and cache['Qwen3_1_7B_Base_max'][index] <= max_input_tokens}
    train = [row for row in splits['train'] if row['id'] in accepted]
    groups = defaultdict(list)
    for row in train:
        groups[row['group_id']].append(row)
    duplicates = json.loads((audit_root / 'results/duplicate_screen.json').read_text())
    training_only = {group for pair in duplicates['near_pairs'] for group in pair}
    training_only.update(group for members in duplicates['train_identical_input_groups'].values() for group in members)
    holdout_pool = {group: rows for group, rows in groups.items() if group not in training_only}
    fresh = set(groups) - historical
    risk_val_groups = choose_groups(holdout_pool, 2000, fresh)
    remaining = {group: rows for group, rows in holdout_pool.items() if group not in risk_val_groups}
    policy_groups = choose_groups(remaining, 2000, fresh)
    holdouts = risk_val_groups | policy_groups
    stage1 = [row for row in train if row['group_id'] not in holdouts]
    stage1_groups = {group: rows for group, rows in groups.items() if group not in holdouts}
    risk_train_groups = choose_groups(stage1_groups, 8000, set(stage1_groups))
    roles = {'stage1': stage1,
        'risk_train': [row for row in stage1 if row['group_id'] in risk_train_groups],
        'risk_val': [row for row in train if row['group_id'] in risk_val_groups],
        'policy_fit': [row for row in train if row['group_id'] in policy_groups]}
    role_groups = {name: {row['group_id'] for row in records} for name, records in roles.items()}
    intersections = {left + '__' + right: len(role_groups[left] & role_groups[right])
        for position, left in enumerate(roles) for right in list(roles)[position + 1:]}
    if any(count for name, count in intersections.items() if name != 'stage1__risk_train'):
        raise ValueError('Forbidden role overlap')
    group_roles = defaultdict(set)
    for name, members in role_groups.items():
        for group in members:
            group_roles[group].add(name)
    near_crossings = []
    for left, right in duplicates['near_pairs']:
        pair_roles = group_roles[left] | group_roles[right]
        if ('risk_val' in pair_roles or 'policy_fit' in pair_roles) and len(pair_roles) > 1:
            near_crossings.append([left, right])
    exact_crossings = []
    for members in duplicates['train_identical_input_groups'].values():
        member_roles = set.union(*(group_roles[group] for group in members))
        if ('risk_val' in member_roles or 'policy_fit' in member_roles) and len(member_roles) > 1:
            exact_crossings.append(members)
    report = {'status': 'draft; awaiting duplicate/coverage review', 'seed': SEED, 'revision': revision,
        'roles': {name: distribution(records) | {'tokens': token_report(records, indexes, cache),
            'historically_used_groups': len(role_groups[name] & historical)} for name, records in roles.items()},
        'intersections_groups': intersections, 'validation_and_policy_unseen_by_stage1': True,
        'near_duplicate_cross_role_pairs': len(near_crossings), 'exact_duplicate_cross_role_keys': len(exact_crossings),
        'cross_role_duplicate_groups': {'near_pairs': near_crossings, 'exact_groups': exact_crossings},
        'available_train': distribution(train), 'historical_inventory': inventory,
        'known_similar_groups_training_only': len(training_only & set(groups)),
        'length_limit': max_input_tokens, 'no_truncation': True, 'stage1_full_pass': True,
        'limits': ['Manifests are drafts until cross-role duplicate screens and source coverage are reviewed',
            'Risk-train overlaps Stage I by design; Risk-train scores are not evidence of unseen-Group generalization',
            'Historical fresh preference cannot create unseen-source coverage',
            'Near duplicates screened only on representative records; semantic equivalence may be missed'],
        'elapsed_seconds': time.perf_counter() - started}
    manifest_hashes = {}
    for name, records in roles.items():
        path = output / 'manifests' / (name + '.json')
        write_json(path, role_manifest(records, historical))
        manifest_hashes[name] = digest(path)
    report['manifest_sha256'] = manifest_hashes
    write_json(output / 'results/metrics.json', report)
    write_json(output / 'results/protocol.json', {'script_sha256': digest(Path(__file__)), 'seed': SEED,
        'entrypoint_sha256': digest(Path('experiments/run.py')), 'audit_sources_verified': source_verification,
        'maximum_input_tokens': max_input_tokens,
        'similarity_rule': 'Keep all Groups flagged by the audited exact or sampled near-duplicate screen in Stage I; select independent development roles from remaining Groups',
        'audit_protocol_sha256': digest(audit_root / 'results/protocol.json'), 'role_rule': 'Stage I includes Risk-train, excludes Risk-val and Policy-fit',
        'manifest_sha256': manifest_hashes, 'frozen': False})
    print(json.dumps({'roles': {name: {'records': item['records'], 'groups': item['groups']} for name, item in report['roles'].items()},
        'near_crossings': len(near_crossings), 'exact_crossings': len(exact_crossings)}, ensure_ascii=False), flush=True)
