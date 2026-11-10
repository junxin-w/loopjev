"""Verify the training-source archive without model weights, data or torch."""
from collections import Counter
from pathlib import Path
import ast
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    checkpoint = json.loads((ROOT / 'provenance/CHECKPOINT_MANIFEST.json').read_bytes())
    protocol = checkpoint['protocol']
    expected_sources = protocol['sources_sha256']
    observed = {'experiments/run.py'} | {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / 'experiments/src/loopjev').rglob('*.py')}
    require(observed == set(expected_sources), 'Training source inventory differs from checkpoint')
    for name, sha in expected_sources.items():
        data = (ROOT / name).read_bytes()
        require(digest(data) == sha, 'Training source mismatch: ' + name)
        ast.parse(data, filename=name)
    config = json.loads((ROOT / protocol['config_path']).read_bytes())
    require(config == checkpoint['config'], 'Configuration differs from checkpoint')
    lock_bytes = (ROOT / protocol['data_lock']).read_bytes()
    require(digest(lock_bytes) == protocol['data_lock_sha256'], 'Frozen data protocol changed')
    lock = json.loads(lock_bytes)
    require(config['input_protocol'] == lock['input_protocol'] == 'noul-dual-v1', 'Wrong input protocol')
    require(digest((ROOT / 'repos/Open-Jev/jev/api.py').read_bytes()) == lock['renderer_sha256'], 'Renderer changed')
    third = json.loads((ROOT / 'repos/Open-Jev/PROVENANCE.json').read_bytes())
    for name, item in third['files'].items():
        expected = item.get('actual_sha256', item['baseline_sha256'])
        require(digest((ROOT / 'repos/Open-Jev' / name).read_bytes()) == expected, 'Third-party file changed: ' + name)

    memberships, groups, counts = {}, {}, {}
    for role, sha in lock['manifest_sha256'].items():
        path = ROOT / lock['manifest_root'] / (role + '.json')
        data = path.read_bytes()
        require(digest(data) == sha, 'Membership checksum mismatch: ' + role)
        rows = json.loads(data)
        memberships[role] = {row['id'] for row in rows}
        groups[role] = {row['group_id'] for row in rows}
        require(len(rows) == len(memberships[role]) == lock['roles'][role]['records'], 'Record count mismatch: ' + role)
        require(len(groups[role]) == lock['roles'][role]['groups'], 'Group count mismatch: ' + role)
        require(all(not {'state', 'question', 'options', 'target'} & row.keys() for row in rows), 'Unexpected raw dataset content in manifest')
        counts[role] = dict(Counter(row['kind'] for row in rows))
    require(memberships['risk_train'] <= memberships['stage1'], 'Risk-train membership changed')
    for i, left in enumerate(groups):
        for right in list(groups)[i + 1:]:
            if {left, right} != {'stage1', 'risk_train'}:
                require(not groups[left] & groups[right], 'Forbidden group overlap')

    sys.path.insert(0, str(ROOT / 'repos/Open-Jev'))
    sys.path.insert(0, str(ROOT / 'experiments/src'))
    from jev.api import candidate_prompts
    from loopjev.input_protocol import validate_checkpoint_manifest, validate_config
    validate_checkpoint_manifest(checkpoint, lock)
    row = {'state': 'The switch is on.', 'question': 'Is the switch on?',
           'kind': 'noul', 'options': ['no', 'yes'], 'target': [0., 1.]}
    prompts = candidate_prompts(row)
    require(len(prompts) == 2 and 'Proposed answer: No\n' in prompts[0]
            and 'Proposed answer: Yes\n' in prompts[1], 'Wrong NOUL prompt order')
    require(candidate_prompts(dict(row, target=[1., 0.])) == prompts, 'Targets enter prompts')
    try:
        validate_config({key: value for key, value in config.items() if key != 'input_protocol'})
    except ValueError:
        pass
    else:
        raise ValueError('Legacy configuration was accepted')
    return {'status': 'passed', 'training_source_files': len(expected_sources),
            'configuration_matches_checkpoint': True, 'data_lock_and_memberships_verified': True,
            'renderer_and_third_party_hashes_verified': True, 'noul_prompt_order_verified': True,
            'targets_absent_from_prompts': True, 'legacy_config_rejected': True,
            'stage1_counts': counts['stage1'], 'gpu_or_training_executed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
