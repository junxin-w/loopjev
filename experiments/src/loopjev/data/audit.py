import hashlib
import importlib
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from loopjev.paths import ROOT
from loopjev.io import digest, write_json


KINDS = ('choice', 'noul', 'score')
SPLITS = ('train', 'calibration', 'validation', 'test', 'ood')
SEED = 20261019


def normalized(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def identity(row, context_only=False):
    fields = [row['kind'], normalized(row['state']), normalized(row['question'])]
    if not context_only:
        fields.append(sorted(normalized(option) for option in row['options']))
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def ordered_key(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def summarize(values):
    if not len(values):
        return {'count': 0}
    values = np.asarray(values)
    return {'count': len(values), 'min': int(values.min()), 'p50': float(np.quantile(values, .5)),
        'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)),
        'max': int(values.max()), 'mean': float(values.mean()), 'sum': int(values.sum())}


def distribution(records):
    groups = defaultdict(set)
    for row in records:
        groups[row['group_id']].add(row['kind'])
    by_kind = {}
    for kind in KINDS:
        selected = [row for row in records if row['kind'] == kind]
        counts = Counter(row['group_id'] for row in selected)
        by_kind[kind] = {'records': len(selected), 'groups': len(counts),
            'sources': dict(sorted(Counter(row['source'] for row in selected).items())),
            'soft_records': sum(not all(value in (0, 1) for value in row['target']) for row in selected),
            'output_classes': summarize([len(row['options']) for row in selected]),
            'encoded_inputs': sum(len(row['options']) for row in selected),
            'records_per_group': dict(sorted(Counter(counts.values()).items()))}
    return {'records': len(records), 'groups': len(groups), 'by_kind': by_kind,
        'group_kind_signatures': dict(sorted(Counter('+'.join(sorted(kinds)) for kinds in groups.values()).items()))}


def historical_inventory():
    import torch
    paths = [Path('preliminary/artifacts/step3/uniform4/cache') / (split + '.pt') for split in ['train', 'dev-cal', 'validation']]
    paths += sorted(Path('preliminary/artifacts/step5/risk_uniform4_v1/cache/manifest').glob('*.json'))
    paths += sorted(Path('preliminary/artifacts/step7/s2_factorial_v1/cache/manifest').glob('*.json'))
    paths += [Path('preliminary/artifacts/step7/multisource_holdout_v1/results') / name for name in ['step7_confirm.json', 'step8_final.json']]
    inventory, used = {}, set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Expected historical evidence missing: {path}')
        if path.suffix == '.pt':
            payload = torch.load(path, map_location='cpu', weights_only=True)
            records = payload['rows']
        else:
            records = json.loads(path.read_text())
        groups = {row['group_id'] for row in records}
        used.update(groups)
        inventory[str(path)] = {'groups': len(groups), 'records': len(records), 'sha256': digest(path)}
        if path.suffix == '.pt':
            del payload
    return used, inventory


def load_release(data_root):
    export = json.loads((data_root / 'export-manifest.json').read_text())
    splits, provenance, revisions, errors = {}, {}, set(), Counter()
    identifiers = set()
    for split in SPLITS:
        records = []
        for path in sorted((data_root / 'data/community-hard-mix-v2-redistributable').glob(split + '-*.parquet')):
            relative = path.relative_to(data_root).as_posix()
            actual = digest(path)
            if actual != export['published_files'][relative]['sha256']:
                raise ValueError(f'Release SHA256 mismatch: {path}')
            metadata = data_root / '.cache/huggingface/download' / (relative + '.metadata')
            revision = metadata.read_text().splitlines()[0]
            revisions.add(revision)
            provenance[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size, 'revision': revision}
            table = parquet.read_table(path, columns=['record_json'])
            records.extend(json.loads(item) for item in table['record_json'].to_pylist())
        if len(records) != export['splits'][split]['count']:
            raise ValueError(f'Release count mismatch: {split}')
        for row in records:
            if row['id'] in identifiers:
                errors['duplicate_id'] += 1
            identifiers.add(row['id'])
            target = row['target']
            if row['split'] != split or row['kind'] not in KINDS:
                errors['split_or_kind'] += 1
            if len(target) != len(row['options']) or len(target) < 2:
                errors['target_shape'] += 1
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in target) or not math.isclose(sum(target), 1, abs_tol=1e-8):
                errors['target_value'] += 1
            if row['kind'] == 'noul' and row['options'] != ['no', 'yes']:
                errors['noul_order'] += 1
            if row['kind'] == 'score':
                levels = row['metadata'].get('score_values', [])
                if len(levels) != len(target) or any(not math.isfinite(value) for value in levels) or any(left >= right for left, right in zip(levels, levels[1:])):
                    errors['score_levels'] += 1
        splits[split] = records
        print(f'release: {split}: {len(records)} records verified', flush=True)
    if len(revisions) != 1 or errors:
        raise ValueError(f'Release integrity failed: revisions={revisions}, errors={errors}')
    return splits, provenance, next(iter(revisions))


def overlap_report(splits):
    indexes = {name: defaultdict(set) for name in ['group', 'input', 'context', 'semantic_context']}
    train_exact = defaultdict(set)
    for split, records in splits.items():
        for row in records:
            indexes['group'][row['group_id']].add(split)
            key = identity(row)
            indexes['input'][key].add(split)
            indexes['context'][identity(row, True)].add(split)
            context = row['metadata'].get('semantic_context_sha256')
            if context:
                indexes['semantic_context'][context].add(split)
            if split == 'train':
                train_exact[key].add(row['group_id'])
    result = {}
    for name, index in indexes.items():
        conflicts = {key: sorted(value) for key, value in index.items() if len(value) > 1}
        result[name] = {'cross_split_keys': len(conflicts), 'examples_hashes_only': dict(list(conflicts.items())[:10])}
    merged = {key: sorted(value) for key, value in train_exact.items() if len(value) > 1}
    result['train_identical_input_across_groups'] = {'keys': len(merged), 'groups': len({group for groups in merged.values() for group in groups})}
    return result, merged


def near_screen(records):
    representatives = {}
    for row in records:
        group = row['group_id']
        if group not in representatives or row['id'] < representatives[group]['id']:
            representatives[group] = row
    generator = np.random.default_rng(SEED)
    multiply = generator.integers(1, 2 ** 30, size=32, dtype=np.uint64)
    offsets = generator.integers(0, 2 ** 30, size=32, dtype=np.uint64)
    prime = np.uint64(4294967291)
    buckets, sets, owners, pairs = defaultdict(list), [], [], set()
    sampled = sorted(representatives.items(), key=lambda item: ordered_key(item[0]))[:8000]
    capped_queries = 0
    for position, (group, row) in enumerate(sampled):
        text = normalized(row['state']) + '\n' + normalized(row['question']) + '\n' + '\n'.join(sorted(normalized(option) for option in row['options']))
        words = re.findall(r'\w+|[^\w\s]', text)
        shingles = [' '.join(words[index:index + 5]) for index in range(max(1, len(words) - 4))]
        tokens = frozenset(int.from_bytes(hashlib.blake2s(value.encode(), digest_size=4).digest(), 'little') for value in shingles)
        values = np.fromiter(tokens, dtype=np.uint64)
        signature = ((values[:, None] * multiply + offsets) % prime).min(axis=0)
        keys = [(band, tuple(signature[band * 4:band * 4 + 4])) for band in range(8)]
        candidates = {member for key in keys for member in buckets[key]}
        if len(candidates) > 64:
            capped_queries += 1
            candidates = set(sorted(candidates)[:64])
        for member in candidates:
            other = sets[member]
            if min(len(tokens), len(other)) < .9 * max(len(tokens), len(other)):
                continue
            if len(tokens & other) / len(tokens | other) >= .9:
                pairs.add(tuple(sorted((group, owners[member]))))
        sets.append(tokens)
        owners.append(group)
        for key in keys:
            buckets[key].append(position)
        if position and position % 2000 == 0:
            print(f'near-screen: {position}/{len(sampled)} sampled group representatives', flush=True)
    return {'scope': 'hash-selected 8000 Train Groups, one smallest-id representative each; normalized state+question+sorted options',
        'candidate_generation': '32-permutation MinHash, 8 bands of 4; approximate retrieval',
        'verification': '5-token shingle Jaccard >= 0.90', 'representatives': len(sampled), 'capped_queries': capped_queries,
        'pairs': len(pairs), 'groups': len({group for pair in pairs for group in pair}),
        'limits': 'Sampled diagnosis, at most 64 candidates/query; not exhaustive or semantic proof; no automatic deletion'}, sorted(pairs)


def tokenize_development(splits, output):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path('repos/Open-Jev').resolve()))
    renderer = importlib.import_module('jev.api').candidate_prompts
    records = splits['train'] + splits['validation']
    features = {key: np.array([row[key] for row in records]) for key in ['id', 'group_id', 'split', 'kind', 'source']}
    reports, tokenizer_hashes = {}, {}
    for model_name in ['Ouro-1.4B', 'Qwen3-1.7B-Base']:
        root = Path('models') / model_name
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
        tokenizer_hashes[model_name] = {str(path): digest(path) for path in root.glob('*') if path.is_file() and path.name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']}
        maxima, totals, counts = [], [], []
        for start in range(0, len(records), 256):
            texts, sizes = [], []
            for row in records[start:start + 256]:
                prompts = renderer(row)
                sizes.append(len(prompts))
                for prompt in prompts:
                    rendered = prompt if model_name == 'Qwen3-1.7B-Base' else tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    texts.append(rendered)
            lengths = tokenizer(texts, add_special_tokens=False, truncation=False, return_length=True, return_attention_mask=False)['length']
            cursor = 0
            for count in sizes:
                values = lengths[cursor:cursor + count]
                maxima.append(max(values))
                totals.append(sum(values))
                counts.append(count)
                cursor += count
            if start % 8192 == 0:
                print(f'tokens: {model_name}: {start}/{len(records)} records', flush=True)
                write_json(output / 'results/status.json', {'status': 'running', 'stage': 'development tokenization',
                    'model': model_name, 'records_processed': start, 'records_total': len(records)})
        maxima, totals, counts = np.array(maxima), np.array(totals), np.array(counts)
        key = model_name.replace('-', '_').replace('.', '_')
        features[key + '_max'] = maxima
        features[key + '_sum'] = totals
        features[key + '_inputs'] = counts
        reports[model_name] = {}
        for split in ['train', 'validation']:
            reports[model_name][split] = {}
            for kind in KINDS:
                mask = (features['split'] == split) & (features['kind'] == kind)
                reports[model_name][split][kind] = {'max_input_tokens': summarize(maxima[mask]),
                    'total_input_tokens_per_record': summarize(totals[mask]), 'encoded_inputs': int(counts[mask].sum()),
                    'over_length_records': {str(limit): int((maxima[mask] > limit).sum()) for limit in [2048, 4096, 8192]}}
    cache = output / 'cache/token_lengths.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **features)
    return reports, features, tokenizer_hashes, {'path': str(cache), 'sha256': digest(cache), 'bytes': cache.stat().st_size}


def apportion(counts, total):
    denominator = sum(counts.values())
    if not 0 <= total <= denominator:
        raise ValueError('Infeasible apportionment')
    assigned = {key: total * value // denominator for key, value in counts.items()}
    ranking = sorted(counts, key=lambda key: (-(total * counts[key] % denominator), str(key)))
    for key in ranking[:total - sum(assigned.values())]:
        assigned[key] += 1
    return assigned


def draft_allocation(records, excluded, name):
    eligible = [row for row in records if row['group_id'] not in excluded]
    groups = defaultdict(list)
    for row in eligible:
        groups[row['group_id']].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        signature = (tuple(sorted({row['source'] for row in members})), tuple(sorted({row['kind'] for row in members})))
        strata[signature].append(group)
    quota = apportion({key: len(value) for key, value in strata.items()}, 12000)
    reserved = {key: sorted(members, key=ordered_key)[:quota[key]] for key, members in strata.items()}
    train_quota = apportion({key: len(value) for key, value in reserved.items()}, 8000)
    val_quota = apportion({key: len(value) - train_quota[key] for key, value in reserved.items()}, 2000)
    roles = {'risk_train': set(), 'risk_val': set(), 'policy_fit': set()}
    for key, members in reserved.items():
        first, second = train_quota[key], train_quota[key] + val_quota[key]
        roles['risk_train'].update(members[:first])
        roles['risk_val'].update(members[first:second])
        roles['policy_fit'].update(members[second:])
    occupied = set.union(*roles.values())
    pool = [row for row in eligible if row['group_id'] not in occupied]
    buckets = defaultdict(list)
    for row in pool:
        buckets[(row['source'], row['kind'])].append(row)
    selected, shortages = [], {}
    for key, count in apportion(Counter((row['source'], row['kind']) for row in records), 64000).items():
        available = sorted(buckets[key], key=lambda row: ordered_key(row['id']))
        selected.extend(available[:count])
        if len(available) < count:
            shortages[' / '.join(key)] = {'target': count, 'available': len(available)}
    return {'condition': name, 'eligible': distribution(eligible), 'after_risk_reservation': distribution(pool),
        'stage1_draft': distribution(selected), 'source_kind_shortages': shortages,
        'risk_roles': {role: distribution([row for group in members for row in groups[group]]) for role, members in roles.items()},
        'seed': SEED, 'frozen': False, 'group_disjoint': not bool({row['group_id'] for row in selected} & occupied),
        'limits': 'Feasibility draft only; all eligible records of each reserved Group kept; no manifests frozen'}


def audit_data(data_root, output):
    started = time.perf_counter()
    results = output / 'results'
    if results.exists():
        raise FileExistsError(f'Use a new empty audit output: {results}')
    results.mkdir(parents=True)
    write_json(results / 'status.json', {'status': 'running', 'stage': 'release integrity'})
    splits, provenance, revision = load_release(data_root)
    used, history = historical_inventory()
    overlaps, identical = overlap_report(splits)
    near, pairs = near_screen(splits['train'])
    write_json(results / 'status.json', {'status': 'running', 'stage': 'development tokenization'})
    tokens, features, tokenizer_hashes, cache = tokenize_development(splits, output)
    mask = (features['Ouro_1_4B_max'] <= 4096) & (features['Qwen3_1_7B_Base_max'] <= 4096)
    accepted = {row_id for row_id, keep in zip(features['id'], mask) if keep}
    train = [row for row in splits['train'] if row['id'] in accepted]
    drafts = {name: draft_allocation(train, exclude, name) for name, exclude in [('retain_historical_train', set()), ('exclude_historical_train', used)]}
    write_json(results / 'allocation_drafts.json', drafts)
    write_json(results / 'duplicate_screen.json', {'exact': overlaps, 'train_identical_input_groups': identical,
        'near': near, 'near_pairs': pairs})
    metrics = {'purpose': 'Main-experiment integrity, coverage and allocation audit; no model evaluation',
        'dataset_revision': revision, 'splits': {name: distribution(records) for name, records in splits.items()},
        'after_historical_exclusion': {name: distribution([row for row in records if row['group_id'] not in used]) for name, records in splits.items()},
        'historical_inventory': history, 'historical_group_count': len(used), 'overlap': overlaps,
        'near_duplicate_screen': near, 'token_lengths': tokens, 'token_cache': cache,
        'common_4096_train': distribution(train), 'common_4096_rejected_train_records': len(splits['train']) - len(train),
        'common_4096_rejected_validation_records': sum(row['id'] not in accepted for row in splits['validation']),
        'elapsed_seconds': time.perf_counter() - started,
        'limits': ['Official Calibration/Test/OOD decoded for integrity and aggregate metadata only; never tokenized or evaluated',
            'Length rejection is a draft; held-out long inputs must remain visible in coverage reporting',
            'Historical inventory covers known U4 caches and downstream manifests, not arbitrary unrecorded manual use',
            'Near screen is approximate on one representative per Train Group; no semantic decontamination proof',
            'No formal manifests or model hyperparameters frozen; allocation drafts exclude no near pairs yet']}
    sources = [Path(__file__).relative_to(ROOT), Path('experiments/run.py'), Path('repos/Open-Jev/jev/api.py'), data_root / 'export-manifest.json']
    write_json(results / 'protocol.json', {'seed': SEED, 'data_sha256': provenance,
        'sources_sha256': {str(path): digest(path) for path in sources}, 'tokenizers_sha256': tokenizer_hashes,
        'dataset_revision': revision, 'maximum_input_tokens_draft': 4096, 'no_truncation': True,
        'kinds': KINDS, 'tokenized_splits': ['train', 'validation'], 'historical_inventory': history,
        'input_formats': {'Ouro-1.4B': 'native chat template, generation prompt, no reasoning',
            'Qwen3-1.7B-Base': 'raw official candidate prompt; base-model baseline, no chat template'},
        'no_model_weights_loaded': True})
    write_json(results / 'metrics.json', metrics)
    write_json(results / 'status.json', {'status': 'completed', 'seconds': metrics['elapsed_seconds']})
    print(json.dumps({'revision': revision, 'seconds': metrics['elapsed_seconds'], 'metrics': str(results / 'metrics.json')}, ensure_ascii=False), flush=True)
