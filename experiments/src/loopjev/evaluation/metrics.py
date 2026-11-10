from collections import defaultdict

import numpy as np
from scipy.special import logsumexp


def summarize(records, logits, temperatures=None):
    if len(records) != len(logits) or not records:
        raise ValueError('Empty or unmatched predictions')
    nll, brier, confidence, correct, ordinal = [], [], [], [], []
    for row, scores in zip(records, logits, strict=True):
        temperature = 1. if temperatures is None else temperatures[row['kind']]
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperatures must be finite and positive')
        scores = np.asarray(scores, dtype=np.float64) / temperature
        target = np.asarray(row['target'], dtype=np.float64)
        if scores.shape != target.shape or not np.isfinite(scores).all():
            raise ValueError('Invalid prediction shape/value')
        log_probability = scores - logsumexp(scores)
        probability = np.exp(log_probability)
        nll.append(float(-target @ log_probability))
        brier.append(float(np.square(probability - target).sum()))
        if np.count_nonzero(target) == 1 and target.max() == 1:
            correct.append(float(probability.argmax() == target.argmax()))
            confidence.append(float(probability.max()))
        if row['kind'] == 'score':
            levels = np.asarray(row['metadata']['score_values'], dtype=np.float64)
            if len(levels) != len(scores) or (np.diff(levels) <= 0).any():
                raise ValueError('Score levels are missing or unordered')
            ordinal.append(float(abs(probability @ levels - target @ levels)))
    result = {'records': len(records), 'groups': len({row['group_id'] for row in records}),
        'nll': float(np.mean(nll)), 'brier': float(np.mean(brier)), 'hard_records': len(correct),
        'accuracy': float(np.mean(correct)) if correct else None,
        'score_expectation_mae': float(np.mean(ordinal)) if ordinal else None, 'ece_15_equal_width': None}
    if correct:
        confidence, correct = np.asarray(confidence), np.asarray(correct)
        bins = np.minimum((15 * confidence).astype(int), 14)
        result['ece_15_equal_width'] = float(sum((bins == index).mean() * abs(correct[bins == index].mean() - confidence[bins == index].mean())
            for index in range(15) if (bins == index).any()))
    return result


def evaluate_predictions(records, logits, temperatures=None):
    overall = summarize(records, logits, temperatures)
    kinds, sources = defaultdict(list), defaultdict(list)
    for index, row in enumerate(records):
        kinds[row['kind']].append(index)
        sources[(row['kind'], row['source'])].append(index)

    def selected(indices):
        return summarize([records[index] for index in indices], [logits[index] for index in indices], temperatures)

    by_kind = {kind: selected(indices) for kind, indices in sorted(kinds.items())}
    by_source = {kind: {} for kind in by_kind}
    for (kind, source), indices in sorted(sources.items()):
        by_source[kind][source] = selected(indices)
    fields = ('nll', 'brier', 'accuracy')
    macros = {}
    for label, members in [('task_macro', list(by_kind.values())),
        ('source_task_macro', [part for kind in by_source.values() for part in kind.values()])]:
        macros[label] = {field: float(np.mean([member[field] for member in members if member[field] is not None]))
            if any(member[field] is not None for member in members) else None for field in fields}
    return {'record_weighted': overall, 'by_kind': by_kind, 'by_source_and_kind': by_source, **macros}


def paired_group_interval(differences, group_ids, repeats=1000, seed=0):
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or len(differences) != len(group_ids) or not np.isfinite(differences).all():
        raise ValueError('Paired finite record differences required')
    groups, inverse = np.unique(group_ids, return_inverse=True)
    if len(groups) < 2 or repeats < 1:
        raise ValueError('At least two Groups and positive repeats required')
    sums = np.bincount(inverse, weights=differences)
    counts = np.bincount(inverse)
    generator = np.random.default_rng(seed)
    means = np.empty(repeats)
    for index in range(repeats):
        members = generator.integers(0, len(groups), len(groups))
        means[index] = sums[members].sum() / counts[members].sum()
    return {'mean': float(differences.mean()), 'two_sided_95': np.quantile(means, [.025, .975]).tolist(),
        'one_sided_95_lower': float(np.quantile(means, .05)), 'one_sided_95_upper': float(np.quantile(means, .95)),
        'groups': len(groups), 'records': len(differences), 'repeats': repeats, 'seed': seed}
