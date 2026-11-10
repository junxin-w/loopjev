def record_batches(records, sequences, max_records, token_budget):
    current, tokens = [], 0
    for row, candidates in zip(records, sequences, strict=True):
        size = sum(map(len, candidates))
        if current and (len(current) >= max_records or tokens + size > token_budget):
            yield current
            current, tokens = [], 0
        current.append((row, candidates))
        tokens += size
    if current:
        yield current


def candidate_batches(sequences, max_candidates, padded_token_budget):
    current, width = [], 0
    for index in sorted(range(len(sequences)), key=lambda member: len(sequences[member])):
        length = len(sequences[index])
        if not length or length > padded_token_budget:
            raise ValueError('Candidate length exceeds padded-token budget')
        next_width = max(width, length)
        if current and (len(current) >= max_candidates or next_width * (len(current) + 1) > padded_token_budget):
            yield current
            current, width = [], 0
        current.append(index)
        width = max(width, length)
    if current:
        yield current
