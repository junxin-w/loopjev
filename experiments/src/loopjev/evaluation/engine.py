"""Full-depth development prediction and checkpoint-bound evaluation."""
import gzip
import json
import time

import numpy as np
import torch

from loopjev.data.encoding_cache import cached_encodings
from loopjev.data.records import RecordEncoder, load_lock, load_records, smoke_records
from loopjev.training.batching import record_batches
from loopjev.evaluation.metrics import evaluate_predictions
from loopjev.io import digest, write_json
from loopjev.models.decision import DecisionModel
from loopjev.paths import ROOT
from loopjev.runtime import configure_runtime, source_hashes
from loopjev.training.checkpoints import checkpoint_reader
from loopjev.training.tracking import TensorBoardRun


def validate_temperatures(temperatures, loops, kinds):
    if temperatures is None:
        return
    for loop in range(1, loops + 1):
        for kind in kinds:
            value = temperatures[str(loop)][kind]
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Temperatures must be finite and positive')


@torch.inference_mode()
def predict(model, records, encoder):
    model.eval()
    predictions, padded, tokens, inputs, forwards = [], 0, 0, 0, 0
    cache_started = time.perf_counter()
    cached = cached_encodings(records, encoder, model.data_lock, ROOT / 'experiments/outputs/cache/tokenized') if model.settings.get('token_cache', False) else None
    cache_setup_seconds = time.perf_counter() - cache_started
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(records), model.settings.get('batch_records', 16)):
        members = records[start:start + model.settings.get('batch_records', 16)]
        encoded = [cached[index] for index in range(start, start + len(members))] if cached is not None else encoder.encode_many(members)
        for batch in record_batches(members, encoded, model.settings.get('record_microbatch', 1),
                model.settings.get('microbatch_token_budget', 4096)):
            rows, sequences = zip(*batch)
            logits, count = model.forward_records(rows, sequences)
            widths = [value.shape[-1] for value in logits]
            packed = torch.cat(logits, dim=-1).cpu().double().numpy()
            if not np.isfinite(packed).all():
                raise ValueError('Nonfinite evaluation prediction')
            predictions.extend(np.split(packed, np.cumsum(widths)[:-1], axis=-1))
            padded += count
            tokens += sum(len(candidate) for candidates in sequences for candidate in candidates)
            inputs += sum(map(len, sequences))
            forwards += model.forward_calls
    torch.cuda.synchronize()
    return predictions, {'wall_seconds_including_encoding': time.perf_counter() - started,
        'token_cache_key': cached.metadata['key'] if cached is not None else None,
        'token_cache_setup_seconds': cache_setup_seconds,
        'records': len(records), 'encoded_inputs': inputs, 'unpadded_tokens': tokens, 'padded_tokens': padded,
        'executed_loops': model.loops, 'backbone_calls': forwards,
        'record_window': model.settings.get('batch_records', 16),
        'record_microbatch': model.settings.get('record_microbatch', 1),
        'microbatch_token_budget': model.settings.get('microbatch_token_budget', 4096),
        'scope': 'full-depth development evaluation; includes online encoding or cached-token lookup; excludes cache preparation; not dynamic early-exit benchmark'}


def save_evaluation(model, records, encoder, output, temperatures=None):
    from loopjev.input_protocol import identity, validate_output
    validate_output(output, model.settings)
    run_identity = identity(model.settings)
    validate_temperatures(temperatures, model.loops, {row['kind'] for row in records})
    predictions, cost = predict(model, records, encoder)
    loops = {}
    for loop in range(model.loops):
        scores = [value[loop] for value in predictions]
        loops[str(loop + 1)] = {'raw': evaluate_predictions(records, scores),
            'scaled': evaluate_predictions(records, scores, temperatures[str(loop + 1)]) if temperatures else None}
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row, logits in zip(records, predictions, strict=True):
            stream.write(json.dumps({key: row[key] for key in ('id', 'group_id', 'kind', 'source', 'target')} | {
                **run_identity, 'score_values': row['metadata'].get('score_values'), 'logits_by_loop': logits.tolist()}, ensure_ascii=False, allow_nan=False) + '\n')
    report = {**run_identity, 'loops': loops, 'cost': cost, 'temperature_status': 'explicit frozen temperature file' if temperatures else 'unavailable; no temperature fitted',
        'evaluation_role': 'development', 'macro_definition': 'equal observed task types; source_task_macro gives equal weight to each observed source-task cell'}
    write_json(output / 'metrics.json', report)
    return report



def evaluate(args):
    lock = load_lock(args.data_lock)
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    records = load_records(args.role, lock)
    if args.smoke:
        records = smoke_records(records, args.smoke_per_kind)
    temperatures, temperature_hash = None, None
    # Protect only loading, not the long inference pass. Provenance is captured
    # here and never reread through the mutable latest link after inference.
    with checkpoint_reader(ROOT / args.checkpoint, lock) as checkpoint:
        manifest = checkpoint.manifest
        from loopjev.input_protocol import validate_output
        validate_output(output, manifest['config'])
        configure_runtime(manifest['config'])
        if args.temperature_file:
            temperature_path = ROOT / args.temperature_file
            calibration = json.loads(temperature_path.read_text())
            temperature_hash = digest(temperature_path)
            if calibration['checkpoint_sha256'] != checkpoint.sha256:
                raise ValueError('Temperatures belong to a different checkpoint')
            temperatures = calibration['temperatures']
            validate_temperatures(temperatures, manifest['config']['loops'], {row['kind'] for row in records})
        model = DecisionModel(manifest['config'], lock, checkpoint.path)
        completed = manifest.get('completed_updates')
        if completed is None:
            completed = torch.load(checkpoint.path / 'training.pt', map_location='cpu', weights_only=True)['completed_updates']
    encoder = RecordEncoder(model.tokenizer, model.family, lock)
    protocol = {'input_protocol': model.settings['input_protocol'], 'experiment_id': model.settings['experiment_id'],
        'checkpoint_sha256': checkpoint.sha256,
        'checkpoint_directory': str(checkpoint.path.relative_to(ROOT)), 'completed_updates': completed,
        'role': args.role, 'smoke_only': args.smoke or manifest['protocol']['smoke_only'],
        'data_lock_sha256': digest(ROOT / args.data_lock), 'sources_sha256': source_hashes(),
        'temperature_file_sha256': temperature_hash}
    write_json(output / 'protocol.json', protocol)
    report = save_evaluation(model, records, encoder, output, temperatures)
    with TensorBoardRun(output / 'tensorboard') as tracking:
        tracking.start(manifest['config'], protocol)
        tracking.evaluation(report, args.role, completed)
    print(json.dumps({'records': report['cost']['records'], 'loops': model.loops, 'output': str(output)}), flush=True)
