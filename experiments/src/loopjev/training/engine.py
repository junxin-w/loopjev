import csv
from importlib.metadata import version
import json
import math
import os
import time

import numpy as np
import torch

from loopjev.io import digest, write_json
from loopjev.data.records import RecordEncoder, load_lock, load_records, smoke_records, verify_roles
from loopjev.data.encoding_cache import cached_encodings
from loopjev.paths import ROOT
from loopjev.models.decision import DecisionModel
from loopjev.training.tracking import TensorBoardRun
from loopjev.training.checkpoints import checkpoint_reader, checkpoint_save
from loopjev.training.step import make_optimizer, batch_step
from loopjev.evaluation.engine import save_evaluation
from loopjev.runtime import load_config, configure_runtime, source_hashes, model_provenance
from loopjev.training.stopping import StopRequest


def preflight(args):
    config, lock = load_config(args.config), load_lock(args.data_lock)
    roles = verify_roles(lock)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], trust_remote_code=True, local_files_only=True)
    encoder = RecordEncoder(tokenizer, config['family'], lock)
    records = smoke_records(load_records('stage1', lock), args.per_kind)
    lengths = [list(map(len, encoder(row))) for row in records]
    print(json.dumps({'status': 'CPU preflight passed; no weights loaded', 'roles': roles,
        'kinds': [row['kind'] for row in records], 'encoded_inputs': sum(map(len, lengths)),
        'max_input_tokens': max(max(value) for value in lengths)}, ensure_ascii=False), flush=True)


def train(args):
    from loopjev.input_protocol import validate_output
    validate_output(ROOT / args.output, load_config(args.config))
    with TensorBoardRun(ROOT / args.output / 'tensorboard') as tracking, StopRequest() as stop:
        _train(args, tracking, stop)


def _train(args, tracking, stop):
    config, lock = load_config(args.config), load_lock(args.data_lock)
    if not args.smoke and not config['approved_for_full_training']:
        raise ValueError('Training hyperparameters are drafts; only --smoke is enabled before approval')
    configure_runtime(config)
    verify_roles(lock)
    output = ROOT / args.output
    completed, previous_state, model = 0, None, None
    if args.resume:
        with checkpoint_reader(output / 'checkpoint', lock, recover=True) as checkpoint:
            saved = checkpoint.manifest
            protocol = saved['protocol']
            if (saved['config'] != config or protocol['smoke_only'] != args.smoke
                    or protocol['smoke_updates'] != args.smoke_updates
                    or protocol.get('smoke_per_kind', 1) != args.smoke_per_kind
                    or protocol['sources_sha256'] != source_hashes()):
                raise ValueError('Resume configuration/code differs from the checkpoint')
            previous_state = torch.load(checkpoint.path / 'training.pt', map_location='cpu', weights_only=True)
            completed = previous_state['completed_updates']
            model = DecisionModel(config, lock, checkpoint.path)
    else:
        if output.exists():
            raise FileExistsError('Use a new output condition')
        output.mkdir(parents=True)
        protocol = {'smoke_only': args.smoke, 'smoke_updates': args.smoke_updates, 'smoke_per_kind': args.smoke_per_kind, 'config_path': args.config,
            'input_protocol': config['input_protocol'], 'experiment_id': config['experiment_id'],
            'data_lock': args.data_lock, 'data_lock_sha256': digest(ROOT / args.data_lock),
            'dataset_revision': lock['dataset_revision'], 'sources_sha256': source_hashes(),
            'model_files_sha256': model_provenance(config),
            'packages': {name: version(name) for name in ('torch', 'transformers', 'peft', 'numpy', 'pyarrow', 'tensorboard')},
            'tracking': {'backend': 'tensorboard', 'directory': 'tensorboard', 'log_every': config['log_every'],
                'train_loss': 'current optimizer batch before update; not an interval average',
                'evaluation_frequency': 'end of training only'},
            'initialization': 'independent base model, Yes-minus-No scalar head, zero trainable bias',
            'loss_reduction': 'sum-class Brier; CE+0.1 Brier; mean over complete Records and supervised loops',
            'gpu': torch.cuda.get_device_name()}
        write_json(output / 'protocol.json', protocol | {'config': config})
    records = load_records('stage1', lock)
    validation = load_records('validation', lock)
    if args.smoke:
        records, validation = smoke_records(records, args.smoke_per_kind), smoke_records(validation, args.smoke_per_kind)
    if model is None:
        model = DecisionModel(config, lock)
    encoder = RecordEncoder(model.tokenizer, config['family'], lock)
    cached = cached_encodings(records, encoder, lock, ROOT / 'experiments/outputs/cache/tokenized') if config.get('token_cache', False) else None
    if not args.resume:
        protocol['encoding'] = {'mode': 'shared memory-mapped tokens' if cached is not None else 'online batched tokenizer',
            'token_cache_key': cached.metadata['key'] if cached is not None else None}
        write_json(output / 'protocol.json', protocol | {'config': config})
    adapters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = make_optimizer(model, config)
    stats = {'records': 0, 'encoded_inputs': 0, 'unpadded_tokens': 0, 'padded_tokens': 0, 'training_wall_seconds': 0.}
    if previous_state:
        optimizer.load_state_dict(previous_state['optimizer'])
        stats = previous_state['statistics']
        torch.set_rng_state(previous_state['cpu_rng'])
        torch.cuda.set_rng_state_all(previous_state['cuda_rng'])
    batch_size = len(records) if args.smoke else config['batch_records']
    epochs = args.smoke_updates if args.smoke else config['epochs']
    total_updates = math.ceil(len(records) / batch_size) * epochs
    if args.stop_after_updates is not None and not completed < args.stop_after_updates <= total_updates:
        raise ValueError('Stop point must follow the saved update and stay within the original schedule')
    curve = output / 'training_curve.csv'
    fields = ['update', 'epoch', 'decision_loss', 'loop_losses', 'gradient_norm', 'step_seconds',
        'elapsed_training_seconds', 'records', 'unpadded_tokens', 'padded_tokens', 'head_lr', 'lora_lr', 'peak_allocated_bytes']
    if args.resume:
        with curve.open() as stream:
            retained = [row for row in csv.DictReader(stream) if int(row['update']) <= completed]
        with curve.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
    torch.cuda.reset_peak_memory_stats()
    update = 0
    head_before = model.head.weight.detach().clone()
    adapter_before = {name: value.detach().cpu().clone() for name, value in model.backbone.named_parameters() if value.requires_grad} if args.smoke else None
    tracking.start(config, {'smoke_only': args.smoke, 'dataset_revision': lock['dataset_revision'],
        'data_lock_sha256': protocol['data_lock_sha256'], 'scheduled_updates': total_updates,
        'validation_role': 'three-task functional subset' if args.smoke else 'official development Validation'}, completed, args.resume)
    write_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'completed_updates': completed, 'scheduled_updates': total_updates})
    with curve.open('a' if args.resume else 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for epoch in range(epochs):
            order = np.random.default_rng(config['seed'] + epoch).permutation(len(records)).tolist()
            for start in range(0, len(order), batch_size):
                update += 1
                if update <= completed:
                    continue
                torch.cuda.synchronize()
                started = time.perf_counter()
                fraction = update / total_updates
                multiplier = min(1., fraction / max(config['warmup_ratio'], 1e-12))
                if fraction > config['warmup_ratio']:
                    multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * (fraction - config['warmup_ratio']) / (1 - config['warmup_ratio'])))
                for group, base in zip(optimizer.param_groups, (config['head_lr'], config['lora_lr']), strict=True):
                    group['lr'] = base * (1. if args.smoke else multiplier)
                members = order[start:start + batch_size]
                tokens_before = stats['unpadded_tokens']
                rows = [records[index] for index in members]
                encoded = [cached[index] for index in members] if cached is not None else encoder.encode_many(rows)
                value = batch_step(model, optimizer, rows, encoded, config)
                stats['records'] += len(rows)
                stats['encoded_inputs'] += sum(map(len, encoded))
                stats['unpadded_tokens'] += sum(len(candidate) for candidates in encoded for candidate in candidates)
                stats['padded_tokens'] += value['padded_tokens']
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                stats['training_wall_seconds'] += elapsed
                stopping = stop.requested or update == args.stop_after_updates
                if update == 1 or update % config['log_every'] == 0 or update == total_updates or stopping:
                    total, loop_losses = float(value['loss']), value['loop_losses'].cpu().numpy()
                    norm = float(value['norm'])
                    writer.writerow({'update': update, 'epoch': epoch + 1, 'decision_loss': total,
                        'loop_losses': json.dumps(loop_losses.tolist()), 'gradient_norm': float(norm), 'step_seconds': elapsed,
                        'elapsed_training_seconds': stats['training_wall_seconds'], **{name: stats[name] for name in ('records', 'unpadded_tokens', 'padded_tokens')},
                        'head_lr': optimizer.param_groups[0]['lr'], 'lora_lr': optimizer.param_groups[1]['lr'],
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                    stream.flush()
                    print(f'update={update}/{total_updates} loss={total:.5f} seconds={elapsed:.2f} total_seconds={stats["training_wall_seconds"]:.2f}', flush=True)
                    tracking.training(update, total, loop_losses, float(norm), epoch + 1,
                        [group['lr'] for group in optimizer.param_groups], stats, elapsed, len(members),
                        stats['unpadded_tokens'] - tokens_before, torch.cuda.max_memory_allocated())
                if update % config['checkpoint_every'] == 0 or update == total_updates or stopping:
                    checkpoint_save(model, optimizer, output, config, protocol, update, stats)
                    tracking.flush()
                if stopping and update < total_updates:
                    write_json(output / 'status.json', {'status': 'paused after signal' if stop.requested else 'paused at requested update', 'completed_updates': update,
                        'scheduled_updates': total_updates, 'resume': True})
                    print(f'Paused after update {update}; resume with the same config and output', flush=True)
                    return
    changed_head = not torch.equal(head_before, model.head.weight.detach())
    changed_adapters = None if adapter_before is None else sum(not torch.equal(adapter_before[name], value.detach().cpu())
        for name, value in model.backbone.named_parameters() if name in adapter_before)
    if args.smoke and completed < total_updates and (not changed_head or not changed_adapters):
        raise ValueError('Smoke must update both Decision Head and LoRA')
    if any(value.grad is not None for value in model.backbone.parameters() if not value.requires_grad):
        raise ValueError('Frozen backbone received gradients')
    evaluation = save_evaluation(model, validation, encoder, output / 'validation')
    tracking.evaluation(evaluation, 'validation', total_updates)
    write_json(output / 'metrics.json', {'status': 'SMOKE ONLY' if args.smoke else 'completed development training',
        'input_protocol': config['input_protocol'], 'experiment_id': config['experiment_id'],
        'optimizer_fused': optimizer.defaults['fused'],
        'updates': total_updates, 'training': stats, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'adapter_parameters': sum(value.numel() for value in adapters), 'head_parameters': sum(value.numel() for value in model.head.parameters()),
        'head_updated': changed_head, 'changed_adapter_tensors': changed_adapters,
        'token_loop_equivalents': stats['unpadded_tokens'] * model.loops,
        'validation_role': 'three-task functional subset' if args.smoke else 'official development Validation',
        'validation_records': len(validation), 'validation_raw_final': evaluation['loops'][str(model.loops)]['raw']['record_weighted'],
        'compute_limit': 'token-loop equivalents omit attention length, backward and checkpoint recomputation; not measured FLOPs'})
    write_json(output / 'status.json', {'status': 'completed smoke' if args.smoke else 'completed', 'completed_updates': total_updates})
