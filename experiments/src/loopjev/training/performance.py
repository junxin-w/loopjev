from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

from loopjev.io import digest, write_json
from loopjev.data.encoding_cache import cached_encodings
from loopjev.data.records import RecordEncoder, load_lock, load_records, smoke_records, verify_roles
from loopjev.models.decision import DecisionModel
from loopjev.paths import ROOT
from loopjev.training.step import make_optimizer, batch_step
from loopjev.runtime import load_config, configure_runtime, source_hashes
from loopjev.training.tasks import supervision_loss


def performance_records(records, count, length_file):
    lengths = np.load(length_file)
    maximum = np.maximum(lengths['Ouro_1_4B_max'], lengths['Qwen3_1_7B_Base_max'])
    total = np.maximum(lengths['Ouro_1_4B_sum'], lengths['Qwen3_1_7B_Base_sum'])
    lookup = {str(identifier): (int(width), int(tokens))
        for identifier, width, tokens in zip(lengths['id'], maximum, total, strict=True)}
    kinds = sorted({row['kind'] for row in records})
    weights = Counter(row['kind'] for row in records)
    allocated = {kind: int(count * weights[kind] / len(records)) for kind in kinds}
    for kind in sorted(kinds, key=lambda member: (-(count * weights[member] / len(records) - allocated[member]), member))[:count - sum(allocated.values())]:
        allocated[kind] += 1
    selected = []
    for kind in kinds:
        members = sorted((row for row in records if row['kind'] == kind),
            key=lambda row: (*lookup[row['id']], hashlib.sha256(row['id'].encode()).hexdigest()))
        positions = np.linspace(0, len(members) - 1, allocated[kind], dtype=int)
        selected.extend(members[index] for index in positions)
    order = np.random.default_rng(617).permutation(len(selected))
    return [selected[index] for index in order]


class GPUSampler:
    def __init__(self):
        self.samples = []
        self.finished = threading.Event()
        self.worker = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.finished.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--id=0',
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=3)
                self.samples.append([float(value.strip()) for value in result.stdout.strip().split(',')])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.finished.wait(1.)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.finished.set()
        self.worker.join(timeout=4)

    def report(self):
        if not self.samples:
            return {'samples': 0}
        means = np.asarray(self.samples).mean(0)
        return {'samples': len(self.samples), **dict(zip(
            ('gpu_busy_percent_mean', 'memory_controller_busy_percent_mean', 'device_memory_mib_mean', 'power_watts_mean'),
            means.tolist(), strict=True)),
            'scope': 'nvidia-smi device-wide busy percentage, not an SM occupancy counter'}


def benchmark(args):
    config, lock = load_config(args.config), load_lock(args.data_lock)
    config = dict(config)
    if args.variant == 'serial':
        config.update(candidate_batch=4, padded_token_budget=8192, record_microbatch=1,
            microbatch_token_budget=4096, token_cache=False, fused_adamw=False)
    elif args.variant != 'current':
        config.update(token_cache=True, fused_adamw=True)
        if args.variant.startswith('causal'):
            config['causal_padding_fastpath'] = True
        if args.variant.startswith('budget') or args.variant in ('causal8k', 'causal16k', 'causal32k'):
            budget = int(args.variant.removeprefix('budget').removeprefix('causal').removesuffix('k')) * 1024
            config.update(record_microbatch=4, microbatch_token_budget=budget,
                candidate_batch=max(1, budget // 1024), padded_token_budget=budget)
        elif args.variant in ('no-checkpoint', 'causal-no-checkpoint'):
            config.update(gradient_checkpointing=False, record_microbatch=1,
                candidate_batch=4, padded_token_budget=8192, microbatch_token_budget=4096)
    verify_roles(lock)
    if args.warmup < 1 or args.updates < 30 or args.records < config['batch_records']:
        raise ValueError('Benchmark requires warmup and at least 30 measured updates')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    records = performance_records(load_records('stage1', lock), args.records,
        ROOT / 'experiments/outputs/data/audit/cache/token_lengths.npz')
    members = [{key: row[key] for key in ('id', 'kind', 'source', 'group_id')} for row in records]
    write_json(output / 'records.json', members)
    protocol = {'benchmark_only': True, 'variant': args.variant, 'config': config, 'warmup_updates': args.warmup,
        'measured_updates': args.updates, 'records_sha256': digest(output / 'records.json'),
        'data_lock_sha256': digest(ROOT / args.data_lock), 'sources_sha256': source_hashes(),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'sampling': 'Stage I only; proportional task counts, evenly spaced length ranks including extremes; fixed shuffle',
        'schedule': 'cyclic fixed record batches; constant LR for timing; no checkpoint/evaluation; not a quality experiment',
        'gpu_metrics': 'device-wide nvidia-smi utilization; no claim of measured SM occupancy'}
    write_json(output / 'protocol.json', protocol)
    configure_runtime(config)
    timings, totals, measured = [], Counter(), 0
    started = time.perf_counter()
    try:
        model = DecisionModel(config, lock)
        encoder = RecordEncoder(model.tokenizer, config['family'], lock)
        cache_started = time.perf_counter()
        cache = cached_encodings(records, encoder, lock, ROOT / 'experiments/outputs/cache/tokenized') if config.get('token_cache', False) else None
        cache_seconds = time.perf_counter() - cache_started
        optimizer = make_optimizer(model, config)
        batches = [list(range(start, min(start + config['batch_records'], len(records))))
            for start in range(0, len(records), config['batch_records'])]
        model.train()
        with GPUSampler() as sampler:
            for update in range(args.warmup + args.updates):
                if update == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                    sampler.samples.clear()
                indices = batches[update % len(batches)]
                rows = [records[index] for index in indices]
                torch.cuda.synchronize()
                step_started = time.perf_counter()
                encoded = [cache[index] for index in indices] if cache is not None else encoder.encode_many(rows)
                value = batch_step(model, optimizer, rows, encoded, config)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - step_started
                if update >= args.warmup:
                    measured += 1
                    timings.append(seconds)
                    totals.update({'records': len(rows), 'candidates': sum(map(len, encoded)),
                        'candidate_tokens': sum(len(sequence) for candidates in encoded for sequence in candidates),
                        'padded_tokens': value['padded_tokens'], 'backbone_calls': value['forward_calls']})
                    if measured == 1 or measured % 5 == 0 or measured == args.updates:
                        print(f'benchmark={measured}/{args.updates} seconds={seconds:.3f} loss={float(value["loss"]):.5f}', flush=True)
            wall = sum(timings)
            report = {'status': 'completed', 'totals': dict(totals), 'measured_updates': measured,
                'warmup_updates': args.warmup, 'update_seconds': timings, 'total_update_seconds': wall,
                'update_seconds_mean': float(np.mean(timings)), 'update_seconds_p50': float(np.median(timings)),
                'update_seconds_p95': float(np.quantile(timings, .95)),
                'records_per_second': totals['records'] / wall, 'candidate_tokens_per_second': totals['candidate_tokens'] / wall,
                'tokens_per_update': totals['candidate_tokens'] / measured,
                'padding_multiplier': totals['padded_tokens'] / totals['candidate_tokens'],
                'backbone_calls_per_update': totals['backbone_calls'] / measured,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
                'gpu': sampler.report(), 'cache_setup_seconds': cache_seconds,
                'token_cache_key': cache.metadata['key'] if cache is not None else None,
                'actual_optimizer_fused': optimizer.defaults['fused']}
        if args.profile:
            indices = batches[0]
            rows = [records[index] for index in indices]
            encoded = [cache[index] for index in indices] if cache is not None else encoder.encode_many(rows)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
                batch_step(model, optimizer, rows, encoded, config)
                torch.cuda.synchronize()
            events = profiler.key_averages()
            (output / 'profile.txt').write_text(events.table(sort_by='self_cuda_time_total', row_limit=25))
            report['attention_events'] = [entry.key for entry in events
                if any(word in entry.key.lower() for word in ('attention', 'fmha', 'flash', 'cudnn', 'sdp'))]
            report['profiler_scope'] = 'one extra update excluded from measured throughput; aggregate only, no large trace'
    except torch.OutOfMemoryError:
        report = {'status': 'out_of_memory', 'completed_measured_updates': measured,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
            'scope': 'this exact configuration; does not rule out all configurations without checkpointing'}
        gc.collect()
        torch.cuda.empty_cache()
    report['process_wall_seconds'] = time.perf_counter() - started
    write_json(output / 'metrics.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def verify_batching(args):
    config, lock = load_config(args.config), load_lock(args.data_lock)
    config = dict(config) | {'causal_padding_fastpath': args.causal_fastpath}
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(config)
    records = smoke_records(load_records('stage1', lock), per_kind=2)
    model = DecisionModel(config, lock)
    if args.fp32:
        model.backbone.float()
    encoder = RecordEncoder(model.tokenizer, config['family'], lock)
    encoded = encoder.encode_many(records)
    cache = cached_encodings(records, encoder, lock, ROOT / 'experiments/outputs/cache/tokenized')
    if any(cache[index] != sequence for index, sequence in enumerate(encoded)):
        raise ValueError('Cached tokens differ from online encoding')
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    tolerances = {'logits_atol': .05, 'logits_rtol': .01, 'probability_atol': .02, 'loss_atol': .005,
        'gradient_relative_l2_max': .05, 'gradient_cosine_min': .995}
    write_json(output / 'protocol.json', {'config': config, 'tolerances': tolerances, 'fp32': args.fp32,
        'sources_sha256': source_hashes(), 'records': [row['id'] for row in records]})
    def capture(grouped):
        model.train()
        model.zero_grad(set_to_none=True)
        all_logits, value = [], 0.
        settings = dict(model.settings)
        if not grouped:
            model.settings = settings | {'candidate_batch': 1, 'padded_token_budget': 2048, 'causal_padding_fastpath': False}
        for start in range(0, len(records), len(records) if grouped else 1):
            rows = records[start:start + (len(records) if grouped else 1)]
            sequences = encoded[start:start + len(rows)]
            logits, _ = model.forward_records(rows, sequences)
            loss = torch.stack([supervision_loss(member, row['target'], config['supervision'], config['brier_weight'])[0]
                for row, member in zip(rows, logits, strict=True)]).sum() / len(records)
            loss.backward()
            value += float(loss.detach())
            all_logits.extend(member.detach().cpu() for member in logits)
        gradients = torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten().cpu()
            for parameter in trainable])
        model.settings = settings
        return all_logits, value, gradients
    single, single_loss, single_gradient = capture(False)
    grouped, grouped_loss, grouped_gradient = capture(True)
    relative = float((single_gradient - grouped_gradient).norm() / single_gradient.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(single_gradient.double(), grouped_gradient.double(), dim=0))
    strict_logits_passed = all(torch.allclose(left, right, atol=tolerances['logits_atol'], rtol=tolerances['logits_rtol'])
            for left, right in zip(single, grouped, strict=True))
    max_probability_difference = max(float((left.softmax(-1) - right.softmax(-1)).abs().max())
        for left, right in zip(single, grouped, strict=True))
    passed = (max_probability_difference <= tolerances['probability_atol']
        and abs(single_loss - grouped_loss) <= tolerances['loss_atol']
        and relative <= tolerances['gradient_relative_l2_max'] and cosine >= tolerances['gradient_cosine_min'])
    report = {'passed': passed, 'strict_logits_passed': strict_logits_passed, 'fp32': args.fp32, 'records': len(records), 'task_types': dict(Counter(row['kind'] for row in records)),
        'max_logit_absolute_difference': max(float((left - right).abs().max()) for left, right in zip(single, grouped, strict=True)),
        'single_candidate_loss': single_loss, 'grouped_loss': grouped_loss,
        'loss_absolute_difference': abs(single_loss - grouped_loss), 'gradient_relative_l2': relative,
        'gradient_cosine': cosine, 'cache_tokens_equal': True, 'tolerances': tolerances,
        'semantic_probability_absolute_max': max_probability_difference,
        'by_record': [{'kind': row['kind'], 'id': row['id'],
            'max_logit_absolute_difference': float((left - right).abs().max()),
            'max_probability_absolute_difference': float((left.softmax(-1) - right.softmax(-1)).abs().max()),
            'max_centered_logit_absolute_difference': float(((left - left.mean(-1, keepdim=True)) - (right - right.mean(-1, keepdim=True))).abs().max())}
            for row, left, right in zip(records, single, grouped, strict=True)],
        'scope': 'real backbone training-mode logits and gradients; one sequence versus right-padded, sorted candidates; no optimizer step'}
    write_json(output / 'metrics.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not passed:
        raise ValueError('Batching equivalence exceeded predeclared tolerances')
