import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardRun:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.writer is not None:
            self.writer.close()

    def start(self, config, context, completed=0, resume=False):
        self.writer = SummaryWriter(str(self.directory), max_queue=100, flush_secs=30,
            purge_step=completed + 1 if resume else None)
        if not resume:
            self.writer.add_text('run/config', '```json\n' + json.dumps(config, indent=2) + '\n```', 0)
            self.writer.add_text('run/context', '```json\n' + json.dumps(context, indent=2) + '\n```', 0)
        self.flush()

    def scalars(self, values, update):
        for name, value in values.items():
            if value is not None and math.isfinite(float(value)):
                self.writer.add_scalar(name, float(value), update)

    def training(self, update, loss, loop_losses, gradient_norm, epoch, learning_rates, stats, step_seconds,
        batch_records, batch_tokens, peak_bytes):
        self.scalars({'train/decision_loss': loss, 'train/gradient_norm_before_clip': gradient_norm,
            'train/epoch': epoch, 'optimizer/head_lr': learning_rates[0], 'optimizer/lora_lr': learning_rates[1],
            'time/update_seconds': step_seconds, 'time/training_seconds': stats['training_wall_seconds'],
            'throughput/records_per_second': batch_records / step_seconds,
            'throughput/input_tokens_per_second': batch_tokens / step_seconds,
            'data/records_seen': stats['records'], 'data/encoded_inputs_seen': stats['encoded_inputs'],
            'data/input_tokens_seen': stats['unpadded_tokens'], 'data/padded_tokens_seen': stats['padded_tokens'],
            'memory/peak_allocated_gib': peak_bytes / 1024 ** 3}, update)
        self.scalars({f'train/loop_{index}/decision_loss': value for index, value in enumerate(loop_losses, 1)}, update)
        self.flush()

    def evaluation(self, report, role, update):
        metrics = ('nll', 'brier', 'accuracy', 'ece_15_equal_width', 'score_expectation_mae')
        for loop, variants in report['loops'].items():
            for variant, result in variants.items():
                if result is None:
                    continue
                scopes = {name: result[name] for name in ('record_weighted', 'task_macro', 'source_task_macro')}
                scopes.update({f'by_kind/{kind}': values for kind, values in result['by_kind'].items()})
                for scope, values in scopes.items():
                    self.scalars({f'eval/{role}/loop_{loop}/{variant}/{scope}/{name}': values.get(name)
                        for name in metrics}, update)
        self.scalars({f'eval/{role}/cost/{name}': report['cost'][name]
            for name in ('wall_seconds_including_encoding', 'records', 'encoded_inputs', 'unpadded_tokens', 'padded_tokens', 'executed_loops')}, update)
        self.flush()

    def flush(self):
        if self.writer is not None:
            self.writer.flush()
