"""The same optimizer update is used by training and performance benchmarks."""
import torch

from loopjev.training.batching import record_batches
from loopjev.training.tasks import supervision_loss


def make_optimizer(model, config):
    return torch.optim.AdamW([
        {'params': list(model.head.parameters()), 'lr': config['head_lr']},
        {'params': [value for value in model.backbone.parameters() if value.requires_grad], 'lr': config['lora_lr']}],
        weight_decay=0., fused=config.get('fused_adamw', False))


def batch_step(model, optimizer, rows, encoded, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=model.head.weight.device)
    loop_losses = torch.zeros(model.loops, device=total.device)
    padded, forwards = 0, 0
    for batch in record_batches(rows, encoded, config.get('record_microbatch', 1), config.get('microbatch_token_budget', 4096)):
        members, sequences = zip(*batch)
        logits, count = model.forward_records(members, sequences)
        values = [supervision_loss(value, row['target'], config['supervision'], config['brier_weight'])
            for row, value in zip(members, logits, strict=True)]
        # Divide by the whole update, including an uneven final microbatch.
        loss = torch.stack([value[0] for value in values]).sum() / len(rows)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        total += loss.detach()
        loop_losses += torch.stack([value[1].detach() for value in values]).sum(0) / len(rows)
        padded += count
        forwards += model.forward_calls
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {'loss': total, 'loop_losses': loop_losses, 'norm': norm,
        'padded_tokens': padded, 'forward_calls': forwards}
