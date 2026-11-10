import torch


def class_logits(candidate_scores, kind):
    if candidate_scores.ndim != 2:
        raise ValueError('Expected candidate-by-loop scores')
    if kind == 'noul' and candidate_scores.shape[0] != 2:
        raise ValueError('Dual-input NOUL requires [No, Yes] candidate scores')
    if kind not in ('choice', 'score', 'noul') or candidate_scores.shape[0] < 2:
        raise ValueError('Invalid task/candidate count')
    return candidate_scores.transpose(0, 1)


def decision_losses(logits, target, brier_weight=.1):
    """Targets are validated on the CPU by the record loader.

    The update checks the combined loss and gradients before stepping. Avoid
    synchronizing CUDA once per record to recheck immutable target values.
    """
    target = torch.as_tensor(target, dtype=logits.dtype, device=logits.device)
    if logits.ndim != 2 or target.shape != logits.shape[1:]:
        raise ValueError('Target shape mismatch')
    log_probabilities = logits.log_softmax(-1)
    return -(target * log_probabilities).sum(-1) + brier_weight * (log_probabilities.exp() - target).square().sum(-1)


def supervision_loss(logits, target, supervision, brier_weight=.1):
    losses = decision_losses(logits, target, brier_weight)
    if supervision == 'uniform':
        return losses.mean(), losses
    if supervision in ('last', 'single'):
        return losses[-1], losses
    raise ValueError('Unknown supervision rule')
