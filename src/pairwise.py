"""Stable all-pairs binary ranking loss and opt-in Train-only accounting."""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def ranking_term(logits, targets):
    # Retain float64 for gradient checking; promote half precision for stable softplus.
    z = logits.flatten()
    if z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    y = targets.flatten()
    if z.shape != y.shape or not len(z) or not torch.all((y == 0) | (y == 1)):
        raise ValueError('Pairwise ranking requires nonempty aligned binary labels')
    positive, negative = z[y == 1], z[y == 0]
    count = positive.numel() * negative.numel()
    return (F.softplus(negative[None, :] - positive[:, None]).mean() if count else z.sum()*0), count


class LossAccounting:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = self.positive = self.negative = self.batches = self.valid_batches = self.pairs = 0
        self.bce = self.pairwise = self.weighted_pairwise = self.pair_sum = 0.

    def observe(self, bce, pairwise, targets, count, weight):
        n = targets.numel(); positives = int((targets == 1).sum())
        b, r = float(bce.detach()), float(pairwise.detach())
        self.samples += n; self.positive += positives; self.negative += n-positives
        self.batches += 1; self.valid_batches += int(count > 0); self.pairs += count
        self.bce += b*n; self.pairwise += r*n; self.weighted_pairwise += weight*r*n
        self.pair_sum += r*count

    def finish_epoch(self):
        if not self.samples:
            raise ValueError('No Train batches recorded')
        out = {'positive_draws': self.positive, 'negative_draws': self.negative,
               'batches': self.batches, 'valid_pair_batches': self.valid_batches,
               'valid_batch_fraction': self.valid_batches/self.batches, 'effective_pair_count': self.pairs,
               'bce_loss': self.bce/self.samples, 'pairwise_loss': self.pairwise/self.samples,
               'weighted_pairwise_loss': self.weighted_pairwise/self.samples,
               'weighted_pairwise_over_bce': self.weighted_pairwise/self.bce if self.bce else None,
               'pairwise_over_bce': self.pairwise/self.bce if self.bce else None,
               'mean_pair_loss_over_valid_pairs': self.pair_sum/self.pairs if self.pairs else 0.,
               'combined_loss': (self.bce+self.weighted_pairwise)/self.samples,
               'loss_mean_scope': 'sample-weighted batch means; absent pair term is zero'}
        self.reset()
        return out


_monitor = None


@contextmanager
def monitor_training(monitor):
    global _monitor
    if _monitor is not None:
        raise RuntimeError('Nested training accounting is unsupported')
    _monitor = monitor
    try:
        yield monitor
    finally:
        _monitor = None


def bce_pairwise(logits, targets, weight):
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid ranking weight')
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    pairwise, count = ranking_term(logits, targets)
    if _monitor is not None:
        _monitor.observe(bce, pairwise, targets, count, weight)
    # Exact BCE return at lambda=0; avoids modifying its autograd graph.
    return bce if weight == 0 or count == 0 else bce + weight*pairwise
