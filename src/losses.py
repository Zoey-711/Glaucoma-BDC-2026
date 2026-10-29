import torch
from torch.nn import functional as F


class BinaryLoss(torch.nn.Module):
    def __init__(self, config, counts):
        super().__init__()
        self.config = config
        n0, n1 = counts
        if min(counts) <= 0:
            raise ValueError('Both training classes required')
        self.register_buffer('pos_weight', torch.tensor(n0 / n1))
        beta = config.get('beta', .9999)
        if not 0 <= beta < 1:
            raise ValueError('Class-balanced beta must be in [0,1)')
        weights = torch.tensor([(1 - beta) / (1 - beta ** n) for n in counts])
        self.register_buffer('class_weights', weights / weights.mean())

    def forward(self, logits, y):
        logits, y = logits.float(), y.float()  # Stable logarithms/clipping under AMP.
        c, name = self.config, self.config['name']
        if name == 'bce': return F.binary_cross_entropy_with_logits(logits, y)
        if name == 'pairwise_auc':
            from .pairwise import bce_pairwise
            return bce_pairwise(logits, y, c['lambda'])
        if name == 'weighted_bce': return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        if name == 'class_balanced': return (bce * self.class_weights[y.long()]).mean()
        p = logits.sigmoid()
        if name == 'focal':
            pt = torch.where(y.bool(), p, 1 - p)
            alpha = torch.where(y.bool(), c.get('alpha', .25), 1 - c.get('alpha', .25))
            return (bce * alpha * (1 - pt).pow(c.get('gamma', 2.))).mean()
        if name == 'asl':
            negative = (1 - p + c.get('clip', .05)).clamp(max=1)
            pt = torch.where(y.bool(), p, negative).clamp(min=1e-8)
            gamma = torch.where(y.bool(), c.get('gamma_pos', 1.), c.get('gamma_neg', 4.))
            return (-(1 - pt).pow(gamma) * pt.log()).mean()
        raise ValueError(f'Unknown loss: {name}')
