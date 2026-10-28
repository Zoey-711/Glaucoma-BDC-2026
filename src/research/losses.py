import torch
from torch import nn
from torch.nn import functional as F


class BinaryObjective(nn.Module):
    def __init__(self, spec, positives, negatives):
        super().__init__();self.spec=spec;self.last_pairs=0
        ratio=negatives/max(positives,1)
        mode=spec.get('weight_mode','none')
        if mode not in {'none','ratio','sqrt'}:raise ValueError('Unknown weight mode')
        self.register_buffer('pos_weight',torch.tensor(1. if mode=='none' else ratio if mode=='ratio' else ratio**.5))

    def forward(self,logits,targets):
        logits=logits.float().flatten();targets=targets.float().flatten();self.last_pairs=0
        name=self.spec['name']
        if name in {'bce','weighted_bce','pairwise_auc'}:
            loss=F.binary_cross_entropy_with_logits(logits,targets,pos_weight=self.pos_weight)
            if name!='pairwise_auc':return loss
            positive=logits[targets>.5];negative=logits[targets<=.5]
            self.last_pairs=positive.numel()*negative.numel()
            if not self.last_pairs:return loss
            ranking=F.softplus(negative[None,:]-positive[:,None]).mean()
            return loss+float(self.spec.get('lambda',0))*ranking
        if name=='focal':
            gamma=float(self.spec.get('gamma',2));alpha=float(self.spec.get('alpha',.25))
            bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
            pt=torch.exp(-bce);weight=alpha*targets+(1-alpha)*(1-targets)
            return (weight*(1-pt)**gamma*bce).mean()
        raise ValueError('Unknown objective')


def bernoulli_kl(student_logits,teacher_logits,temperature):
    t=float(temperature)
    if t<=0:raise ValueError('Temperature must be positive')
    p=torch.sigmoid(teacher_logits.detach().float()/t).clamp(1e-6,1-1e-6)
    q=student_logits.float()/t
    return (p*(p.log()-F.logsigmoid(q))+(1-p)*((1-p).log()-F.logsigmoid(-q))).mean()*t*t
