"""Synthetic V2-S forward/backward and sampler check; never reads official images."""
import argparse
import json

import numpy as np
import torch

from src.research.config import ExperimentConfig
from src.research.data import BalancedBatchSampler,EpochSamplingDiagnostics
from src.research.models import make_model,parameter_count


def preflight(config,device='cuda'):
    cfg=ExperimentConfig.load(config)
    if cfg.experiment_id not in {'H04_RANDOM','H04_BALANCED'}:
        raise ValueError('H04 config required')
    torch.set_num_threads(4)
    model=make_model(cfg,smoke=True).to(device).train()
    x=torch.randn(1,3,cfg.image_size,cfg.image_size,device=device)
    y=torch.ones(1,device=device)
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        z=model(x).flatten()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    loss.backward()
    if z.shape!=(1,) or not torch.isfinite(loss) or not any(p.grad is not None for p in model.parameters()):
        raise ValueError('V2-S synthetic forward/backward failed')
    labels=np.array([1]*4+[0]*28)
    tracker=EpochSamplingDiagnostics(labels)
    if cfg.sampler=='balanced':
        for indices in BalancedBatchSampler(labels,8,cfg.seed):tracker.update(indices)
    else:
        rng=np.random.default_rng(cfg.seed)
        for indices in np.array_split(rng.permutation(len(labels)),4):tracker.update(indices)
    return {'experiment_id':cfg.experiment_id,'status':'TESTED_SYNTHETIC_ONLY',
        'config_hash':cfg.identity(),'amp_enabled':device=='cuda',
        'input_shape':list(x.shape),'output_shape':list(z.shape),
        'parameters':parameter_count(model),'sampling_example':tracker.summary(),
        'weights_loaded':False,'formal_training_started':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    print(json.dumps(preflight(a.config,a.device),indent=2))
