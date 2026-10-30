"""Small synthetic forward/backward/AMP check and model-cost estimate, no formal training."""
import argparse
import json
from pathlib import Path

import torch

from src.research.config import ExperimentConfig
from src.research.models import make_model,parameter_count
from src.utils import write_json


def preflight(config,device='cuda',output=None):
    cfg=ExperimentConfig.load(config)
    if not cfg.experiment_id.startswith('H03'):
        raise ValueError('H03 config required')
    torch.set_num_threads(4)
    model=make_model(cfg,smoke=True).to(device)
    x=torch.randn(1,3,cfg.image_size,cfg.image_size,device=device)
    label=torch.tensor([1.],device=device)
    model.train()
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        logits=model(x).flatten()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,label)
    loss.backward()
    if logits.shape!=(1,) or not torch.isfinite(loss) or not all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Synthetic H03 forward/backward failed')
    if hasattr(model,'cost_estimate'):
        cost=model.cost_estimate()
    else:
        cost={'basis':'ConvNeXt-Tiny model card approximate 4.5 GMAC at 224; original single-linear classifier',
            'backbone_macs_estimate':4_500_000_000,
            'head_linear_macs_estimate':model.num_features,
            'total_macs_approximate':4_500_000_000+model.num_features,
            'head_parameters':sum(p.numel() for p in model.get_classifier().parameters()),
            'total_parameters':parameter_count(model),'feature_dimensions':{'final_channels':model.num_features},
            'measured_latency_ms':None}
    report={'status':'TESTED_SYNTHETIC_ONLY','experiment_id':cfg.experiment_id,
        'config_hash':cfg.identity(),'input_shape':list(x.shape),'output_shape':list(logits.shape),
        'amp_enabled':device=='cuda','loss_finite':True,'cost':cost,'formal_training_started':False,
        'weights_loaded':False}
    if output:
        out=Path(output)
        if out.exists():raise FileExistsError(out)
        write_json(out,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--output');a=p.parse_args()
    print(json.dumps(preflight(a.config,a.device,a.output),indent=2))
