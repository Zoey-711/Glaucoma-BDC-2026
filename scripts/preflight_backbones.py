"""Synthetic architecture/backward/memory preflight; no formal training."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.research.config import ExperimentConfig
from src.research.models import make_model,parameter_count
from src.utils import write_json


CASES=['G02','G03','G04','G05','G06','G07','G08','G09']


def probe(cfg,entry,batch_size):
    if cfg.weights_manifest:cfg.assert_pretraining()
    model=make_model(cfg,smoke=not bool(cfg.weights_manifest)).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    x=torch.randn(batch_size,3,cfg.image_size,cfg.image_size,device='cuda');y=torch.ones(batch_size,device='cuda')
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        z=model(x).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    if z.shape!=(batch_size,) or not torch.isfinite(loss):raise ValueError('Invalid binary output/loss')
    loss.backward();opt.step();torch.cuda.synchronize()
    entry.update(status='SYNTHETIC_BACKWARD_PASSED',parameters=parameter_count(model),
        output_shape=list(z.shape),loss=float(loss.detach()),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        weight_and_architecture_ready=bool(cfg.weights_manifest) and cfg.status!='BLOCKED',
        full_formal_gate_checked=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/research_preflight/backbones.json')
    p.add_argument('--cases',nargs='*',default=CASES)
    p.add_argument('--use-config-batch',action='store_true',help='Test full configured microbatch on target GPU')
    a=p.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA preflight requires GPU')
    torch.set_num_threads(4);results=[]
    for eid in a.cases:
        cfg=ExperimentConfig.load(f'configs/research/{eid}.yaml');entry={'experiment_id':eid,
            'backbone':cfg.backbone,'image_size':cfg.image_size,'gpu':torch.cuda.get_device_name(),
            'tested_batch_size':cfg.batch_size if a.use_config_batch else 1,
            'weight_review':'approved' if cfg.weights_manifest else 'BLOCKED_PENDING_PROVENANCE'}
        try:
            probe(cfg,entry,entry['tested_batch_size'])
        except Exception as exc:entry.update(status='FAILED',error=repr(exc),weight_and_architecture_ready=False)
        results.append(entry);print(json.dumps(entry),flush=True)
        gc.collect();torch.cuda.empty_cache()
    write_json(a.output,results)


if __name__=='__main__':main()
