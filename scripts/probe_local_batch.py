"""Measure real Train-image throughput and AMP memory without writing a model checkpoint."""
import argparse
import json
import time

import torch

from src.data import read_manifest
from src.research.config import ExperimentConfig
from src.research.data import ImageDataset
from src.research.trainer import loader
from src.research.losses import BinaryObjective
from src.research.models import make_model
from src.utils import seed_everything,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/research/G02.yaml')
    p.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    p.add_argument('--val-manifest',default='data/official/manifests/valid.csv')
    p.add_argument('--root',default='data/official')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--output',default='outputs/research_preflight/G02_actual_data.json')
    a=p.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    cfg=ExperimentConfig.load(a.config)
    cfg.assert_data(a.train_manifest,a.val_manifest)
    cfg.assert_pretraining()
    seed_everything(cfg.seed)
    torch.set_num_threads(int(cfg.extra.get('torch_threads',4)))
    frame=read_manifest(a.train_manifest,a.root)
    model=make_model(cfg).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.backbone_lr,weight_decay=cfg.weight_decay)
    objective=BinaryObjective(cfg.loss,int(frame.label.sum()),len(frame)-int(frame.label.sum())).cuda()
    ds=ImageDataset(frame,cfg,train=True)
    train_loader,_=loader(ds,cfg,True)
    scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    records=[];start=time.perf_counter();iterator=iter(train_loader)
    for step in range(a.steps):
        t0=time.perf_counter();x,y,_=next(iterator)
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            z=model(x).flatten();loss=objective(z,y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite probe loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm) and scaler.get_scale()<=1:raise FloatingPointError('Persistent probe gradient failure')
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        records.append({'step':step+1,'images':len(y),'seconds':time.perf_counter()-t0,
                        'loss':float(loss.detach()),'scale':scaler.get_scale(),
                        'gradient_finite':bool(torch.isfinite(norm))})
        print(json.dumps(records[-1]),flush=True)
    steady=records[2:] if len(records)>2 else records
    report={'experiment_id':cfg.experiment_id,'config_hash':cfg.identity(),'batch_size':cfg.batch_size,
            'steps':a.steps,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'images_per_second_steady':sum(x['images'] for x in steady)/sum(x['seconds'] for x in steady),
            'probe_seconds':time.perf_counter()-start,
            'full_train_images':len(frame),'steps_detail':records,'official_val_used':False,
            'checkpoint_written':False}
    write_json(a.output,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
