import csv
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.metrics import metrics
from src.utils import seed_everything, sha256, write_json
from .data import BalancedBatchSampler,ImageDataset,load_official
from .losses import BinaryObjective,bernoulli_kl
from .models import make_model,split_parameters


def utc():return datetime.now(timezone.utc).isoformat()


def loader(ds,cfg,train=False):
    workers=int(cfg.extra.get('num_workers',0))
    if train and cfg.sampler=='balanced':
        sampler=BalancedBatchSampler(ds.frame.label,cfg.batch_size,cfg.seed)
        return DataLoader(ds,batch_sampler=sampler,num_workers=workers,pin_memory=True),sampler
    return DataLoader(ds,batch_size=cfg.batch_size,shuffle=train,num_workers=workers,pin_memory=True,
                      generator=torch.Generator().manual_seed(cfg.seed+ds.epoch)),None


def make_scheduler(optimizer,steps_per_epoch,cfg):
    total=steps_per_epoch*cfg.epochs;warm=steps_per_epoch*cfg.warmup
    def scale(step):
        if warm==0:
            epoch=min(cfg.epochs,step//steps_per_epoch)
            return .5*(1+math.cos(math.pi*epoch/cfg.epochs))
        if warm and step<warm:return max(1e-3,(step+1)/warm)
        return .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def mixup(x,y,alpha,seed):
    if alpha<=0:return x,y
    rng=np.random.default_rng(seed);lam=float(rng.beta(alpha,alpha));perm=torch.randperm(len(y),device=y.device,generator=torch.Generator(device=y.device).manual_seed(seed))
    return lam*x+(1-lam)*x[perm],lam*y+(1-lam)*y[perm]


def evaluate(model,ds,cfg,device,source='official_val'):
    model.eval();dl,_=loader(ds,cfg,False);records=[]
    with torch.inference_mode():
        for x,y,index in dl:
            x=x.to(device,non_blocking=True)
            # Match frozen B02/G01 validation: FP32, no random augmentation.
            z=model(x).float().flatten().cpu().numpy()
            for i,logit,label in zip(index.numpy(),z,y.numpy()):
                records.append((int(i),float(logit),float(label)))
    records.sort(key=lambda row:row[0])
    if [r[0] for r in records]!=list(range(len(ds))):raise ValueError('Validation missing or duplicate indices')
    frame=pd.DataFrame({'Image_name':ds.frame.image_id,'Label':[int(r[2]) for r in records],
                        'logit':[r[1] for r in records]})
    frame['Probability']=torch.sigmoid(torch.tensor(frame.logit.to_numpy())).numpy()
    frame['experiment_id']=cfg.experiment_id;frame['source_split']=source
    score=metrics(frame.Label,frame.Probability)
    tn,fp,fn,tp=confusion_matrix(frame.Label,frame.Probability>=.5,labels=[0,1]).ravel()
    score['confusion_matrix']={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    return frame,score


def save_checksums(directory):
    files=[p for p in Path(directory).iterdir() if p.is_file() and p.name!='checksums.sha256']
    (Path(directory)/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(files)),encoding='utf-8')


def run(cfg,root,train_manifest,val_manifest,*,smoke=False,resume=False,group_manifest=None,teacher_path=None):
    if cfg.status=='BLOCKED':raise ValueError('Experiment explicitly blocked')
    if not smoke:
        cfg.assert_code()
        if not (cfg.extra.get('ssl_initialization')=='random_then_ssl_then_finetune' and cfg.extra.get('ssl_encoder_checkpoint')):
            cfg.assert_pretraining()
    cfg.assert_data(train_manifest,val_manifest)
    train_frame,val_frame,oof_frame=load_official(cfg,root,train_manifest,val_manifest,group_manifest)
    if smoke and len(train_frame)>32:raise ValueError('Smoke accepts synthetic small dataset only')
    out=Path(cfg.output_dir);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not resume:raise FileExistsError(f'Output already exists: {out}')
    if resume and not (out/'last.pth').exists():raise FileNotFoundError('No resume checkpoint')
    if resume:
        previous=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
        if previous.get('status')=='COMPLETED':raise ValueError('Completed experiment cannot be resumed or overwritten')
        old_cfg=ExperimentConfig.load(out/'config.yaml')
        if old_cfg.identity()!=cfg.identity() or old_cfg.git_commit!=cfg.git_commit:
            raise ValueError('Resume result config/code mismatch; no files changed')
        pre_resume=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        if (pre_resume['config_hash']!=cfg.identity() or
            pre_resume['data_manifest_hash']!=cfg.data_manifest_hash or
            pre_resume['git_commit']!=cfg.git_commit):
            raise ValueError('Resume checkpoint config/data/code mismatch; no files changed')
    out.mkdir(exist_ok=resume)
    started=previous['started_utc'] if resume else utc()
    seed_everything(cfg.seed);torch.set_num_threads(int(cfg.extra.get('torch_threads',4)))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not smoke and device.type!='cuda':raise ValueError('Formal runs require CUDA AMP')
    manifest={'schema':'glaucoma_research_v1','experiment_id':cfg.experiment_id,'status':'RUNNING','git_commit':cfg.git_commit,
        'config_hash':cfg.identity(),'data_manifest_hash':cfg.data_manifest_hash,
        'pretrained_sha256':cfg.pretrained_sha256,'seed':cfg.seed,'fold':cfg.fold,
        'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
        'started_utc':started,'finished_utc':None,'smoke':smoke}
    write_json(out/'run_manifest.json',manifest)
    import yaml
    if not resume:
        (out/'config.yaml').write_text(yaml.safe_dump(cfg.as_dict(),sort_keys=False,allow_unicode=True),encoding='utf-8')
        (out/'environment.txt').write_text(f'python={platform.python_version()}\ntorch={torch.__version__}\ncuda={torch.version.cuda}\ngpu={manifest["gpu"]}\n',encoding='utf-8')
        (out/'training.log').write_text('',encoding='utf-8')
    model=make_model(cfg,smoke=smoke).to(device)
    backbone,head=split_parameters(model)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':cfg.backbone_lr},{'params':head,'lr':cfg.head_lr}],weight_decay=cfg.weight_decay)
    train_ds=ImageDataset(train_frame,cfg,True);val_ds=ImageDataset(val_frame,cfg,False)
    val_loader,_=loader(val_ds,cfg)
    steps=math.ceil(math.ceil(len(train_ds)/cfg.batch_size)/cfg.gradient_accumulation)
    scheduler=make_scheduler(optimizer,steps,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    counts=np.bincount(train_frame.label,minlength=2)
    objective=BinaryObjective(cfg.loss,int(counts[1]),int(counts[0])).to(device)
    teacher=None
    if teacher_path:
        if cfg.experiment_id!='D01':raise ValueError('Teacher only in D01')
        from .distill import load_approved_teacher
        teacher=load_approved_teacher(teacher_path,cfg,device,train_manifest)
    start_epoch=0;best=(-1.,-1.);stale=0;history=[]
    if resume:
        ck=pre_resume
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1;best=tuple(ck['best']);stale=ck['stale'];history=ck['history']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
    try:
        for epoch in range(start_epoch,cfg.epochs):
            train_ds.set_epoch(epoch);train_loader,sampler=loader(train_ds,cfg,True)
            model.train();loss_total=0.;pair_total=0;seen=0
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            epoch_start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            for step,(x,y,index) in enumerate(train_loader):
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                if cfg.extra.get('mixup_alpha'):
                    x,y=mixup(x,y,float(cfg.extra['mixup_alpha']),cfg.seed+epoch*100000+step)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    logits=model(x).flatten()
                    if logits.shape!=y.shape:raise ValueError('Binary logit shape mismatch')
                    loss=objective(logits,y)
                    if teacher is not None:
                        with torch.inference_mode():teacher_logits=teacher(x).flatten()
                        alpha=float(cfg.extra['distill_alpha']);t=float(cfg.extra['temperature'])
                        loss=(1-alpha)*loss+alpha*bernoulli_kl(logits,teacher_logits,t)
                    group_start=(step//cfg.gradient_accumulation)*cfg.gradient_accumulation
                    group_size=min(cfg.gradient_accumulation,len(train_loader)-group_start)
                    loss=loss/group_size
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward();pair_total+=objective.last_pairs
                loss_total+=float(loss.detach())*len(y)*cfg.gradient_accumulation;seen+=len(y)
                end=(step+1)%cfg.gradient_accumulation==0 or step+1==len(train_loader)
                if end:
                    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.extra.get('grad_clip_norm',1.)))
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old_scale:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            if device.type=='cuda':torch.cuda.synchronize(device)
            train_seconds=time.perf_counter()-epoch_start
            peak_allocated=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None
            peak_reserved=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None
            predictions,score=evaluate(model,val_ds,cfg,device)
            if oof_frame is not None:
                fold_ds=ImageDataset(oof_frame,cfg,False)
                fold_pred,fold_score=evaluate(model,fold_ds,cfg,device,'train_oof')
                fold_pred.to_csv(out/'oof_train.csv',index=False)
            row={'epoch':epoch+1,'train_loss':loss_total/seen,'train_seconds':train_seconds,
                 'train_peak_allocated_mib':peak_allocated,'train_peak_reserved_mib':peak_reserved,
                 'effective_pair_count':pair_total,'lr':optimizer.param_groups[0]['lr'],**score}
            if sampler:row.update(sampler.last_stats)
            history.append(row)
            pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
            with (out/'training.log').open('a',encoding='utf-8') as log:log.write(json.dumps(row)+'\n')
            key=(score['auroc'],score['auprc'])
            if key>best:
                best=key;stale=0
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                    'config':cfg.as_dict(),'metrics':score,'epoch':epoch+1,'provenance':cfg.weights_manifest,
                    'smoke':smoke},out/'best.pth')
                predictions.to_csv(out/'validation.csv',index=False)
                write_json(out/'metrics.json',{'best_epoch':epoch+1,**score})
            else:stale+=1
            torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'best':best,'stale':stale,'history':history,
                'config_hash':cfg.identity(),'data_manifest_hash':cfg.data_manifest_hash,'git_commit':cfg.git_commit,
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},out/'last.pth')
            print(json.dumps(row),flush=True)
            if stale>=cfg.patience:break
        manifest.update(status='COMPLETED',finished_utc=utc(),best_epoch=int(pd.read_json(out/'metrics.json',typ='series')['best_epoch']))
        write_json(out/'run_manifest.json',manifest);save_checksums(out)
        from .results import package_result
        package_result(out,out.parent/f'{cfg.experiment_id}_seed{cfg.seed}_fold{cfg.fold if cfg.fold is not None else "none"}.tar.gz')
        return out
    except (torch.cuda.OutOfMemoryError,FloatingPointError,Exception) as exc:
        manifest.update(status='FAILED',finished_utc=utc(),error=repr(exc))
        write_json(out/'run_manifest.json',manifest);save_checksums(out)
        raise
