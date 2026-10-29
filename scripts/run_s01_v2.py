"""G01-controlled S01 configs, smoke and QC-gated single formal run."""
import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from scripts.s01_v2_qc import guard
from src.analysis import training_plot
from src.benchmark import local_benchmark
from src.bootstrap import paired_bootstrap
from src.data import FundusDataset, read_manifest, open_rgb
from src.experiment_guard import checksums
from src.experiment_training import train
from src.fov_v2 import crop_fov_v2 as crop_fov
from src.metrics import metrics
from src.model import build_model, verified_weights
from src.utils import load_config, seed_everything, sha256, write_json

ROOT=Path('outputs/S01_v2_execution')
CONFIG=Path('configs/experiments/S01v2.yaml')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);guard()
    original=json.loads(Path('outputs/runs/G01/config.json').read_text())
    for name,preprocess in [('S01v2','safe_fov_v2')]:
        c=copy.deepcopy(original)
        c.update(experiment_id=name,preprocess=preprocess,output=f'results/{name}/seed_42/fold_none',
                 plan={'reference':'G01','factor':'deterministic_fov_crop','status':'QC_REQUIRED'})
        path=Path(f'configs/experiments/{name}.yaml')
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf-8')
    write_json(ROOT/'control_equivalence.json',{'G01_trainer_unchanged':sha256('src/experiment_training.py')==sha256('outputs/phase2_g01_execution/source_snapshot/src/experiment_training.py'),
        'S01a_control':'G01 reused; resize code branch unchanged; no retraining',
        'changed_fields':['experiment_id','preprocess','output','plan']})


def gate(require_review=True):
    guard();c=load_config(CONFIG)
    original=json.loads(Path('outputs/runs/G01/config.json').read_text())
    assert all(c[k]==v for k,v in original.items() if k not in {'experiment_id','preprocess','output','plan'})
    assert c['preprocess']=='safe_fov_v2'
    assert sha256('src/experiment_training.py')==sha256('outputs/phase2_g01_execution/source_snapshot/src/experiment_training.py')
    verified_weights(c['model']['weights_manifest'],'convnext_tiny')
    if require_review:
        for split in ['train','val']:
            root=Path('outputs/S01_v2')/split
            audit=json.loads((root/'report.json').read_text())
            review=json.loads((root/'visual_review.json').read_text())
            assert audit['rows']==(51396 if split=='train' else 5849)
            assert audit['final_truncation_candidates']==0
            assert audit['crop_code_sha256']==sha256('src/fov_v2.py')==review['crop_code_sha256']
            assert review['status']=='PASSED', 'Visual QC not passed'
        regression=json.loads(Path('outputs/S01_v2/regression/report.json').read_text())
        assert regression['status']=='PASSED' and regression['crop_code_sha256']==sha256('src/fov_v2.py')
        locked=json.loads(Path('outputs/S01_v2/policy_locked.json').read_text())
        assert locked['crop_code_sha256']==sha256('src/fov_v2.py')
    return c


def smoke():
    c=gate(False);root=ROOT/'smoke_v2';root.mkdir(parents=True,exist_ok=False)
    images=root/'images';images.mkdir()
    for split,n in [('train',8),('val',4)]:
        rows=[]
        for i in range(n):
            im=Image.new('RGB',(400,300));ImageDraw.Draw(im).ellipse((80,30,320,270),fill=(70+i*5,25 if split=='train' else 45,10))
            name=f'{split}_{i}.png';im.save(images/name)
            rows.append({'image_id':name,'path':name,'label':i%2})
        pd.DataFrame(rows).to_csv(root/f'{split}.csv',index=False)
    synthetic=copy.deepcopy(c)
    synthetic.update(smoke=True,experiment_id='SMOKE_S01',epochs=1,batch_size=4,num_workers=0,
                     output=str(root/'run'),cache=False)
    synthetic['model']['weights_manifest']=None
    synthetic['data']={'source':'synthetic','root':str(images.resolve()),'train_csv':str(root/'train.csv'),'val_csv':str(root/'val.csv')}
    train(synthetic)
    seed_everything(42);torch.cuda.empty_cache()
    model=build_model(c['model']['weights_manifest'],model_config=c['model']).cuda().train()
    frame=read_manifest(c['data']['train_csv'],c['data']['root']).sample(n=256,random_state=42)
    ds=FundusDataset(frame,224,True,preprocess=c['preprocess'])
    loader=DataLoader(ds,batch_size=32,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');times=[];torch.cuda.reset_peak_memory_stats()
    for step,(x,y) in enumerate(loader):
        start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x.cuda()).flatten(),y.cuda())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize()
        if step>=2:times.append(time.perf_counter()-start)
    write_json(ROOT/'smoke_result.json',{'status':'PASSED','batch_size':32,'real_batches':8,
        'images_per_second_excluding_decode':32/np.mean(times),'probe_discarded':True,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'config_sha256':sha256(CONFIG),'crop_code_sha256':sha256('src/fov_v2.py')})


def run():
    c=gate();sm=json.loads((ROOT/'smoke_result.json').read_text())
    assert sm['status']=='PASSED' and sm['config_sha256']==sha256(CONFIG) and sm['crop_code_sha256']==sha256('src/fov_v2.py')
    out=Path(c['output'])
    if out.exists() or (ROOT/'run_manifest.json').exists():raise FileExistsError('Already started; no overwrite')
    snapshot=ROOT/'source_snapshot'
    snapshot.mkdir()
    for base in ['src','scripts']:
        for p in Path(base).rglob('*.py'):
            target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    manifest={'status':'RUNNING','started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*.py')},
        'config_sha256':sha256(CONFIG),'seed':42,
        'data_sha256':{k:sha256(c['data'][k]) for k in ['train_csv','val_csv']}}
    write_json(ROOT/'run_manifest.json',manifest);history=[]
    def on_epoch(row,*args):
        history.append(row);pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
        with (out/'training.log').open('a') as f:f.write(json.dumps(row)+'\n')
        training_plot(out)
    try:
        checkpoint=train(c,on_epoch=on_epoch)
        pred=pd.read_csv(out/'validation.csv');ref=pd.read_csv('outputs/runs/G01/validation.csv')
        assert pred.image_id.tolist()==ref.image_id.tolist() and np.array_equal(pred.label,ref.label)
        scores=metrics(pred.label,pred.probability)
        timing=local_benchmark(checkpoint,c,out/'latency')
        records=[]
        for row in read_manifest(c['data']['val_csv'],c['data']['root']).itertuples():
            _,m=crop_fov(open_rgb(row.resolved_path));records.append(dict(image_id=row.image_id,**m))
        pd.DataFrame(records).to_csv(out/'validation_crop_statistics.csv',index=False)
        report={'metrics':scores,'paired_bootstrap':paired_bootstrap(ref,pred,repeats=2000,seed=42),
            'best_epoch':max(history,key=lambda r:(r['auroc'],r['auprc']))['epoch'],
            'epochs_completed':len(history),'model_bytes':checkpoint.stat().st_size,
            'latency_4060_ms':timing['mean_ms'],'T4_measured':False,
            'val_fallback_fraction':float(np.mean([m['fallback'] for m in records])),
            'val_mean_area_ratio':float(np.mean([m['area_ratio'] for m in records]))}
        write_json(out/'comparison.json',report);write_json(out/'metrics.json',scores)
        shutil.copy2(CONFIG,out/'config.yaml')
        shutil.copy2('outputs/S01_v2/preprocessing.yaml',out/'preprocessing.yaml')
        manifest.update(status='COMPLETED',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),frozen_files_verified=guard())
        write_json(out/'run_manifest.json',manifest);checksums(out)
        write_json(ROOT/'run_manifest.json',manifest);print(json.dumps(report),flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',error=repr(exc));write_json(ROOT/'run_manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','run']);args=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'run':run}[args.action]()
