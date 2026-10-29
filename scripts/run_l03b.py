"""G01 trainer, original sampling, reviewed ImageNet init; only L03b is runnable."""
import argparse
import copy
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.analysis import training_plot
from src.benchmark import local_benchmark
from src.bootstrap import paired_bootstrap
from src.data import FundusDataset, read_manifest
from src.experiment_guard import freeze_or_verify, checksums
from src.experiment_training import train
from src.losses import BinaryLoss
from src.model import build_model, verified_weights
from src.pairwise import LossAccounting, monitor_training
from src.utils import load_config, seed_everything, sha256, write_json
from scripts.run_m02 import utc, ranking

ROOT=Path('outputs/L03b_execution')
CONFIG=Path('configs/experiments/L03b.yaml')


def gate():
    freeze_or_verify()
    c=load_config(CONFIG)
    assert c['experiment_id']=='L03b' and c['loss']=={'name':'pairwise_auc','lambda':.05}
    original=json.loads(Path('outputs/runs/G01/config.json').read_text(encoding='utf-8'))
    assert all(c[k]==value for k,value in original.items() if k not in {'experiment_id','output','loss','plan'})
    assert sha256('src/experiment_training.py')==sha256('outputs/phase2_g01_execution/source_snapshot/src/experiment_training.py')
    verified_weights(c['model']['weights_manifest'],'convnext_tiny')
    audit=json.loads(Path('outputs/M02/age_audit.json').read_text())
    for split in ['train','valid']:
        assert sha256(f'data/official/glaucoma_first/{split}.csv')==audit['raw_csv_sha256'][split]
    return c


def smoke():
    c=gate();root=ROOT/'smoke';root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42);images=root/'images';images.mkdir()
    for split,n in [('train',8),('val',4)]:
        rows=[]
        for i in range(n):
            name=f'{split}_{i}.png'
            Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)).save(images/name)
            rows.append({'image_id':name,'path':name,'label':i%2})
        pd.DataFrame(rows).to_csv(root/f'{split}.csv',index=False)
    synthetic=copy.deepcopy(c)
    synthetic.update(experiment_id='SMOKE_L03b',output=str(root/'run'),smoke=True,epochs=1,batch_size=4,num_workers=0,cache=False)
    synthetic['model']['weights_manifest']=None
    synthetic['data']={'source':'synthetic','root':str(images.resolve()),'train_csv':str(root/'train.csv'),'val_csv':str(root/'val.csv')}
    monitor=LossAccounting();record=[]
    with monitor_training(monitor):
        train(synthetic,on_epoch=lambda *args:record.append(monitor.finish_epoch()))
    assert record[0]['positive_draws']==4 and record[0]['negative_draws']==4
    gc.collect();torch.cuda.empty_cache()
    seed_everything(42);torch.set_num_threads(4)
    model=build_model(c['model']['weights_manifest'],model_config=c['model']).cuda().train()
    frame=read_manifest(c['data']['train_csv'],c['data']['root']).iloc[:256]
    cache=json.loads(Path('outputs/runs/G01/cache.json').read_text())['train']
    ds=FundusDataset(frame,224,True,cache_dir=cache)
    dl=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');objective=BinaryLoss(c['loss'],[49739,1657]).cuda()
    times=[];torch.cuda.reset_peak_memory_stats();monitor=LossAccounting()
    with monitor_training(monitor):
        for step,(x,y) in enumerate(dl):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            x=x.cuda();y=y.cuda()
            with torch.autocast('cuda'):loss=objective(model(x).flatten(),y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            torch.cuda.synchronize()
            if step>=2:times.append(time.perf_counter()-start)
    report={'status':'PASSED','synthetic':record,'real_train_probe':monitor.finish_epoch(),
            'batch_size':32,'probe_steps':8,'measured_steps_after_warmup':6,'images_per_second':32/float(np.mean(times)),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'probe_discarded':True,'config_sha256':sha256(CONFIG),'gpu':torch.cuda.get_device_name(),
            'source_sha256':{p:sha256(p) for p in ['src/losses.py','src/pairwise.py','src/experiment_training.py']}}
    write_json(ROOT/'smoke_result.json',report)
    print(json.dumps(report),flush=True)


def run():
    c=gate();sm=json.loads((ROOT/'smoke_result.json').read_text())
    assert sm['status']=='PASSED' and sm['config_sha256']==sha256(CONFIG)
    assert all(sha256(p)==h for p,h in sm['source_sha256'].items())
    out=Path(c['output'])
    if out.exists() or (ROOT/'run_manifest.json').exists():raise FileExistsError('L03b already started; no repeat/overwrite')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for directory in ['src','scripts']:
        (snapshot/directory).mkdir()
        for file in Path(directory).glob('*.py'):shutil.copy2(file,snapshot/directory/file.name)
    shutil.copy2(CONFIG,snapshot/'L03b.yaml')
    manifest={'status':'RUNNING','started_utc':utc(),'experiment_id':'L03b','seed':42,
              'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'source_sha256':{str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*') if p.is_file()},
              'config_sha256':sha256(CONFIG),'reference':'G01','sampler':'original_shuffle',
              'pretrained_sha256':verified_weights(c['model']['weights_manifest'])['sha256'],
              'data_sha256':{key:sha256(c['data'][key]) for key in ['train_csv','val_csv']}}
    write_json(ROOT/'run_manifest.json',manifest)
    monitor=LossAccounting();history=[]
    def epoch_done(row,checkpoint,elapsed):
        accounting=monitor.finish_epoch()
        assert accounting['positive_draws']==1657 and accounting['negative_draws']==49739
        assert abs(accounting['combined_loss']-row['train_loss'])<1e-5
        record={**row,**accounting,'elapsed_seconds':elapsed}
        history.append(record)
        pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
        with (out/'training.log').open('a',encoding='utf-8') as log:log.write(json.dumps(record)+'\n')
        training_plot(out)
    try:
        with monitor_training(monitor):checkpoint=train(c,on_epoch=epoch_done)
        prediction=pd.read_csv(out/'validation.csv');reference=pd.read_csv('outputs/runs/G01/validation.csv')
        assert list(prediction.image_id)==list(reference.image_id) and np.array_equal(prediction.label,reference.label)
        from src.metrics import metrics
        scores=metrics(prediction.label,prediction.probability)
        boot=paired_bootstrap(reference,prediction,repeats=2000,seed=42)
        timing=local_benchmark(checkpoint,c,out/'latency')
        best=max(history,key=lambda r:(r['auroc'],r['auprc']))
        report={'experiment_id':'L03b','best_epoch':best['epoch'],'epochs_completed':len(history),
                'metrics':scores,'paired_bootstrap':boot,'model_bytes':checkpoint.stat().st_size,
                'latency_4060_ms':timing['mean_ms'],'T4_measured':False,
                'ranking':ranking(reference.probability.to_numpy(),prediction.probability.to_numpy(),prediction.label.to_numpy()),
                'peak_allocated_mib':max(r['train_peak_allocated_mib'] for r in history),
                'peak_reserved_mib':max(r['train_peak_reserved_mib'] for r in history)}
        write_json(out/'comparison.json',report);write_json(out/'metrics.json',scores)
        shutil.copy2(CONFIG,out/'config.yaml')
        manifest.update(status='COMPLETED',finished_utc=utc(),frozen_files_verified=freeze_or_verify())
        write_json(ROOT/'run_manifest.json',manifest);write_json(out/'run_manifest.json',manifest);checksums(out)
        print(json.dumps(report),flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',finished_utc=utc(),error=repr(exc))
        write_json(ROOT/'run_manifest.json',manifest)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','run']);a=p.parse_args()
    {'smoke':smoke,'run':run}[a.action]()
