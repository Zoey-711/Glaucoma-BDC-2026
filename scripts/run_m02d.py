"""M02c protocol-matched ablation: two constant-zero metadata columns, no Age read."""
import ast
import csv
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from scripts.run_m02 import C, fit_head, head_logits, utc
from src.bootstrap import paired_bootstrap
from src.data import read_manifest
from src.experiment_guard import freeze_or_verify, checksums
from src.metrics import metrics
from src.multimodal import FusionHead, M02Predictor
from src.submission import COLUMNS
from src.utils import sha256, write_json

OUT=Path('results/M02d/seed_42/fold_none')


def ast_function(path, name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


def main():
    torch.set_num_threads(4)
    freeze_or_verify()
    if OUT.exists(): raise FileExistsError('M02d already exists; no overwrite/repeat')
    previous_audit=json.loads(Path('outputs/M02/age_audit.json').read_text())
    for split in ['train','valid']:
        assert sha256(f'data/official/glaucoma_first/{split}.csv')==previous_audit['raw_csv_sha256'][split]
    assert ast_function('scripts/run_m02.py','fit_head')==ast_function('outputs/M02/source_snapshot/run_m02.py','fit_head')
    train=pd.read_csv('data/official/glaucoma_first/train.csv',usecols=['Image_name','True_Label'])
    validation=pd.read_csv('data/official/glaucoma_first/valid.csv',usecols=['Image_name','True_Label'])
    matrices={}
    for split in ['train','valid']:
        meta=json.loads(Path(f'outputs/M02/features/{split}.json').read_text())
        path=Path(f'outputs/M02/features/{split}.npy')
        assert sha256(path)==meta['sha256'] and meta['identity']['checkpoint_sha256']==C['checkpoint_sha256']
        matrices[split]=np.load(path,mmap_mode='r')
    y=train.True_Label.to_numpy(); ids=np.arange(len(y))
    fit_idx,hold_idx=train_test_split(ids,test_size=.2,random_state=42,stratify=y)
    old_split=pd.read_csv('outputs/M02/training/internal_split.csv')
    assert list(old_split.Image_name)==list(train.Image_name)
    assert np.array_equal(np.isin(ids,hold_idx),old_split.internal_holdout.astype(bool))
    OUT.mkdir(parents=True)
    config=yaml.safe_load(Path('configs/multimodal/M02d.yaml').read_text())
    (OUT/'config.yaml').write_text(yaml.safe_dump({'experiment':config,'protocol':C},sort_keys=False),encoding='utf-8')
    manifest={'status':'RUNNING','started_utc':utc(),'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'source_sha256':{p:sha256(p) for p in ['scripts/run_m02d.py','scripts/run_m02.py','src/multimodal.py']},
              'G01_sha256':C['checkpoint_sha256'],'age_columns_read':False,'placeholder':[0.,0.],
              'internal_split_sha256':sha256('outputs/M02/training/internal_split.csv'),'seed':42,
              'official_val_used_for_selection':False}
    write_json(OUT/'run_manifest.json',manifest)
    age=np.zeros((len(y),2),dtype=np.float32);x=matrices['train']; trials=[]
    for lr in C['head']['learning_rates']:
        model,summary=fit_head(x[fit_idx],age[fit_idx],y[fit_idx],lr,C['head']['epochs'],OUT/f'internal_lr_{lr}.csv',
                               (x[hold_idx],age[hold_idx],y[hold_idx]))
        trials.append(summary);del model;gc.collect();torch.cuda.empty_cache()
    selected=max(trials,key=lambda r:(r['internal_auroc'],r['internal_auprc']))
    write_json(OUT/'selection_locked.json',{'trials':trials,'selected':selected,'official_val_used':False,'locked_utc':utc()})
    model,_=fit_head(x,age,y,selected['lr'],selected['best_epoch'],OUT/'train_history.csv')
    original=torch.load(C['checkpoint'],map_location='cpu',weights_only=True)
    torch.save({'backbone_state_dict':original['state_dict'],'model_config':original['config']['model'],
                'head_state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'image_dim':768,
                'age_input':'zeros','G01_sha256':C['checkpoint_sha256'],'protocol':C,'selected':selected,
                'provenance':original['provenance']},OUT/'best.pth')
    z=head_logits(model,matrices['valid'],np.zeros((len(validation),2),dtype=np.float32))
    pred=pd.DataFrame({'image_id':validation.Image_name,'label':validation.True_Label,'logit':z,
                       'probability':torch.sigmoid(torch.tensor(z)).numpy(),'experiment_id':'M02d'})
    pred.to_csv(OUT/'validation.csv',index=False)
    ref=pd.read_csv('results/M02c/seed_42/fold_none/validation.csv')
    score=metrics(pred.label,pred.probability)
    score.update(M02d_minus_M02c=paired_bootstrap(ref,pred,repeats=2000,seed=42),
                 M02c_minus_M02d=paired_bootstrap(pred,ref,repeats=2000,seed=42),
                 M02d_minus_G01=paired_bootstrap(pd.read_csv('outputs/runs/G01/validation.csv'),pred,repeats=2000,seed=42))
    write_json(OUT/'metrics.json',score)
    del model,original;gc.collect();torch.cuda.empty_cache()
    predictor=M02Predictor(OUT/'best.pth',{})
    frame=read_manifest('data/official/manifests/valid.csv','data/official')
    chosen=np.random.default_rng(42).choice(len(frame),105,replace=False)
    timing=[]
    with (OUT/'latency_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(COLUMNS)
        for row in frame.iloc[chosen[:5]].itertuples():predictor.predict_one(row.resolved_path,row.image_id,writer,f)
        for _ in range(3):
            values=[]
            for row in frame.iloc[chosen[5:]].itertuples():
                predictor.synchronize();start=time.perf_counter_ns()
                predictor.predict_one(row.resolved_path,row.image_id,writer,f)
                predictor.synchronize();values.append((time.perf_counter_ns()-start)/1e6)
            timing.append(values)
    write_json(OUT/'latency_4060.json',{'mean_ms':float(np.mean(timing)),'latency_ms_by_round':timing,
               'model_bytes':(OUT/'best.pth').stat().st_size,'sample_ids':frame.iloc[chosen[5:]].image_id.tolist(),
               'gpu':torch.cuda.get_device_name(),'T4_measured':False,'scope':'batch1 decode through CSV write/flush, 5 warmup, 100 samples x3'})
    manifest.update(status='COMPLETED',finished_utc=utc(),selected=selected,frozen_files_verified=freeze_or_verify())
    write_json(OUT/'run_manifest.json',manifest);checksums(OUT)
    print(json.dumps({'M02d':score,'selected':selected}),flush=True)


if __name__=='__main__':main()
