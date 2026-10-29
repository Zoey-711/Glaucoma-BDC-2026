import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.research.config import ExperimentConfig, manifest_semantic_hash
from src.research.trainer import loader


def test_retired_experiments_and_unsafe_crop_cannot_run():
    for name in ['G02','G04','G09','L03','L03_LAMBDA0','L03_LAMBDA02','S01']:
        with pytest.raises(ValueError):
            ExperimentConfig.load(f'configs/research/{name}.yaml').assert_execution_policy()
    ExperimentConfig.load('configs/research/G01_REPRO.yaml').assert_execution_policy()


def test_balanced_loader_changes_epoch_reproducibly():
    cfg=dataclasses.replace(ExperimentConfig.load('configs/research/L04.yaml'),extra={'num_workers':0})
    class Dataset:
        frame=pd.DataFrame({'label':[0]*100+[1]*10})
        epoch=0
        def __len__(self):return len(self.frame)
    ds=Dataset()
    _,a=loader(ds,cfg,True);first=list(a)
    ds.epoch=1
    _,b=loader(ds,cfg,True);second=list(b)
    _,c=loader(ds,cfg,True)
    assert first!=second and second==list(c)


def test_synthetic_training_resume_package_and_nan_rejection(tmp_path,monkeypatch):
    from src.research import trainer,results
    cfg=ExperimentConfig.load('configs/research/G01_REPRO.yaml')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Linear(3,4);self.head=torch.nn.Linear(4,1)
        def get_classifier(self):return self.head
        def forward(self,x):return self.head(self.encoder(x.mean((2,3))))
    monkeypatch.setattr(trainer,'make_model',lambda *a,**kw:Tiny())
    for split,n in [('train',6),('val',4)]:
        rows=[]
        for i in range(n):
            name=f'{split}_{i}.png';Image.new('RGB',(32,32),(20+i*25,40,60)).save(name)
            rows.append({'image_id':name,'path':name,'label':i%2})
        pd.DataFrame(rows).to_csv(f'{split}.csv',index=False)
    cfg=dataclasses.replace(cfg,experiment_id='SMOKE_SERVER_REVIEW',image_size=32,batch_size=2,
        gradient_accumulation=2,epochs=1,patience=1,config_hash=None,git_commit=None,
        output_dir='results/SMOKE_SERVER_REVIEW/seed_42/fold_none',
        extra={'num_workers':0},data_manifest_hash={
            'train':manifest_semantic_hash('train.csv'),'val':manifest_semantic_hash('val.csv'),'archive':'synthetic'})
    package=results.package_result
    def fail_package(*args):raise RuntimeError('simulated packaging interruption')
    monkeypatch.setattr(results,'package_result',fail_package)
    with pytest.raises(RuntimeError,match='interruption'):
        trainer.run(cfg,tmp_path,'train.csv','val.csv',smoke=True)
    monkeypatch.setattr(results,'package_result',package)
    out=trainer.run(cfg,tmp_path,'train.csv','val.csv',smoke=True,resume=True)
    results.validate_result(out,'val.csv',allow_smoke=True)
    bundle=next(out.parent.glob('*.tar.gz'))
    imported=results.safe_import(bundle,tmp_path/'imported')
    results.validate_result(imported,'val.csv',allow_smoke=True)
    score=json.loads((out/'metrics.json').read_text());score['auroc']=float('nan')
    (out/'metrics.json').write_text(json.dumps(score));trainer.save_checksums(out)
    with pytest.raises(ValueError,match='Metric mismatch'):
        results.validate_result(out,'val.csv',allow_smoke=True)
