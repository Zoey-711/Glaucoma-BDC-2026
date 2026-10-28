import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.create_research_configs import matrix
from src.research.config import ExperimentConfig
from src.research.data import BalancedBatchSampler
from src.research.fusion import fixed_fusion
from src.research.losses import BinaryObjective,bernoulli_kl
from src.research.queue import load_matrix,run_queue


def test_matrix_and_frozen_config_contract():
    groups,variants,extras,references=matrix()
    assert len(groups)==20 and len(extras)==6 and len(variants)>=12
    assert {c['experiment_id'] for c in references}=={'G01_REPRO','B02_REPRO'}
    assert {c['experiment_id'] for c in groups}=={*(f'G{i:02d}' for i in range(2,10)),
        *(f'L{i:02d}' for i in range(1,5)),*(f'S{i:02d}' for i in range(1,5)),*(f'R{i:02d}' for i in range(1,5))}
    for path in Path('configs/research').glob('*.yaml'):
        if path.name=='first_batch_8gpu.yaml':continue
        ExperimentConfig.load(path)
    base=ExperimentConfig.load('configs/research/G02.yaml')
    broken=dataclasses.replace(base,image_size=512)
    with pytest.raises(ValueError,match='hash'):broken.validate()
    assert base.status=='PENDING'
    assert ExperimentConfig.load('configs/research/S03.yaml').status=='BLOCKED'


def test_balanced_batches_have_both_classes_and_report_reuse():
    sampler=BalancedBatchSampler([0]*100+[1]*3,batch_size=8,seed=42)
    batches=list(iter(sampler))
    assert all(set(np.asarray([0]*100+[1]*3)[b])=={0,1} for b in batches)
    assert sampler.last_stats['duplicate_draws']>0
    assert 0<sampler.last_stats['independent_sample_coverage']<=1


def test_pairwise_single_class_and_mixup_soft_label_stable():
    loss=BinaryObjective({'name':'pairwise_auc','lambda':.2},3,100)
    z=torch.tensor([100.,-100.],requires_grad=True)
    y=torch.tensor([1.,1.]);out=loss(z,y);out.backward()
    assert torch.isfinite(out) and loss.last_pairs==0 and torch.isfinite(z.grad).all()
    z2=torch.tensor([1.,-.1],requires_grad=True)
    out2=BinaryObjective({'name':'bce'},3,100)(z2,torch.tensor([.3,.7]));out2.backward()
    assert torch.isfinite(out2) and torch.isfinite(z2.grad).all()
    kl=bernoulli_kl(torch.tensor([0.],requires_grad=True),torch.tensor([1.],requires_grad=True),2)
    kl.backward();assert kl>=0


def test_fusion_rejects_label_mismatch_and_equal_weights(tmp_path):
    a=pd.DataFrame({'Image_name':['a','b'],'Label':[0,1],'logit':[-2.,2.],'Probability':[.12,.88]})
    b=a.copy();b['logit']=[-1.,1.];b['Probability']=[.27,.73]
    a.to_csv(tmp_path/'a.csv',index=False);b.to_csv(tmp_path/'b.csv',index=False)
    out,m=fixed_fusion([tmp_path/'a.csv',tmp_path/'b.csv'],'probability')
    assert np.allclose(out.Probability,[.195,.805]) and m['auroc']==1
    b.loc[0,'Label']=1;b.to_csv(tmp_path/'b.csv',index=False)
    with pytest.raises(ValueError,match='labels'):fixed_fusion([tmp_path/'a.csv',tmp_path/'b.csv'])


def test_queue_reports_blocked_dependencies_and_all_eight_gpu_ids():
    plan=load_matrix('configs/research/first_batch_8gpu.yaml')
    assert {t['gpu'] for t in plan['tasks']}==set(range(8))
    report=run_queue('configs/research/first_batch_8gpu.yaml',archive=None,dry_run=True)
    assert len(report)==8 and all(r['status']=='BLOCKED' for r in report)
    assert any('weight' in (r['reason'] or '').lower() for r in report if r['experiment_id']=='G05')


def test_queue_launches_one_process_per_eight_gpus_without_collision(monkeypatch,tmp_path):
    from src.research import queue
    active=set();maximum=[0];devices=[]
    class FakeProcess:
        def __init__(self,args,env,stdout,stderr):
            self.pid=1000+len(devices);self.device=env['CUDA_VISIBLE_DEVICES'];self.polls=0
            assert self.device not in active
            active.add(self.device);devices.append(self.device);maximum[0]=max(maximum[0],len(active))
        def poll(self):
            self.polls+=1
            if self.polls==1:return None
            active.remove(self.device);return 0
    monkeypatch.setattr(queue,'preflight',lambda task,completed,archive:None)
    monkeypatch.setattr(queue.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(queue.time,'sleep',lambda _:None)
    result=queue.run_queue('configs/research/first_batch_8gpu.yaml',archive='synthetic',dry_run=False,
                           status_file=tmp_path/'queue.json')
    assert maximum[0]==8 and set(devices)==set(map(str,range(8)))
    assert all(x['status']=='COMPLETED' for x in result['tasks'].values())


def test_result_validator_rejects_tampered_prediction(tmp_path):
    from shutil import copy2
    from src.research.results import validate_result
    from src.research.trainer import save_checksums
    source=Path('results/SMOKE_RESEARCH_V2/seed_42/fold_none')
    if not source.exists():pytest.skip('Synthetic research smoke has not run')
    for name in ['best.pth','validation.csv','metrics.json','config.yaml','train_history.csv',
                 'training.log','run_manifest.json','environment.txt','checksums.sha256']:
        copy2(source/name,tmp_path/name)
    save_checksums(tmp_path)
    frame=pd.read_csv(tmp_path/'validation.csv');frame.loc[0,'Label']=1-frame.loc[0,'Label']
    frame.to_csv(tmp_path/'validation.csv',index=False)
    with pytest.raises(ValueError,match='checksum'):
        validate_result(tmp_path,'outputs/research_smoke_v2/val.csv',freeze_inventory=None,allow_smoke=True)


def test_s04_requires_reviewed_locator_and_supports_three_binary_modes():
    from src.research.branches import GlobalLocalClassifier
    cfg=dataclasses.replace(ExperimentConfig.load('configs/research/G02.yaml'),image_size=64,config_hash=None)
    with pytest.raises(ValueError,match='reviewed locator'):
        GlobalLocalClassifier(cfg,cfg,None,smoke=True)
    class ReviewedLocator:
        reviewed=True
        def locate(self,image):return (image.width//2,image.height//2,.99)
    for mode in ('global_only','local_only','dual_branch'):
        model=GlobalLocalClassifier(cfg,cfg,ReviewedLocator(),mode,smoke=True)
        output=model(torch.randn(1,3,64,64),torch.randn(1,3,64,64))
        assert output.shape==(1,1)
        output.square().mean().backward()


def test_s04_dual_dataset_has_one_label_and_deterministic_validation():
    from src.research.data import DualImageDataset
    from src.research.trainer import forward_inputs
    class Locator:
        reviewed=True
        def locate(self,image):return (image.width//2,image.height//2,.99)
    cfg=dataclasses.replace(ExperimentConfig.load('configs/research/S04.yaml'),
        image_size=64,config_hash=None)
    row=pd.DataFrame([{'image_id':'synthetic','resolved_path':str(Path('outputs/research_smoke_v3/images/train_0.png')),
                       'label':1}])
    ds=DualImageDataset(row,cfg,cfg,Locator(),train=False)
    (g1,l1),y,i=ds[0];(g2,l2),_,_=ds[0]
    assert g1.shape==l1.shape==(3,64,64) and y.item()==1 and i==0
    assert torch.equal(g1,g2) and torch.equal(l1,l2)


def test_t4_certification_rejects_local_report(tmp_path):
    from src.research.deployment import certify_t4_report
    checkpoint=tmp_path/'model.pth';checkpoint.write_bytes(b'synthetic')
    report=tmp_path/'report.json';report.write_text(json.dumps({'official_protocol':False,'gpu':'NVIDIA RTX 4090'}),encoding='utf-8')
    with pytest.raises(ValueError,match='T4'):certify_t4_report(checkpoint,report)


def test_completed_smoke_resume_does_not_rewrite_artifacts():
    from src.research.trainer import run
    from src.utils import sha256
    out=Path('results/SMOKE_RESEARCH_V3/seed_42/fold_none')
    if not out.exists():pytest.skip('Synthetic research smoke v3 has not run')
    cfg=ExperimentConfig.load(out/'config.yaml')
    before=sha256(out/'training.log')
    with pytest.raises(ValueError,match='Completed experiment'):
        run(cfg,'outputs/research_smoke_v3/images','outputs/research_smoke_v3/train.csv',
            'outputs/research_smoke_v3/val.csv',smoke=True,resume=True)
    assert sha256(out/'training.log')==before
