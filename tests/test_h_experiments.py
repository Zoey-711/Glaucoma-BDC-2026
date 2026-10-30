import numpy as np
import pytest
import torch
from PIL import Image,ImageFilter

from src.research.config import ExperimentConfig,manifest_component_hashes
from src.research.data import BalancedBatchSampler,EpochSamplingDiagnostics
from src.research.h01_roi import ConservativeDiscLocator
from src.research.models import make_model
from scripts.h_gpu_task import release,status


@pytest.mark.parametrize('id', ['H03_G01_HEAD','H03_MATCHED_MLP','H03_MULTISCALE'])
def test_h03_single_logit_amp_backward_and_head_cost(id):
    cfg=ExperimentConfig.load(f'configs/h/{id}.yaml')
    model=make_model(cfg,smoke=True).cuda().train()
    x=torch.randn(1,3,224,224,device='cuda')
    with torch.autocast('cuda'):
        z=model(x).flatten()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.ones_like(z))
    loss.backward()
    assert z.shape==(1,) and torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    if hasattr(model,'cost_estimate'):
        assert model.cost_estimate()['feature_dimensions']['final_channels']==768


def test_h03_mlp_matches_multiscale_head_parameters():
    a=make_model(ExperimentConfig.load('configs/h/H03_MATCHED_MLP.yaml'),smoke=True)
    b=make_model(ExperimentConfig.load('configs/h/H03_MULTISCALE.yaml'),smoke=True)
    assert abs(a.cost_estimate()['head_parameters']-b.cost_estimate()['head_parameters'])<=128


def test_h04_sampler_diagnostics_train_only():
    labels=np.array([1]*12+[0]*36)
    sampler=BalancedBatchSampler(labels,8,42)
    tracker=EpochSamplingDiagnostics(labels)
    for indices in sampler:
        tracker.update(indices)
    result=tracker.summary()
    assert result['effective_batch_ratio']==1.
    assert result['positive_draws']==24 and result['epoch_positive_ratio']==.5
    assert result['positive_duplicate_draws']>0
    assert 0<result['independent_sample_coverage']<=1
    assert ExperimentConfig.load('configs/h/H04_RANDOM.yaml').status=='BLOCKED'
    assert ExperimentConfig.load('configs/h/H04_BALANCED.yaml').status=='BLOCKED'
    random_cfg=ExperimentConfig.load('configs/h/H04_RANDOM.yaml')
    balanced_cfg=ExperimentConfig.load('configs/h/H04_BALANCED.yaml')
    assert random_cfg.sampler=='shuffle' and balanced_cfg.sampler=='balanced'
    for field in ('backbone','image_size','loss','optimizer','backbone_lr','head_lr',
                  'weight_decay','epochs','patience','augmentation','batch_size','seed'):
        assert getattr(random_cfg,field)==getattr(balanced_cfg,field)


def test_h01_falls_back_on_dark_blur_and_edge_image():
    locator=ConservativeDiscLocator()
    dark=Image.new('RGB',(256,256),(4,4,4))
    cropped,p=locator.crop_or_fallback(dark)
    assert cropped is dark and not p.applied and p.reason in {'no_field','field_geometry'}
    blurred=Image.new('RGB',(256,256),(20,60,20)).filter(ImageFilter.GaussianBlur(10))
    assert not locator.propose(blurred).applied
    assert locator.reviewed is False
    with pytest.raises(ValueError):ConservativeDiscLocator(confidence_threshold=.1)


def test_h_launcher_releases_only_finished_owned_lock(monkeypatch,tmp_path):
    import json
    from scripts import h_gpu_task
    monkeypatch.setattr(h_gpu_task,'LOCK_DIR',tmp_path)
    marker=tmp_path/'H03_MULTISCALE.json'
    marker.write_text(json.dumps({'pid':99999999,'gpu':2}),encoding='utf-8')
    lock=tmp_path/'gpu_2.lock';lock.write_text('H03_MULTISCALE',encoding='utf-8')
    monkeypatch.setattr(h_gpu_task,'_alive',lambda pid:False)
    assert next(x for x in status() if x['experiment_id']=='H03_MULTISCALE')['state']=='FAILED'
    assert release('H03_MULTISCALE')['gpu_lock_released']
    assert not marker.exists() and not lock.exists()
    with pytest.raises(FileNotFoundError):release('H03_MULTISCALE')


def test_manifest_annotation_component_includes_age(tmp_path):
    import pandas as pd
    file=tmp_path/'train.csv'
    pd.DataFrame([{'image_id':'a.jpg','path':'train/a.jpg','label':1,'Age':65}]).to_csv(file,index=False)
    before=manifest_component_hashes(file)
    pd.DataFrame([{'image_id':'a.jpg','path':'train/a.jpg','label':1,'Age':66}]).to_csv(file,index=False)
    after=manifest_component_hashes(file)
    assert before['images']==after['images']
    assert before['annotations']!=after['annotations']
