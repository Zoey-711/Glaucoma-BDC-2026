"""S04 feature fusion; formal use requires a reviewed S03 locator."""
import torch
from torch import nn
import time
import dataclasses
import importlib
import json
from pathlib import Path

from src.data import open_rgb
from src.submission import write_row
from .preprocess import transform_image

from .models import make_model
from .preprocess import crop_disc
from src.model import verified_weights
from src.utils import sha256


def load_reviewed_locator(cfg):
    """Load only a locator with a separate, immutable human review record."""
    spec=cfg.extra.get('locator_plugin');report_path=cfg.extra.get('locator_review_report')
    if not spec or not report_path:raise ValueError('S03 reviewed locator plugin and review report required')
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    module_name,class_name=spec.split(':',1)
    module=importlib.import_module(module_name)
    source=Path(module.__file__)
    if (report.get('status')!='approved' or report.get('source_sha256')!=sha256(source)
        or report.get('reliability_reviewed') is not True
        or not isinstance(report.get('reviewed_samples'),int) or report['reviewed_samples']<100
        or report.get('training_source') not in {'official_train_only','no_training'}
        or report.get('official_val_used_for_training') is not False
        or report.get('test_used') is not False
        or not report.get('reviewer')):
        raise ValueError('S03 locator source/reliability review incomplete')
    locator=getattr(module,class_name)()
    if not getattr(locator,'reviewed',False):raise ValueError('S03 locator did not declare reviewed status')
    return locator


def local_config(cfg):
    """S04 local branch uses the independently reviewed B02 ImageNet weights."""
    path=cfg.extra.get('local_weights_manifest','configs/weights_b02.local.json')
    m=verified_weights(path,'efficientnet_b0')
    return dataclasses.replace(cfg,backbone='efficientnet_b0',weights_manifest=path,
        pretrained_source=m['source_url'],pretrained_sha256=m['sha256'],license='Apache-2.0',
        config_hash=None)


class GlobalLocalClassifier(nn.Module):
    def __init__(self,global_cfg,local_cfg,locator,mode='dual_branch',smoke=False):
        super().__init__()
        if mode not in {'global_only','local_only','dual_branch'}:raise ValueError('Unknown ablation mode')
        if locator is None or not getattr(locator,'reviewed',False):raise ValueError('S03 reviewed locator required')
        self.global_encoder=make_model(global_cfg,smoke=smoke,features=True)
        self.local_encoder=make_model(local_cfg,smoke=smoke,features=True)
        self.mode=mode;self.locator=locator
        gd=self.global_encoder.num_features;ld=self.local_encoder.num_features
        self.head=nn.Linear(gd if mode=='global_only' else ld if mode=='local_only' else gd+ld,1)

    def forward(self,global_tensor,local_tensor):
        g=self.global_encoder(global_tensor).flatten(1) if self.mode!='local_only' else None
        l=self.local_encoder(local_tensor).flatten(1) if self.mode!='global_only' else None
        x=g if self.mode=='global_only' else l if self.mode=='local_only' else torch.cat([g,l],dim=1)
        return self.head(x)

    def get_classifier(self):return self.head

    def prepare_local(self,image):
        # Caller times this together with decode, feature extraction and CSV write.
        return crop_disc(image,self.locator)


def predict_s04_one(model,path,image_id,global_cfg,local_cfg,writer,handle,device='cuda'):
    """One timed raw-image operation including locator and ROI fallback."""
    if not getattr(model.locator,'reviewed',False):raise ValueError('Unreviewed locator')
    if device.startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter_ns()
    image=open_rgb(path)
    local,roi=model.prepare_local(image)
    global_tensor=transform_image(image,global_cfg,False).unsqueeze(0).to(device)
    local_tensor=transform_image(local,local_cfg,False).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():probability=float(model(global_tensor,local_tensor).flatten().sigmoid().item())
    write_row(writer,image_id,probability,.5);handle.flush()
    if device.startswith('cuda'):torch.cuda.synchronize()
    return {'probability':probability,'latency_ms':(time.perf_counter_ns()-start)/1e6,'roi':roi}
