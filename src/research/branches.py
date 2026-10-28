"""S04 feature fusion; formal use requires a reviewed S03 locator."""
import torch
from torch import nn
import time

from src.data import open_rgb
from src.submission import write_row
from .preprocess import transform_image

from .models import make_model
from .preprocess import crop_disc


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
        g=self.global_encoder(global_tensor).flatten(1)
        l=self.local_encoder(local_tensor).flatten(1)
        x=g if self.mode=='global_only' else l if self.mode=='local_only' else torch.cat([g,l],dim=1)
        return self.head(x)

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
