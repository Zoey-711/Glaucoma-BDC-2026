"""Train-only SimSiam; encoder can be transferred into a binary classifier."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import read_manifest
from src.utils import sha256,write_json
from .config import manifest_semantic_hash
from .models import make_model
from .preprocess import transform_image


class SimSiam(nn.Module):
    def __init__(self,encoder,feature_dim,projection_dim=512,prediction_dim=128):
        super().__init__();self.encoder=encoder
        self.projector=nn.Sequential(nn.Linear(feature_dim,projection_dim,bias=False),nn.BatchNorm1d(projection_dim),nn.ReLU(),
                                     nn.Linear(projection_dim,projection_dim,bias=False),nn.BatchNorm1d(projection_dim))
        self.predictor=nn.Sequential(nn.Linear(projection_dim,prediction_dim,bias=False),nn.BatchNorm1d(prediction_dim),nn.ReLU(),nn.Linear(prediction_dim,projection_dim))
    def forward(self,x1,x2):
        z1=self.projector(self.encoder(x1).flatten(1));z2=self.projector(self.encoder(x2).flatten(1))
        p1=self.predictor(z1);p2=self.predictor(z2)
        return -.5*(F.cosine_similarity(p1,z2.detach()).mean()+F.cosine_similarity(p2,z1.detach()).mean())


def assert_ssl_source(cfg,train_manifest,root):
    if Path(train_manifest).name!='train.csv' or cfg.data_manifest_hash['train']!=manifest_semantic_hash(train_manifest):
        raise ValueError('SimSiam may read only frozen official Train manifest')
    frame=read_manifest(train_manifest,root)
    if any('valid' in p.lower() or 'test' in p.lower() for p in frame.path):
        raise ValueError('Val/Test image in SSL source')
    return frame


def train_simsiam(cfg,train_manifest,root,output,initialization='imagenet_then_ssl_then_finetune',epochs=5,smoke=False):
    if initialization not in {'imagenet_then_ssl_then_finetune','random_then_ssl_then_finetune'}:
        raise ValueError('Unknown SSL initialization')
    frame=assert_ssl_source(cfg,train_manifest,root)
    from src.utils import seed_everything
    seed_everything(cfg.seed)
    if smoke and len(frame)>32:raise ValueError('Synthetic SSL smoke max32 images')
    if not smoke and initialization.startswith('imagenet'):cfg.assert_pretraining()
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset
    class PairDataset(Dataset):
        epoch=0
        def __len__(self):return len(frame)
        def __getitem__(self,i):
            row=frame.iloc[i]
            with Image.open(row.resolved_path) as im:
                offset=self.epoch*len(frame)*2
                return (transform_image(im,cfg,True,np.random.default_rng(cfg.seed+offset+2*i)),
                        transform_image(im,cfg,True,np.random.default_rng(cfg.seed+offset+2*i+1)))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if not smoke and device!='cuda':raise ValueError('Formal SimSiam requires CUDA')
    encoder=make_model(cfg,smoke=smoke or initialization.startswith('random'),features=True)
    model=SimSiam(encoder,encoder.num_features).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.backbone_lr,weight_decay=cfg.weight_decay)
    dataset=PairDataset()
    loader=DataLoader(dataset,batch_size=max(2,cfg.batch_size),shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(cfg.seed),drop_last=True)
    if not len(loader):raise ValueError('SimSiam needs at least one complete batch')
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');history=[]
    for epoch in range(epochs):
        dataset.epoch=epoch
        model.train();total=0
        for x1,x2 in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,enabled=device=='cuda'):
                loss=model(x1.to(device),x2.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('SimSiam nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();total+=float(loss.detach())
        history.append({'epoch':epoch+1,'loss':total/len(loader)})
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.save({'encoder':{k:v.cpu() for k,v in model.encoder.state_dict().items()},
                'training_data_sha256':manifest_semantic_hash(train_manifest),'pretraining_sha256':cfg.pretrained_sha256,
                'initialization':initialization,'test_used':False,'validation_used':False},out/'encoder.pth')
    write_json(out/'history.json',history)
    return out/'encoder.pth'
