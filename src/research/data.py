import math
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data import read_manifest
from .preprocess import transform_image


class ImageDataset(Dataset):
    def __init__(self,frame,cfg,train=False,locator=None):
        self.frame=frame.reset_index(drop=True);self.cfg=cfg;self.train=train;self.epoch=0;self.locator=locator
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.frame)
    def __getitem__(self,idx):
        row=self.frame.iloc[int(idx)]
        rng=np.random.default_rng(self.cfg.seed+self.epoch*len(self.frame)+int(idx))
        with Image.open(row.resolved_path) as im:x=transform_image(im,self.cfg,self.train,rng,self.locator)
        return x,torch.tensor(float(row.label)),int(idx)


class BalancedBatchSampler(Sampler):
    def __init__(self,labels,batch_size,seed):
        self.labels=np.asarray(labels,dtype=int);self.batch_size=batch_size;self.seed=seed;self.epoch=0
        self.pos=np.flatnonzero(self.labels==1);self.neg=np.flatnonzero(self.labels==0)
        if len(self.pos)==0 or len(self.neg)==0 or batch_size<2:raise ValueError('Balanced sampler needs both classes and batch>=2')
        self.last_stats={}
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(len(self.labels)/self.batch_size)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);seen=[];pos_n=max(1,self.batch_size//2);neg_n=self.batch_size-pos_n
        for _ in range(len(self)):
            p=rng.choice(self.pos,pos_n,replace=len(self.pos)<pos_n)
            n=rng.choice(self.neg,neg_n,replace=len(self.neg)<neg_n)
            indices=np.r_[p,n];rng.shuffle(indices);seen.extend(indices.tolist())
            yield indices.tolist()
        self.last_stats={'draws':len(seen),'unique_samples':len(set(seen)),
                         'duplicate_draws':len(seen)-len(set(seen)),
                         'independent_sample_coverage':len(set(seen))/len(self.labels),
                         'positive_draws':sum(self.labels[i] for i in seen)}


def load_official(cfg,root,train_manifest,val_manifest,group_manifest=None):
    cfg.assert_data(train_manifest,val_manifest)
    train=read_manifest(train_manifest,root);val=read_manifest(val_manifest,root)
    if set(train.resolved_path)&set(val.resolved_path):raise ValueError('Train/Val path overlap')
    if cfg.fold is not None:
        if not group_manifest:raise ValueError('Fold requires reviewed group/index manifest')
        import pandas as pd
        folds=pd.read_csv(group_manifest)
        if set(folds.image_id)!=set(train.image_id) or folds.image_id.duplicated().any():raise ValueError('Fold coverage mismatch')
        if cfg.extra.get('fold_manifest_hash')!=__import__('src.utils',fromlist=['sha256']).sha256(group_manifest):raise ValueError('Fold manifest hash mismatch')
        train=train.merge(folds[['image_id','fold']],on='image_id',validate='one_to_one')
        val_fold=train[train.fold==cfg.fold].drop(columns='fold')
        train=train[train.fold!=cfg.fold].drop(columns='fold')
        return train.reset_index(drop=True),val, val_fold.reset_index(drop=True)
    return train,val,None
