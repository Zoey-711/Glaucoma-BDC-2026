import math
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data import read_manifest
from .preprocess import transform_image
from .preprocess import crop_disc


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


class DualImageDataset(ImageDataset):
    """S04 source-image decode, reviewed ROI and separate deterministic validation views."""
    def __init__(self,frame,cfg,local_cfg,locator,train=False):
        super().__init__(frame,cfg,train,locator)
        if not getattr(locator,'reviewed',False):raise ValueError('Reviewed S03 locator required')
        self.local_cfg=local_cfg

    def __getitem__(self,idx):
        row=self.frame.iloc[int(idx)]
        with Image.open(row.resolved_path) as im:
            image=im.convert('RGB')
            local,roi=crop_disc(image,self.locator)
            if not roi['applied'] and self.cfg.extra.get('require_roi',True):
                raise ValueError(f'ROI locator failed for {row.image_id}; run S03 QC first')
            seed=self.cfg.seed+self.epoch*len(self.frame)+int(idx)
            global_tensor=transform_image(image,self.cfg,self.train,np.random.default_rng(seed))
            local_tensor=transform_image(local,self.local_cfg,self.train,np.random.default_rng(seed))
        return (global_tensor,local_tensor),torch.tensor(float(row.label)),int(idx)


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


class EpochSamplingDiagnostics:
    """Observe actual Train-only draws; never alter the sampler or Val loader."""
    def __init__(self, labels):
        self.labels=np.asarray(labels,dtype=int)
        if set(np.unique(self.labels))!={0,1}:
            raise ValueError('Sampler diagnostics require binary Train labels')
        self.seen=[];self.valid_batches=0;self.batches=0

    def update(self, indices):
        drawn=np.asarray(indices,dtype=int).reshape(-1)
        if (drawn<0).any() or (drawn>=len(self.labels)).any():
            raise ValueError('Sampler drew index outside official Train')
        values=self.labels[drawn]
        self.valid_batches+=int(0<values.sum()<len(values))
        self.batches+=1
        self.seen.extend(drawn.tolist())

    def summary(self):
        seen=np.asarray(self.seen,dtype=int)
        pos=seen[self.labels[seen]==1]
        unique_pos=len(np.unique(pos));unique_total=len(np.unique(seen))
        repeat=max(0,len(pos)-unique_pos)
        coverage=unique_total/len(self.labels)
        repeat_rate=repeat/len(pos) if len(pos) else 0.
        return {'sample_draws':len(seen),'positive_draws':len(pos),
            'epoch_positive_ratio':len(pos)/len(seen) if len(seen) else 0.,
            'positive_duplicate_draws':repeat,'positive_repeat_rate':repeat_rate,
            'positive_unique_coverage':unique_pos/int((self.labels==1).sum()),
            'independent_sample_coverage':coverage,
            'effective_batch_ratio':self.valid_batches/self.batches if self.batches else 0.,
            'single_class_batches':self.batches-self.valid_batches,
            'sampling_risk_flag':bool(repeat_rate>.80 or coverage<.65)}


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
