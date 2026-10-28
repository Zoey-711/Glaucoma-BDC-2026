"""Train-image QC sheets for S01/S02/R01/R02/R03. Never uses Val/Test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

from src.data import read_manifest,open_rgb
from src.research.config import ExperimentConfig
from src.research.preprocess import fov_crop,color_enhance,jpeg_compress,transform_image
from src.research.trainer import mixup
from src.utils import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    p.add_argument('--root',default='data/official');p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=12)
    a=p.parse_args();cfg=ExperimentConfig.load(a.config)
    if Path(a.train_manifest).name!='train.csv':raise ValueError('QC must use Train only')
    frame=read_manifest(a.train_manifest,a.root).iloc[:a.count]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sheet=Image.new('RGB',(640,180*a.count),'white');draw=ImageDraw.Draw(sheet);records=[]
    for k,r in enumerate(frame.itertuples()):
        original=open_rgb(r.resolved_path);changed=original;details={}
        if cfg.experiment_id.startswith('S01'):changed,details=fov_crop(original,**cfg.extra.get('fov',{}))
        elif cfg.experiment_id.startswith('S02'):changed=color_enhance(original,**cfg.extra['enhancement_params'])
        elif cfg.experiment_id.startswith('R02') and cfg.augmentation.get('jpeg_quality'):
            changed=jpeg_compress(original,cfg.augmentation['jpeg_quality'])
        elif cfg.experiment_id.startswith('R03'):
            # Visualize the exact soft-label interpolation used by the trainer.
            other=open_rgb(frame.iloc[(k+1)%len(frame)].resolved_path).resize(original.size)
            alpha=cfg.extra['mixup_alpha'];lam=float(np.random.default_rng(42+k).beta(alpha,alpha))
            changed=Image.blend(original,other,1-lam);details={'lambda':lam,'soft_label':lam*r.label+(1-lam)*frame.iloc[(k+1)%len(frame)].label}
        else:
            _,_,changed=transform_image(original,cfg,True,np.random.default_rng(42+k),return_qc=True)
        for x,im in [(0,original),(320,changed)]:
            im=im.copy();im.thumbnail((310,150));sheet.paste(im,(x,k*180+25))
        draw.text((0,k*180),r.image_id,fill='black');records.append({'image_id':r.image_id,**details})
    sheet.save(out/'qc_contact_sheet.jpg')
    pd.DataFrame(records).to_csv(out/'qc.csv',index=False)
    write_json(out/'report.json',{'experiment_id':cfg.experiment_id,'source':'official_train_only','count':len(frame),
                                  'manual_artifact_review':'pending','validation_used':False,'test_used':False})
    print(json.dumps({'output':str(out),'count':len(frame)}))


if __name__=='__main__':main()
