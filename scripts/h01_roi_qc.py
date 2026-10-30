"""Train-only optic-disc proposal QC; no automatic review approval or training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

from src.data import read_manifest,open_rgb
from src.research.h01_roi import ConservativeDiscLocator
from src.utils import sha256,write_json


def run(train_manifest,root,output,count=64,seed=42):
    train_manifest=Path(train_manifest);root=Path(root);out=Path(output)
    if out.exists():raise FileExistsError(out)
    if train_manifest.name!='train.csv' or len(pd.read_csv(train_manifest,nrows=0).columns)<3:
        raise ValueError('H01 QC requires the official labeled Train manifest')
    frame=read_manifest(train_manifest,root)
    if len(frame)!=51396:raise ValueError('Official Train must contain 51396 images')
    if 'train_05402.jpg' not in set(frame.image_id):raise ValueError('Required S01 regression sample absent')
    locator=ConservativeDiscLocator()
    chosen=frame.sample(n=min(count,len(frame)),random_state=seed)
    chosen=pd.concat([frame[frame.image_id=='train_05402.jpg'],chosen],ignore_index=True).drop_duplicates('image_id')
    out.mkdir(parents=True)
    rows=[];cards=[]
    for row in chosen.itertuples():
        image=open_rgb(row.resolved_path)
        local,proposal=locator.crop_or_fallback(image)
        mark=image.copy();draw=ImageDraw.Draw(mark)
        if proposal.box:draw.rectangle(proposal.box,outline=(255,0,0),width=max(2,image.width//400))
        thumb=Image.new('RGB',(768,256))
        for i,panel in enumerate((image,mark,local)):
            copy=panel.copy();copy.thumbnail((256,256));thumb.paste(copy,(256*i,0))
        cards.append(thumb)
        rows.append({'Image_name':row.image_id,'applied':proposal.applied,
            'confidence':proposal.confidence,'reason':proposal.reason,
            'box':json.dumps(proposal.box),'roi_latency_ms':proposal.latency_ms,
            **proposal.metrics})
    pd.DataFrame(rows).to_csv(out/'proposals.csv',index=False)
    sheet=Image.new('RGB',(768,256*len(cards)))
    for i,card in enumerate(cards):sheet.paste(card,(0,256*i))
    sheet.save(out/'contact_sheet.jpg',quality=90)
    failure=pd.DataFrame(rows).loc[lambda x:~x.applied]
    failure.to_csv(out/'fallback_candidates.csv',index=False)
    write_json(out/'qc_report.json',{'status':'BLOCKED','reason':'Heuristic candidates require independent manual localization review before classifier training',
        'source':'official_train_only','train_manifest_sha256':sha256(train_manifest),
        'locator_source_sha256':sha256(Path(__import__('src.research.h01_roi',fromlist=['x']).__file__)),
        'test_used':False,'val_used_for_thresholds':False,'reviewed':False,
        'qc_images':len(rows),'accepted_proposals':sum(x['applied'] for x in rows),
        'fallbacks':len(failure),'regression_train_05402_included':True,
        'roi_latency_ms_mean':float(np.mean([x['roi_latency_ms'] for x in rows]))})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    p.add_argument('--root',default='data/official');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=64);a=p.parse_args()
    print(run(a.train_manifest,a.root,a.output,a.count))
