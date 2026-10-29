"""S01-v2 full Train QC, permanent native-signal regression and frozen Val safety."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageChops, ImageDraw, ImageOps

from src.data import read_manifest,open_rgb
from src.fov_v2 import crop_fov_v2,POLICY
from src.utils import sha256,write_json

ROOT=Path('outputs/S01_v2')
KNOWN_ID='train_05402.jpg'
KNOWN_SHA='906995c5fba32f64e1b7edb28c102eb1f133dbd5016a4ccf97689c55299213c5'


def guard():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'frozen_inventory.json'
    if not path.exists():
        inventory=json.loads(Path('outputs/S01_qc/frozen_inventory.json').read_text())
        for base in ['outputs/S01_qc','outputs/S01_execution']:
            for p in Path(base).rglob('*'):
                if p.is_file():inventory[str(p)]=sha256(p)
        for name in ['src/fov.py','scripts/s01_qc.py','scripts/run_s01.py','scripts/report_s01.py',
                     'configs/experiments/S01a.yaml','configs/experiments/S01b.yaml','docs/S01_experiment_report.md']:
            inventory[name]=sha256(name)
        write_json(path,inventory)
    inventory=json.loads(path.read_text())
    bad=[p for p,h in inventory.items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise ValueError(f'Frozen assets changed: {bad}')
    return len(inventory)


def source(split):
    c=json.loads(Path('outputs/runs/G01/config.json').read_text())
    path=c['data']['train_csv' if split=='train' else 'val_csv']
    # Explicitly discard labels; no QC decision depends on labels or predictions.
    frame=read_manifest(path,c['data']['root'],labeled=False)[['image_id','path','resolved_path']]
    return frame,path


def independent_safety(original,output,box,check_pixels=False):
    x0,y0,x1,y1=box;w,h=original.size
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):return {'safety_verified':False,'removed_signal_pixels':-1}
    strips=[(0,0,w,y0),(0,y1,w,h),(0,y0,x0,y1),(x1,y0,w,y1)]
    removed=0
    for box_ in strips:
        if box_[0]==box_[2] or box_[1]==box_[3]:continue
        strip=original.crop(box_)
        if max(pair[1] for pair in strip.getextrema())>0:
            removed+=int(np.any(np.asarray(strip)!=0,axis=2).sum())
    matches=output.size==(x1-x0,y1-y0)
    if check_pixels and matches:matches=ImageChops.difference(original.crop(box),output).getbbox() is None
    return {'safety_verified':bool(removed==0 and matches),'removed_signal_pixels':removed,
            'output_matches_source_checked':check_pixels,'output_matches_source':bool(matches)}


def pair_image(original,output,meta,title,size=(480,220)):
    panel=Image.new('RGB',size,'white');draw=ImageDraw.Draw(panel)
    draw.text((3,3),title,fill='black');draw.text((3,17),f"{meta['reason']} area={meta['area_ratio']:.3f}",fill='black')
    before=original.copy();d=ImageDraw.Draw(before)
    d.rectangle(meta['proposed_box'],outline='orange',width=max(2,original.width//200))
    d.rectangle(meta['box'],outline='lime',width=max(2,original.width//200))
    for x,im in [(0,before),(size[0]//2,output)]:
        panel.paste(ImageOps.contain(im,(size[0]//2-5,size[1]-40)),(x,36))
    return panel


def sheets(paths,folder,prefix):
    for start in range(0,len(paths),10):
        sheet=Image.new('RGB',(960,1100),'white')
        for j,p in enumerate(paths[start:start+10]):
            with Image.open(p) as im:sheet.paste(im,((j%2)*480,(j//2)*220))
        sheet.save(folder/f'{prefix}_{start//10:04d}.jpg',quality=90)


def regression():
    guard();frame,_=source('train');lookup=frame.set_index('image_id')
    expected=pd.read_csv('outputs/S01_qc/extreme_review.csv').image_id.tolist()
    if KNOWN_ID not in expected:expected.append(KNOWN_ID)
    out=ROOT/'regression';out.mkdir(exist_ok=False)
    records=[];panels=[]
    for name in expected:
        path=lookup.loc[name,'resolved_path'];im=open_rgb(path)
        changed,m,masks=crop_fov_v2(im,diagnostics=True)
        safety=independent_safety(im,changed,m['box'],True)
        if name==KNOWN_ID:
            assert sha256(path)==KNOWN_SHA,'Known source changed'
            assert safety['safety_verified'] and m['fallback'] and changed.size==im.size
            folder=out/name.replace('.jpg','');folder.mkdir()
            im.save(folder/'original.png');changed.save(folder/'final_output.png')
            lifted=Image.fromarray((np.power(np.asarray(im)/255.,.35)*255).astype('uint8'))
            lifted.save(folder/'brightness_diagnostic.png')
            d=ImageDraw.Draw(lifted);d.rectangle(m['proposed_box'],outline='orange',width=12);d.rectangle(m['box'],outline='lime',width=12)
            lifted.save(folder/'boxes.png')
            for key,mask in masks.items():Image.fromarray(mask).save(folder/f'mask_{key}.png')
            write_json(folder/'evidence.json',{'image_sha256':KNOWN_SHA,**m,**safety})
        records.append({'image_id':name,**m,**safety})
        p=out/f'{name}.png';pair_image(im,changed,m,name).save(p);panels.append(p)
    sheets(panels,out,'extremes')
    df=pd.DataFrame(records);df.to_csv(out/'regression.csv',index=False)
    assert df.safety_verified.all(),'Regression truncation detected'
    write_json(out/'report.json',{'status':'PASSED','samples':len(df),'known_sample':KNOWN_ID,
        'crop_code_sha256':sha256('src/fov_v2.py'),'all_nonzero_signal_preserved':True})
    print(df[['image_id','reason','area_ratio','safety_verified']].to_string(index=False),flush=True)


def audit(split):
    guard();frame,manifest=source(split)
    assert len(frame)==(51396 if split=='train' else 5849)
    lock=ROOT/'policy_locked.json'
    if split=='val':
        fixed=json.loads(lock.read_text())
        assert fixed['crop_code_sha256']==sha256('src/fov_v2.py') and fixed['policy']==POLICY
        assert json.loads((ROOT/'train'/'visual_review.json').read_text())['status']=='PASSED'
    out=ROOT/split;out.mkdir(exist_ok=False)
    anomalies=out/'anomaly_pairs';anomalies.mkdir()
    old=pd.read_csv('outputs/S01_qc/train_crop_statistics.csv').set_index('image_id') if split=='train' else None
    known=set(pd.read_csv('outputs/S01_qc/extreme_review.csv').image_id) if split=='train' else set()
    def scan(item):
        i,row=item
        try:
            im=open_rgb(row.resolved_path);changed,m=crop_fov_v2(im)
            v1_risk=bool(old.loc[row.image_id,'fallback'] or old.loc[row.image_id,'eccentricity']>.15) if old is not None else False
            anomaly=bool(m['fallback'] or m['boundary_risk'] or m['eccentricity']>.15 or v1_risk or row.image_id in known)
            check=independent_safety(im,changed,m['box'],check_pixels=anomaly)
            if anomaly or not check['safety_verified']:
                pair_image(im,changed,m,row.image_id).save(anomalies/f'{i:06d}.jpg',quality=85)
            return dict(image_id=row.image_id,**m,**check,anomaly=anomaly,
                        targeted_resolution='unchanged_fallback' if m['fallback'] else 'native_zero_only_removal',
                        source_bytes=Path(row.resolved_path).stat().st_size)
        except Exception as exc:
            return dict(image_id=row.image_id,reason='audit_error',error=repr(exc),safety_verified=False,anomaly=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(pool.map(scan,enumerate(frame.itertuples())),1):
            rows.append(row)
            if i%2000==0:
                print(f'{split} QC {i}/{len(frame)}',flush=True)
                write_json(out/'progress.json',{'processed':i,'total':len(frame)})
    df=pd.DataFrame(rows);df.to_csv(out/'statistics.csv',index=False)
    df[df.anomaly].to_csv(out/'all_anomaly_targeted_review.csv',index=False)
    df[df.fallback.fillna(True) | ~df.safety_verified].to_csv(out/'failures.csv',index=False)
    df[~df.safety_verified].to_csv(out/'unresolved_severe_risks.csv',index=False)
    sheets(sorted(anomalies.glob('*.jpg')),out,'anomalies')
    sample_count=160 if split=='train' else 40
    pools={'dark':df.nsmallest(2000,'brightness'),'eccentric':df.nlargest(2000,'eccentricity'),
           'black_border':df.nlargest(2000,'black_fraction'),
           'normal':df[(df.brightness>=df.brightness.median()) & (df.eccentricity<.1)]}
    chosen=[];used=set();pairs=out/'sample_pairs';pairs.mkdir();lookup=frame.set_index('image_id')
    for group,pool in pools.items():
        for name in pool[~pool.image_id.isin(used)].sample(sample_count//4,random_state=42).image_id:
            used.add(name);im=open_rgb(lookup.loc[name,'resolved_path']);changed,m=crop_fov_v2(im)
            p=pairs/f'{len(chosen):03d}.jpg';pair_image(im,changed,m,f'{group} {name}').save(p,quality=95)
            chosen.append({'image_id':name,'group':group,'file':str(p)})
    pd.DataFrame(chosen).to_csv(out/'stratified_sample.csv',index=False)
    sheets([Path(r['file']) for r in chosen],out,'sample')
    report={'status':'PENDING_VISUAL_REVIEW','split':split,'rows':len(df),'labels_used':False,
        'crop_code_sha256':sha256('src/fov_v2.py'),'manifest_sha256':sha256(manifest),
        'crop_fraction':float(df.applied.fillna(False).mean()),'fallback_fraction':float(df.fallback.fillna(True).mean()),
        'boundary_risk_proposals':int(df.boundary_risk.fillna(False).sum()),
        'connected_weak_structure_proposals':int(df.continuous_structure_candidate.fillna(False).sum()),
        'final_truncation_candidates':int((~df.safety_verified).sum()),
        'anomaly_candidates':int(df.anomaly.sum()),'all_anomalies_independently_checked':True,
        'reason_counts':df.reason.value_counts().to_dict(),'area_ratio_quantiles':df.area_ratio.quantile([0,.01,.25,.5,.75,.99,1]).to_dict(),
        'mean_area_ratio':float(df.area_ratio.mean()),'stratified_samples':sample_count}
    if old is not None:
        report['v1_comparison']={'crop_fraction':float(old.applied.mean()),'fallback_fraction':float(old.fallback.mean()),
                                'mean_area_ratio':float(old.area_ratio.mean())}
    write_json(out/'report.json',report)
    if split=='train':
        assert not lock.exists()
        write_json(lock,{'locked_before_val':True,'crop_code_sha256':sha256('src/fov_v2.py'),'policy':POLICY,
            'train_audit_sha256':sha256(out/'report.json'),'train_manifest_sha256':sha256(manifest)})
        (ROOT/'preprocessing.yaml').write_text(yaml.safe_dump(POLICY,sort_keys=False),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['regression','train','val']);a=p.parse_args()
    regression() if a.action=='regression' else audit(a.action)
