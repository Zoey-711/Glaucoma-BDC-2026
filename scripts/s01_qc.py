"""Official Train only: full crop audit and stratified reproducible QC sheets."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.data import read_manifest, open_rgb
from src.experiment_guard import freeze_or_verify
from src.fov import crop_fov
from src.utils import sha256, write_json

ROOT = Path('outputs/S01_qc')


def guard():
    freeze_or_verify()
    path = ROOT/'frozen_inventory.json'
    if not path.exists():
        inventory = json.loads(Path('outputs/M02d_L03/frozen_inventory.json').read_text())
        for base in ['results/M02d', 'results/L03b', 'outputs/L03b_execution']:
            for file in Path(base).rglob('*'):
                if file.is_file(): inventory[str(file)] = sha256(file)
        write_json(path, inventory)
    inventory = json.loads(path.read_text())
    assert all(sha256(p)==h for p,h in inventory.items()), 'Frozen assets changed'
    return len(inventory)


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    guard()
    c = json.loads(Path('outputs/runs/G01/config.json').read_text())
    frame = read_manifest(c['data']['train_csv'], c['data']['root'])
    assert len(frame)==51396
    target = ROOT/'train_crop_statistics.csv'
    if target.exists(): raise FileExistsError('Audit already exists; do not overwrite')
    def check(row):
        try:
            _, meta = crop_fov(open_rgb(row.resolved_path))
            return dict(image_id=row.image_id, **meta)
        except Exception as exc:
            return dict(image_id=row.image_id, reason='decode_error', error=repr(exc))
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(check, frame.itertuples()), 1):
            records.append(result)
            if i%2000==0: print(f'Train QC {i}/{len(frame)}', flush=True)
    df=pd.DataFrame(records); df.to_csv(target,index=False)
    df[df.reason!='cropped'].to_csv(ROOT/'fallback_and_uncropped.csv',index=False)
    suspicious=df[df.fallback.fillna(True) | (df.area_ratio < .4) | (df.eccentricity > .15)]
    suspicious.to_csv(ROOT/'anomaly_candidates.csv',index=False)
    used=set(); selected=[]
    pools={'dark':df.nsmallest(2000,'brightness'),
           'eccentric':df.nlargest(2000,'eccentricity'),
           'black_border':df.nlargest(2000,'black_fraction'),
           'normal':df[(df.brightness>=df.brightness.median()) & (df.eccentricity<.1)]}
    for group,pool in pools.items():
        sample=pool[~pool.image_id.isin(used)].sample(n=30,random_state=42)
        for row in sample.itertuples():
            used.add(row.image_id); selected.append({'image_id':row.image_id,'group':group})
    pd.DataFrame(selected).to_csv(ROOT/'selected_120.csv',index=False)
    lookup=frame.set_index('image_id'); pairs=ROOT/'pairs';pairs.mkdir()
    for start in range(0,len(selected),10):
        sheet=Image.new('RGB',(1000,5*240),'white'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected[start:start+10]):
            im=open_rgb(lookup.loc[r['image_id'],'resolved_path']);cropped,meta=crop_fov(im)
            before=im.copy();ImageDraw.Draw(before).rectangle(meta['box'],outline='lime',width=max(2,im.width//200))
            pair=Image.new('RGB',(500,240),'white');d=ImageDraw.Draw(pair)
            d.text((3,3),f"{r['group']} {r['image_id']}",fill='black')
            d.text((3,18),f"{meta['reason']} area={meta['area_ratio']:.3f}",fill='black')
            for x,pic in [(0,before),(250,cropped)]:
                pair.paste(ImageOps.contain(pic,(245,200)),(x,36))
            pair.save(pairs/f"{start+j:03d}.jpg",quality=95)
            sheet.paste(pair,((j%2)*500,(j//2)*240))
        sheet.save(ROOT/f'sheet_{start//10:02d}.jpg',quality=95)
    report={'status':'PENDING_VISUAL_REVIEW','source':'official_train_only','rows':len(df),
            'test_used':False,'val_used':False,'crop_code_sha256':sha256('src/fov.py'),
            'train_manifest_sha256':sha256(c['data']['train_csv']),
            'reason_counts':df.reason.value_counts().to_dict(),'fallback_fraction':float(df.fallback.fillna(True).mean()),
            'crop_fraction':float(df.applied.fillna(False).mean()),
            'area_ratio_quantiles':df.area_ratio.quantile([0,.01,.25,.5,.75,.99,1]).to_dict(),
            'dimensions':df[['width','height']].describe().to_dict(),'qc_images':120,
            'frozen_files_verified':guard()}
    write_json(ROOT/'audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
