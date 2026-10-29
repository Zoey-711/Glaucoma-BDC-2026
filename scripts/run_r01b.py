"""Train-only light-color experiment controlled against frozen G01."""
import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms as T

from scripts.s01_v2_qc import guard as frozen_guard
from src.analysis import training_plot
from src.benchmark import local_benchmark
from src.bootstrap import paired_bootstrap
from src.data import FundusDataset, open_rgb, read_manifest, transform
from src.experiment_guard import checksums
from src.experiment_training import train, training_data
from src.metrics import metrics
from src.model import build_model, verified_weights
from src.utils import load_config, seed_everything, sha256, write_json

ROOT=Path('outputs/R01b_execution')
CONFIG=Path('configs/experiments/R01b.yaml')
OUT=Path('results/R01b/seed_42/fold_none')
G01=Path('outputs/runs/G01/config.json')
AUGMENT='r01b_light_color'


def controlled_config():
    old=json.loads(G01.read_text(encoding='utf-8'))
    new=copy.deepcopy(old)
    new.update(experiment_id='R01b',augmentation=AUGMENT,output=str(OUT).replace('\\','/'),
        plan={'reference':'G01','only_factor':'train_color_jitter','frozen_control':True})
    return old,new


def gate(visual=True):
    frozen_guard()
    old,new=controlled_config()
    current=load_config(CONFIG)
    assert current==new, 'R01b differs from controlled G01 configuration'
    assert old['augmentation']=='basic' and old['preprocess']=='resize'
    assert sha256('src/experiment_training.py')==sha256('outputs/phase2_g01_execution/source_snapshot/src/experiment_training.py')
    verified_weights(new['model']['weights_manifest'],new['model']['name'])
    assert sha256(new['data']['train_csv'])==sha256(old['data']['train_csv'])
    assert sha256(new['data']['val_csv'])==sha256(old['data']['val_csv'])
    if visual:
        qc=json.loads((ROOT/'qc_report.json').read_text(encoding='utf-8'))
        review=json.loads((ROOT/'visual_review.json').read_text(encoding='utf-8'))
        assert qc['status']=='READY_FOR_VISUAL_REVIEW' and qc['pairs']>=64
        assert review['status']=='PASSED' and review['config_sha256']==sha256(CONFIG)
        assert review['data_code_sha256']==sha256('src/data.py')
        assert qc['config_sha256']==sha256(CONFIG) and qc['data_code_sha256']==sha256('src/data.py')
    return current


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():raise FileExistsError(CONFIG)
    old,new=controlled_config()
    assert old['augmentation']=='basic' and old['preprocess']=='resize'
    CONFIG.write_text(yaml.safe_dump(new,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write_json(ROOT/'control.json',{'reference':str(G01),'reference_sha256':sha256(G01),
        'changed_config_fields':['experiment_id','augmentation','output','plan'],
        'G01_augmentation':'basic: horizontal flip 0.5, rotation +/-10 degrees; no color jitter',
        'R01b_augmentation':'same geometry + RandomApply(ColorJitter(.10,.10,.08,.02), p=.6)',
        'trainer_sha256':sha256('src/experiment_training.py'),'data_code_sha256':sha256('src/data.py'),
        'pretrained_manifest_sha256':sha256(new['model']['weights_manifest'])})
    print('R01b controlled configuration written',flush=True)


def qc():
    c=gate(False)
    folder=ROOT/'qc';folder.mkdir(parents=True,exist_ok=False)
    frame=read_manifest(c['data']['train_csv'],c['data']['root'],labeled=False)
    stats=pd.read_csv('outputs/S01_v2/train/statistics.csv')
    assert frame.image_id.tolist()==stats.image_id.tolist()
    groups={'dark':stats.nsmallest(1000,'brightness'),
        'black_border':stats.nlargest(1000,'black_fraction'),
        'eccentric':stats.nlargest(1000,'eccentricity'),
        'ordinary':stats[(stats.brightness.between(stats.brightness.quantile(.4),stats.brightness.quantile(.6))) &
                         (stats.eccentricity<.1)]}
    used=set();selected=[]
    for name,pool in groups.items():
        choices=pool[~pool.image_id.isin(used)].sample(n=16,random_state=42).image_id.tolist()
        for image_id in choices:used.add(image_id);selected.append((name,image_id))
    lookup=frame.set_index('image_id')
    rows=[];max_mean_delta=0.
    for i,(group,image_id) in enumerate(selected):
        im=open_rgb(lookup.loc[image_id,'resolved_path']).resize((224,224),Image.Resampling.BILINEAR)
        seed_everything(42000+i)
        aug=T.RandomApply([T.ColorJitter(brightness=.10,contrast=.10,saturation=.08,hue=.02)],p=.6)(im)
        orig=np.asarray(im).astype(np.float32);changed=np.asarray(aug).astype(np.float32)
        mean_delta=float(np.abs(changed-orig).mean());max_mean_delta=max(max_mean_delta,mean_delta)
        panel=Image.new('RGB',(448,248),'white');panel.paste(im,(0,24));panel.paste(aug,(224,24))
        ImageDraw.Draw(panel).text((2,3),f'{group} {image_id} original | enhanced',fill='black')
        panel.save(folder/f'pair_{i:03d}.jpg',quality=94)
        rows.append({'image_id':image_id,'group':group,'mean_abs_rgb_delta':mean_delta,
                     'original_rgb_mean':orig.mean(axis=(0,1)).tolist(),
                     'enhanced_rgb_mean':changed.mean(axis=(0,1)).tolist()})
    for start in range(0,64,8):
        sheet=Image.new('RGB',(896,992),'white')
        for j in range(8):
            with Image.open(folder/f'pair_{start+j:03d}.jpg') as im:
                sheet.paste(im,((j%2)*448,(j//2)*248))
        sheet.save(folder/f'sheet_{start//8:02d}.jpg',quality=92)
    write_json(ROOT/'qc_report.json',{'status':'READY_FOR_VISUAL_REVIEW','split':'official_train',
        'labels_used':False,'test_used':False,'pairs':64,'groups':{k:16 for k in groups},
        'max_mean_abs_rgb_delta':max_mean_delta,'config_sha256':sha256(CONFIG),
        'data_code_sha256':sha256('src/data.py'),'contact_sheets':8})
    print('64 QC pairs and 8 contact sheets generated',flush=True)


def smoke():
    c=gate();folder=ROOT/'smoke_v2';folder.mkdir(parents=True,exist_ok=False)
    images=folder/'images';images.mkdir()
    for split,n in [('train',8),('val',4)]:
        rows=[]
        for i in range(n):
            im=Image.new('RGB',(320,240),(15,15,15));ImageDraw.Draw(im).ellipse((40,10,280,230),
                fill=(80+i*8,35 if split=='train' else 55,20))
            image_id=f'{split}_{i}.png';im.save(images/image_id)
            rows.append({'image_id':image_id,'path':image_id,'label':i%2})
        pd.DataFrame(rows).to_csv(folder/f'{split}.csv',index=False)
    tiny=copy.deepcopy(c)
    tiny.update(smoke=True,experiment_id='SMOKE_R01b',epochs=1,batch_size=4,num_workers=0,
        cache=False,output=str(folder/'run'))
    tiny['model']['weights_manifest']=None
    tiny['data']={'source':'synthetic','root':str(images.resolve()),
                  'train_csv':str(folder/'train.csv'),'val_csv':str(folder/'val.csv')}
    train(tiny)
    seed_everything(42);torch.cuda.empty_cache()
    model=build_model(c['model']['weights_manifest'],model_config=c['model']).cuda().train()
    sample=read_manifest(c['data']['train_csv'],c['data']['root']).sample(n=128,random_state=42)
    ds=FundusDataset(sample,224,True,AUGMENT,'resize')
    dl=DataLoader(ds,batch_size=32,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');times=[];torch.cuda.reset_peak_memory_stats()
    for x,y in dl:
        start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x.cuda()).flatten(),y.cuda())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    write_json(ROOT/'smoke_result.json',{'status':'PASSED','real_train_batches':len(times),
        'batch_size':32,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'images_per_second_excluding_decode':32/np.mean(times[1:]),'probe_discarded':True,
        'config_sha256':sha256(CONFIG),'data_code_sha256':sha256('src/data.py')})
    print('R01b smoke PASSED',flush=True)


def run():
    c=gate();sm=json.loads((ROOT/'smoke_result.json').read_text(encoding='utf-8'))
    assert sm['status']=='PASSED' and sm['config_sha256']==sha256(CONFIG) and sm['data_code_sha256']==sha256('src/data.py')
    if OUT.exists() or (ROOT/'run_manifest.json').exists():raise FileExistsError('R01b already started')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for base in ['src','scripts']:
        for p in Path(base).rglob('*.py'):
            target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    manifest={'status':'RUNNING','started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'config_sha256':sha256(CONFIG),'seed':42,
        'data_sha256':{k:sha256(c['data'][k]) for k in ['train_csv','val_csv']},
        'source_sha256':{str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*.py')}}
    write_json(ROOT/'run_manifest.json',manifest);history=[]
    def on_epoch(row,*_):
        history.append(row);pd.DataFrame(history).to_csv(OUT/'train_history.csv',index=False)
        with (OUT/'training.log').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        training_plot(OUT)
    try:
        best=train(c,on_epoch=on_epoch)
        pred=pd.read_csv(OUT/'validation.csv');ref=pd.read_csv('outputs/runs/G01/validation.csv')
        assert pred.image_id.tolist()==ref.image_id.tolist() and np.array_equal(pred.label,ref.label)
        score=metrics(pred.label,pred.probability)
        timing=local_benchmark(best,c,OUT/'latency')
        result={'metrics':score,'paired_bootstrap':paired_bootstrap(ref,pred,repeats=2000,seed=42),
            'best_epoch':max(history,key=lambda r:(r['auroc'],r['auprc']))['epoch'],
            'epochs_completed':len(history),'model_bytes':best.stat().st_size,
            'latency_4060_ms':timing['mean_ms'],'T4_measured':False}
        write_json(OUT/'comparison.json',result);write_json(OUT/'metrics.json',score)
        shutil.copy2(CONFIG,OUT/'config.yaml')
        manifest.update(status='COMPLETED',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                        frozen_files_verified=frozen_guard())
        write_json(OUT/'run_manifest.json',manifest);checksums(OUT)
        write_json(ROOT/'run_manifest.json',manifest)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),error=repr(exc))
        write_json(ROOT/'run_manifest.json',manifest)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','qc','smoke','run']);a=p.parse_args()
    {'prepare':prepare,'qc':qc,'smoke':smoke,'run':run}[a.action]()
