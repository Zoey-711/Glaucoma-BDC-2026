"""G01-only execution with audit gates, discarded preflight and frozen B02 checks."""
import argparse
import copy
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.analysis import training_plot
from src.benchmark import local_benchmark
from src.bootstrap import paired_bootstrap
from src.data import FundusDataset, read_manifest
from src.experiment_training import train
from src.metrics import metrics
from src.model import build_model, verified_weights
from src.registry import Registry
from src.utils import load_config, seed_everything, sha256, write_json

ROOT=Path('outputs/phase2_g01_execution')
CONFIG=Path('configs/experiments/G01.yaml')


def freeze_check():
    inventory=json.loads(Path('outputs/migration_review/frozen_inventory.json').read_text(encoding='utf-8'))
    assert all(sha256(p)==h for p,h in inventory.items()), 'Frozen B02/Round01 changed'
    source=json.loads(Path('outputs/migration_review/source_inventory.json').read_text(encoding='utf-8'))
    assert all(sha256(Path(source['root'])/p)==h for p,h in source['files'].items()), 'Old sources changed'


def gate():
    summary=json.loads(Path('outputs/phase2_diagnostics_v3/summary.json').read_text())
    counts=summary['categories']
    assert not any(counts.get(k,0) for k in ['identical_file','identical_decoded_image','visual_near_duplicate'])
    assert summary['high_risk_label_disagreements']==0 and summary['pairs']==430
    controls=json.loads(Path('outputs/phase2_audit_controls/report.json').read_text())
    assert controls['known_derived_detected']==4 and controls['random_pairs_flagged']==0
    freeze_check()
    return {'status':'pass_with_disclosed_linkage_risk',
            'basis':'Dual-track authorization; no confirmed exact/visual duplicate or supported label conflict in reviewed candidates.',
            'remaining_risk':'One suspected same-eye different capture (same label); patient IDs absent. Official validation is not claimed patient-independent.',
            'summary_sha256':sha256('outputs/phase2_diagnostics_v3/summary.json'),
            'pair_review_sha256':sha256('outputs/phase2_diagnostics_v3/pair_review.csv'),
            'official_split_unchanged':True}


def prepare():
    ROOT.mkdir(exist_ok=False)
    write_json(ROOT/'audit_gate.json',gate())
    c=load_config('configs/experiments/B02.yaml')
    c.update(experiment_id='G01',output='outputs/runs/G01')
    c['model']={'name':'convnext_tiny','backend':'timm','architecture':'convnext_tiny.fb_in1k',
                'weights_manifest':'configs/weights_g01.local.json'}
    c['plan']={'stage':'G','kind':'train','reference':'B02','factor':'backbone_and_nonmedical_weights',
               'initial_status':'planned','authorization':'User explicitly authorized G01 only after audit/smoke'}
    verified_weights(c['model']['weights_manifest'],'convnext_tiny')
    if CONFIG.exists(): raise FileExistsError(CONFIG)
    CONFIG.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print('G01 configuration and audit gate prepared',flush=True)


def smoke():
    gate(); c=load_config(CONFIG); out=ROOT/'smoke'; out.mkdir(exist_ok=False)
    image_root=out/'images';image_root.mkdir();rng=np.random.default_rng(42)
    for split,n in [('train',8),('val',4)]:
        rows=[]
        for i in range(n):
            name=f'{split}_{i}.png';Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)).save(image_root/name)
            rows.append({'image_id':name,'path':name,'label':i%2})
        pd.DataFrame(rows).to_csv(out/f'{split}.csv',index=False)
    synthetic=copy.deepcopy(c)
    synthetic.update(experiment_id='SMOKE_G01',output=str(out/'run'),smoke=True,epochs=1,batch_size=4,num_workers=0,cache=False)
    synthetic['model']['weights_manifest']=None
    synthetic['data']={'source':'synthetic','root':str(image_root.resolve()),
                       'train_csv':str(out/'train.csv'),'val_csv':str(out/'val.csv')}
    checkpoint=train(synthetic)
    from src.inference import Predictor
    predictor=Predictor(checkpoint,'cuda',allow_smoke=True)
    assert predictor.model(torch.zeros(1,3,224,224,device='cuda')).shape==(1,1)
    probability=predictor.predict_image(Image.open(image_root/'val_0.png'))
    assert 0 <= probability <= 1
    del predictor;gc.collect();torch.cuda.empty_cache()
    # Discard all resource-probe updates; formal training reloads approved initialization and seed.
    seed_everything(42);torch.set_num_threads(4)
    model=build_model(c['model']['weights_manifest'],model_config=c['model']).cuda().train()
    frame=read_manifest(c['data']['train_csv'],c['data']['root']).iloc[:256]
    cache=json.loads(Path('outputs/runs/B02/cache.json').read_text())['train']
    ds=FundusDataset(frame,224,True,cache_dir=cache)
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');times=[];torch.cuda.reset_peak_memory_stats()
    for i,(x,y) in enumerate(loader):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        x=x.cuda();y=y.cuda()
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        if i>=2:times.append(time.perf_counter()-start)
    report={'status':'passed','synthetic_pipeline':'passed','gpu':torch.cuda.get_device_name(),
            'batch_size':32,'input_size':224,'amp':True,'real_train_probe_steps':8,'warmup_steps':2,
            'measured_steps':len(times),'mean_step_seconds':float(np.mean(times)),
            'images_per_second':32/float(np.mean(times)),
            'speed_scope':'cached official training batch H2D+forward+backward+optimizer; excludes DataLoader wait',
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'probe_checkpoint_saved':False,'formal_initialization_reset':True}
    # Check FP32 evaluation at the unchanged validation batch size too.
    model.eval();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():assert torch.isfinite(model(x)).all()
    report['fp32_val_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    write_json(ROOT/'smoke_result.json',report);freeze_check();print(json.dumps(report),flush=True)


def compare_results(c):
    out=Path(c['output']);pred=pd.read_csv(out/'validation.csv');ref=pd.read_csv('outputs/runs/B02/validation.csv')
    assert list(pred.image_id)==list(ref.image_id) and np.array_equal(pred.label,ref.label)
    m=metrics(pred.label,pred.probability);b=metrics(ref.label,ref.probability)
    ci=paired_bootstrap(ref,pred,repeats=2000,seed=42)
    y=pred.label.to_numpy();b_error=(ref.probability.to_numpy()>=.5)!=y;g_error=(pred.probability.to_numpy()>=.5)!=y
    table=pd.DataFrame({'image_id':pred.image_id,'label':y,'b02_probability':ref.probability,
                        'g01_probability':pred.probability,'b02_error':b_error,'g01_error':g_error})
    table.to_csv(out/'error_complementarity.csv',index=False)
    complement={}
    for name,mask in [('all',np.ones(len(y),dtype=bool)),('positive',y==1),('negative',y==0)]:
        complement[name]={'n':int(mask.sum()),'both_wrong':int((mask&b_error&g_error).sum()),
                         'b02_only_wrong':int((mask&b_error&~g_error).sum()),
                         'g01_only_wrong':int((mask&~b_error&g_error).sum()),
                         'both_correct':int((mask&~b_error&~g_error).sum())}
    h=json.loads((out/'history.json').read_text());best=max(h,key=lambda r:(r['auroc'],r['auprc']))
    latency=json.loads((out/'latency/local_fp32.json').read_text())
    artifact=json.loads((out/'artifact.json').read_text())
    report={'experiment_id':'G01','best_epoch':best['epoch'],'metrics':m,'B02':b,'paired_bootstrap':ci,
            'train_duration':artifact['train_duration'],'model_size_mb':artifact['bytes']/1e6,
            'local_latency_ms':latency['mean_ms'],'gpu':latency['gpu'],'t4_latency_ms':None,
            'train_peak_allocated_mib':max(r['train_peak_allocated_mib'] for r in h),
            'train_peak_reserved_mib':max(r['train_peak_reserved_mib'] for r in h),
            'val_peak_allocated_mib':max(r['val_peak_allocated_mib'] for r in h),
            'complementarity_at_threshold_0_5':complement,'single_seed':True,
            'validation_independence_not_established':True,'test_used':False,'submission_generated':False}
    write_json(out/'comparison.json',report);print(json.dumps(report),flush=True)


def run():
    gate();sm=json.loads((ROOT/'smoke_result.json').read_text());assert sm['status']=='passed'
    c=load_config(CONFIG);reg=Registry();reg.register(c)
    if not reg.claim('G01'):
        print('G01 already completed; training not repeated',flush=True);return
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for directory in ['src','scripts']:
        (snapshot/directory).mkdir()
        for p in Path(directory).glob('*.py'):shutil.copy2(p,snapshot/directory/p.name)
    shutil.copy2(CONFIG,snapshot/'G01.yaml')
    try:
        def epoch_done(result,checkpoint,elapsed):
            from scripts.experiments import best_record
            reg.update('G01',**best_record(checkpoint),train_duration=elapsed,reason=f'Finished epoch {result["epoch"]}')
            reg.export();training_plot(c['output'])
        checkpoint=train(c,on_epoch=epoch_done)
        from scripts.experiments import best_record
        artifact=json.loads((Path(c['output'])/'artifact.json').read_text())
        reg.update('G01',**best_record(checkpoint),train_duration=artifact['train_duration'],
                   status='completed',checkpoint_sha256=sha256(checkpoint),reason=None)
        latency=local_benchmark(checkpoint,c,Path(c['output'])/'latency')
        reg.update('G01',local_latency_ms=latency['mean_ms'],local_latency_device=latency['gpu'])
        reg.export();compare_results(c);freeze_check()
        write_json(ROOT/'completion.json',{'status':'completed','freeze_verified':True,'other_experiments_started':False})
    except Exception as exc:
        status='completed' if reg.get('G01')['status']=='completed' else 'failed'
        reg.update('G01',status=status,reason=str(exc));reg.export();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','run']);a=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'run':run}[a.action]()
