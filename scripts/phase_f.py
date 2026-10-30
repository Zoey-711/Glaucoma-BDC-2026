"""Phase F: frozen offline analysis, inference preflight and Train-only diagnostics."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image,ImageEnhance,ImageFilter
from scipy.stats import spearmanr

from scripts.s01_v2_qc import guard as history_guard
from src.bootstrap import paired_bootstrap
from src.data import read_manifest,open_rgb
from src.inference import Predictor
from src.metrics import metrics
from src.model import verified_weights
from src.phase_f import validate_predictions,fuse,complementarity,precision_comparison
from src.research.config import ExperimentConfig
from src.utils import sha256,write_json

ROOT=Path('outputs/phase_f')
PLAN=Path('configs/phase_f.yaml')
TRAIN=Path('data/official/manifests/train.csv')
VAL=Path('data/official/manifests/valid.csv')
DATA=Path('data/official')


def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def plan():return yaml.safe_load(PLAN.read_text(encoding='utf-8'))


def checkpoint(directory):
    p=Path(directory)
    return p/('best.pt' if (p/'best.pt').exists() else 'best.pth')


def verify_frozen():
    inventory=json.loads((ROOT/'frozen_inventory.json').read_text(encoding='utf-8'))
    bad=[p for p,h in inventory.items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise ValueError(f'Frozen assets changed: {bad}')
    lock=json.loads((ROOT/'protocol_locked.json').read_text(encoding='utf-8'))
    assert lock['plan_sha256']==sha256(PLAN)
    assert lock['train_sha256']==sha256(TRAIN) and lock['val_sha256']==sha256(VAL)
    return len(inventory)


def prepare():
    history_guard();ROOT.mkdir(exist_ok=False)
    inventory=json.loads(Path('outputs/S01_v2/frozen_inventory.json').read_text(encoding='utf-8'))
    for directory in [*plan()['models'].values(),'results/S01v2']:
        for p in Path(directory).rglob('*'):
            if p.is_file():inventory[str(p)]=sha256(p)
    write_json(ROOT/'frozen_inventory.json',inventory)
    write_json(ROOT/'protocol_locked.json',{'plan':plan(),'plan_sha256':sha256(PLAN),
        'train_sha256':sha256(TRAIN),'val_sha256':sha256(VAL),
        'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'test_used':False,'weights_searched':False,'train_subset_seen_during_training':True})
    train=read_manifest(TRAIN,DATA)
    s=plan()['stress'];reserved=pd.concat([train[train.label==1].sample(s['positives'],random_state=s['seed']),
        train[train.label==0].sample(s['negatives'],random_state=s['seed'])]).sort_values('image_id')
    reserved[['image_id','path','label']].to_csv(ROOT/'train_diagnostic_subset.csv',index=False)
    d=plan()['deployment']
    train.sample(d['measured']+d['warmup'],random_state=d['seed'])[['image_id','path','label']].to_csv(ROOT/'train_timing_subset.csv',index=False)
    print('Protocol and Train subsets locked; no Test accessed',flush=True)


def load_models():
    verify_frozen();official=pd.read_csv(VAL);predictions={};evidence=[]
    for name,folder in plan()['models'].items():
        p=Path(folder);ck=checkpoint(p)
        if (p/'checksums.sha256').exists():
            for line in (p/'checksums.sha256').read_text().splitlines():
                digest,filename=line.split('  ',1)
                assert sha256(p/filename)==digest,filename
        if (p/'artifact.json').exists():
            assert json.loads((p/'artifact.json').read_text())['checkpoint_sha256']==sha256(ck)
        obj=torch.load(ck,map_location='cpu',weights_only=True)
        assert not obj['smoke']
        cfgpath=p/('config.json' if (p/'config.json').exists() else 'config.yaml')
        cfg=(json.loads(cfgpath.read_text(encoding='utf-8')) if cfgpath.suffix=='.json'
             else yaml.safe_load(cfgpath.read_text(encoding='utf-8')))
        assert cfg==obj['config'], f'{name}: checkpoint config differs from saved config'
        if name=='G02':
            c=ExperimentConfig(**cfg);c.validate()
            meta=json.loads((p/'run_manifest.json').read_text())
            assert c.identity()==meta['config_hash']
            config_hash=c.identity();basis='historical research config hash + checkpoint equality'
        else:
            config_hash=canonical(cfg);basis='canonical hash computed at Phase F; saved/embedded config equality + frozen file hashes'
            if (p/'config.yaml').exists():
                assert yaml.safe_load((p/'config.yaml').read_text(encoding='utf-8'))==cfg
                meta=json.loads((p/'run_manifest.json').read_text())
                if 'config_sha256' in meta:assert meta['config_sha256']==sha256(p/'config.yaml')
        provenance=verified_weights(obj['provenance'])
        frame=validate_predictions(pd.read_csv(p/'validation.csv'),official,5849)
        scores=metrics(frame.label,frame.probability)
        for key in ['auroc','auprc']:
            assert abs(scores[key]-obj['metrics'][key])<1e-8,(name,key)
        predictions[name]=frame
        evidence.append({'model':name,'status':'PASS','rows':len(frame),'config_hash':config_hash,
            'config_hash_basis':basis,'saved_config_sha256':sha256(cfgpath),
            'checkpoint':str(ck),'checkpoint_sha256':sha256(ck),'prediction_sha256':sha256(p/'validation.csv'),
            'model_bytes':ck.stat().st_size,'metrics':scores,'pretraining':provenance})
        del obj
        print('Verified',name,flush=True)
    return predictions,evidence


def analyze():
    folder=ROOT/'analysis';folder.mkdir(exist_ok=False)
    predictions,evidence=load_models();write_json(folder/'model_evidence.json',evidence)
    rows=[];pairs=[];reference=predictions['G01']
    for name,p in predictions.items():
        rows.append({'experiment_id':name,'method':'single','members':name,**metrics(p.label,p.probability),
                     'delta_auroc':metrics(p.label,p.probability)['auroc']-metrics(reference.label,reference.probability)['auroc']})
        if name!='G01':pairs.append({'reference':'G01','candidate':name,**complementarity(reference,p)})
    for eid,spec in plan()['fusions'].items():
        assert spec['weights']==[.5,.5]
        a,b=[predictions[n] for n in spec['members']];p=fuse(a,b,spec['method'])
        p['experiment_id']=eid;p.to_csv(folder/f'{eid}_validation.csv',index=False)
        ci=paired_bootstrap(reference,p,repeats=plan()['bootstrap']['replicates'],seed=plan()['bootstrap']['seed'])
        rows.append({'experiment_id':eid,'method':spec['method'],'members':'+'.join(spec['members']),
                     **metrics(p.label,p.probability),**ci})
        pairs.append({'reference':'G01','candidate':eid,**complementarity(reference,p)})
        print(eid,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'fusion_comparison.csv',index=False)
    pd.DataFrame(pairs).to_csv(folder/'complementarity.csv',index=False)
    pd.DataFrame({k:v.probability for k,v in predictions.items()}).corr().to_csv(folder/'probability_correlation.csv')
    # Deploy the predefined F01 probability rule only; other formulas remain analysis outputs.
    write_json(ROOT/'F01_bundle.json',{'members':[str(checkpoint(plan()['models'][n]).resolve()) for n in ['G01','B02']],
        'weights':[.5,.5],'threshold':.5,'tta':['identity']})
    write_json(folder/'verification.json',{'status':'PASS','frozen_files_verified':verify_frozen(),
        'test_used':False,'weight_search':False,'multiple_comparisons':'5 exploratory comparisons; unadjusted paired CIs; no seed-stability claim'})


def timed(predictor,frame,folder):
    folder.mkdir();rows=[];probs=[]
    with (folder/'warmup.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        for r in frame.iloc[:5].itertuples():predictor.predict_one(r.resolved_path,r.image_id,w,f)
    for repeat in range(3):
        times=[];values=[]
        with (folder/f'diagnostic_round_{repeat+1}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for r in frame.iloc[5:].itertuples():
                predictor.synchronize();start=time.perf_counter_ns()
                values.append(predictor.predict_one(r.resolved_path,r.image_id,w,f));predictor.synchronize()
                times.append((time.perf_counter_ns()-start)/1e6)
        rows.append(times);probs.append(values)
    report={**predictor.metadata,'mean_ms':float(np.mean(rows)),'p95_ms':float(np.percentile(rows,95)),
        'latency_ms_by_round':rows,'sample_ids':frame.iloc[5:].image_id.tolist(),
        'warmup_ids':frame.iloc[:5].image_id.tolist(),'source':'official_train',
        'batch_size':1,'rounds':3,'T4_measured':False,'official_certified':False,
        'repeat_max_probability_delta':float(np.max(np.abs(np.asarray(probs)-probs[0]))),
        'scope':'raw file decode, preprocess, transfer, forward, probability/label, CSV write+flush, CUDA sync; no OS cache eviction'}
    write_json(folder/'timing.json',report)
    return report


def deploy():
    verify_frozen();folder=ROOT/'deployment_run';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    frame=read_manifest(ROOT/'train_timing_subset.csv',DATA)
    val=read_manifest(VAL,DATA).sample(32,random_state=20261030)
    evidence=json.loads((ROOT/'analysis/model_evidence.json').read_text())
    candidates={k:checkpoint(v) for k,v in plan()['models'].items()};candidates['F01']=ROOT/'F01_bundle.json'
    rows=[];precision_rows=[];loaded={}
    for name,ck in candidates.items():
        probabilities={};val_checks={}
        for precision in ['fp32','fp16']:
            predictor=Predictor(ck,'cuda',precision)
            loaded[name]=predictor.metadata['inference_files']
            probabilities[precision]=[predictor.predict_image(open_rgb(r.resolved_path)) for r in frame.itertuples()]
            if name in plan()['models'] and precision=='fp32':
                old=pd.read_csv(Path(plan()['models'][name])/'validation.csv').rename(columns={'Image_name':'image_id','Probability':'probability'})
                old=old.set_index('image_id').loc[val.image_id]
                actual=np.array([predictor.predict_image(open_rgb(r.resolved_path)) for r in val.itertuples()])
                diff=np.abs(actual-old.probability.to_numpy());val_checks={'val_probe_count':32,'val_fp32_max_error':float(diff.max()),
                    'val_fp32_saved_predictions_match_1e4':bool(diff.max()<=1e-4),
                    'val_fp32_saved_predictions_match_1e3':bool(diff.max()<=plan()['deployment']['probability_tolerance']),
                    'val_fp32_threshold_flips':int(((actual>=.5)!=(old.probability.to_numpy()>=.5)).sum())}
                if diff.max()>plan()['deployment']['probability_tolerance']:
                    raise ValueError(f'{name}: current inference exceeds locked probability tolerance against frozen Val predictions')
            size=sum(f['bytes'] for f in loaded[name])
            row={'model':name,'precision':precision,'loaded_bytes':size,'model_size_mb':size/1e6,
                'size_status':'PASS' if size<=500_000_000 else 'FAIL','t4_status':'UNVERIFIED','t4_latency_ms':None}
            if name in plan()['deployment']['timed']:
                report=timed(predictor,frame,folder/f'{name}_{precision}')
                row.update(local_latency_ms=report['mean_ms'],local_gpu=report['gpu'])
            rows.append(row);del predictor;torch.cuda.empty_cache()
        comparison=precision_comparison(probabilities['fp32'],probabilities['fp16'])
        precision_rows.append({'model':name,**comparison,**val_checks,'sample_count':len(frame)})
        pd.DataFrame({'image_id':frame.image_id,'fp32':probabilities['fp32'],'fp16':probabilities['fp16']}).to_csv(folder/f'{name}_precision.csv',index=False)
        print('Deployment checked',name,comparison,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'model_latency_report.csv',index=False)
    pd.DataFrame(precision_rows).to_csv(folder/'precision_consistency.csv',index=False)
    write_json(folder/'loaded_files.json',loaded)
    write_json(folder/'pretraining_audit.json',{e['model']:e['pretraining'] for e in evidence})
    write_json(folder/'interface_audit.json',{'predict_shared_import':'from src.inference import Predictor' in Path('predict.py').read_text(),
        'cost_shared_import':'from src.inference import Predictor' in Path('cost.py').read_text(),
        'predict_uses_predict_one':'predictor.predict_one(' in Path('predict.py').read_text(),
        'cost_uses_predict_one':'predictor.predict_one(' in Path('cost.py').read_text(),
        'predict_sha256':sha256('predict.py'),'cost_sha256':sha256('cost.py'),
        'inference_sha256':sha256('src/inference.py'),'frozen_files_verified':verify_frozen(),
        'FP16_scope':'105 fixed Train images; 32 Val FP32 archive checks; not exhaustive FP16 certification'})


def perturbations(image):
    yield 'clean',image
    for x in plan()['stress']['brightness']:yield f'brightness_{x}',ImageEnhance.Brightness(image).enhance(x)
    for x in plan()['stress']['contrast']:yield f'contrast_{x}',ImageEnhance.Contrast(image).enhance(x)
    for x in plan()['stress']['blur_sigma']:yield f'blur_{x}',image.filter(ImageFilter.GaussianBlur(x))
    for x in plan()['stress']['jpeg_quality']:
        stream=io.BytesIO();image.save(stream,format='JPEG',quality=x);stream.seek(0)
        with Image.open(stream) as im:yield f'jpeg_{x}',im.convert('RGB')


def stress():
    verify_frozen();folder=ROOT/'stress';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);frame=read_manifest(ROOT/'train_diagnostic_subset.csv',DATA)
    assert set(frame.image_id).issubset(set(pd.read_csv(TRAIN).image_id))
    assert not set(frame.image_id)&set(pd.read_csv(VAL).image_id)
    summaries=[]
    for name,directory in plan()['models'].items():
        model=Predictor(checkpoint(directory),'cuda','fp32');records=[]
        for i,r in enumerate(frame.itertuples()):
            im=open_rgb(r.resolved_path).resize((plan()['stress']['canvas_size'],)*2,Image.Resampling.BILINEAR)
            for condition,changed in perturbations(im):
                p=model.predict_image(changed)
                records.append({'image_id':r.image_id,'label':r.label,'condition':condition,'probability':p,'logit':model.last_logits[0]})
            if (i+1)%64==0:print('Stress',name,i+1,len(frame),flush=True)
        table=pd.DataFrame(records);table.to_csv(folder/f'{name}_diagnostic_predictions.csv',index=False)
        clean=table[table.condition=='clean'].set_index('image_id')
        for condition,group in table.groupby('condition',sort=False):
            group=group.set_index('image_id').loc[clean.index]
            delta=np.abs(group.probability-clean.probability);flips=(group.probability>=.5)!=(clean.probability>=.5)
            summaries.append({'model':name,'condition':condition,'samples':len(group),
                'mean_abs_probability_change':float(delta.mean()),'p95_abs_probability_change':float(delta.quantile(.95)),
                'max_abs_probability_change':float(delta.max()),'threshold_flip_fraction':float(flips.mean()),
                'positive_flip_fraction':float(flips[group.label==1].mean()),'negative_flip_fraction':float(flips[group.label==0].mean()),
                'mean_abs_logit_change':float(np.abs(group.logit-clean.logit).mean()),
                'spearman_vs_clean':float(spearmanr(group.probability,clean.probability).statistic)})
        del model;torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(folder/'robustness_comparison.csv',index=False)
    write_json(folder/'protocol.json',{'source':'official_train','count':len(frame),
        'positive_count':int(frame.label.sum()),'seen_during_model_training':True,
        'not_independent_holdout':True,'official_Val_AUROC_reported':False,'test_used':False,
        'no_model_or_strength_selection':True,'canvas_size':384,'frozen_files_verified':verify_frozen()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze','deploy','stress']);a=p.parse_args()
    {'prepare':prepare,'analyze':analyze,'deploy':deploy,'stress':stress}[a.action]()
