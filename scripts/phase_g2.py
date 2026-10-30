"""Freeze F01/F04 inference packages and run interleaved shared-decode RTX timing."""
import argparse
import csv
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import open_rgb, read_manifest
from src.inference import Predictor
from src.phase_g import compare_frozen_to_inference
from src.phase_f import validate_predictions
from src.submission import COLUMNS
from src.utils import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
ORDER = [['F01', 'F04'], ['F04', 'F01'], ['F01', 'F04']]


def prepare(root=ROOT):
    root = Path(root)
    out = root / 'outputs/phase_g2'
    if out.exists():
        raise FileExistsError('Phase G2 is locked; refusing to overwrite existing outputs')
    evidence = {v['model']: v for v in json.loads((root / 'outputs/phase_f/analysis/model_evidence.json').read_text(encoding='utf-8'))}
    official = read_manifest(root / 'data/official/manifests/valid.csv', root / 'data/official')
    if len(official) != 5849:
        raise ValueError('Expected complete 5849-row official Val manifest')
    out.mkdir(parents=True)
    frozen = out / 'freeze'; frozen.mkdir()
    phase_f = root / 'outputs/phase_f/analysis'
    for name in ('F01', 'F04'):
        p = phase_f / f'{name}_validation.csv'
        frame = validate_predictions(pd.read_csv(p), official, 5849)
        if frame.experiment_id.ne(name).any():
            raise ValueError(f'{name} validation prediction ID mismatch')
        shutil.copyfile(p, frozen / p.name)
    sample_file = root / 'outputs/phase_g/freeze/t4_validation_105.csv'
    sample = pd.read_csv(sample_file, dtype={'image_id': str})
    if len(sample) != 105 or sample.image_id.duplicated().any():
        raise ValueError('Expected locked 5 warmup + 100 measured Val images')
    sample.to_csv(frozen / 't4_validation_105.csv', index=False)
    packages = {}
    for name, members in {'F01': ('G01', 'B02'), 'F04': ('G01', 'G02')}.items():
        folder = out / 'candidates' / name
        folder.mkdir(parents=True)
        member_records = []
        for member in members:
            ev = evidence[member]
            src = root / ev['checkpoint']
            if sha256(src) != ev['checkpoint_sha256']:
                raise ValueError(f'{member} checkpoint SHA mismatch')
            dst = folder / f'{member}{src.suffix}'
            shutil.copyfile(src, dst)
            pre = dict(ev['pretraining'])
            for key in ('path', 'license_file', 'model_card_file', 'upstream_license_file'):
                if pre.get(key):
                    pre[key] = (root / pre[key].replace('\\', '/')).resolve().as_posix()
            member_records.append({'model': member, 'config_hash': ev['config_hash'],
                'checkpoint_bytes': dst.stat().st_size, 'checkpoint_sha256': sha256(dst),
                'pretraining': pre})
        bundle = folder / 'inference.json'
        bundle.write_text(json.dumps({'members': [f'{m}{Path(evidence[m]["checkpoint"]).suffix}' for m in members],
            'weights': [.5, .5], 'threshold': .5, 'tta': ['identity']}, indent=2) + '\n', encoding='utf-8')
        compliance = folder / 'compliance'; compliance.mkdir()
        materials = []
        for member in members:
            pre = evidence[member]['pretraining']
            for key in ('license_file', 'model_card_file', 'upstream_license_file'):
                if pre.get(key):
                    source = root / pre[key].replace('\\', '/')
                    target = compliance / f'{member}_{source.name}'
                    shutil.copyfile(source, target)
                    expected = pre[{'license_file':'license_sha256','model_card_file':'model_card_sha256','upstream_license_file':'upstream_license_sha256'}[key]]
                    if sha256(target) != expected:
                        raise ValueError(f'{member} compliance file mismatch')
                    materials.append({'file': str(target.relative_to(folder)), 'sha256': sha256(target)})
        loaded = [bundle] + [folder / f'{m}{Path(evidence[m]["checkpoint"]).suffix}' for m in members]
        record = {'candidate': name, 'precision': 'fp32', 'fusion': {'method':'probability_mean','weights':[.5,.5]},
            'checkpoint': bundle.name, 'members': member_records,
            'loaded_files': [{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in loaded],
            'loaded_bytes':sum(p.stat().st_size for p in loaded),
            'loaded_files_limit_bytes':500_000_000, 'compliance_materials':materials,
            'frozen_validation_sha256':sha256(frozen / f'{name}_validation.csv'),
            'source_preprocessing': {m: evidence[m]['config_hash'] for m in members},
            'status':'PREPARED_T4_UNVERIFIED'}
        write_json(folder / 'candidate_manifest.json', record)
        if record['loaded_bytes'] > record['loaded_files_limit_bytes']:
            raise ValueError(f'{name} exceeds model-size limit')
        packages[name] = record
    definitions = {'F01': {'members':['G01','B02'],'formula':'0.5*sigmoid(G01_logit)+0.5*sigmoid(B02_logit)'},
                   'F04': {'members':['G01','G02'],'formula':'0.5*sigmoid(G01_logit)+0.5*sigmoid(G02_logit)'}}
    source_files = [root/'configs/phase_g.yaml', root/'data/official/manifests/valid.csv',
        root/'outputs/phase_f/analysis/model_evidence.json', root/'outputs/phase_f/final_verification.json']
    source_files += [phase_f/f'{n}_validation.csv' for n in ('F01','F04')]
    source_files += [root/evidence[m]['checkpoint'] for m in ('B02','G01','G02')]
    write_json(frozen/'fusion_definitions.json', definitions)
    write_json(frozen/'source_hashes.json', {str(p.relative_to(root)):sha256(p) for p in source_files})
    write_json(out/'run_config.json', {'phase':'G2','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':'07fcc3f','device_required':'RTX 4060 Laptop','precision':'fp32',
        'shared_decode':True,'warmup_images':5,'measured_images':100,'rounds_per_model':3,
        'batch_size':1,'run_order':ORDER,'latency_accounting':'shared image read/decode measured once and split equally across two model paths; each model timing includes its own transform, forward, fusion/output path as implemented, and CSV write/flush',
        'cache_policy':'normal OS filesystem cache; first warmup pass excluded; no artificial cache clearing',
        'candidates':{k:{'loaded_bytes':v['loaded_bytes'],'members':v['members']} for k,v in packages.items()}})
    return out


class GpuMonitor:
    def __init__(self): self.samples=[]; self.stop=threading.Event(); self.thread=None
    def sample(self):
        try:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.gr',
                '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
            self.samples.append({'time':time.time(),'temperature_c':float(values[0]),
                'utilization_pct':float(values[1]),'memory_used_mb':float(values[2]),
                'power_w':float(values[3]),'graphics_clock_mhz':float(values[4])})
        except Exception as exc: self.samples.append({'time':time.time(),'error':str(exc)})
    def run(self):
        while not self.stop.wait(.5): self.sample()
    def start(self): self.sample(); self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def close(self): self.stop.set();self.thread.join();self.sample()


def benchmark(root=ROOT):
    root=Path(root); out=root/'outputs/phase_g2'
    if not out.exists(): raise FileNotFoundError('Run prepare() before benchmark()')
    sample=pd.read_csv(out/'freeze/t4_validation_105.csv',dtype={'image_id':str})
    measured=sample.iloc[5:].reset_index(drop=True)
    warmup=sample.iloc[:5]
    predictors={n:Predictor(out/'candidates'/n/'inference.json','cuda','fp32') for n in ('F01','F04')}
    gpu=predictors['F01'].metadata['gpu']
    if not gpu or '4060' not in gpu:
        raise ValueError(f'Expected local RTX 4060 for this Phase G2 run, found {gpu!r}')
    if predictors['F04'].precision!='fp32' or predictors['F01'].precision!='fp32':raise ValueError('FP32 required')
    for n,pred in predictors.items():
        recorded=read_json(out/'candidates'/n/'candidate_manifest.json')
        checks={v['name']:v for v in recorded['loaded_files']}
        for item in pred.metadata['inference_files']:
            match=next((x for x in checks.values() if x['sha256']==item['sha256']),None)
            if not match:raise ValueError(f'{n} loaded weights changed')
    # Warmup exactly five images per candidate, with each decoded once and shared.
    for row in warmup.itertuples():
        im=open_rgb(root/'data/official'/row.path)
        for name in ('F01','F04'):
            with (out/f'.warm_{name}.csv').open('a',newline='',encoding='utf-8') as handle:
                w=csv.writer(handle);predictors[name].predict_decoded_one(im,row.image_id,w,handle)
    for p in predictors.values():p.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor=GpuMonitor();monitor.start()
    events=[]; per_round={n:[] for n in predictors}; pred_frames={n:[] for n in predictors}
    for round_idx,order in enumerate(ORDER,1):
        handles={};writers={}
        for name in predictors:
            path=out/f'{name}_round_{round_idx}.csv'
            handle=path.open('w',newline='',encoding='utf-8');writer=csv.writer(handle);writer.writerow(COLUMNS)
            handles[name]=handle;writers[name]=writer
        try:
            for row in measured.itertuples():
                path=root/'data/official'/row.path
                t0=time.perf_counter_ns();im=open_rgb(path);decode_ms=(time.perf_counter_ns()-t0)/1e6
                model_ms_by_name={};event_start=len(events)
                for name in order:
                    predictor=predictors[name];predictor.synchronize();start=time.perf_counter_ns()
                    p=predictor.predict_decoded_one(im,row.image_id,writers[name],handles[name])
                    predictor.synchronize();model_ms=(time.perf_counter_ns()-start)/1e6
                    model_ms_by_name[name]=model_ms
                    events.append({'round':round_idx,'order':'>'.join(order),'image_id':row.image_id,
                        'model':name,'decode_ms_shared':decode_ms,'model_path_ms':model_ms,
                        'allocated_end_to_end_ms':model_ms+decode_ms/2,'probability':p})
                pair_total=decode_ms+sum(model_ms_by_name.values())
                for event in events[event_start:]:event['pair_total_ms']=pair_total
        finally:
            for h in handles.values():h.close()
        for name in predictors:
            frame=pd.read_csv(out/f'{name}_round_{round_idx}.csv').rename(columns={'Image_name':'image_id','Probability':'probability'})
            if frame.image_id.tolist()!=measured.image_id.tolist():raise ValueError(f'{name} output order mismatch')
            pred_frames[name].append(frame)
            expected=pd.read_csv(out/f'freeze/{name}_validation.csv')
            check=compare_frozen_to_inference(expected,frame,1e-3,.5)
            per_round[name].append({'round':round_idx,**check})
    monitor.close()
    event_frame=pd.DataFrame(events);event_frame.to_csv(out/'per_image_timings.csv',index=False)
    metrics=[]
    for name in predictors:
        values=event_frame.loc[event_frame.model.eq(name),'allocated_end_to_end_ms'].to_numpy()
        metrics.append({'model':name,'round_count':3,'image_count_per_round':100,
            'per_image_mean_ms':float(values.mean()),'per_image_p95_ms':float(np.percentile(values,95)),
            'overall_mean_ms':float(values.mean()),'round_mean_ms':[float(v) for v in event_frame.loc[event_frame.model.eq(name)].groupby('round').allocated_end_to_end_ms.mean()],
            'size_bytes':predictors[name].metadata['inference_files'] and sum(x['bytes'] for x in predictors[name].metadata['inference_files']),
            'size_mb_decimal':sum(x['bytes'] for x in predictors[name].metadata['inference_files'])/1e6,
            'within_500mb':sum(x['bytes'] for x in predictors[name].metadata['inference_files'])<=500_000_000,
            'within_100ms_4060':float(values.mean())<=100,
            'frozen_consistency_by_round':per_round[name]})
    pd.DataFrame(metrics).to_csv(out/'latency_comparison.csv',index=False)
    delta=metrics[1]['overall_mean_ms']-metrics[0]['overall_mean_ms']
    report={'status':'LOCAL_4060_ONLY','gpu':gpu,'gpu_metadata':predictors['F01'].metadata,
        'torch_cuda_device_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'peak_allocated_bytes_process_both_models_loaded':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes_process_both_models_loaded':torch.cuda.max_memory_reserved(),
        'gpu_samples':monitor.samples,'round_order':[{'round':i+1,'sequence':o} for i,o in enumerate(ORDER)],
        'shared_decode':True,'precision':'fp32','batch_size':1,'warmup_images':5,
        'measured_images_per_round':100,'rounds_per_model':3,'per_model':metrics,
        'F04_minus_F01_mean_latency_ms':delta,'F04_slower_than_F01_pct':100*delta/metrics[0]['overall_mean_ms'],
        'timing_scope':'shared disk read+decode once per image pair, decode divided equally; each model includes its own preprocessing, forward, fusion/postprocessing, CSV write+flush; CUDA synchronized',
        'validation_consistency':'Measured subset checked each round; full Val checked separately by --full-val',
        't4_status':'UNVERIFIED','official_acceptance':False}
    write_json(out/'latency_report.json',report)
    return report


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def full_val(root=ROOT):
    """Recompute both bundles over full Val from one decode per image; compare frozen predictions."""
    root=Path(root);out=root/'outputs/phase_g2';official=read_manifest(root/'data/official/manifests/valid.csv',root/'data/official')
    predictors={n:Predictor(out/'candidates'/n/'inference.json','cuda','fp32') for n in ('F01','F04')}
    frames={n:[] for n in predictors}
    for idx,row in enumerate(official.itertuples(),1):
        im=open_rgb(row.resolved_path)
        for name,pred in predictors.items():
            p=pred.predict_image(im);frames[name].append((row.image_id,row.label,p))
        if idx%500==0:print(f'Full Val shared-decode consistency: {idx}/{len(official)}',flush=True)
    results={}
    for name,values in frames.items():
        actual=pd.DataFrame(values,columns=['image_id','label','probability'])
        expected=pd.read_csv(out/f'freeze/{name}_validation.csv')
        joined=expected[['image_id','label','probability']].merge(actual,on='image_id',suffixes=('_frozen','_new'),validate='one_to_one',sort=False)
        if not joined.image_id.tolist()==expected.image_id.tolist():raise ValueError('Full Val order mismatch')
        d=np.abs(joined.probability_frozen-joined.probability_new)
        actual.to_csv(out/f'{name}_fullval_recomputed.csv',index=False)
        results[name]={'rows':len(actual),'max_probability_error':float(d.max()),'mean_probability_error':float(d.mean()),
            'over_1e3':int((d>1e-3).sum()),'pred_label_mismatches':int(((joined.probability_frozen>=.5)!=(joined.probability_new>=.5)).sum()),
            'frozen_prediction_sha256':sha256(out/f'freeze/{name}_validation.csv'),
            'recomputed_prediction_sha256':sha256(out/f'{name}_fullval_recomputed.csv'),
            'within_probability_tolerance_1e3':bool(d.max()<=1e-3)}
    write_json(out/'fullval_consistency.json',{'precision':'fp32','decode_shared_between_models':True,
        'validation_count':len(official),'results':results,'interpretation':'Batch-1 implementation comparison to frozen validation predictions; differences can arise from historical batch inference and GPU kernels.'})
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--full-val',action='store_true');a=p.parse_args()
    if a.prepare:print(json.dumps({'prepared':str(prepare())},indent=2))
    if a.benchmark:print(json.dumps(benchmark(),indent=2))
    if a.full_val:print(json.dumps(full_val(),indent=2))
    if not (a.prepare or a.benchmark or a.full_val):p.error('Choose --prepare, --benchmark, and/or --full-val')

if __name__=='__main__':main()
