"""D02 candidate checks. Official T4 timing remains cost.py's responsibility."""
from pathlib import Path
import csv
import json
import time

import numpy as np
import pandas as pd
import torch

from src.data import open_rgb,read_manifest
from src.inference import Predictor
from src.utils import write_json
from src.utils import sha256
from src.submission import COLUMNS


def compare_precisions(checkpoint,val_manifest,root,output,count=100):
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    frame=read_manifest(val_manifest,root).iloc[:count]
    values={}
    inference_files=None
    for precision in ['fp32','fp16']:
        model=Predictor(checkpoint,'cuda',precision)
        inference_files=model.metadata['inference_files']
        probs=np.array([model.predict_image(open_rgb(r.resolved_path)) for r in frame.itertuples()])
        values[precision]=probs
    diff=np.abs(values['fp32']-values['fp16'])
    rank_changed=bool(np.any(np.sign(values['fp32'][:,None]-values['fp32'][None,:])!=
                             np.sign(values['fp16'][:,None]-values['fp16'][None,:])))
    report={'samples':len(frame),'max_absolute_probability_difference':float(diff.max()),
            'label_mismatch':int(np.sum((values['fp32']>=.5)!=(values['fp16']>=.5))),
            'pairwise_ranking_changed':rank_changed,'precision_gate':bool(diff.max()<=1e-3 and not rank_changed),
            'device':torch.cuda.get_device_name(),'t4_certified':False,
            'inference_bytes':sum(f['bytes'] for f in inference_files),
            'within_500mb':sum(f['bytes'] for f in inference_files)<=500_000_000}
    write_json(output,report)
    return report


def local_latency(checkpoint,val_manifest,root,output,count=100,precision='fp32'):
    frame=read_manifest(val_manifest,root)
    if len(frame)<count+5:raise ValueError('Need 5 distinct warmup and 100 measured validation images')
    chosen=np.random.default_rng(42).choice(len(frame),count+5,replace=False)
    warm=frame.iloc[chosen[:5]];measured=frame.iloc[chosen[5:]]
    predictor=Predictor(checkpoint,'cuda',precision)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    values=[]
    with output.with_suffix('.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(COLUMNS)
        for row in warm.itertuples():predictor.predict_one(row.resolved_path,row.image_id,writer,f)
        for repeat in range(3):
            round_ms=[]
            for row in measured.itertuples():
                predictor.synchronize();start=time.perf_counter_ns()
                predictor.predict_one(row.resolved_path,row.image_id,writer,f)
                predictor.synchronize();round_ms.append((time.perf_counter_ns()-start)/1e6)
            values.append(round_ms)
    report={**predictor.metadata,'sample_ids':measured.image_id.tolist(),
            'latency_ms_by_round':values,'mean_ms':float(np.mean(values)),
            'p95_ms':float(np.percentile(values,95)),'official_t4':False,
            'scope':'raw decode to CSV write+flush, batch1, 5 warmup, 100 samples, 3 rounds'}
    write_json(output,report)
    return report


def certify_t4_report(checkpoint, report_path):
    """Independently check a cost.py report; never infer T4 timing from local GPUs."""
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    rounds=report.get('latency_ms_by_round',[])
    ids=report.get('sample_ids',[])
    if not report.get('official_protocol') or 'T4' not in str(report.get('gpu','')):
        raise ValueError('Official NVIDIA T4 cost.py report required')
    if len(ids)!=100 or len(set(ids))!=100 or len(rounds)!=3 or any(len(x)!=100 for x in rounds):
        raise ValueError('T4 timing must cover 100 distinct images in 3 rounds')
    if report.get('checkpoint_sha256')!=sha256(checkpoint):
        raise ValueError('T4 report checkpoint SHA256 mismatch')
    measured=float(np.asarray(rounds,dtype=float).mean())
    if not np.isfinite(measured) or abs(measured-float(report.get('mean_ms',float('nan'))))>1e-6:
        raise ValueError('T4 latency cannot be independently reproduced')
    files=report.get('inference_files') or []
    if not files or not any(f.get('sha256')==sha256(checkpoint) for f in files):
        raise ValueError('T4 report loaded-file inventory is missing checkpoint')
    if len({f.get('path') for f in files})!=len(files) or any(not isinstance(f.get('bytes'),int) or f['bytes']<1 for f in files):
        raise ValueError('Invalid T4 loaded-file inventory')
    model_bytes=sum(f['bytes'] for f in files)
    passed=(model_bytes<=500_000_000 and measured<=100 and
            report.get('passed') is True and report.get('mismatches_by_round')==[0,0,0])
    return {'t4_certified':passed,'mean_ms':measured,'loaded_model_bytes':model_bytes,
            'within_500mb':model_bytes<=500_000_000,'within_100ms':measured<=100}
