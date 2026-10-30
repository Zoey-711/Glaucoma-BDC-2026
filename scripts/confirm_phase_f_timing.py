"""Interleaved local timing diagnostic for inconsistent standalone measurements."""
import csv
import json
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from scripts.phase_f import ROOT,DATA,checkpoint,plan,verify_frozen
from src.data import read_manifest
from src.inference import Predictor
from src.utils import write_json


def main():
    verify_frozen();torch.set_num_threads(4)
    out=ROOT/'deployment_run/interleaved_fp32';out.mkdir(exist_ok=False)
    frame=read_manifest(ROOT/'train_timing_subset.csv',DATA)
    models={name:Predictor(checkpoint(plan()['models'][name]),'cuda','fp32') for name in ['B02','G01']}
    models['F01']=Predictor(ROOT/'F01_bundle.json','cuda','fp32')
    names=list(models);records=[]
    with ExitStack() as stack:
        handles={name:stack.enter_context((out/f'{name}_diagnostic.csv').open('w',newline='',encoding='utf-8')) for name in names}
        writers={name:csv.writer(h) for name,h in handles.items()}
        for name in names:
            writers[name].writerow(['Image_name','Probability','Pred_Label'])
            for r in frame.iloc[:5].itertuples():models[name].predict_one(r.resolved_path,r.image_id,writers[name],handles[name])
        for repeat in range(3):
            for i,r in enumerate(frame.iloc[5:].itertuples()):
                order=names[(i+repeat)%3:]+names[:(i+repeat)%3]
                for name in order:
                    model=models[name];model.synchronize();start=time.perf_counter_ns()
                    model.predict_one(r.resolved_path,r.image_id,writers[name],handles[name]);model.synchronize()
                    records.append({'model':name,'round':repeat+1,'image_id':r.image_id,'ms':(time.perf_counter_ns()-start)/1e6})
            print('Interleaved local round',repeat+1,flush=True)
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'timings.csv',index=False)
    write_json(out/'report.json',{'purpose':'diagnose order/system-state sensitivity; not official consecutive-candidate timing',
        'source':'official_train','same_100_plus_5':True,'batch_size':1,'rounds':3,'precision':'fp32',
        'gpu':models['G01'].metadata['gpu'],'T4_measured':False,
        'mean_ms':{name:float(np.mean([r['ms'] for r in records if r['model']==name])) for name in names},
        'frozen_files_verified':verify_frozen()})


if __name__=='__main__':main()
