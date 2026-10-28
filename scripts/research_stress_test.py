"""R02 fixed validation robustness diagnostic; does not select hyperparameters."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import open_rgb,read_manifest
from src.inference import Predictor
from src.metrics import metrics
from src.research.preprocess import jpeg_compress
from src.utils import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--val-manifest',default='data/official/manifests/valid.csv');p.add_argument('--root',default='data/official')
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
    frame=read_manifest(a.val_manifest,a.root);predictor=Predictor(a.checkpoint,'cuda')
    for quality in [100,75,50]:
        rows=[]
        for r in frame.itertuples():
            im=open_rgb(r.resolved_path);prob=predictor.predict_image(jpeg_compress(im,quality))
            rows.append({'Image_name':r.image_id,'Label':r.label,'Probability':prob})
        table=pd.DataFrame(rows);table.to_csv(out/f'jpeg_q{quality}.csv',index=False)
        write_json(out/f'jpeg_q{quality}.json',metrics(table.Label,table.Probability))
    write_json(out/'protocol.json',{'qualities':[100,75,50],'fixed_before_evaluation':True,
                                    'validation_diagnostic_only':True,'test_used':False})


if __name__=='__main__':main()
