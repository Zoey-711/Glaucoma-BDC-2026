"""Local-only AUROC and complementarity gate for accepted H predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap import paired_bootstrap
from src.metrics import metrics
from src.phase_f import complementarity
from src.research.results import validate_result
from src.utils import write_json


def normalized(path,official):
    frame=pd.read_csv(path).rename(columns={'Image_name':'image_id','True_Label':'label',
        'Label':'legacy_label','Probability':'probability'})
    label='label' if 'label' in frame else 'legacy_label'
    if (len(frame)!=len(official) or frame.image_id.duplicated().any()
        or frame.image_id.tolist()!=official.image_id.tolist()
        or not np.array_equal(frame[label],official.label)):
        raise ValueError('Reference/candidate Val IDs or labels disagree')
    return pd.DataFrame({'image_id':frame.image_id,'label':frame[label],
        'probability':frame.probability,'logit':frame.logit})


def analyze(candidate,output,reference_g01,reference_f01):
    root=Path(candidate);out=Path(output)
    if out.exists():raise FileExistsError(out)
    official=pd.read_csv('data/official/manifests/valid.csv')
    validated=validate_result(root,'data/official/manifests/valid.csv',
        f'configs/h/{root.parents[1].name}.yaml',official_train='data/official/manifests/train.csv')
    current=normalized(root/'validation.csv',official)
    report={'candidate':validated,'references':{}}
    for name,path in [('G01',reference_g01),('F01',reference_f01)]:
        ref=normalized(path,official)
        comparison=paired_bootstrap(ref,current,repeats=2000,seed=42)
        complement=complementarity(ref,current)
        report['references'][name]={'paired_auroc':comparison,'complementarity':complement,
            'candidate_metrics':metrics(current.label,current.probability)}
    out.mkdir(parents=True)
    write_json(out/'comparison.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--output',required=True)
    p.add_argument('--g01',default='outputs/runs/G01/validation.csv')
    p.add_argument('--f01',default='outputs/phase_g2/freeze/F01_validation.csv')
    a=p.parse_args();print(json.dumps(analyze(a.candidate,a.output,a.g01,a.f01),indent=2))
