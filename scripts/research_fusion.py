import argparse
import json
from pathlib import Path

import pandas as pd

from src.research.fusion import fixed_fusion
from src.research.results import validate_result
from src.utils import write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probability','logit'],required=True)
    p.add_argument('--members',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--val',default='data/official/manifests/valid.csv');a=p.parse_args()
    if len(a.members)<2:raise ValueError('At least two independently completed models required')
    info=[validate_result(x,a.val) for x in a.members]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    pred,score=fixed_fusion([str(Path(x)/'validation.csv') for x in a.members],a.mode)
    pred.to_csv(output/'validation.csv',index=False)
    write_json(output/'metrics.json',score)
    write_json(output/'fusion.json',{'mode':a.mode,'weights':[1/len(a.members)]*len(a.members),
        'members':[{'path':x,'checkpoint_sha256':sha256(Path(x)/'best.pth')} for x in a.members],
        'validation_only':True,'test_used':False,'selection':'fixed_equal_no_search'})
    print(json.dumps(score))


if __name__=='__main__':main()
