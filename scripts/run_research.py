import argparse
import dataclasses
import json

from src.research.config import ExperimentConfig,git_commit
from src.research.trainer import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--data-root',default='data/official')
    p.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    p.add_argument('--val-manifest',default='data/official/manifests/valid.csv')
    p.add_argument('--archive',help='Local copy of official zip for SHA256 check')
    p.add_argument('--group-manifest');p.add_argument('--teacher')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=ExperimentConfig.load(a.config)
    if not a.smoke:
        if not a.archive:p.error('--archive required for formal dataset version verification')
        cfg.assert_data(a.train_manifest,a.val_manifest,a.archive)
        cfg=dataclasses.replace(cfg,git_commit=git_commit())
    out=run(cfg,a.data_root,a.train_manifest,a.val_manifest,smoke=a.smoke,resume=a.resume,
            group_manifest=a.group_manifest,teacher_path=a.teacher)
    print(json.dumps({'result_dir':str(out),'status':'COMPLETED'}))


if __name__=='__main__':main()
