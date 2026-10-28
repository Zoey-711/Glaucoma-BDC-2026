import argparse
import dataclasses

from src.research.config import ExperimentConfig,git_commit
from src.research.ssl import train_simsiam


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    p.add_argument('--root',default='data/official');p.add_argument('--output',required=True)
    p.add_argument('--initialization',choices=['imagenet_then_ssl_then_finetune','random_then_ssl_then_finetune'],required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=ExperimentConfig.load(a.config)
    if not a.smoke:
        if cfg.status=='BLOCKED':raise ValueError(cfg.extra.get('block_reason','SSL experiment blocked'))
        cfg=dataclasses.replace(cfg,git_commit=git_commit());cfg.assert_code()
    print(train_simsiam(cfg,a.train_manifest,a.root,a.output,a.initialization,a.epochs,a.smoke))


if __name__=='__main__':main()
