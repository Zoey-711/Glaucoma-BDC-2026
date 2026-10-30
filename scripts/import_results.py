import argparse
import json
from pathlib import Path

from src.research.results import safe_import,validate_result
from src.research.config import ExperimentConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--experiment-id',required=True)
    p.add_argument('--destination-root',default='remote_results');p.add_argument('--official-val',default='data/official/manifests/valid.csv')
    p.add_argument('--official-train',default='data/official/manifests/train.csv')
    a=p.parse_args()
    config=(Path('configs/h')/f'{a.experiment_id}.yaml' if a.experiment_id.startswith('H')
            else Path('configs/research')/f'{a.experiment_id}.yaml')
    cfg=ExperimentConfig.load(config)
    dest=(Path(a.destination_root)/a.experiment_id/f'seed_{cfg.seed}'/
          f'fold_{cfg.fold if cfg.fold is not None else "none"}' if a.experiment_id.startswith('H')
          else Path(a.destination_root)/a.experiment_id)
    staging=dest.with_name(dest.name+'.staging')
    staging.parent.mkdir(parents=True,exist_ok=True)
    safe_import(a.bundle,staging)
    result=validate_result(staging,a.official_val,config,
                           freeze_inventory='outputs/migration_review/frozen_inventory.json',
                           official_train=a.official_train)
    if result['experiment_id']!=a.experiment_id:raise ValueError('Experiment ID mismatch')
    if dest.exists():raise FileExistsError(dest)
    staging.rename(dest)
    print(json.dumps({'accepted':str(dest),**result},ensure_ascii=False))


if __name__=='__main__':main()
