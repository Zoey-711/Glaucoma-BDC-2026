import argparse
import json
from pathlib import Path

from src.research.results import safe_import,validate_result


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--experiment-id',required=True)
    p.add_argument('--destination-root',default='remote_results');p.add_argument('--official-val',default='data/official/manifests/valid.csv')
    a=p.parse_args();dest=Path(a.destination_root)/a.experiment_id
    staging=dest.with_name(dest.name+'.staging')
    safe_import(a.bundle,staging)
    result=validate_result(staging,a.official_val,f'configs/research/{a.experiment_id}.yaml',
                           freeze_inventory='outputs/migration_review/frozen_inventory.json')
    if result['experiment_id']!=a.experiment_id:raise ValueError('Experiment ID mismatch')
    if dest.exists():raise FileExistsError(dest)
    staging.rename(dest)
    print(json.dumps({'accepted':str(dest),**result},ensure_ascii=False))


if __name__=='__main__':main()
