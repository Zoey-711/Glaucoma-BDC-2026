"""V01 prepares three immutable seed configs; execution stays with GPU queue."""
import argparse
import dataclasses
from pathlib import Path

import yaml

from src.research.config import ExperimentConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True)
    p.add_argument('--output-dir',default='configs/research/replicates');a=p.parse_args()
    candidate=ExperimentConfig.load(a.candidate)
    if candidate.status=='BLOCKED':raise ValueError('Cannot replicate blocked candidate')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for seed in [42,2026,3407]:
        eid=f'V01_{candidate.experiment_id}_S{seed}'
        cfg=dataclasses.replace(candidate,experiment_id=eid,parent_experiment_id=candidate.experiment_id,
            seed=seed,output_dir=f'results/{eid}/seed_{seed}/fold_none',config_hash=None,git_commit=None)
        data=cfg.as_dict();data['config_hash']=cfg.identity();file=out/f'{eid}.yaml'
        if file.exists():raise FileExistsError(file)
        file.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
        print(file)


if __name__=='__main__':main()
