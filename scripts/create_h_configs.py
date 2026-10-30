"""Create isolated H03/H04 protocol configs without touching historical YAML."""
from dataclasses import replace
from pathlib import Path

import yaml

from src.research.config import ExperimentConfig


def create(directory='configs/h'):
    target=Path(directory)
    target.mkdir(parents=True,exist_ok=True)
    g01=ExperimentConfig.load('configs/research/G01_REPRO.yaml')
    g05=ExperimentConfig.load('configs/research/G05.yaml')
    specifications=[
        ('H03_G01_HEAD',g01,{'head_variant':'standard'},'G01_REPRO','shuffle'),
        ('H03_MATCHED_MLP',g01,{'head_variant':'matched_mlp','middle_stage':2,'head_width':128},'H03_G01_HEAD','shuffle'),
        ('H03_MULTISCALE',g01,{'head_variant':'multiscale','middle_stage':2,'head_width':128},'H03_G01_HEAD','shuffle'),
        ('H04_RANDOM',g05,{'sampling_comparison':'random'},'G05','shuffle'),
        ('H04_BALANCED',g05,{'sampling_comparison':'balanced'},'H04_RANDOM','balanced')]
    written=[]
    for eid,base,extra,parent,sampler in specifications:
        cfg=replace(base,experiment_id=eid,parent_experiment_id=parent,
            output_dir=f'results/{eid}/seed_42/fold_none',sampler=sampler,
            extra={**base.extra,**extra},config_hash=None,git_commit=None,
            status='BLOCKED' if eid.startswith('H04') else 'PENDING',dependencies=[])
        cfg=replace(cfg,config_hash=cfg.identity())
        path=target/f'{eid}.yaml'
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg.as_dict(),sort_keys=False,allow_unicode=True),encoding='utf-8')
        written.append(str(path))
    return written


if __name__=='__main__':print('\n'.join(create()))
