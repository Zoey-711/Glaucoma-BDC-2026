"""Synthetic end-to-end training + result contract; no official images or weights."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.research.config import ExperimentConfig,manifest_semantic_hash
from src.research.results import package_result,safe_import,validate_result
from src.research.trainer import run
from src.utils import sha256,write_json


def main():
    base=Path('outputs/research_smoke_v3')
    if base.exists():raise FileExistsError('Smoke already complete/partial; do not overwrite')
    base.mkdir(parents=True);images=base/'images';images.mkdir()
    rng=np.random.default_rng(42)
    for split,n in [('train',8),('val',4)]:
        rows=[]
        for i in range(n):
            name=f'{split}_{i}.png'
            Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(images/name)
            rows.append({'image_id':name,'path':name,'label':i%2})
        pd.DataFrame(rows).to_csv(base/f'{split}.csv',index=False)
    cfg=ExperimentConfig.load('configs/research/G02.yaml')
    eid='SMOKE_RESEARCH_V3'
    cfg=dataclasses.replace(cfg,experiment_id=eid,image_size=64,batch_size=2,gradient_accumulation=2,
        epochs=1,patience=1,pretrained_source=None,pretrained_sha256=None,license=None,weights_manifest=None,
        output_dir=f'results/{eid}/seed_42/fold_none',config_hash=None,git_commit=None,
        extra={**cfg.extra,'num_workers':0},
        data_manifest_hash={'train':manifest_semantic_hash(base/'train.csv'),'val':manifest_semantic_hash(base/'val.csv'),'archive':'synthetic'})
    output=run(cfg,images,base/'train.csv',base/'val.csv',smoke=True)
    info=validate_result(output,base/'val.csv',freeze_inventory=None,allow_smoke=True)
    bundle=package_result(output,base/'result.tar.gz')
    imported=safe_import(bundle,base/'imported')
    info2=validate_result(imported,base/'val.csv',freeze_inventory=None,allow_smoke=True)
    assert info['AUROC']==info2['AUROC']
    report={'status':'PASSED','synthetic_only':True,'training':info,'bundle_bytes':bundle.stat().st_size,
            'checksum_import_verified':True,'official_training_started':False}
    write_json(base/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
