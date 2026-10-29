"""One-time correction of YAML numeric parsing in M02 age strata; no fitting."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.metrics import metrics
from src.multimodal import AgePreprocessor, age_group
from src.utils import sha256, write_json


def main():
    base=Path('outputs/M02')
    archive=base/'strata_serialization_correction'
    archive.mkdir(exist_ok=False)
    before=yaml.safe_load((base/'source_snapshot/M02_protocol.yaml').read_text())
    after=yaml.safe_load(Path('configs/multimodal/M02_protocol.yaml').read_text())
    assert {k:v for k,v in before.items() if k!='age_strata'} == {k:v for k,v in after.items() if k!='age_strata'}
    val=pd.read_csv('data/official/glaucoma_first/valid.csv')
    processor=AgePreprocessor(**yaml.safe_load((base/'age_preprocessing.yaml').read_text())['parameters'])
    groups=age_group(val.Age,processor)
    proof={'reason':'YAML interprets unquoted 40_59 and 60_79 as integers; repair stratified summaries only.',
           'original_protocol_sha256':sha256(base/'source_snapshot/M02_protocol.yaml'),
           'corrected_protocol_sha256':sha256('configs/multimodal/M02_protocol.yaml'),
           'predictions_and_checkpoints_unchanged':True,'results':{}}
    for eid in ['M02a','M02b','M02c']:
        directory=Path(f'results/{eid}/seed_42/fold_none')
        pred_hash=sha256(directory/'validation.csv')
        shutil.copy2(directory/'metrics.json',archive/f'{eid}_metrics_before.json')
        shutil.copy2(directory/'checksums.sha256',archive/f'{eid}_checksums_before.sha256')
        score=json.loads((directory/'metrics.json').read_text())
        prediction=pd.read_csv(directory/'validation.csv')
        assert list(prediction.image_id)==list(val.Image_name)
        strata={}
        for group in after['age_strata']:
            mask=groups==group; labels=val.True_Label.to_numpy()[mask]
            positive=int(labels.sum());negative=len(labels)-positive
            strata[group]={'n':len(labels),'positive':positive,'negative':negative,
                'auroc':metrics(labels,prediction.probability.to_numpy()[mask])['auroc']
                if positive>=20 and negative>=20 else None}
        score['age_strata']=strata
        write_json(directory/'metrics.json',score)
        files=sorted(p for p in directory.iterdir() if p.is_file() and p.name!='checksums.sha256')
        (directory/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files),encoding='utf-8')
        assert sha256(directory/'validation.csv')==pred_hash
        proof['results'][eid]={'validation_sha256':pred_hash,'stratum_total':sum(s['n'] for s in strata.values())}
        assert proof['results'][eid]['stratum_total']==len(val)
    write_json(archive/'correction.json',proof)


if __name__=='__main__':
    main()
