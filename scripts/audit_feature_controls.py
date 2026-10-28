"""Positive perturbation and unrelated-image controls for the audit detector."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import ImageEnhance

import scripts.phase2_multilevel_audit as audit
from src.data import open_rgb
from src.utils import write_json


def main():
    audit.ENHANCED=True; audit.OUT=Path('outputs/phase2_audit_controls')
    audit.OUT.mkdir(exist_ok=False); (audit.OUT/'matches').mkdir()
    f=pd.read_csv('outputs/phase2_g01/near_duplicate_audit/image_hashes.csv')
    rows=[]
    for i in range(4):
        r=f.iloc[i*100]; im=open_rgb(r.path); im.thumbnail((1500,1500))
        # Known derived image: resize/recompress, and optionally brightness/rotation.
        if i%2: im=ImageEnhance.Brightness(im.rotate(2)).enhance(.9)
        p=audit.OUT/f'positive_{i}.jpg'; im.save(p,quality=85)
        pair=SimpleNamespace(path_train=r.path,path_val=str(p),image_id_train=r.image_id,
                             image_id_val=p.name,label_train=r.label,label_val=r.label)
        result=audit.compare(pair,i); result['control']='known_derived';rows.append(result)
    rng=np.random.default_rng(42)
    for i in range(20):
        a,b=f.iloc[rng.choice(len(f),2,replace=False)].itertuples()
        pair=SimpleNamespace(path_train=a.path,path_val=b.path,image_id_train=a.image_id,
                             image_id_val=b.image_id,label_train=a.label,label_val=b.label)
        result=audit.compare(pair,i+4);result['control']='random_pair_not_proven_different_patients';rows.append(result)
    pd.DataFrame(rows).to_csv(audit.OUT/'controls.csv',index=False)
    report={'known_derived_detected':sum(r['high_risk'] for r in rows[:4]),'known_derived_count':4,
            'random_pairs_flagged':sum(r['high_risk'] for r in rows[4:]),'random_pair_count':20,
            'interpretation':'Small sensitivity sanity check; not a clinical identity validation or proof of independence'}
    write_json(audit.OUT/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__': main()
