"""Train-only grouped folds. Reviewed duplicate-group data is mandatory."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import sha256,write_json


def create_folds(train_manifest,reviewed_groups,output,seed=42):
    frame=pd.read_csv(train_manifest);groups=pd.read_csv(reviewed_groups)
    if not {'image_id','group_id','review_status'}.issubset(groups):raise ValueError('Reviewed group columns required')
    if groups.review_status.ne('reviewed').any() or groups.image_id.duplicated().any() or set(groups.image_id)!=set(frame.image_id):
        raise ValueError('Groups incomplete, repeated or unreviewed')
    if 'patient_id' in groups and groups.patient_id.notna().all():unit='patient'
    else:unit='reviewed_duplicate_group_not_patient'
    joined=frame.merge(groups[['image_id','group_id']],on='image_id',validate='one_to_one')
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    fold=np.full(len(joined),-1,dtype=int)
    for k,(_,val) in enumerate(split.split(joined,joined.label,joined.group_id)):fold[val]=k
    if (fold<0).any() or any(joined.iloc[fold==k].label.nunique()!=2 for k in range(5)):
        raise ValueError('Fold coverage or class failure')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    pd.DataFrame({'image_id':joined.image_id,'fold':fold,'group_id':joined.group_id}).to_csv(out,index=False)
    write_json(str(out)+'.metadata.json',{'source':'official_train_only','train_sha256':sha256(train_manifest),
        'reviewed_groups_sha256':sha256(reviewed_groups),'fold_sha256':sha256(out),
        'seed':seed,'unit':unit,'patient_level_claim':unit=='patient','validation_used':False})
    return out


def assemble_oof(fold_dirs,train_manifest,output):
    train=pd.read_csv(train_manifest);tables=[pd.read_csv(Path(p)/'oof_train.csv') for p in fold_dirs]
    all_rows=pd.concat(tables,ignore_index=True)
    if all_rows.Image_name.duplicated().any() or set(all_rows.Image_name)!=set(train.image_id):
        raise ValueError('OOF missing/duplicated Train images')
    all_rows=all_rows.set_index('Image_name').loc[train.image_id].reset_index()
    if not np.array_equal(all_rows.Label.to_numpy(),train.label.to_numpy()):raise ValueError('OOF labels mismatch')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    all_rows.to_csv(output,index=False)
    return output
