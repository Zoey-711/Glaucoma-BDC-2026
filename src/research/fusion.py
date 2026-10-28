import numpy as np
import pandas as pd

from src.metrics import metrics


def fixed_fusion(paths,mode='probability'):
    if mode not in {'probability','logit'} or len(paths)<2:raise ValueError('Need fixed probability/logit fusion with >=2 members')
    frames=[pd.read_csv(p) for p in paths]
    base=frames[0]
    if base.Image_name.duplicated().any():raise ValueError('Duplicate reference IDs')
    aligned=[]
    for f in frames:
        if f.Image_name.duplicated().any() or set(f.Image_name)!=set(base.Image_name):raise ValueError('Fusion ID mismatch')
        f=f.set_index('Image_name').loc[base.Image_name].reset_index()
        if not np.array_equal(f.Label,base.Label):raise ValueError('Fusion labels mismatch')
        aligned.append(f)
    if mode=='probability':p=np.mean([f.Probability.to_numpy(dtype=float) for f in aligned],axis=0)
    else:
        z=np.mean([f.logit.to_numpy(dtype=float) for f in aligned],axis=0)
        p=1/(1+np.exp(-np.clip(z,-700,700)))
    out=pd.DataFrame({'Image_name':base.Image_name,'Label':base.Label,'Probability':p})
    out['logit']=np.log(np.clip(p,1e-12,1-1e-12)/(1-np.clip(p,1e-12,1-1e-12)))
    return out,metrics(out.Label,out.Probability)
