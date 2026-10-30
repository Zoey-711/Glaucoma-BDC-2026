import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from src.phase_f import validate_predictions,fuse,complementarity,precision_comparison


def table(z):
    return pd.DataFrame({'image_id':['a','b','c','d'],'label':[1,1,0,0],
        'logit':np.asarray(z,dtype=float),'probability':expit(z)})


def test_fixed_fusion_and_pair_count_definition():
    a=table([2,-1,1,-2]);b=table([1,2,-1,-2])
    f=fuse(a,b,'probability');g=fuse(a,b,'logit')
    assert np.allclose(f.probability,(a.probability+b.probability)/2)
    assert np.allclose(g.probability,expit((a.logit+b.logit)/2))
    c=complementarity(a,b)
    assert c['positive_negative_pairs']==4 and c['reference_wrong_candidate_correct']==1
    assert c['reference_pair_loss']==1-roc_auc_score(a.label,a.probability)


def test_prediction_integrity_and_precision_order():
    a=table([2,-1,1,-2]);validate_predictions(a,a,4)
    for field,value in [('probability',np.nan),('logit',np.inf),('label',0),('image_id','b')]:
        broken=a.copy();broken.loc[0,field]=value
        with pytest.raises(ValueError):validate_predictions(broken,a,4)
    report=precision_comparison([.1,.10001,.9],[.10001,.1,.9])
    assert report['probability_threshold_consistent'] and not report['strict_order_consistent']
