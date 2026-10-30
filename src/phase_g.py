"""Fixed-prediction comparison and locked T4 consistency checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def strict_pair_comparison(reference,candidate):
    if reference.image_id.tolist()!=candidate.image_id.tolist() or not np.array_equal(reference.label,candidate.label):
        raise ValueError('Comparison requires identical sample order and labels')
    y=reference.label.to_numpy()
    ar=reference.probability.to_numpy()[y==1,None]-reference.probability.to_numpy()[y==0][None,:]
    ac=candidate.probability.to_numpy()[y==1,None]-candidate.probability.to_numpy()[y==0][None,:]
    loss_r=(ar<0)+.5*(ar==0);loss_c=(ac<0)+.5*(ac==0)
    result={'positive_negative_pairs':int(ar.size),
        'reference_strict_inversions':int((ar<0).sum()),
        'candidate_strict_inversions':int((ac<0).sum()),
        'both_strict_inversions':int(((ar<0)&(ac<0)).sum()),
        'F04_fixes_F01_pairs':int(((ar<0)&(ac>0)).sum()),
        'F04_breaks_F01_pairs':int(((ar>0)&(ac<0)).sum()),
        'reference_ties':int((ar==0).sum()),'candidate_ties':int((ac==0).sum()),
        'net_pair_loss_reduction':float(loss_r.sum()-loss_c.sum())}
    assert np.isclose(roc_auc_score(y,reference.probability),1-loss_r.mean())
    assert np.isclose(roc_auc_score(y,candidate.probability),1-loss_c.mean())
    return result


def compare_frozen_to_inference(expected,actual,tolerance=.001,threshold=.5):
    """Compare one 100-image same-precision round with the frozen 5849-row Val table."""
    if actual.image_id.duplicated().any() or not actual.image_id.isin(expected.image_id).all():
        raise ValueError('Inference IDs are duplicated or absent from frozen validation')
    if not np.isfinite(actual.probability.to_numpy(float)).all():
        raise ValueError('Inference probabilities must be finite')
    e=expected.set_index('image_id').loc[actual.image_id]
    p=e.probability.to_numpy(float);q=actual.probability.to_numpy(float)
    delta=np.abs(p-q)
    i,j=np.triu_indices(len(p),1)
    order=np.sign(p[i]-p[j])!=np.sign(q[i]-q[j])
    flips=(p>=threshold)!=(q>=threshold)
    return {'count':len(p),'max_probability_error':float(delta.max()),
        'mean_probability_error':float(delta.mean()),'probability_mismatches':int((delta>tolerance).sum()),
        'pred_label_mismatches':int(flips.sum()),'pairwise_order_mismatches':int(order.sum()),
        'pairwise_compared':int(len(i)),
        'passed':bool(delta.max()<=tolerance and not flips.any() and not order.any())}
