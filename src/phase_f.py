"""Fixed offline ensemble analysis; no training or weight optimization."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr


def validate_predictions(frame, official, expected_count=None):
    frame=frame.rename(columns={'Image_name':'image_id','Label':'label','Probability':'probability'}).copy()
    required={'image_id','label','logit','probability'}
    if not required.issubset(frame) or frame.columns.duplicated().any():
        raise ValueError('Missing or ambiguous prediction columns')
    if expected_count is not None and len(frame)!=expected_count:raise ValueError('Wrong sample count')
    if frame.image_id.duplicated().any() or frame.image_id.tolist()!=official.image_id.tolist():
        raise ValueError('Prediction ID/order mismatch')
    if not np.array_equal(frame.label,official.label):raise ValueError('Label mismatch')
    z=frame.logit.to_numpy(float);p=frame.probability.to_numpy(float)
    if not np.isfinite(z).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Nonfinite logits/probabilities or range violation')
    if not np.allclose(expit(z),p,atol=1e-6,rtol=0):raise ValueError('Logit/probability mismatch')
    return frame


def fuse(a,b,method):
    if a.image_id.tolist()!=b.image_id.tolist() or not np.array_equal(a.label,b.label):
        raise ValueError('Fusion requires aligned IDs and labels')
    if method=='probability':
        p=(a.probability.to_numpy()+b.probability.to_numpy())/2
        z=logit(np.clip(p,1e-15,1-1e-15))
    elif method=='logit':
        z=(a.logit.to_numpy()+b.logit.to_numpy())/2;p=expit(z)
    else:raise ValueError('Unsupported fixed fusion')
    return pd.DataFrame({'image_id':a.image_id,'label':a.label,'logit':z,'probability':p})


def complementarity(a,b):
    if a.image_id.tolist()!=b.image_id.tolist() or not np.array_equal(a.label,b.label):
        raise ValueError('Complementarity requires aligned samples')
    y=a.label.to_numpy();pa=a.probability.to_numpy();pb=b.probability.to_numpy()
    da=pa[y==1,None]-pa[y==0][None,:]
    db=pb[y==1,None]-pb[y==0][None,:]
    ea=(pa>=.5)!=y;eb=(pb>=.5)!=y
    return {'pearson_probability':float(np.corrcoef(pa,pb)[0,1]),
        'spearman_probability':float(spearmanr(pa,pb).statistic),
        'pearson_logit':float(np.corrcoef(a.logit,b.logit)[0,1]),
        'positive_negative_pairs':int(da.size),
        'reference_strict_inversions':int((da<0).sum()),'candidate_strict_inversions':int((db<0).sum()),
        'reference_ties':int((da==0).sum()),'candidate_ties':int((db==0).sum()),
        'both_strict_inversions':int(((da<0)&(db<0)).sum()),
        'reference_wrong_candidate_correct':int(((da<0)&(db>0)).sum()),
        'reference_correct_candidate_wrong':int(((da>0)&(db<0)).sum()),
        'reference_pair_loss':float(((da<0)+.5*(da==0)).mean()),
        'candidate_pair_loss':float(((db<0)+.5*(db==0)).mean()),
        'threshold_reference_errors':int(ea.sum()),'threshold_candidate_errors':int(eb.sum()),
        'threshold_both_errors':int((ea&eb).sum()),
        'threshold_reference_only_errors':int((ea&~eb).sum()),
        'threshold_candidate_only_errors':int((~ea&eb).sum()),
        'reference_false_negatives':int((ea&(y==1)).sum()),
        'candidate_false_negatives':int((eb&(y==1)).sum()),
        'both_false_negatives':int((ea&eb&(y==1)).sum()),
        'reference_false_positives':int((ea&(y==0)).sum()),
        'candidate_false_positives':int((eb&(y==0)).sum())}


def precision_comparison(fp32,fp16,tolerance=.001):
    a=np.asarray(fp32);b=np.asarray(fp16)
    delta=np.abs(a-b)
    flips=int(((a>=.5)!=(b>=.5)).sum())
    i,j=np.triu_indices(len(a),1)
    order=int((np.sign(a[i]-a[j])!=np.sign(b[i]-b[j])).sum())
    return {'max_probability_error':float(delta.max()),'mean_probability_error':float(delta.mean()),
        'over_tolerance':int((delta>tolerance).sum()),'threshold_flips':flips,
        'pairwise_order_disagreements':order,'total_pairs':len(i),
        'probability_threshold_consistent':bool(delta.max()<=tolerance and flips==0),
        'strict_order_consistent':order==0,'spearman':float(spearmanr(a,b).statistic)}
