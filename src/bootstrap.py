import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_bootstrap(reference, candidate, repeats=2000, seed=42):
    a, b = reference.copy(), candidate.copy()
    for f in [a, b]:
        if f.image_id.duplicated().any() or not {'image_id', 'label', 'probability'}.issubset(f):
            raise ValueError('Prediction IDs must be unique and labeled')
    if set(a.image_id) != set(b.image_id): raise ValueError('Paired bootstrap requires identical sample IDs')
    b = b.set_index('image_id').loc[a.image_id].reset_index()
    if not np.array_equal(a.label, b.label): raise ValueError('Paired labels disagree')
    y, pa, pb = a.label.to_numpy(), a.probability.to_numpy(), b.probability.to_numpy()
    if set(np.unique(y)) != {0, 1}: raise ValueError('Both classes required')
    rng = np.random.default_rng(seed)
    patient = 'patient_id' in a and 'patient_id' in b and a.patient_id.notna().all() and b.patient_id.notna().all() and a.patient_id.ne('').all() and b.patient_id.ne('').all()
    if patient and not np.array_equal(a.patient_id, b.patient_id): raise ValueError('Patient IDs disagree')
    groups = [g.index.to_numpy() for _, g in a.reset_index(drop=True).groupby('patient_id')] if patient else None
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        if patient:
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        else:
            idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        if len(np.unique(y[idx])) < 2: continue
        deltas.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    if len(deltas) != repeats: raise ValueError('Insufficient valid cluster bootstrap replicates')
    lo, hi = np.quantile(deltas, [.025, .975])
    return {'delta_auroc': float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            'ci95_low': float(lo), 'ci95_high': float(hi), 'replicates': repeats, 'seed': seed,
            'unit': 'patient_cluster' if patient else 'class_stratified_image',
            'interpretation': 'Exploratory paired validation CI; does not establish seed stability or correct model-selection/multiple-comparison bias'}
