import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve


def metrics(labels, probabilities, threshold=.5):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Validation requires both classes and finite probabilities')
    precision, recall, _ = precision_recall_curve(y, p)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    sensitivity90 = float(tpr[fpr <= .1 + 1e-12].max())
    return {'auroc': float(roc_auc_score(y, p)), 'auprc': float(auc(recall, precision)),
            'recall': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)),
            'sensitivity_at_90_specificity': sensitivity90, 'threshold': threshold}
