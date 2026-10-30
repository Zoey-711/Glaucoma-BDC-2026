import numpy as np

from scripts.f01_deployment import cross_class_pair_effect, strict_ranking_changes


def test_pair_effect_reproduces_auc_credit_change():
    labels = np.array([1, 1, 0, 0])
    historical = np.array([.9, .2, .8, .1])
    deployment = np.array([.9, .7, .3, .1])
    effect = cross_class_pair_effect(labels, historical, deployment)
    assert effect['total_positive_negative_pairs'] == 4
    assert effect['improved_pairs'] == 1
    assert effect['worsened_pairs'] == 0
    assert effect['delta_auroc_from_pairs'] == .25


def test_same_implementation_gate_detects_any_rank_change():
    a = np.array([.2, .2000001, .9])
    b = np.array([.2000002, .2, .9])
    assert strict_ranking_changes(a, a.copy()) == 0
    assert strict_ranking_changes(a, b) == 1
