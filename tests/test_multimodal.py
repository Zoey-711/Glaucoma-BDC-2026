import numpy as np
import pytest
import torch

from src.multimodal import AgePreprocessor, FusionHead, age_group


def test_age_train_only_and_fixed_missing_rule():
    p = AgePreprocessor.fit([20, 40, 60, None, 0], 'official_train')
    assert p.median == 40 and p.lower == 20 and p.upper == 60
    before = p.as_dict()
    x = p.apply([None, 0, 137, float('inf'), 40])
    assert np.array_equal(x[:, 1], [1, 1, 1, 1, 0])
    assert np.allclose(x[:, 0], 0)
    assert p.as_dict() == before
    with pytest.raises(ValueError, match='only be fitted on Train'):
        AgePreprocessor.fit([1, 1000], 'test')


def test_head_gradient_and_eval_determinism():
    head = FusionHead(8)
    features = torch.randn(4, 8, requires_grad=False)
    ages = torch.randn(4, 2)
    output = head(torch.cat([features, ages], dim=1))
    torch.nn.functional.binary_cross_entropy_with_logits(output.flatten(), torch.tensor([0., 1., 0., 1.])).backward()
    assert features.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    head.eval()
    assert torch.equal(head(torch.cat([features, ages], 1)), head(torch.cat([features, ages], 1)))


def test_age_strata_invalid_values_not_used_as_real_age():
    p = AgePreprocessor.fit([20, 90], 'internal_train')
    assert list(age_group([21, 45, 70, 90, 137], p)) == ['under_40', '40_59', '60_79', '80_plus', 'missing']


def test_protocol_age_strata_remain_strings():
    from pathlib import Path
    import yaml
    config = yaml.safe_load(Path('configs/multimodal/M02_protocol.yaml').read_text())
    assert config['age_strata'] == ['under_40', '40_59', '60_79', '80_plus', 'missing']


def test_placeholder_age_ignores_both_real_values_and_missing_records():
    from src.multimodal import M02Predictor
    predictor=M02Predictor.__new__(M02Predictor)
    predictor.age_input='zeros'
    predictor.ages={'a':137,'b':None}
    assert np.array_equal(predictor.age_features('a'),predictor.age_features('b'))
    assert np.array_equal(predictor.age_features('absent'),np.zeros((1,2),dtype=np.float32))
