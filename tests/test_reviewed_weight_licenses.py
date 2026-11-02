import dataclasses

import pytest

from src.research.config import ExperimentConfig


@pytest.mark.parametrize('config_license,manifest_license,redistribution,accepted', [
    ('MIT', 'MIT', True, True),
    ('Apache-2.0', 'Apache-2.0', True, True),
    ('Apache-2.0', None, True, True),
    ('MIT', None, True, False),
    ('Apache-2.0', 'MIT', True, False),
    ('MIT', 'Apache-2.0', True, False),
    ('CC-BY-NC-4.0', 'CC-BY-NC-4.0', True, False),
    ('MIT', 'MIT', False, False),
])
def test_training_requires_a_matching_reviewed_redistributable_license(
        monkeypatch, config_license, manifest_license, redistribution, accepted):
    from src import model
    cfg = dataclasses.replace(
        ExperimentConfig.load('configs/research/G07.yaml'),
        license=config_license, weights_manifest='fixture.json',
        pretrained_source='https://example.org/pinned/model.safetensors',
        pretrained_sha256='a' * 64, config_hash=None)
    manifest = {'source_url': cfg.pretrained_source, 'sha256': cfg.pretrained_sha256,
                'allows_competition_and_redistribution': redistribution}
    if manifest_license is not None:
        manifest['license'] = manifest_license
    monkeypatch.setattr(model, 'verified_weights', lambda *_: manifest)
    if accepted:
        assert cfg.assert_pretraining() == manifest
    else:
        with pytest.raises(ValueError, match='Weight license not approved'):
            cfg.assert_pretraining()
