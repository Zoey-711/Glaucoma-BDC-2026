from types import SimpleNamespace

import pytest
import torch

from src.research.models import make_model


@pytest.mark.parametrize('mutation', ['legacy_buffers', 'missing_parameter', 'unknown_parameter'])
def test_swin_legacy_buffers_do_not_weaken_strict_parameter_loading(monkeypatch, mutation):
    import timm
    from safetensors import torch as safetensors_torch
    from timm.models.swin_transformer import SwinTransformer

    def tiny():
        return SwinTransformer(img_size=32, patch_size=4, window_size=2,
                               embed_dim=8, depths=(1, 1, 1, 1),
                               num_heads=(1, 2, 4, 8), num_classes=1000)

    reference = tiny()
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    state['layers.0.blocks.0.attn.relative_position_index'] = torch.zeros(4, 4, dtype=torch.long)
    state['layers.0.blocks.0.attn_mask'] = torch.zeros(1, 4, 4)
    if mutation == 'missing_parameter':
        state.pop('patch_embed.proj.weight')
    elif mutation == 'unknown_parameter':
        state['unknown_trainable.weight'] = torch.ones(1)
    monkeypatch.setattr(timm, 'create_model', lambda *a, **kw: tiny())
    monkeypatch.setattr(safetensors_torch, 'load_file', lambda _: state)
    manifest = {'backend': 'timm', 'architecture': 'swin_tiny_patch4_window7_224.ms_in1k',
                'path': 'fixture.safetensors'}
    cfg = SimpleNamespace(backbone='swin_tiny', image_size=32, extra={},
                          assert_pretraining=lambda: manifest)
    if mutation != 'legacy_buffers':
        with pytest.raises(RuntimeError, match='state_dict'):
            make_model(cfg)
    else:
        loaded = make_model(cfg)
        assert loaded.get_classifier().out_features == 1
        for key, value in reference.state_dict().items():
            if not key.startswith('head.'):
                assert torch.equal(loaded.state_dict()[key], value)
