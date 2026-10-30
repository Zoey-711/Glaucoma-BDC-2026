import torch
from torch import nn


ARCHITECTURES = {
    'convnext_tiny':'convnext_tiny.fb_in1k',
    'convnext_small':'convnext_small.fb_in1k',
    'efficientnet_b0':'efficientnet_b0.ra_in1k',
    'efficientnet_v2_s':'tf_efficientnetv2_s.in21k_ft_in1k',
    'swin_tiny':'swin_tiny_patch4_window7_224.ms_in1k',
    'densenet121':'densenet121.ra_in1k',
}


def make_model(cfg, smoke=False, features=False):
    import timm
    if not features and cfg.extra.get('head_variant') in {'matched_mlp', 'multiscale'}:
        from .h03 import ConvNeXtHeadExperiment
        return ConvNeXtHeadExperiment(cfg, smoke=smoke)
    if cfg.backbone not in ARCHITECTURES:
        raise ValueError('Unknown backbone')
    random_ssl_transfer=cfg.extra.get('ssl_initialization')=='random_then_ssl_then_finetune' and cfg.extra.get('ssl_encoder_checkpoint')
    if not smoke and not random_ssl_transfer:
        m = cfg.assert_pretraining()
        architecture = m['architecture']
        if m['backend'] != 'timm': raise ValueError('This runner accepts reviewed timm weights only')
    else:
        m = None; architecture = ARCHITECTURES[cfg.backbone]
    kwargs={'img_size':cfg.image_size} if cfg.backbone=='swin_tiny' else {}
    model = timm.create_model(architecture, pretrained=False, num_classes=1000, **kwargs)
    if m:
        if m['path'].endswith('.safetensors'):
            from safetensors.torch import load_file
            state=load_file(m['path'])
        else: state=torch.load(m['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
    model.reset_classifier(0 if features else 1)
    transfer=cfg.extra.get('ssl_encoder_checkpoint')
    if transfer and not features:
        obj=torch.load(transfer,map_location='cpu',weights_only=True)
        if obj.get('training_data_sha256')!=cfg.data_manifest_hash['train'] or obj.get('test_used') or obj.get('validation_used'):
            raise ValueError('SSL encoder source is not official Train only')
        missing,unexpected=model.load_state_dict(obj['encoder'],strict=False)
        if unexpected or not all('head' in x or 'classifier' in x for x in missing):
            raise ValueError('SSL encoder topology mismatch')
    return model


def split_parameters(model):
    head = model.get_classifier()
    if not isinstance(head, nn.Module): raise ValueError('Classifier head unavailable')
    ids={id(p) for p in head.parameters()}
    return [p for p in model.parameters() if id(p) not in ids], list(head.parameters())


def parameter_count(model): return sum(p.numel() for p in model.parameters())
