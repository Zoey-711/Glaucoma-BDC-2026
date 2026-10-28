import json
from pathlib import Path

import torch
from torchvision.models import efficientnet_b0

from .utils import sha256


SUPPORTED = {'efficientnet_b0', 'efficientnet_b2', 'resnet50', 'convnext_tiny', 'convnext_small', 'swin_tiny', 'efficientnet_v2_s', 'densenet121'}


def verified_weights(manifest, model_name=None):
    m = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text(encoding='utf-8'))
    # Manifests are shared between Windows and Linux; store relative paths with '/'.
    m = {**m}
    for key in ('path', 'license_file', 'model_card_file', 'upstream_license_file'):
        if m.get(key):
            m[key] = m[key].replace('\\', '/')
    if (m.get('status') != 'approved' or m.get('model') not in SUPPORTED
        or m.get('dataset') != 'ImageNet-1K' or m.get('medical_pretraining') is not False
        or m.get('allows_competition_and_redistribution') is not True
        or m.get('non_commercial_or_gated') is not False):
        raise ValueError('Pretraining provenance/license not approved')
    if model_name and m['model'] != model_name:
        raise ValueError('Weight manifest backbone mismatch')
    for field in ['source_url', 'training_description_url', 'license_url', 'reviewer', 'review_date']:
        if not m.get(field):
            raise ValueError(f'Missing compliance field: {field}')
    for path, digest in [('path', 'sha256'), ('license_file', 'license_sha256')]:
        if not m.get(path) or not m.get(digest) or sha256(m[path]) != m[digest]:
            raise ValueError(f'Invalid SHA256: {path}')
    if m.get('model_card_file') and sha256(m['model_card_file']) != m.get('model_card_sha256'):
        raise ValueError('Model card SHA256 mismatch')
    if m.get('upstream_license_file') and sha256(m['upstream_license_file']) != m.get('upstream_license_sha256'):
        raise ValueError('Upstream license SHA256 mismatch')
    return m


def build_model(manifest=None, allow_random=False, model_config=None):
    config = model_config or {'name': 'efficientnet_b0', 'backend': 'torchvision'}
    name = config['name']
    if name not in SUPPORTED:
        raise ValueError(f'Unsupported backbone {name}')
    m = verified_weights(manifest, name) if manifest else None
    if config.get('backend', 'torchvision') == 'timm':
        import timm
        architecture = config['architecture']
        if m and (m.get('backend') != 'timm' or m.get('architecture') != architecture):
            raise ValueError('Pretraining architecture/backend mismatch')
        model = timm.create_model(architecture, pretrained=False, num_classes=1000)
    else:
        if name != 'efficientnet_b0':
            raise ValueError('Legacy torchvision interface supports B0 only; experiments use timm')
        model = efficientnet_b0(weights=None)
    if manifest:
        if str(m['path']).endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(m['path'])
        else:
            state = torch.load(m['path'], map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    elif not allow_random:
        raise ValueError('Reviewed nonmedical weights required; random initialization is smoke-only')
    if config.get('backend', 'torchvision') == 'timm':
        model.reset_classifier(1)
    else:
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    return model
