"""ConvNeXt-Tiny head controls with the same frozen ImageNet encoder source."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtHeadExperiment(nn.Module):
    def __init__(self, cfg, smoke=False):
        super().__init__()
        from .models import make_model
        if cfg.backbone != 'convnext_tiny' or cfg.image_size != 224:
            raise ValueError('H03 controls require ConvNeXt-Tiny at 224 pixels')
        self.encoder = make_model(cfg, smoke=smoke, features=True)
        if not hasattr(self.encoder, 'stages') or len(self.encoder.stages) != 4:
            raise ValueError('Expected four ConvNeXt feature stages')
        self.variant = cfg.extra['head_variant']
        if self.variant not in {'matched_mlp', 'multiscale'}:
            raise ValueError('Unknown H03 head variant')
        self.mid_stage = int(cfg.extra.get('middle_stage', 2))
        if self.mid_stage not in (1, 2):
            raise ValueError('Middle stage must be 1 or 2')
        self.middle = None
        self.encoder.stages[self.mid_stage].register_forward_hook(self._capture)
        # timm ConvNeXt stages expose output channels through their final block.
        mid_channels = self.encoder.stages[self.mid_stage].blocks[-1].conv_dw.out_channels
        final_channels = self.encoder.num_features
        width = int(cfg.extra.get('head_width', 128))
        if width < 1:
            raise ValueError('Head width must be positive')
        if self.variant == 'multiscale':
            in_features = mid_channels + final_channels
        else:
            # Match trainable head parameters to the multiscale head within one width step.
            target = (mid_channels + final_channels + 2) * width + 1
            width = max(1, round((target - 1) / (final_channels + 2)))
            in_features = final_channels
        self.head = nn.Sequential(nn.Linear(in_features, width), nn.GELU(), nn.Linear(width, 1))
        self.feature_dimensions = {'middle_stage': self.mid_stage,
            'middle_channels': mid_channels, 'final_channels': final_channels,
            'fused_channels': in_features, 'head_width': width}

    def _capture(self, _module, _inputs, output):
        self.middle = output

    def get_classifier(self):
        return self.head

    def forward(self, x):
        self.middle = None
        final = self.encoder.forward_features(x)
        if final.ndim != 4 or self.middle is None or self.middle.ndim != 4:
            raise RuntimeError('ConvNeXt stage features not captured')
        pooled = F.adaptive_avg_pool2d(final, 1).flatten(1)
        if self.variant == 'multiscale':
            middle = F.adaptive_avg_pool2d(self.middle, 1).flatten(1)
            pooled = torch.cat((middle, pooled), dim=1)
        return self.head(pooled)

    def cost_estimate(self):
        d = self.feature_dimensions
        head_macs = d['fused_channels']*d['head_width'] + d['head_width']
        return {'basis': 'ConvNeXt-Tiny ImageNet model card: approximately 4.5 GMAC at 224px; linear head analytic MAC. Excludes pooling/activation overhead.',
            'backbone_macs_estimate': 4_500_000_000,
            'head_linear_macs_estimate': head_macs,
            'total_macs_approximate': 4_500_000_000 + head_macs,
            'head_parameters': sum(p.numel() for p in self.head.parameters()),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'feature_dimensions': d, 'measured_latency_ms': None}
