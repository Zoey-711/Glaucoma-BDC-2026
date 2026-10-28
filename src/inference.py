from pathlib import Path
import platform
import subprocess
import json
import math

import torch
import torchvision

from .data import open_rgb, transform
from .model import build_model
from .submission import write_row
from .utils import seed_everything, sha256


class Predictor:
    def __init__(self, checkpoint, device='cuda', precision='fp32', allow_smoke=False):
        seed_everything(2026)
        self.device = torch.device(device)
        if precision not in ['fp32', 'fp16'] or (precision == 'fp16' and self.device.type != 'cuda'):
            raise ValueError('FP16 requires CUDA; supported precision: fp32/fp16')
        self.precision = precision
        self.checkpoint = Path(checkpoint).resolve()
        self.children = None
        self.tta = ['identity']
        self.channels_last = False
        if self.checkpoint.suffix == '.json':
            bundle = json.loads(self.checkpoint.read_text(encoding='utf-8'))
            members = [(self.checkpoint.parent / p).resolve() for p in bundle['members']]
            if not members or any(p.suffix == '.json' for p in members):
                raise ValueError('Bundle requires nonrecursive checkpoint members')
            self.children = [Predictor(p, device, precision, allow_smoke) for p in members]
            self.tta = bundle.get('tta', ['identity'])
            if not self.tta or any(v not in ['identity', 'hflip'] for v in self.tta):
                raise ValueError('Only frozen identity/hflip TTA supported')
            self.threshold = bundle.get('threshold', .5)
            self.config = self.children[0].config
            self.provenance = {'members': [c.provenance for c in self.children]}
            weights = bundle.get('weights', [1 / len(members)] * len(members))
            if len(weights) != len(members) or min(weights) < 0 or abs(sum(weights) - 1) > 1e-8:
                raise ValueError('Invalid frozen ensemble weights')
            self.weights = weights
            if bundle.get('channels_last'):
                for child in self.children:
                    child.model.to(memory_format=torch.channels_last); child.channels_last = True
            files = {str(self.checkpoint): {'path': str(self.checkpoint), 'bytes': self.checkpoint.stat().st_size, 'sha256': sha256(self.checkpoint)}}
            for child in self.children:
                for f in child.metadata['inference_files']: files[f['path']] = f
            if sum(f['bytes'] for f in files.values()) > 500_000_000: raise ValueError('Bundle exceeds 500MB')
            self.metadata = {**self.children[0].metadata, 'checkpoint_sha256': sha256(self.checkpoint),
                'inference_files': list(files.values()), 'tta': self.tta, 'ensemble_weights': weights,
                'threshold': self.threshold, 'channels_last': bundle.get('channels_last', False)}
            return
        obj = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if obj['smoke'] and not allow_smoke:
            raise ValueError('Synthetic smoke checkpoint cannot produce an official submission')
        self.config = obj['config']
        self.provenance = obj.get('provenance', {})
        if 'backbone' in self.config:
            from .research.config import ExperimentConfig
            from .research.models import make_model
            from .research.preprocess import transform_image
            research_config = ExperimentConfig(**self.config)
            self.model = make_model(research_config, smoke=True)
            self.transform = lambda im: transform_image(im, research_config, train=False)
            self.threshold = .5
        else:
            self.model = build_model(allow_random=True, model_config=obj['config'].get('model'))
            self.threshold = obj['config']['threshold']
            self.transform = transform(obj['config']['image_size'], preprocess=obj['config'].get('preprocess', 'resize'))
        self.model.load_state_dict(obj['state_dict'])
        self.model.to(self.device).eval().requires_grad_(False)
        self.metadata = {'checkpoint_sha256': sha256(self.checkpoint), 'precision': precision,
            'threshold': self.threshold, 'smoke': obj['smoke'], 'batch_size': 1,
            'device': str(self.device), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
            'python': platform.python_version(), 'platform': platform.platform(),
            'torchvision': torchvision.__version__, 'cudnn': torch.backends.cudnn.version()}
        if self.device.type == 'cuda':
            try:
                self.metadata['nvidia_smi'] = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                    text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                self.metadata['nvidia_smi'] = 'unavailable; collect driver evidence manually'
        # Configuration, normalization and class logic needed for inference are embedded in the checkpoint/code.
        self.metadata['inference_files'] = [{'path': str(self.checkpoint), 'bytes': self.checkpoint.stat().st_size,
                                           'sha256': self.metadata['checkpoint_sha256']}]
        if self.checkpoint.stat().st_size > 500_000_000:
            raise ValueError('Inference artifacts exceed 500 MB (decimal)')

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_image(self, im):
        self.last_logits = []
        if self.children:
            from PIL import ImageOps
            probability = 0.
            for child, weight in zip(self.children, self.weights):
                values = []
                for view in self.tta:
                    values.append(child.predict_image(im if view == 'identity' else ImageOps.mirror(im)))
                    self.last_logits.extend(child.last_logits)
                probability += weight * sum(values) / len(values)
            return probability
        x = self.transform(im).unsqueeze(0).to(self.device)
        if self.channels_last: x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.precision == 'fp16'):
            logit = self.model(x).flatten().float()
            p = float(logit.sigmoid().item())
            self.last_logits = [float(logit.item())]
        return p

    @torch.inference_mode()
    def predict_one(self, path, image_id, writer, handle):
        # Shared end-to-end operation: disk decode -> preprocessing -> H2D -> forward
        # -> probability/label -> CSV write and flush. No cached image tensors.
        p = self.predict_image(open_rgb(path))
        write_row(writer, image_id, p, self.threshold)
        handle.flush()
        return p
