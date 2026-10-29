"""Train-only age preprocessing and frozen ConvNeXt feature fusion for M02."""
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import open_rgb, transform
from src.model import build_model
from src.submission import write_row
from src.utils import seed_everything, sha256


def numeric_age(values):
    return pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class AgePreprocessor:
    lower: float
    upper: float
    median: float
    mean: float
    std: float
    fit_rows: int
    fit_source: str = 'official_train'

    @classmethod
    def fit(cls, values, source):
        if source not in {'official_train', 'internal_train'}:
            raise ValueError('Age preprocessing may only be fitted on Train')
        age = numeric_age(values)
        valid = np.isfinite(age) & (age > 0)
        if not valid.any():
            raise ValueError('No finite positive Train ages')
        median = float(np.median(age[valid]))
        imputed = np.where(valid, age, median)
        return cls(float(age[valid].min()), float(age[valid].max()), median,
                   float(imputed.mean()), max(float(imputed.std(ddof=0)), 1e-8), len(age), source)

    def apply(self, values):
        age = numeric_age(values)
        missing = ~np.isfinite(age) | (age < self.lower) | (age > self.upper)
        imputed = np.where(missing, self.median, age)
        return np.column_stack(((imputed - self.mean) / self.std, missing.astype(float))).astype(np.float32)

    def as_dict(self):
        return asdict(self)


class FusionHead(nn.Sequential):
    def __init__(self, image_dim=768):
        super().__init__(nn.LayerNorm(image_dim + 2), nn.Linear(image_dim + 2, 128),
                         nn.GELU(), nn.Dropout(.2), nn.Linear(128, 1))


def age_summary(values, processor):
    age = numeric_age(values)
    finite = age[np.isfinite(age)]
    quantiles = np.quantile(finite, [0, .01, .05, .25, .5, .75, .95, .99, 1]) if len(finite) else None
    return {'n': len(age), 'missing_nonfinite': int((~np.isfinite(age)).sum()),
            'nonpositive': int((np.isfinite(age) & (age <= 0)).sum()),
            'outside_train_positive_envelope': int((np.isfinite(age) & ((age < processor.lower) | (age > processor.upper))).sum()),
            'missing_or_abnormal_after_rule': int(processor.apply(age)[:, 1].sum()),
            'mean_finite': float(finite.mean()) if len(finite) else None,
            'std_finite': float(finite.std()) if len(finite) else None,
            'quantiles': dict(zip(['min','p01','p05','p25','median','p75','p95','p99','max'],
                                  map(float, quantiles))) if quantiles is not None else None}


def age_group(values, processor):
    age = numeric_age(values)
    group = np.select([age < 40, age < 60, age < 80], ['under_40','40_59','60_79'], default='80_plus')
    group[processor.apply(age)[:, 1].astype(bool)] = 'missing'
    return group


class M02Predictor:
    """Single-image decode -> frozen backbone/head -> CSV; age is an input lookup."""
    def __init__(self, checkpoint, ages, device='cuda'):
        seed_everything(42)
        self.device = torch.device(device)
        self.ages = dict(ages)
        self.path = Path(checkpoint)
        self.model = None
        if self.path.suffix == '.json':
            obj = json.loads(self.path.read_text(encoding='utf-8'))
            self.coef = np.asarray(obj['coef'], dtype=np.float64)
            self.intercept = float(obj['intercept'])
            self.mode = 'age_only'
        else:
            obj = torch.load(self.path, map_location='cpu', weights_only=True)
            self.model = build_model(allow_random=True, model_config=obj['model_config'])
            self.model.load_state_dict(obj['backbone_state_dict'], strict=True)
            self.model.to(self.device).eval().requires_grad_(False)
            self.head = FusionHead(obj['image_dim'])
            self.head.load_state_dict(obj['head_state_dict'], strict=True)
            self.head.to(self.device).eval().requires_grad_(False)
            self.transform = transform(224, train=False, preprocess='resize')
            self.mode = 'fusion'
        self.age_input = obj.get('age_input', 'real')
        if self.age_input not in {'real', 'zeros'}:
            raise ValueError('Unknown age input policy')
        self.processor = AgePreprocessor(**obj['age_preprocessing']) if self.age_input == 'real' else None
        self.metadata = {'checkpoint_sha256': sha256(self.path), 'model_bytes': self.path.stat().st_size,
                         'precision': 'fp32', 'batch_size': 1, 'age_lookup_in_timer': True,
                         'gpu': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None}

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def age_features(self, image_id):
        if self.age_input == 'zeros':
            return np.zeros((1, 2), dtype=np.float32)
        if image_id not in self.ages:
            raise ValueError('Missing official age record; absent Age value differs from absent image ID')
        return self.processor.apply([self.ages[image_id]])

    @torch.inference_mode()
    def predict_one(self, image_path, image_id, writer, handle):
        # Decode is included even for the age-only comparator to retain the same E2E scope.
        image = open_rgb(image_path)
        age = self.age_features(image_id)
        if self.mode == 'age_only':
            logit = float((age.astype(np.float64) @ self.coef + self.intercept).item())
        else:
            x = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
            joined = torch.cat([features, torch.from_numpy(age).to(self.device)], dim=1)
            logit = float(self.head(joined).item())
        probability = float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64)))
        write_row(writer, image_id, probability, .5)
        handle.flush()
        return probability
