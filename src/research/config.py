import dataclasses
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml
import pandas as pd

from src.utils import sha256


def git_commit():
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def manifest_semantic_hash(path):
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    payload=json.dumps({'columns':list(frame.columns),'rows':frame.to_dict('records')},
                       ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def manifest_component_hashes(path):
    """Separate order-sensitive image list and annotation hashes; never image content hashes."""
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    if not {'image_id','path','label'}.issubset(frame):
        raise ValueError('Labeled manifest required for component hashes')
    def digest(columns):
        rows=frame[columns].to_dict('records')
        return hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    annotation_columns=['image_id']+[c for c in frame.columns if c not in {'image_id','path'}]
    return {'images':digest(['image_id','path']),'annotations':digest(annotation_columns)}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    parent_experiment_id: str | None
    backbone: str
    pretrained_source: str | None
    pretrained_sha256: str | None
    license: str | None
    weights_manifest: str | None
    image_size: int
    crop_policy: str
    normalization: str
    optimizer: str
    backbone_lr: float
    head_lr: float
    weight_decay: float
    scheduler: str
    warmup: int
    epochs: int
    patience: int
    loss: dict
    sampler: str
    augmentation: dict
    batch_size: int
    gradient_accumulation: int
    seed: int
    fold: int | None
    data_manifest_hash: dict
    output_dir: str
    config_hash: str | None = None
    git_commit: str | None = None
    status: str = 'PENDING'
    dependencies: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def identity(self):
        raw = dataclasses.asdict(self)
        raw.pop('config_hash'); raw.pop('status'); raw.pop('git_commit')
        return hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate(self):
        import re
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', self.experiment_id):
            raise ValueError('Unsafe experiment ID')
        if self.config_hash is not None and self.config_hash != self.identity():
            raise ValueError('Config hash mismatch')
        if self.image_size < 32 or self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError('Invalid image/batch settings')
        if self.optimizer != 'AdamW' or self.scheduler != 'cosine' or self.normalization != 'imagenet':
            raise ValueError('Unknown optimizer/scheduler/normalization')
        if self.status not in {'PENDING','RUNNING','COMPLETED','FAILED','BLOCKED'}:
            raise ValueError('Unknown status')
        if self.output_dir != f'results/{self.experiment_id}/seed_{self.seed}/fold_{self.fold if self.fold is not None else "none"}':
            raise ValueError('Output directory must match experiment/seed/fold')
        if any(Path(self.output_dir).resolve().is_relative_to(Path(p).resolve()) for p in ['outputs/runs/B02','outputs/runs/G01','submission/round_01_B02']):
            raise ValueError('Frozen output destination')

    def assert_data(self, train_manifest, val_manifest, archive=None):
        if manifest_semantic_hash(train_manifest) != self.data_manifest_hash['train'] or manifest_semantic_hash(val_manifest) != self.data_manifest_hash['val']:
            raise ValueError('Official manifest version mismatch')
        if archive is not None and sha256(archive) != self.data_manifest_hash['archive']:
            raise ValueError('Official archive SHA256 mismatch')
        if self.fold is not None and not self.extra.get('fold_manifest_hash'):
            raise ValueError('Fold requires frozen grouping/index hash')

    def assert_code(self):
        if not self.git_commit or not git_commit() or self.git_commit != git_commit():
            raise ValueError('Runner requires pinned Git commit')
        r = subprocess.run(['git','status','--porcelain','--untracked-files=no'], capture_output=True,text=True)
        if r.stdout.strip(): raise ValueError('Tracked code changed since pinned commit')

    def assert_execution_policy(self):
        # Applies to both direct CLI and queue; historical YAML cannot bypass QC.
        if self.crop_policy == 'fov':
            raise ValueError('Legacy research FOV has no accepted safety QC; local S01-v2 is a separate implementation')
        if self.experiment_id in {'G02','G04','G09','L03','L03_LAMBDA0','L03_LAMBDA02'}:
            raise ValueError('Experiment retired from server execution after local results; explicit new protocol required')

    def assert_pretraining(self):
        if not self.weights_manifest:
            raise ValueError('Reviewed nonmedical weight manifest is required')
        from src.model import verified_weights
        m = verified_weights(self.weights_manifest, self.backbone)
        if m['sha256'] != self.pretrained_sha256 or m['source_url'] != self.pretrained_source:
            raise ValueError('Weight SHA256/source differs from config')
        # Legacy reviewed manifests predate the explicit license field and are
        # Apache-2.0. New MIT weights must identify MIT in both config and manifest.
        reviewed_license = m.get('license', 'Apache-2.0')
        if (self.license not in {'Apache-2.0', 'MIT'}
            or reviewed_license != self.license
            or not m['allows_competition_and_redistribution']):
            raise ValueError('Weight license not approved for redistribution')
        return m

    def as_dict(self):
        return dataclasses.asdict(self)
