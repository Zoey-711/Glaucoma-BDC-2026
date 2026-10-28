"""D01 teacher provenance and Bernoulli-KL guardrails."""
import json
from pathlib import Path

import torch

from src.model import verified_weights
from src.utils import sha256
from .config import manifest_semantic_hash
from .models import make_model


def load_approved_teacher(path,student_cfg,device,train_manifest):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    proof=checkpoint.get('teacher_provenance') or checkpoint.get('provenance')
    if not isinstance(proof,dict):raise ValueError('Teacher provenance missing')
    weight=proof.get('pretraining') or proof
    verified_weights(weight)
    if proof.get('training_data_sha256')!=manifest_semantic_hash(train_manifest):
        raise ValueError('Teacher not certified as official-Train-only')
    if proof.get('used_validation_for_training') or proof.get('used_test') or proof.get('external_medical_data'):
        raise ValueError('Teacher used forbidden data')
    cfg=checkpoint.get('config')
    if not cfg or cfg.get('image_size')!=student_cfg.image_size or cfg.get('crop_policy')!=student_cfg.crop_policy:
        raise ValueError('Teacher/student preprocessing mismatch')
    from .config import ExperimentConfig
    teacher_cfg=ExperimentConfig(**cfg)
    model=make_model(teacher_cfg,smoke=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False)
