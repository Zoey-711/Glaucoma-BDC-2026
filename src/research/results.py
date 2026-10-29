import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.metrics import metrics
from src.utils import sha256
from .config import ExperimentConfig


REQUIRED={'best.pth','validation.csv','metrics.json','config.yaml','train_history.csv',
          'training.log','run_manifest.json','environment.txt','checksums.sha256'}
VERSION='glaucoma_research_v1'


def validate_result(directory,official_val,expected_config=None,freeze_inventory=None,allow_smoke=False):
    root=Path(directory);missing=REQUIRED-{p.name for p in root.iterdir()}
    if missing:raise ValueError(f'Missing required result files: {sorted(missing)}')
    hashes={}
    for line in (root/'checksums.sha256').read_text(encoding='utf-8').splitlines():
        digest,filename=line.split('  ',1)
        if Path(filename).name!=filename or filename in hashes:raise ValueError('Invalid checksum filename')
        hashes[filename]=digest
    required_hashes={p.name for p in root.iterdir() if p.is_file()}-{'checksums.sha256'}
    if set(hashes)!=required_hashes:raise ValueError('Checksum inventory incomplete')
    if any(sha256(root/f)!=h for f,h in hashes.items()):raise ValueError('File checksum mismatch')
    checkpoint=torch.load(root/'best.pth',map_location='cpu',weights_only=True)
    if (not isinstance(checkpoint,dict) or not checkpoint.get('state_dict')
        or checkpoint.get('config',{}).get('experiment_id')!=json.loads((root/'run_manifest.json').read_text(encoding='utf-8')).get('experiment_id')):
        raise ValueError('Invalid or mismatched model checkpoint')
    cfg=ExperimentConfig.load(root/'config.yaml')
    expected=ExperimentConfig.load(expected_config) if expected_config else cfg
    if cfg.identity()!=expected.identity():raise ValueError('Config version mismatch')
    if expected.git_commit is not None and cfg.git_commit!=expected.git_commit:
        raise ValueError('Expected Git commit mismatch')
    meta=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    if meta.get('schema')!=VERSION or meta.get('status')!='COMPLETED' or meta.get('config_hash')!=cfg.identity() or meta.get('git_commit')!=cfg.git_commit:
        raise ValueError('Manifest status/config/code mismatch')
    if meta.get('data_manifest_hash')!=cfg.data_manifest_hash or meta.get('pretrained_sha256')!=cfg.pretrained_sha256:
        raise ValueError('Data/pretraining provenance mismatch')
    if meta.get('seed')!=cfg.seed or meta.get('fold')!=cfg.fold or not meta.get('finished_utc'):
        raise ValueError('Seed/fold/time mismatch')
    if cfg.git_commit is None and not (allow_smoke and meta.get('smoke')):raise ValueError('No pinned Git commit')
    official=pd.read_csv(official_val)
    pred=pd.read_csv(root/'validation.csv')
    if not {'Image_name','Label','logit','Probability'}.issubset(pred):raise ValueError('Prediction columns missing')
    if pred.Image_name.duplicated().any() or list(pred.Image_name)!=list(official.image_id):
        raise ValueError('Validation ID/order mismatch')
    if not np.array_equal(pred.Label.to_numpy(),official.label.to_numpy()):raise ValueError('Validation labels mismatch')
    logits=pred.logit.to_numpy(dtype=float);p=pred.Probability.to_numpy(dtype=float)
    if not np.isfinite(logits).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Nonfinite logit or invalid probability')
    if not np.allclose(1/(1+np.exp(-np.clip(logits,-700,700))),p,atol=1e-6):
        raise ValueError('Probabilities do not match logits')
    score=metrics(pred.Label,p)
    reported=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    for key in ['auroc','auprc','recall','specificity']:
        if not np.isfinite(reported[key]) or abs(score[key]-reported[key])>1e-8:raise ValueError(f'Metric mismatch: {key}')
    history=pd.read_csv(root/'train_history.csv')
    if reported['best_epoch'] not in set(history.epoch):raise ValueError('Best epoch absent from history')
    if freeze_inventory:
        frozen=json.loads(Path(freeze_inventory).read_text(encoding='utf-8'))
        if any(sha256(path)!=digest for path,digest in frozen.items()):raise ValueError('Frozen asset changed')
    return {'experiment_id':cfg.experiment_id,'seed':cfg.seed,'fold':cfg.fold,
            'AUROC':score['auroc'],'AUPRC':score['auprc'],
            'model_size_mb':(root/'best.pth').stat().st_size/1e6,
            'config_hash':cfg.identity(),'git_commit':cfg.git_commit,'status':'COMPLETED'}


def package_result(directory,target):
    root=Path(directory);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    if not REQUIRED.issubset({p.name for p in root.iterdir()}):raise ValueError('Incomplete result; checkpoint missing')
    with tarfile.open(target,'w:gz') as tar:
        for p in sorted(root.iterdir()):
            if p.is_file():tar.add(p,arcname=p.name,recursive=False)
    return target


def safe_import(bundle,destination):
    bundle=Path(bundle);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        with tarfile.open(bundle,'r:gz') as tar:
            if sum(x.size for x in tar.getmembers())>4_000_000_000:raise ValueError('Result archive too large')
            for member in tar.getmembers():
                if not member.isfile() or Path(member.name).name!=member.name or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member')
            for member in tar.getmembers():
                with tar.extractfile(member) as source,(destination/member.name).open('xb') as sink:
                    while block:=source.read(1024*1024):sink.write(block)
    except Exception:
        # Preserve failed staging directory for forensic review; never merge into accepted results.
        raise
    return destination
