import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audit import audit
from .data import FundusDataset, read_manifest
from .metrics import metrics
from .model import build_model, verified_weights
from .utils import load_config, seed_everything, seed_worker, sha256, write_json


def train(c):
    if not c.get('smoke', False):
        raise ValueError('Formal training must use scripts.experiments run with a registered experiment YAML')
    seed_everything(c['seed'])
    smoke = c.get('smoke', False)
    if smoke and (c['data']['source'] != 'synthetic' or c['epochs'] > 2):
        raise ValueError('Smoke must use synthetic data and <=2 epochs')
    if not smoke and c['data']['source'] != 'official':
        raise ValueError('Only official data permitted for training')
    if not 0 < c['threshold'] < 1 or c['epochs'] < 1 or c['patience'] < 1:
        raise ValueError('Invalid threshold/epochs/patience')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite experiments.
    write_json(out / 'config.json', c)
    d = c['data']
    report = audit(d['train_csv'], d['val_csv'], d['root'])
    write_json(out / 'audit.json', report)
    if not report['passed']:
        raise ValueError('Audit failed; repair leakage/missing files before training')
    if c['model']['name'] != 'efficientnet_b0':
        raise ValueError('Unsupported architecture')
    manifest = c['model'].get('weights_manifest')
    provenance = verified_weights(manifest) if manifest else {'initialization': 'random_smoke_only'}
    write_json(out / 'provenance.json', provenance)
    model = build_model(manifest, allow_random=smoke)
    device = torch.device(c['device'])
    model.to(device)
    frames = [read_manifest(d[k], d['root']) for k in ['train_csv', 'val_csv']]
    loaders = [DataLoader(FundusDataset(f, c['image_size'], train=i == 0),
                batch_size=c['batch_size'], shuffle=i == 0, num_workers=c['num_workers'],
                worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(c['seed']),
                pin_memory=device.type == 'cuda') for i, f in enumerate(frames)]
    amp = c['amp'] and device.type == 'cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
        'device': str(device), 'cuda': torch.version.cuda, 'amp_training': amp,
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'environment.json', environment)
    write_json(out / 'code_sha256.json', {str(p): sha256(p) for p in
        [*Path('src').glob('*.py'), Path('predict.py'), Path('cost.py'), Path('requirements.txt')]})
    (out / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    best, stale, history = (-1., -1.), 0, []
    for epoch in range(c['epochs']):
        model.train()
        total = 0.
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(x).flatten(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        model.eval()
        labels, probs = [], []
        with torch.inference_mode():
            for x, y in loaders[1]:
                # Validation FP32 matches default submitted inference precision.
                probs.extend(torch.sigmoid(model(x.to(device)).flatten()).cpu().tolist())
                labels.extend(y.tolist())
        result = metrics(labels, probs, c['threshold'])
        result.update(epoch=epoch + 1, train_loss=total / len(frames[0]), lr=optimizer.param_groups[0]['lr'])
        history.append(result)
        write_json(out / 'history.json', history)
        key = (result['auroc'], result['auprc'])
        if key > best:
            best, stale = key, 0
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'config': c, 'metrics': result, 'provenance': provenance,
                'smoke': smoke}, out / 'best.pt')
            import pandas as pd
            pd.DataFrame({'image_id': frames[1].image_id, 'label': labels, 'probability': probs}).to_csv(out / 'validation.csv', index=False)
        else:
            stale += 1
        scheduler.step()
        print(json.dumps(result), flush=True)
        if stale >= c['patience']:
            break
    checkpoint = out / 'best.pt'
    write_json(out / 'artifact.json', {'checkpoint_sha256': sha256(checkpoint), 'bytes': checkpoint.stat().st_size})
    register = Path('outputs/experiments.csv')
    exists = register.exists()
    with register.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['started_utc', 'output', 'smoke', 'seed', 'best_auroc', 'best_auprc', 'checkpoint_sha256'])
        w.writerow([environment['started_utc'], str(out), smoke, c['seed'], *best, sha256(checkpoint)])
    return checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    train(load_config(p.parse_args().config))


if __name__ == '__main__':
    main()
