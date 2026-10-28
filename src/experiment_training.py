import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .audit import audit
from .cache import build_cache
from .data import FundusDataset, read_manifest
from .losses import BinaryLoss
from .metrics import metrics
from .model import build_model, verified_weights
from .utils import load_config, seed_everything, seed_worker, sha256, write_json


def training_data(c):
    d = c['data']
    if not c.get('smoke'):
        official = load_config('configs/official.local.yaml')['data']
        if d['source'] != 'official' or Path(d['root']).resolve() != Path(official['root']).resolve():
            raise ValueError('Formal experiments must use the registered official image root')
        for key in ['train_csv', 'val_csv']:
            if sha256(d[key]) != sha256(official[key]):
                raise ValueError(f'{key}: official split changed; test/external data forbidden')
    return [read_manifest(d[k], d['root']) for k in ['train_csv', 'val_csv']]


def checked_audit(c):
    d = c['data']
    saved = Path('outputs/official_ingest/audit.json')
    if not c.get('smoke') and saved.exists():
        r = json.loads(saved.read_text(encoding='utf-8'))
        if r.get('passed') and r['manifest_sha256'] == {'train': sha256(d['train_csv']), 'val': sha256(d['val_csv'])}:
            return {**r, 'reuse_basis': 'Existing full audit; same official manifest SHA256; source data must remain immutable'}
    r = audit(d['train_csv'], d['val_csv'], d['root'])
    if not r['passed']: raise ValueError('Data audit failed')
    return r


def train(c, on_epoch=None):
    seed_everything(c['seed'])
    torch.set_num_threads(c.get('torch_threads', 4))
    smoke = c.get('smoke', False)
    if smoke and (c['data']['source'] != 'synthetic' or c['epochs'] > 2):
        raise ValueError('Smoke requires synthetic data and <=2 epochs')
    if not 0 < c['threshold'] < 1 or c['epochs'] < 1 or c['patience'] < 1:
        raise ValueError('Invalid threshold/epochs/patience')
    manifest = c['model'].get('weights_manifest')
    provenance = verified_weights(manifest, c['model']['name']) if manifest else {'initialization': 'random_smoke_only'}
    if not smoke and not manifest: raise ValueError('Formal training requires reviewed pretraining weights')
    frames = training_data(c)
    report = checked_audit(c)
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    for name, obj in [('config', c), ('audit', report), ('provenance', provenance)]:
        write_json(out / f'{name}.json', obj)
    caches = [None, None]
    if c.get('cache'):
        for i, split in enumerate(['train', 'val']):
            caches[i] = build_cache(frames[i], c['data'][split + '_csv'], c['image_size'],
                                    c.get('preprocess', 'resize'), split)
        write_json(out / 'cache.json', {'train': str(caches[0]), 'val': str(caches[1])})
    model = build_model(manifest, allow_random=smoke, model_config=c['model'])
    device = torch.device(c['device'])
    model.to(device)
    counts = np.bincount(frames[0].label, minlength=2).tolist()
    sampler = None
    if c.get('sampler') == 'weighted':
        w = torch.tensor([1 / counts[y] for y in frames[0].label], dtype=torch.double)
        sampler = WeightedRandomSampler(w, len(w), replacement=True, generator=torch.Generator().manual_seed(c['seed']))
    loaders = []
    for i, f in enumerate(frames):
        ds = FundusDataset(f, c['image_size'], i == 0, c.get('augmentation', 'basic'), c.get('preprocess', 'resize'), caches[i])
        workers = c['num_workers']
        loaders.append(DataLoader(ds, batch_size=c['batch_size'], shuffle=i == 0 and sampler is None,
            sampler=sampler if i == 0 else None, num_workers=workers, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(c['seed']), pin_memory=device.type == 'cuda',
            persistent_workers=workers > 0, **({'prefetch_factor': 2} if workers else {})))
    amp = c['amp'] and device.type == 'cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loss_fn = BinaryLoss(c.get('loss', {'name': 'bce'}), counts).to(device)
    teacher = None
    if c.get('distillation'):
        from .inference import Predictor
        kd = c['distillation']
        teacher = Predictor(kd['teacher_checkpoint'], c['device'], allow_smoke=smoke)
        if teacher.config['image_size'] != c['image_size'] or teacher.config.get('preprocess', 'resize') != c.get('preprocess', 'resize'):
            raise ValueError('Distillation requires matching teacher/student preprocessing')
        if not smoke and (teacher.provenance.get('status') != 'approved' or teacher.provenance.get('medical_pretraining') is not False):
            raise ValueError('Teacher must have approved nonmedical pretraining provenance')
        if not smoke:
            verified_weights(teacher.provenance, teacher.config['model']['name'])
        provenance = {**provenance, 'teacher': {'checkpoint_sha256': sha256(kd['teacher_checkpoint']),
            'pretraining_provenance': teacher.provenance, 'training_config': teacher.config}}
        write_json(out / 'provenance.json', provenance)
        write_json(out / 'teacher_provenance.json', teacher.provenance)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': str(device),
        'cuda': torch.version.cuda, 'amp_training': amp, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'environment.json', environment)
    write_json(out / 'code_sha256.json', {str(p): sha256(p) for p in [*Path('src').glob('*.py'), Path('predict.py'), Path('cost.py')]})
    (out / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    best, stale, history = (-1., -1.), 0, []
    started = time.perf_counter()
    for epoch in range(c['epochs']):
        epoch_start = time.perf_counter()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total = 0.
        for step, (x, y) in enumerate(loaders[0], 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x).flatten()
                loss = loss_fn(logits, y)
                if teacher:
                    temperature = c['distillation'].get('temperature', 2.)
                    with torch.no_grad(): target = (teacher.model(x).flatten() / temperature).sigmoid()
                    soft_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, target) * temperature ** 2
                    alpha = c['distillation'].get('alpha', .5)
                    loss = (1 - alpha) * loss + alpha * soft_loss
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            if step % c.get('log_every', 100) == 0:
                write_json(out / 'progress.json', {'epoch': epoch + 1, 'step': step, 'steps': len(loaders[0]),
                    'elapsed_seconds': time.perf_counter() - started, 'train_loss_so_far': total / min(step * c['batch_size'], len(frames[0]))})
                print(f"{c.get('experiment_id', 'smoke')} epoch={epoch+1} step={step}/{len(loaders[0])}", flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            train_memory = {'train_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                            'train_peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}
            torch.cuda.reset_peak_memory_stats(device)
        else:
            train_memory = {'train_peak_allocated_mib': None, 'train_peak_reserved_mib': None}
        train_seconds = time.perf_counter() - epoch_start
        model.eval()
        labels, raw_logits, probs = [], [], []
        val_loss = 0.
        with torch.inference_mode():
            for x, y in loaders[1]:
                logits = model(x.to(device)).flatten().float()
                val_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), reduction='sum').item()
                raw_logits.extend(logits.cpu().tolist())
                probs.extend(logits.sigmoid().cpu().tolist())
                labels.extend(y.tolist())
        result = metrics(labels, probs, c['threshold'])
        result.update(epoch=epoch + 1, train_loss=total / len(frames[0]), val_loss=val_loss / len(frames[1]),
                      lr=optimizer.param_groups[0]['lr'], epoch_seconds=time.perf_counter() - epoch_start)
        result.update(**train_memory, train_epoch_seconds=train_seconds,
                      train_images_per_second=len(frames[0]) / train_seconds,
                      val_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                      val_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else None)
        history.append(result)
        write_json(out / 'history.json', history)
        key = (result['auroc'], result['auprc'])
        if key > best:
            best, stale = key, 0
            tmp = out / 'best.tmp'
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'config': c, 'metrics': result, 'provenance': provenance, 'smoke': smoke}, tmp)
            tmp.replace(out / 'best.pt')
            pred = pd.DataFrame({'image_id': frames[1].image_id, 'logit': raw_logits, 'probability': probs,
                                 'label': labels, 'experiment_id': c.get('experiment_id', 'smoke')})
            if 'patient_id' in frames[1]: pred['patient_id'] = frames[1].patient_id
            pred.to_csv(out / 'validation.csv', index=False)
        else: stale += 1
        scheduler.step()
        if on_epoch: on_epoch(result, out / 'best.pt', time.perf_counter() - started)
        print(json.dumps(result), flush=True)
        if stale >= c['patience']: break
    checkpoint = out / 'best.pt'
    write_json(out / 'artifact.json', {'checkpoint_sha256': sha256(checkpoint), 'bytes': checkpoint.stat().st_size,
        'train_duration': time.perf_counter() - started, 'duration_scope': 'training + validation; excludes cache preparation'})
    return checkpoint
