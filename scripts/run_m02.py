"""Execute frozen G01 age-fusion controls. Never train on or predict official Test."""
import argparse
import csv
import gc
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.bootstrap import paired_bootstrap
from src.data import FundusDataset, read_manifest
from src.inference import Predictor
from src.metrics import metrics
from src.multimodal import AgePreprocessor, FusionHead, M02Predictor, age_group, age_summary
from src.model import verified_weights
from src.submission import COLUMNS
from src.utils import seed_everything, sha256, write_json

PROTOCOL = Path('configs/multimodal/M02_protocol.yaml')
C = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
OUT = Path(C['output'])


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_frozen():
    original = json.loads(Path('outputs/migration_review/frozen_inventory.json').read_text(encoding='utf-8'))
    extra_path = OUT / 'frozen_assets.json'
    if extra_path.exists():
        original.update(json.loads(extra_path.read_text(encoding='utf-8')))
    bad = [str(p) for p, digest in original.items() if not Path(p).is_file() or sha256(p) != digest]
    if bad:
        raise ValueError(f'Frozen assets changed: {bad}')
    if sha256(C['checkpoint']) != C['checkpoint_sha256']:
        raise ValueError('Frozen G01 checkpoint SHA256 mismatch')
    return len(original)


def raw(split):
    return pd.read_csv(Path(C['raw_root']) / f'{split}.csv', dtype={'Image_name': str})


def audit():
    if (OUT / 'age_audit.json').exists():
        saved = json.loads((OUT / 'age_audit.json').read_text(encoding='utf-8'))
        if saved['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Existing M02 protocol differs')
        for split, digest in saved['raw_csv_sha256'].items():
            if sha256(Path(C['raw_root']) / f'{split}.csv') != digest:
                raise ValueError('Age CSV changed since audit')
        return AgePreprocessor(**yaml.safe_load((OUT / 'age_preprocessing.yaml').read_text(encoding='utf-8'))['parameters'])
    OUT.mkdir(parents=True, exist_ok=True)
    verify_frozen()
    frozen = {str(p): sha256(p) for p in Path('results/G02').rglob('*') if p.is_file()}
    write_json(OUT / 'frozen_assets.json', frozen)
    train = raw('train')
    processor = AgePreprocessor.fit(train.Age, 'official_train')
    report = {'created_utc': utc(), 'protocol_sha256': sha256(PROTOCOL),
              'raw_csv_sha256': {}, 'splits': {}, 'fit_source': 'official_train',
              'anomaly_rule': C['age_rule'], 'test_statistics_diagnostic_only': True,
              'patient_ids_available': False}
    for split in ['train', 'valid', 'test']:
        f = raw(split)
        report['raw_csv_sha256'][split] = sha256(Path(C['raw_root']) / f'{split}.csv')
        row = {'all': age_summary(f.Age, processor), 'columns': list(f.columns)}
        if split != 'test':
            row['by_class'] = {str(y): age_summary(f.loc[f.True_Label == y, 'Age'], processor) for y in [0, 1]}
            row['age_strata'] = []
            groups = pd.Series(age_group(f.Age, processor))
            for group, indices in groups.groupby(groups).groups.items():
                labels = f.iloc[list(indices)].True_Label
                row['age_strata'].append({'group': group, 'n': len(labels), 'positive': int(labels.sum()),
                                          'positive_rate': float(labels.mean())})
        report['splits'][split] = row
    (OUT / 'age_preprocessing.yaml').write_text(yaml.safe_dump({
        'parameters': processor.as_dict(), 'rule': C['age_rule'], 'fit_csv_sha256': report['raw_csv_sha256']['train'],
        'columns': ['normalized_age', 'missing_or_abnormal_age'], 'test_fit': False,
        'note': 'Positive Train age envelope; out-of-envelope ages become missing, no Val/Test fitted clipping.'
    }, sort_keys=False), encoding='utf-8')
    write_json(OUT / 'age_audit.json', report)
    print(json.dumps({'age_audit': 'COMPLETED', 'preprocessing': processor.as_dict()}), flush=True)
    return processor


def features(split):
    if split not in {'train', 'valid'}:
        raise ValueError('Official Test feature extraction forbidden in M02')
    folder = OUT / 'features'; folder.mkdir(exist_ok=True)
    path = folder / f'{split}.npy'; meta_path = folder / f'{split}.json'
    manifest = Path(C['manifest_root']) / f'{split}.csv'
    identity = {'checkpoint_sha256': C['checkpoint_sha256'], 'manifest_sha256': sha256(manifest),
                'precision': 'fp32', 'image_size': 224, 'augmentation': 'none'}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['identity'] != identity or meta['sha256'] != sha256(path):
            raise ValueError('Feature cache identity mismatch')
        return np.load(path, mmap_mode='r')
    if path.exists():
        raise ValueError('Incomplete feature cache; refusing overwrite')
    frame = read_manifest(manifest, C['image_root'])
    raw_frame = raw(split)
    assert list(frame.image_id) == list(raw_frame.Image_name)
    assert np.array_equal(frame.label, raw_frame.True_Label)
    caches = json.loads(Path('outputs/runs/G01/cache.json').read_text())
    cached = Path(caches['train' if split == 'train' else 'val'])
    marker = json.loads((cached / 'complete.json').read_text())
    if marker['manifest_sha256'] != sha256(manifest) or marker['size'] != 224:
        raise ValueError('G01 image cache version mismatch')
    ds = FundusDataset(frame, 224, False, cache_dir=cached)
    source_ds = FundusDataset(frame, 224, False)
    for i in np.random.default_rng(42).choice(len(frame), 16, replace=False):
        if not torch.equal(ds[int(i)][0], source_ds[int(i)][0]):
            raise ValueError('Cached/raw G01 preprocessing mismatch')
    dl = DataLoader(ds, batch_size=C['feature_batch'], shuffle=False, num_workers=C['workers'], pin_memory=True)
    predictor = Predictor(C['checkpoint'], 'cuda', 'fp32')
    model = predictor.model
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Encoder is not frozen')
    partial = path.with_suffix('.partial.npy')
    if partial.exists():
        raise FileExistsError(partial)
    array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(frame), C['feature_dim']))
    logits = np.empty(len(frame), dtype=np.float32)
    start = time.perf_counter(); offset = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch, (x, _) in enumerate(dl):
            x = x.cuda(non_blocking=True)
            h = model.forward_head(model.forward_features(x), pre_logits=True)
            z = model.get_classifier()(h).flatten()
            if h.shape[1] != 768 or not torch.isfinite(h).all():
                raise ValueError('Invalid image features')
            if batch == 0 and not torch.allclose(z, model(x).flatten(), atol=1e-6, rtol=1e-5):
                raise ValueError('Pooled features do not reproduce G01 classifier')
            array[offset:offset+len(x)] = h.cpu().numpy()
            logits[offset:offset+len(x)] = z.cpu().numpy()
            offset += len(x)
            if batch % 128 == 0:
                print(json.dumps({'feature_split': split, 'done': offset, 'total': len(frame),
                                  'seconds': round(time.perf_counter()-start, 1)}), flush=True)
    array.flush(); del array
    partial.rename(path)
    np.save(folder / f'{split}_g01_logits.npy', logits)
    write_json(meta_path, {'identity': identity, 'sha256': sha256(path), 'rows': offset,
                          'seconds': time.perf_counter()-start,
                          'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
    del model, predictor; gc.collect(); torch.cuda.empty_cache()
    return np.load(path, mmap_mode='r')


def head_logits(head, images, ages):
    head.eval(); values = []
    with torch.inference_mode():
        for start in range(0, len(images), 1024):
            joined = np.concatenate([images[start:start+1024], ages[start:start+1024]], axis=1)
            values.extend(head(torch.from_numpy(joined).cuda()).flatten().float().cpu().tolist())
    return np.asarray(values)


def fit_head(images, ages, labels, lr, epochs, history_path, holdout=None):
    seed_everything(42)
    head = FusionHead(C['feature_dim']).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=C['head']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C['head']['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    dataset = TensorDataset(torch.from_numpy(np.concatenate([images, ages], 1)), torch.tensor(labels, dtype=torch.float32))
    dl = DataLoader(dataset, batch_size=C['head']['batch_size'], shuffle=True,
                    generator=torch.Generator().manual_seed(42), num_workers=0)
    best = (-1., -1.); best_epoch = 0; stale = 0; history = []
    for epoch in range(epochs):
        head.train(); total = 0.; start = time.perf_counter()
        for x, y in dl:
            x, y = x.cuda(), y.cuda(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                z = head(x).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite fusion loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.detach())*len(y)
        scheduler.step()
        row = {'epoch': epoch+1, 'train_bce': total/len(labels), 'seconds': time.perf_counter()-start}
        if holdout is not None:
            vx, va, vy = holdout
            z = head_logits(head, vx, va)
            score = metrics(vy, torch.sigmoid(torch.tensor(z)).numpy())
            row.update(internal_auroc=score['auroc'], internal_auprc=score['auprc'])
            key = (score['auroc'], score['auprc'])
            if key > best:
                best = key; best_epoch = epoch+1; stale = 0
            else:
                stale += 1
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(json.dumps({'phase': history_path.stem, 'lr': lr, **row}), flush=True)
        if holdout is not None and stale >= C['head']['patience']:
            break
    return head.eval(), {'lr': lr, 'best_epoch': best_epoch, 'internal_auroc': best[0], 'internal_auprc': best[1]}


def train_models(processor, images):
    train = raw('train'); y = train.True_Label.to_numpy()
    root = OUT / 'training'; root.mkdir(exist_ok=True)
    if (root / 'selection_locked.json').exists():
        raise FileExistsError('Selection already locked; do not repeat fitting')
    indices = np.arange(len(train))
    fit_idx, hold_idx = train_test_split(indices, test_size=C['internal_holdout_fraction'], random_state=42, stratify=y)
    pd.DataFrame({'Image_name': train.Image_name, 'internal_holdout': np.isin(indices, hold_idx).astype(int)}).to_csv(root/'internal_split.csv', index=False)
    internal_processor = AgePreprocessor.fit(train.Age.iloc[fit_idx], 'internal_train')
    internal_ages = internal_processor.apply(train.Age)
    write_json(root/'internal_age_preprocessing.json', internal_processor.as_dict())
    trials = []
    for lr in C['head']['learning_rates']:
        head, summary = fit_head(images[fit_idx], internal_ages[fit_idx], y[fit_idx], lr,
                                 C['head']['epochs'], root/f'lr_{lr}.csv',
                                 (images[hold_idx], internal_ages[hold_idx], y[hold_idx]))
        trials.append(summary); del head; gc.collect(); torch.cuda.empty_cache()
    selected = max(trials, key=lambda x: (x['internal_auroc'], x['internal_auprc']))
    write_json(root/'selection_locked.json', {'locked_utc': utc(), 'trials': trials, 'selected': selected,
               'official_val_used': False, 'internal_split_sha256': sha256(root/'internal_split.csv'),
               'limitation': 'G01 encoder saw all Train labels; holdout only isolates fusion-head fitting, no patient IDs.'})
    age = processor.apply(train.Age)
    lr_model = LogisticRegression(C=C['logistic']['C'], solver='lbfgs', max_iter=C['logistic']['max_iter'], random_state=42)
    lr_model.fit(age, y)
    if int(lr_model.n_iter_.max()) >= C['logistic']['max_iter']:
        raise ValueError('Logistic regression did not converge')
    write_json(root/'M02b.json', {'coef': lr_model.coef_.flatten().tolist(), 'intercept': float(lr_model.intercept_[0]),
               'age_preprocessing': processor.as_dict(), 'n_iter': lr_model.n_iter_.tolist(), 'train_only': True})
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    head, _ = fit_head(images, age, y, selected['lr'], selected['best_epoch'], root/'refit_history.csv')
    original = torch.load(C['checkpoint'], map_location='cpu', weights_only=True)
    torch.save({'backbone_state_dict': original['state_dict'], 'model_config': original['config']['model'],
                'head_state_dict': {k: v.cpu() for k,v in head.state_dict().items()},
                'age_preprocessing': processor.as_dict(), 'image_dim': 768,
                'G01_sha256': C['checkpoint_sha256'], 'protocol': C, 'selected': selected,
                'provenance': original['provenance'], 'official_val_used_for_head_selection': False}, root/'M02c.pth')
    write_json(root/'refit_resources.json', {'seconds': time.perf_counter()-start,
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
    del head, original; gc.collect(); torch.cuda.empty_cache()


def ranking(reference, candidate, labels):
    pos = np.flatnonzero(labels == 1); neg = np.flatnonzero(labels == 0)
    a = np.sign(reference[pos,None]-reference[neg][None,:])
    b = np.sign(candidate[pos,None]-candidate[neg][None,:])
    return {'positive_negative_pairs': int(a.size), 'changed_pair_orders_including_ties': int((a != b).sum()),
            'corrected_pair_orders': int(((a <= 0) & (b > 0)).sum()),
            'newly_wrong_pair_orders': int(((a > 0) & (b <= 0)).sum())}


def evaluate(processor, images):
    val = raw('valid'); labels = val.True_Label.to_numpy()
    reference = pd.read_csv('outputs/runs/G01/validation.csv')
    assert list(reference.image_id) == list(val.Image_name)
    assert np.array_equal(reference.label, labels)
    age = processor.apply(val.Age)
    age_model = json.loads((OUT/'training/M02b.json').read_text())
    age_z = age @ np.asarray(age_model['coef']) + age_model['intercept']
    obj = torch.load(OUT/'training/M02c.pth', map_location='cpu', weights_only=True)
    head = FusionHead(768).cuda(); head.load_state_dict(obj['head_state_dict']); head.eval()
    fusion_z = head_logits(head, images, age)
    zs = {'M02a': reference.logit.to_numpy(), 'M02b': age_z, 'M02c': fusion_z}
    summary = {}
    group_names = age_group(val.Age, processor)
    for eid, z in zs.items():
        path = Path(f'results/{eid}/seed_42/fold_none'); path.mkdir(parents=True, exist_ok=False)
        probability = reference.probability.to_numpy() if eid == 'M02a' else torch.sigmoid(torch.tensor(z)).numpy()
        pred = pd.DataFrame({'image_id': val.Image_name, 'label': labels, 'logit': z,
                             'probability': probability, 'experiment_id': eid})
        pred.to_csv(path/'validation.csv', index=False)
        score = metrics(labels, probability)
        ci = paired_bootstrap(reference, pred, repeats=C['bootstrap_repeats'], seed=42)
        strata = {}
        for group in C['age_strata']:
            mask = group_names == group; yy = labels[mask]
            npos = int(yy.sum()); nneg = int(len(yy)-npos)
            strata[group] = {'n': len(yy), 'positive': npos, 'negative': nneg,
                             'auroc': metrics(yy, probability[mask])['auroc'] if npos >= C['stratum_min_positive'] and nneg >= C['stratum_min_negative'] else None}
        score.update(paired_bootstrap=ci, age_strata=strata,
                     ranking=ranking(reference.probability.to_numpy(), probability, labels))
        write_json(path/'metrics.json', score)
        config = yaml.safe_load(Path(f'configs/multimodal/{eid}.yaml').read_text())
        (path/'config.yaml').write_text(yaml.safe_dump({'experiment': config, 'protocol': C}, sort_keys=False), encoding='utf-8')
        if eid == 'M02a':
            write_json(path/'checkpoint_reference.json', {'path': C['checkpoint'], 'sha256': C['checkpoint_sha256']})
        else:
            shutil.copy2(OUT/f'training/{eid}.{ "json" if eid == "M02b" else "pth"}', path/('model.json' if eid == 'M02b' else 'best.pth'))
        summary[eid] = score
    robustness = {}
    rng = np.random.default_rng(42)
    perturbations = {'all_age_missing': [None]*len(val), 'all_age_abnormal': [1e6]*len(val),
                     'age_shuffled_diagnostic_only': val.Age.to_numpy()[rng.permutation(len(val))]}
    for name, values in perturbations.items():
        a = processor.apply(values)
        z = head_logits(head, images, a)
        lp = torch.sigmoid(torch.tensor(a @ np.asarray(age_model['coef']) + age_model['intercept'])).numpy()
        robustness[name] = {'M02c': metrics(labels, torch.sigmoid(torch.tensor(z)).numpy()),
                            'M02b': metrics(labels, lp), 'used_for_selection': False}
    write_json(OUT/'robustness.json', robustness)
    write_json(OUT/'comparison.json', summary)
    del head, obj; gc.collect(); torch.cuda.empty_cache()
    return summary


def benchmark():
    val = read_manifest(Path(C['manifest_root'])/'valid.csv', C['image_root'])
    ages = dict(zip(raw('valid').Image_name, raw('valid').Age))
    chosen = np.random.default_rng(42).choice(len(val), 105, replace=False)
    warm, samples = val.iloc[chosen[:5]], val.iloc[chosen[5:]]
    reports = {}
    for eid in ['M02a', 'M02b', 'M02c']:
        result = Path(f'results/{eid}/seed_42/fold_none')
        predictor = Predictor(C['checkpoint'], 'cuda') if eid == 'M02a' else M02Predictor(result/('model.json' if eid == 'M02b' else 'best.pth'), ages)
        predictions = pd.read_csv(result/'validation.csv').set_index('image_id')
        timings = []; differences = []
        with (result/'latency_4060_predictions.csv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f); writer.writerow(COLUMNS)
            for row in warm.itertuples():
                predictor.predict_one(row.resolved_path, row.image_id, writer, f)
            for repeat in range(3):
                values = []
                for row in samples.itertuples():
                    predictor.synchronize(); start = time.perf_counter_ns()
                    p = predictor.predict_one(row.resolved_path, row.image_id, writer, f)
                    predictor.synchronize(); values.append((time.perf_counter_ns()-start)/1e6)
                    differences.append(abs(float(p)-float(predictions.loc[row.image_id, 'probability'])))
                timings.append(values)
        size = Path(C['checkpoint']).stat().st_size if eid == 'M02a' else (result/('model.json' if eid == 'M02b' else 'best.pth')).stat().st_size
        report = {'gpu': torch.cuda.get_device_name(), 'mean_ms': float(np.mean(timings)),
                  'p95_ms': float(np.percentile(timings,95)), 'latency_ms_by_round': timings,
                  'sample_ids': samples.image_id.tolist(), 'model_bytes': size,
                  'max_probability_difference_vs_cached_val': max(differences), 'T4_measured': False,
                  'scope': 'FP32 batch1, raw image decode + age lookup/preprocess + model + CSV write/flush, 5 warmup, 100 samples x3'}
        write_json(result/'latency_4060.json', report); reports[eid] = report
        print(json.dumps({'latency': eid, 'mean_ms': report['mean_ms'], 'model_bytes': size}), flush=True)
        del predictor; gc.collect(); torch.cuda.empty_cache()
    return reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['audit', 'run']); args = p.parse_args()
    torch.set_num_threads(4)
    processor = audit()
    if args.action == 'audit':
        return
    if not torch.cuda.is_available():
        raise RuntimeError('M02 feature extraction requires the authorized local CUDA GPU')
    if (OUT/'run_manifest.json').exists():
        raise FileExistsError('M02 run exists; do not repeat or overwrite')
    verify_frozen(); verified_weights('configs/weights_g01.local.json', 'convnext_tiny')
    source_files = [Path('src/multimodal.py'), Path('scripts/run_m02.py'), PROTOCOL]
    snapshot = OUT/'source_snapshot'; snapshot.mkdir(exist_ok=False)
    for source in source_files:
        shutil.copy2(source, snapshot/source.name)
    manifest = {'status': 'RUNNING', 'started_utc': utc(), 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'worktree_dirty_at_start': bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                'code_identity': 'Base Git commit plus source_snapshot and source_sha256; no claim of clean commit',
                'source_sha256': {str(p): sha256(p) for p in source_files}, 'G01_sha256': C['checkpoint_sha256'],
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(), 'test_used_for_training': False, 'official_val_used_for_selection': False}
    write_json(OUT/'run_manifest.json', manifest)
    try:
        train_x = features('train')
        train_models(processor, train_x)
        del train_x
        val_x = features('valid')
        replay = torch.sigmoid(torch.tensor(np.load(OUT/'features/valid_g01_logits.npy'))).numpy()
        reference = pd.read_csv('outputs/runs/G01/validation.csv').probability.to_numpy()
        if np.abs(replay-reference).max() > 1e-3:
            raise ValueError('G01 replay differs materially from frozen predictions')
        write_json(OUT/'G01_replay.json', {'max_absolute_probability_difference': float(np.abs(replay-reference).max()),
                   'samples': len(replay)})
        evaluate(processor, val_x)
        benchmark()
        manifest.update(status='COMPLETED', finished_utc=utc(), frozen_files_verified=verify_frozen())
        write_json(OUT/'run_manifest.json', manifest)
        for eid in ['M02a','M02b','M02c']:
            folder = Path(f'results/{eid}/seed_42/fold_none')
            write_json(folder/'run_manifest.json', {**manifest, 'experiment_id': eid})
            if eid == 'M02c':
                shutil.copy2(OUT/'training/refit_history.csv', folder/'train_history.csv')
            files = sorted(p for p in folder.iterdir() if p.is_file())
            (folder/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files), encoding='utf-8')
    except Exception as exc:
        manifest.update(status='FAILED', error=repr(exc), finished_utc=utc())
        write_json(OUT/'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
