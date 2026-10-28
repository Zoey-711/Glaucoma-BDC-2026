"""Read-only source/data review; writes draft plans only. Never loads a model."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import open_rgb, transform
from src.metrics import metrics
from src.utils import sha256, write_json


def main():
    out = Path('outputs/migration_review')
    out.mkdir(exist_ok=False)
    frozen = [*Path('outputs/runs/B02').rglob('*'), *Path('submission/round_01_B02').rglob('*')]
    frozen = {str(p): sha256(p) for p in frozen if p.is_file()}
    write_json(out / 'frozen_inventory.json', frozen)
    old = Path('E:/计挑赛/cxr-multilabel-challenge')
    sources = ['AGENTS.md', 'src/cxr/models/factory.py', 'scripts/train.py',
               'src/cxr/engine/early_stopping.py', 'src/cxr/utils/training_resume.py',
               'src/cxr/batch/plan.py', 'src/cxr/batch/delivery.py',
               'src/cxr/batch/results.py', 'src/cxr/utils/cloud_paths.py',
               'configs/m01_convnext_tiny_384.yaml', 'configs/t01_swin_tiny_384.yaml',
               'configs/n01_efficientnet_v2_s_384.yaml']
    evidence = {s: sha256(old / s) for s in sources}
    write_json(out / 'source_inventory.json', {'root': str(old), 'files': evidence,
                                             'checkpoint_loaded': False, 'copied_code': False})
    torch.set_num_threads(4)
    audit = json.loads(Path('outputs/official_ingest/audit.json').read_text(encoding='utf-8'))
    cache = json.loads(Path('outputs/runs/B02/cache.json').read_text(encoding='utf-8'))
    split_checks = {}
    frames = {}
    for split, original in [('train', 'train'), ('val', 'valid')]:
        path = Path(f'data/official/manifests/{original}.csv')
        f = pd.read_csv(path)
        raw = pd.read_csv(f'data/official/glaucoma_first/{original}.csv')
        assert list(raw.Image_name) == list(f.image_id)
        assert np.array_equal(raw.True_Label, f.label)
        assert set(f.label) == {0, 1}
        assert sha256(path) == audit['manifest_sha256'][split]
        missing = [str(p) for p in f.path if not (Path('data/official') / p).is_file()]
        assert not missing
        ids = np.random.default_rng(42).choice(len(f), 16, replace=False)
        for i in ids:
            direct = transform(224)(open_rgb(Path('data/official') / f.iloc[i].path))
            cached = transform(224, preprocessed=True)(open_rgb(Path(cache[split]) / f'{i:06d}.png'))
            assert torch.equal(direct, cached)
        split_checks[split] = {'count': len(f), 'positives': int(f.label.sum()),
            'original_labels_and_order_match': True, 'manifest_sha256': sha256(path),
            'missing_files': len(missing), 'cached_vs_direct_exact_matches': len(ids),
            'patient_id_present': 'patient_id' in raw.columns,
            'source_columns': list(raw.columns)}
        frames[split] = f
    pred = pd.read_csv('outputs/runs/B02/validation.csv')
    assert list(pred.image_id) == list(frames['val'].image_id)
    assert np.array_equal(pred.label, frames['val'].label)
    assert np.allclose(1 / (1 + np.exp(-pred.logit)), pred.probability, atol=1e-7)
    recalculated = metrics(pred.label, pred.probability)
    history = json.loads(Path('outputs/runs/B02/history.json').read_text())
    best = max(history, key=lambda r: (r['auroc'], r['auprc']))
    assert abs(recalculated['auroc'] - best['auroc']) < 1e-12
    assert abs(recalculated['auprc'] - best['auprc']) < 1e-12
    write_json(out / 'audit_review.json', {'splits': split_checks, 'metrics_recalculated': recalculated,
        'best_epoch': best['epoch'], 'saved_predictions_match_labels': True,
        'historical_full_decoded_audit': {'path': 'outputs/official_ingest/audit.json',
            'sha256': sha256('outputs/official_ingest/audit.json'),
            'duplicate_groups': len(audit['duplicate_groups']),
            'cross_split_duplicate_groups': len(audit['cross_split_duplicate_groups'])},
        'limits': ['Full image-content audit not repeated in this planning review',
                   'No official patient IDs: patient leakage cannot be excluded',
                   'Perceptual near-duplicate audit remains pending',
                   '32 sampled cache comparisons do not prove full cache integrity'],
        'official_test_result_user_reported': {'auroc': .9732, 'rank': '600+ / 700+',
                                              'independently_verified': False}})
    base = yaml.safe_load(Path('configs/experiments/B02.yaml').read_text())
    candidates = [('B02-320', 'efficientnet_b0', 320), ('B02-384', 'efficientnet_b0', 384),
                  ('B02-448', 'efficientnet_b0', 448),
                  ('ConvNeXt-Tiny-224', 'convnext_tiny', 224),
                  ('ConvNeXt-Tiny-384', 'convnext_tiny', 384),
                  ('EfficientNetV2-S-384', 'efficientnet_v2_s', 384),
                  ('DenseNet-121-384', 'densenet121', 384), ('Swin-Tiny-384', 'swin_t', 384)]
    folder = Path('configs/experiments/migration_drafts')
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for eid, name, size in candidates:
        c = copy.deepcopy(base)
        c.update(experiment_id=eid, image_size=size, output=f'outputs/runs/{eid}')
        c['data'].update(root='data/official', train_csv='data/official/manifests/train.csv',
                         val_csv='data/official/manifests/valid.csv')
        if name != 'efficientnet_b0':
            c['model'] = {'name': name, 'backend': 'torchvision', 'architecture': name,
                          'weights_manifest': None}
        c['plan'] = {'stage': 'migration', 'kind': 'migration_draft', 'reference': 'B02',
                     'initial_status': 'blocked', 'factor': 'resolution' if name == 'efficientnet_b0' else 'architecture_and_pretraining',
                     'reason': 'Review-only draft; implementation, provenance review, resource check and explicit authorization required'}
        c['migration'] = {'enabled': False, 'num_classes': 1, 'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR', 'gradient_accumulation_steps': 1,
            'effective_batch_size': 32, 'validation_precision': 'fp32', 'inference_batch_size': 1,
            'weights_status': 'existing_B02_review' if name == 'efficientnet_b0' else 'pending_new_review',
            'paired_resolution_reference': 'ConvNeXt-Tiny-224' if eid == 'ConvNeXt-Tiny-384' else None,
            'priority': 1 if eid == 'ConvNeXt-Tiny-384' else 2,
            'allow_cxr_checkpoint': False, 'allow_automatic_download': False}
        p = folder / f'{eid}.yaml'
        p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding='utf-8')
        records.append({'experiment_id': eid, 'config': str(p), 'config_sha256': sha256(p),
            'status': 'draft_blocked', 'AUROC': None, 'AUPRC': None, 'delta_auroc': None,
            'ci95_low': None, 'ci95_high': None, 'model_size_mb': None, 'local_latency_ms': None,
            't4_latency_ms': None, 'train_peak_allocated_mib': None, 'train_peak_reserved_mib': None,
            'inference_peak_allocated_mib': None})
    pd.DataFrame(records).to_csv(out / 'candidate_plan.csv', index=False)
    write_json(out / 'collaboration_contract.json', {
        'status': 'design_only_not_implemented', 'auto_start': False, 'seed': 42,
        'data_archive_sha256': json.loads(Path('outputs/official_ingest/summary.json').read_text())['archive_sha256'],
        'split_sha256': audit['manifest_sha256'],
        'code_sha256': {p: sha256(p) for p in ['src/data.py', 'src/metrics.py', 'src/bootstrap.py',
                                            'src/experiment_training.py', 'src/inference.py']},
        'runtime_lock_sha256': sha256('requirements-training.lock.txt'),
        'reference_predictions_sha256': sha256('outputs/runs/B02/validation.csv'),
        'tasks': records, 'machines': {'machine_a': None, 'machine_b': None},
        'forbid_shared_sqlite_over_network': True, 'import_requires_hash_and_prediction_alignment': True})
    assert all(sha256(p) == digest for p, digest in frozen.items())
    assert all(sha256(old / p) == digest for p, digest in evidence.items())
    write_json(out / 'verification.json', {'frozen_files_unchanged': len(frozen),
        'inspected_old_sources_unchanged': len(evidence), 'draft_configs': len(records),
        'training_started': False, 'checkpoint_loaded': False, 'new_weights_downloaded': False})
    print(json.dumps({'metrics': recalculated, 'configs': len(records), 'frozen_files': len(frozen)}))


if __name__ == '__main__':
    main()
