import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.create_experiments import matrix
from src.bootstrap import paired_bootstrap
from src.losses import BinaryLoss
from src.metrics import metrics
from src.registry import Registry, training_hash


def test_matrix_control_variables():
    configs = {c['experiment_id']: c for c in matrix()}
    assert len(configs) == 30
    base = configs['B02']
    for eid, c in configs.items():
        if eid[0] in 'BIP': assert c['seed'] == 42
        if eid[0] in 'BI':
            assert c['image_size'] == base['image_size']
            for key in ['batch_size', 'lr', 'epochs', 'augmentation', 'preprocess']:
                assert c[key] == base[key]
    assert training_hash(base) == training_hash(configs['I01']) == training_hash(configs['P01'])
    assert configs['P05']['plan']['initial_status'] == configs['P06']['plan']['initial_status'] == 'skipped'


def test_registry_deduplicates_and_rejects_mutations(tmp_path):
    reg = Registry(tmp_path / 'registry.sqlite')
    a = next(c for c in matrix() if c['experiment_id'] == 'B02')
    reg.register(a); assert reg.claim('B02')
    with pytest.raises(ValueError): reg.claim('B02')
    reg.update('B02', status='completed', AUROC=.7, AUPRC=.2)
    assert not reg.claim('B02')
    b = copy.deepcopy(a); b['experiment_id'] = 'I01'; b['output'] = str(tmp_path / 'alias')
    reg.register(b); assert not reg.claim('I01')
    assert reg.get('I01')['status'] == 'reused'
    assert reg.get('I01')['t4_latency_ms'] is None
    a['lr'] *= 2
    with pytest.raises(ValueError, match='config changed'): reg.register(a)


@pytest.mark.parametrize('name', ['bce', 'weighted_bce', 'focal', 'asl', 'class_balanced'])
def test_losses_finite_backward(name):
    x = torch.tensor([-20., -.3, .8, 20.], requires_grad=True)
    y = torch.tensor([0., 1., 0., 1.])
    loss = BinaryLoss({'name': name}, [100, 3])(x, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_bce_matches_torch():
    x = torch.tensor([-2., 2.]); y = torch.tensor([0., 1.])
    assert torch.allclose(BinaryLoss({'name': 'bce'}, [2, 1])(x, y), torch.nn.functional.binary_cross_entropy_with_logits(x, y))


def test_bootstrap_pairing_and_clusters():
    df = pd.DataFrame({'image_id': ['a', 'b', 'c', 'd'], 'label': [0, 1, 0, 1],
                       'probability': [.1, .8, .2, .9], 'patient_id': ['p1', 'p1', 'p2', 'p2']})
    result = paired_bootstrap(df, df.iloc[::-1], repeats=40)
    assert result['delta_auroc'] == result['ci95_low'] == result['ci95_high'] == 0
    assert result['unit'] == 'patient_cluster'
    with pytest.raises(ValueError, match='identical sample IDs'): paired_bootstrap(df, df.iloc[:2])
    altered = df.copy(); altered.loc[0, 'label'] = 1
    with pytest.raises(ValueError, match='labels disagree'): paired_bootstrap(df, altered)


def test_sensitivity90():
    result = metrics([0, 0, 1, 1], [.1, .2, .3, .4])
    assert result['sensitivity_at_90_specificity'] == 1


def test_local_report_cannot_certify_t4(tmp_path):
    from scripts.deployment import import_t4
    reg = Registry(tmp_path / 'r.sqlite'); c = matrix()[1]; reg.register(c)
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'official_protocol': False, 'gpu': 'RTX 4060', 'mean_ms': 10}))
    with pytest.raises(ValueError, match='T4'): import_t4(reg, c['experiment_id'], report)


def test_bundle_uses_frozen_shared_inference_and_deduplicates_files(tmp_path):
    from PIL import Image
    from src.model import build_model
    from src.inference import Predictor
    torch.set_num_threads(2)
    model = build_model(allow_random=True)
    checkpoint = tmp_path / 'synthetic.pt'
    torch.save({'state_dict': model.state_dict(), 'smoke': True, 'config': {'threshold': .5, 'image_size': 64}}, checkpoint)
    bundle = tmp_path / 'bundle.json'
    bundle.write_text(json.dumps({'members': ['synthetic.pt', 'synthetic.pt'], 'weights': [.5, .5], 'tta': ['identity', 'hflip']}))
    single = Predictor(checkpoint, 'cpu', allow_smoke=True)
    ensemble = Predictor(bundle, 'cpu', allow_smoke=True)
    image = Image.new('RGB', (64, 64), 'red')
    assert abs(single.predict_image(image) - ensemble.predict_image(image)) < 1e-7
    assert len(ensemble.last_logits) == 4
    assert len(ensemble.metadata['inference_files']) == 2
    assert all(not c.model.training for c in ensemble.children)


def test_cache_matches_raw_preprocessing(tmp_path):
    from PIL import Image
    from src.cache import build_cache
    from src.data import FundusDataset, read_manifest
    image = np.random.default_rng(42).integers(0, 256, (79, 91, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / 'a.png')
    manifest = tmp_path / 'val.csv'
    pd.DataFrame({'image_id': ['a.png'], 'path': ['a.png'], 'label': [1]}).to_csv(manifest, index=False)
    frame = read_manifest(manifest, tmp_path)
    cache = build_cache(frame, manifest, 64, 'resize', 'val', root=tmp_path / 'cache')
    raw, _ = FundusDataset(frame, 64)[0]
    cached, _ = FundusDataset(frame, 64, cache_dir=cache)[0]
    assert torch.equal(raw, cached)
