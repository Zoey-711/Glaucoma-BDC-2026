import csv
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.audit import audit
from src.data import read_manifest
from src.metrics import metrics
from src.model import verified_weights
from src.submission import validate_submission
from cost import consistency


def fixture_data(tmp_path):
    for i in range(4):
        Image.new('RGB', (12 + i, 14), (i, 20, 30)).save(tmp_path / f'{i}.png')
    for s, ids in [('train', [0, 1]), ('val', [2, 3])]:
        pd.DataFrame({'image_id': [f'{i}.png' for i in ids], 'path': [f'{i}.png' for i in ids],
            'label': [0, 1], 'patient_id': [f'p{i}' for i in ids]}).to_csv(tmp_path / f'{s}.csv', index=False)
    return tmp_path / 'train.csv', tmp_path / 'val.csv', tmp_path


def test_audit_clean_and_leak(tmp_path):
    a, b, root = fixture_data(tmp_path)
    r = audit(a, b, root)
    assert r['passed'] and r['splits']['train']['positive_ratio'] == .5
    Image.open(root / '0.png').save(root / '2.png')
    df = pd.read_csv(b)
    df.loc[0, 'patient_id'] = 'p0'
    df.to_csv(b, index=False)
    r = audit(a, b, root)
    assert not r['passed'] and r['cross_split_duplicate_groups']
    assert r['patient_check']['overlap'] == ['p0']


def test_missing_corrupt_and_no_patient(tmp_path):
    a, b, root = fixture_data(tmp_path)
    (root / '0.png').unlink()
    (root / '1.png').write_bytes(b'not an image')
    df = pd.read_csv(b).drop(columns=['patient_id'])
    df.to_csv(b, index=False)
    r = audit(a, b, root)
    assert not r['passed'] and r['splits']['train']['missing'] == ['0.png']
    assert r['splits']['train']['corrupt']
    assert r['patient_check']['status'] == 'unavailable_or_incomplete'


def test_manifest_rejects_escape_and_labels(tmp_path):
    with pytest.raises(ValueError):
        read_manifest(None, None)
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'image_id': ['a'], 'path': ['../outside.png'], 'label': [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='outside root'):
        read_manifest(path, tmp_path)
    pd.DataFrame({'image_id': ['a'], 'path': ['a.png'], 'label': [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Labels'):
        read_manifest(path, tmp_path)


def test_metrics():
    r = metrics([0, 0, 1, 1], [.1, .2, .8, .9])
    assert all(r[x] == 1 for x in ['auroc', 'auprc', 'recall', 'specificity'])
    with pytest.raises(ValueError):
        metrics([0, 0], [.1, .2])


@pytest.mark.parametrize('prob,label', [(float('nan'), 0), (1.1, 1), (.7, 0)])
def test_submission_invalid(tmp_path, prob, label):
    p = tmp_path / 'sub.csv'
    pd.DataFrame({'Image_name': ['a'], 'Probability': [prob], 'Pred_Label': [label]}).to_csv(p, index=False)
    with pytest.raises((ValueError, TypeError)):
        validate_submission(p, ['a'])


def test_submission_ids(tmp_path):
    p = tmp_path / 'sub.csv'
    pd.DataFrame({'Image_name': ['001.png', '002.png'], 'Probability': [.2, .8], 'Pred_Label': [0, 1]}).to_csv(p, index=False)
    assert len(validate_submission(p, ['001.png', '002.png'])) == 2
    with pytest.raises(ValueError):
        validate_submission(p, ['002.png', '001.png'])


def test_cost_ordering():
    a = pd.DataFrame({'Image_name': ['a', 'b'], 'Probability': [.8001, .8002], 'Pred_Label': [1, 1]})
    b = a.copy()
    b['Probability'] = [.8002, .8001]
    assert consistency(a, b) == 2
    assert consistency(a, a) == 0


def test_unreviewed_weights_blocked():
    with pytest.raises(ValueError, match='not approved'):
        verified_weights('configs/weights_manifest.template.json')


def test_weight_hash_tampering(tmp_path):
    from src.utils import sha256
    weights, license = tmp_path / 'weights.pt', tmp_path / 'license.txt'
    weights.write_bytes(b'synthetic test bytes, not a real weight')
    license.write_text('synthetic test license', encoding='utf-8')
    m = json.loads(__import__('pathlib').Path('configs/weights_manifest.template.json').read_text())
    m.update(status='approved', allows_competition_and_redistribution=True,
        license_url='https://example.invalid/test-only', reviewer='test fixture', review_date='2026-10-28',
        path=str(weights), sha256=sha256(weights), license_file=str(license), license_sha256=sha256(license))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(m))
    assert verified_weights(path)['sha256'] == sha256(weights)
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA256'):
        verified_weights(path)


def test_compliance_xlsx(tmp_path):
    import zipfile
    import xml.etree.ElementTree as ET
    from scripts.compliance import xlsx
    path = tmp_path / 'materials.xlsx'
    xlsx(path, [['字段', '值'], ['来源', '尚未提供 & 未核验']])
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name in z.namelist():
            ET.fromstring(z.read(name))


def test_frozen_inference(tmp_path):
    import torch
    from src.inference import Predictor
    from src.model import build_model
    torch.set_num_threads(2)
    model = build_model(allow_random=True)
    path = tmp_path / 'smoke.pt'
    torch.save({'state_dict': model.state_dict(), 'smoke': True,
        'config': {'threshold': .5, 'image_size': 64}}, path)
    with pytest.raises(ValueError, match='official submission'):
        Predictor(path, 'cpu')
    predictor = Predictor(path, 'cpu', allow_smoke=True)
    before = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    image = tmp_path / 'a.png'
    Image.new('RGB', (64, 64), 'red').save(image)
    with (tmp_path / 'pred.csv').open('w', newline='') as f:
        w = csv.writer(f)
        p = predictor.predict_one(image, 'a', w, f)
        q = predictor.predict_one(image, 'a', w, f)
    assert p == q
    assert not predictor.model.training and not any(x.requires_grad for x in predictor.model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in predictor.model.state_dict().items())
