import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from scripts.create_experiments import matrix
from scripts.experiments import run
from src.analysis import analyze
from src.registry import Registry
from src.utils import write_json


def main():
    base = Path('outputs/smoke_experiments')
    base.mkdir(parents=True, exist_ok=False)
    root = base / 'images'; root.mkdir()
    rng = np.random.default_rng(42)
    for split in ['train', 'val']:
        rows = []
        for i in range(4):
            name = f'{split}_{i}.png'
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / name)
            rows.append({'image_id': name, 'path': name, 'label': i % 2, 'patient_id': f'{split}_{i}'})
        pd.DataFrame(rows).to_csv(base / f'{split}.csv', index=False)
    c = copy.deepcopy(next(c for c in matrix() if c['experiment_id'] == 'B02'))
    c.update(experiment_id='SMOKE_B02', output=str(base / 'run'), device='cuda' if torch.cuda.is_available() else 'cpu',
        smoke=True, epochs=1, batch_size=2, num_workers=0, image_size=64, cache=False)
    c['model']['weights_manifest'] = None
    c['data'] = {'source': 'synthetic', 'root': str(root.resolve()), 'train_csv': str(base / 'train.csv'), 'val_csv': str(base / 'val.csv')}
    config = base / 'config.yaml'; config.write_text(yaml.safe_dump(c), encoding='utf-8')
    reg = Registry(base / 'registry.sqlite')
    run(config, reg)
    before = (base / 'run/best.pt').stat().st_mtime_ns
    run(config, reg)
    assert (base / 'run/best.pt').stat().st_mtime_ns == before
    pred = pd.read_csv(base / 'run/validation.csv')
    assert {'logit', 'probability', 'label', 'experiment_id'}.issubset(pred)
    assert reg.get('SMOKE_B02')['t4_latency_ms'] is None
    analyze(reg, str(base / 'analysis'))
    write_json(base / 'result.json', {'status': 'passed', 'gpu_smoke': torch.cuda.is_available(), 'formal_experiments_launched': [],
        'checks': ['one-epoch synthetic train/validation', 'raw logits', 'registry', 'analysis plots', 'local end-to-end timing', 'duplicate run prevented']})


if __name__ == '__main__': main()
