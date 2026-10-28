import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.train import train
from src.utils import load_config, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/smoke')
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    base = Path(a.output).resolve()
    base.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026)
    root = base / 'synthetic'
    root.mkdir()
    for split, count in [('train', 4), ('val', 4), ('test', 2), ('warmup', 1)]:
        folder = root / split
        folder.mkdir()
        rows = []
        for i in range(count):
            name = f'{split}_{i}.png'
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(folder / name)
            rows.append({'image_id': name, 'path': f'{split}/{name}', 'label': i % 2, 'patient_id': f'{split}_patient_{i}'})
        pd.DataFrame(rows).to_csv(base / f'{split}.csv', index=False)
    c = load_config('configs/baseline.yaml')
    c.update(smoke=True, epochs=1, batch_size=2, image_size=64, device='cpu', output=str(base / 'run'))
    c['data'] = {'source': 'synthetic', 'root': str(root), 'train_csv': str(base / 'train.csv'), 'val_csv': str(base / 'val.csv')}
    write_json(base / 'smoke_config.json', c)
    checkpoint = train(c)
    submission = base / 'submission.csv'
    subprocess.run([sys.executable, 'predict.py', '--checkpoint', str(checkpoint), '--csv', str(base / 'test.csv'),
        '--root', str(root), '--output', str(submission), '--device', 'cpu', '--allow-smoke'], check=True)
    subprocess.run([sys.executable, 'cost.py', str(submission), str(root / 'test'), str(root / 'warmup'),
        '--checkpoint', str(checkpoint), '--device', 'cpu', '--smoke', '--output', str(base / 'cost')], check=True)
    write_json(base / 'result.json', {'status': 'passed', 'data': 'synthetic random pixels only',
        'formal_training_started': False, 'official_t4_certified': False})


if __name__ == '__main__':
    main()
