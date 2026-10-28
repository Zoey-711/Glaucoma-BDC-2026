import csv
import time
from pathlib import Path

import numpy as np

from .data import read_manifest
from .inference import Predictor
from .submission import COLUMNS
from .utils import write_json


def local_benchmark(checkpoint, config, output, precision='fp32', count=100):
    """Validation images only. This NEVER certifies official T4 performance."""
    df = read_manifest(config['data']['val_csv'], config['data']['root'])
    if len(df) < count + 5:
        if not config.get('smoke'): raise ValueError('Need 100 measured + 5 distinct warmup validation images')
        count = max(1, len(df) - 1)
        warm = df.iloc[:1]; samples = df.iloc[1:count + 1]
    else:
        # Frozen sample plan, no labels used to choose timing samples.
        chosen = np.random.default_rng(42).choice(len(df), count + 5, replace=False)
        warm = df.iloc[chosen[:5]]; samples = df.iloc[chosen[5:]]
    predictor = Predictor(checkpoint, config['device'], precision, allow_smoke=config.get('smoke', False))
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    timing = []
    with (out / f'local_{precision}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(COLUMNS)
        for row in warm.itertuples(): predictor.predict_one(row.resolved_path, row.image_id, w, f)
        for repeat in range(3):
            values = []
            for row in samples.itertuples():
                predictor.synchronize(); start = time.perf_counter_ns()
                predictor.predict_one(row.resolved_path, row.image_id, w, f)
                predictor.synchronize(); values.append((time.perf_counter_ns() - start) / 1e6)
            timing.append(values)
    report = {**predictor.metadata, 'sample_ids': samples.image_id.tolist(), 'source_split': 'official_validation' if not config.get('smoke') else 'synthetic_validation',
        'latency_ms_by_round': timing, 'mean_ms': float(np.asarray(timing).mean(axis=0).mean()),
        'official_protocol': False, 't4_certified': False, 'scope': 'raw file decode through CSV write+flush; batch=1; 3 rounds'}
    write_json(out / f'local_{precision}.json', report)
    return report
