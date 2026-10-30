"""Locked official-Val T4 acceptance; never reads Test or claims a local GPU is T4."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cost import require_official_environment
from src.data import read_manifest
from src.inference import Predictor
from src.phase_g import compare_frozen_to_inference
from src.submission import COLUMNS
from src.utils import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def run(candidate, output, root=ROOT):
    root = Path(root)
    cfg = yaml.safe_load((root / 'configs/phase_g.yaml').read_text(encoding='utf-8'))['t4_validation']
    candidate = Path(candidate).resolve()
    manifest = json.loads((candidate / 'candidate_manifest.json').read_text(encoding='utf-8'))
    if manifest['candidate'] not in {'F01', 'B02', 'G01'} or manifest['precision'] != 'fp32':
        raise ValueError('Only locked FP32 F01/B02/G01 candidates are accepted')
    if manifest['phase_g_config_sha256'] != sha256(root / 'configs/phase_g.yaml'):
        raise ValueError('Phase G protocol changed')
    for item in manifest['loaded_files']:
        p = candidate / item['name']
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            raise ValueError(f"Loaded file changed: {item['name']}")
    model = Predictor(candidate / manifest['checkpoint'], 'cuda', 'fp32')
    if 'T4' not in str(model.metadata.get('gpu', '')):
        raise ValueError('Official acceptance requires an actual NVIDIA T4')
    if torch.cuda.get_device_properties(model.device).total_memory < 15_000_000_000:
        raise ValueError('Official acceptance requires the 16GB T4 configuration')
    require_official_environment(model.metadata)
    if sorted(x['sha256'] for x in model.metadata['inference_files']) != sorted(x['sha256'] for x in manifest['loaded_files']):
        raise ValueError('Predictor loaded files differ from frozen candidate')
    official = read_manifest(root / 'data/official/manifests/valid.csv', root / 'data/official')
    frozen_sample = root / 'outputs/phase_g/freeze/t4_validation_105.csv'
    sample = pd.read_csv(frozen_sample, dtype={'image_id': str})
    if len(official) != 5849 or len(sample) != 105 or sample.image_id.duplicated().any():
        raise ValueError('Locked official-Val sample is invalid')
    expected_sample = official.sample(n=105, random_state=cfg['sample_seed']).sort_index()
    if not sample[['image_id', 'path', 'label']].reset_index(drop=True).equals(
            expected_sample[['image_id', 'path', 'label']].reset_index(drop=True)):
        raise ValueError('Locked T4 sample differs from deterministic Val selection')
    frozen_file = (root / 'outputs/phase_g/freeze/F01_validation.csv') if manifest['candidate'] == 'F01' else root / ('outputs/runs/' + manifest['candidate'] + '/validation.csv')
    if sha256(frozen_file) != manifest['frozen_validation_sha256']:
        raise ValueError('Frozen validation prediction hash changed')
    frozen = pd.read_csv(frozen_file).rename(columns={'Image_name': 'image_id', 'Probability': 'probability'})
    if frozen.image_id.tolist() != official.image_id.tolist() or not np.array_equal(frozen.label, official.label):
        raise ValueError('Frozen Val IDs or labels differ from official manifest')
    if cfg['warmup'] != 5 or cfg['measured'] != 100 or cfg['rounds'] != 3 or cfg['batch_size'] != 1:
        raise ValueError('T4 protocol must be 5 warmup, 100 measured, 3 rounds, batch 1')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    warmup = sample.iloc[:5]
    measured = sample.iloc[5:]
    with (out / 'warmup.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in warmup.itertuples():
            model.predict_one(root / 'data/official' / row.path, row.image_id, writer, f)
    timings, comparisons = [], []
    for n in range(3):
        elapsed = []
        target = out / f'round_{n + 1}.csv'
        with target.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for row in measured.itertuples():
                model.synchronize()
                start = time.perf_counter_ns()
                model.predict_one(root / 'data/official' / row.path, row.image_id, writer, f)
                model.synchronize()
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
        round_frame = pd.read_csv(target).rename(columns={'Image_name': 'image_id', 'Probability': 'probability'})
        if round_frame.image_id.tolist() != measured.image_id.tolist():
            raise ValueError('Round output order differs from locked measured sample')
        if (not np.isfinite(round_frame.probability.to_numpy(float)).all()
            or not round_frame.probability.between(0, 1).all()
            or not np.array_equal(round_frame.Pred_Label.to_numpy(),
                                  (round_frame.probability.to_numpy() >= model.threshold).astype(int))):
            raise ValueError('Invalid probability or Pred_Label in T4 output')
        comparisons.append(compare_frozen_to_inference(frozen, round_frame,
            cfg['probability_tolerance'], model.threshold))
        timings.append(elapsed)
    times = np.asarray(timings)
    report = {'candidate': manifest['candidate'], 'status': 'PASS' if times.mean() <= cfg['mean_latency_limit_ms'] and all(c['passed'] for c in comparisons) and manifest['loaded_limit_pass'] else 'FAIL',
        'actual_gpu': model.metadata['gpu'], 'gpu_memory_bytes': torch.cuda.get_device_properties(model.device).total_memory,
        'python': model.metadata['python'], 'torch': model.metadata['torch'], 'torchvision': model.metadata['torchvision'],
        'cuda': model.metadata['cuda'], 'precision': 'fp32', 'batch_size': 1, 'warmup_count': 5,
        'measured_count': 100, 'rounds': 3, 'mean_ms': float(times.mean()),
        'p95_ms': float(np.percentile(times, 95)), 'round_mean_ms': times.mean(axis=1).tolist(),
        'loaded_bytes': manifest['loaded_bytes'], 'loaded_files': model.metadata['inference_files'],
        'frozen_prediction_comparisons': comparisons, 'sample_sha256': sha256(frozen_sample),
        'timing_scope': 'per-image file read, decode, preprocessing, H2D, forward, fusion, postprocessing, CSV write/flush, CUDA synchronize; OS page cache not cleared',
        'note': 'Official-Val deployment precheck, not official Test submission or organizer certification'}
    write_json(out / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.candidate, args.output)
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)
