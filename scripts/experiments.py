import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.analysis import analyze, training_plot
from src.benchmark import local_benchmark
from src.experiment_training import train, training_data
from src.inference import Predictor
from src.metrics import metrics
from src.model import verified_weights
from src.registry import Registry, config_hash
from src.utils import load_config, sha256, write_json


def best_record(checkpoint):
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
    m = obj['metrics']
    return {'best_epoch': m['epoch'], 'AUROC': m['auroc'], 'AUPRC': m['auprc'],
            'sensitivity_at_90_specificity': m['sensitivity_at_90_specificity'],
            'checkpoint_path': str(checkpoint), 'model_size_mb': Path(checkpoint).stat().st_size / 1e6}


def run(config_path, registry, authorize=None):
    c = load_config(config_path); eid = c['experiment_id']
    registry.register(c)
    if eid != 'B02' and not c.get('smoke') and authorize != eid:
        raise ValueError(f'Only B02 is authorized automatically. Explicit user authorization required for {eid}')
    if c['plan']['kind'] not in ['train', 'distill']:
        raise ValueError('Use evaluate/deployment for nontraining experiments')
    if c['plan']['initial_status'] == 'skipped': raise ValueError(c['plan']['reason'])
    try:
        if not c.get('smoke'):
            verified_weights(c['model']['weights_manifest'], c['model']['name'])
            if c['device'].startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        if not registry.claim(eid):
            print(f'{eid}: completed equivalent config reused; no training launched')
            registry.export(); return
        registry.update(eid, pid=os.getpid(), reason='Preparing audited train/validation data and deterministic cache')
        def epoch_done(result, checkpoint, elapsed):
            registry.update(eid, **best_record(checkpoint), train_duration=elapsed, reason=f"Finished epoch {result['epoch']}")
            registry.export(); training_plot(c['output'])
        checkpoint = train(c, on_epoch=epoch_done)
        artifact = json.loads((Path(c['output']) / 'artifact.json').read_text())
        registry.update(eid, **best_record(checkpoint), train_duration=artifact['train_duration'],
                        status='completed', reason=None, checkpoint_sha256=sha256(checkpoint))
        try:
            measured = local_benchmark(checkpoint, c, Path(c['output']) / 'latency')
            registry.update(eid, local_latency_ms=measured['mean_ms'], local_latency_device=measured['gpu'],
                            local_latency_precision='fp32')
        except Exception as e:
            registry.update(eid, reason=f'Training completed; local benchmark failed: {e}')
        analyze(registry, out_dir=str(registry.path.parent / 'analysis'))
    except Exception as e:
        row = registry.get(eid)
        if row['status'] != 'completed':
            registry.update(eid, status='failed' if row['status'] == 'running' else 'blocked', reason=str(e))
        registry.export()
        raise


def bind(config_path, candidate_ids, registry):
    c = load_config(config_path); eid = c['experiment_id']
    candidates = [registry.get(x) for x in candidate_ids]
    if any(r['status'] not in ['completed', 'reused'] or r['config'].get('smoke') for r in candidates):
        raise ValueError('Bind only completed real validation experiments')
    frozen = copy.deepcopy(c)
    if eid.startswith('R'):
        if len(candidates) != 1: raise ValueError('Seed replication requires one frozen single model')
        frozen = copy.deepcopy(candidates[0]['config'])
        frozen.update(experiment_id=eid + '_bound', seed=c['seed'], output=f'outputs/runs/{eid}_bound', plan=c['plan'])
        frozen['plan'].update(initial_status='planned', reason=None)
    elif eid == 'F06':
        if len(candidates) != 1: raise ValueError('Distillation requires one teacher')
        frozen['distillation'] = {'teacher_checkpoint': candidates[0]['checkpoint_path'], 'temperature': 2., 'alpha': .5}
        frozen['experiment_id'] = eid + '_bound'; frozen['output'] = f'outputs/runs/{eid}_bound'
    else:
        expected = c.get('selection', {}).get('required_members', 1)
        if len(candidates) != expected: raise ValueError(f'{eid} requires {expected} candidates')
        if eid == 'F03':
            from src.registry import training_hash
            settings = []
            for r in candidates:
                settings.append(training_hash({k: v for k, v in r['config'].items() if k != 'seed'}))
            if len(set(settings)) != 1 or len({r['seed'] for r in candidates}) != 3:
                raise ValueError('F03 requires same training settings and three different seeds')
        frozen['experiment_id'] = eid + '_bound'; frozen['output'] = f'outputs/runs/{eid}_bound'
        frozen['bound_members'] = [r['checkpoint_path'] for r in candidates]
    frozen['bound_candidates'] = candidate_ids
    if not eid.startswith('R') and eid != 'F06':
        for key in ['model', 'image_size', 'augmentation', 'preprocess']:
            frozen[key] = copy.deepcopy(candidates[0]['config'][key])
    if eid in ['R02', 'R03']:
        frozen['plan']['reference'] = 'R01_bound'
    if eid in ['F02', 'F03', 'F04', 'F05']:
        frozen['plan']['reference'] = candidate_ids[0]
    if eid == 'F06':
        frozen['plan'].update(reference='B02', factor='distillation')
    frozen['plan']['initial_status'] = 'planned'; frozen['plan']['reason'] = None
    path = Path('configs/experiments') / f"{frozen['experiment_id']}.yaml"
    if path.exists() and load_config(path) != frozen: raise ValueError('Bound config already exists; version the experiment explicitly')
    path.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding='utf-8')
    registry.register(frozen); registry.export()
    print(path)


def evaluate(config_path, registry):
    from src.data import open_rgb
    import numpy as np
    c = load_config(config_path); eid = c['experiment_id']
    if not c.get('bound_members'): raise ValueError('Bind candidates before evaluation')
    registry.register(c)
    if not registry.claim(eid): return
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=False)
    bundle = out / 'inference.json'
    members = [os.path.relpath(Path(p).resolve(), out.resolve()) for p in c['bound_members']]
    write_json(bundle, {'members': members, 'weights': [1 / len(members)] * len(members),
                       'tta': c.get('selection', {}).get('tta', ['identity']), 'threshold': c['threshold']})
    predictor = Predictor(bundle, c['device'])
    _, df = training_data(c)
    rows = []
    for row in df.itertuples():
        p = predictor.predict_image(open_rgb(row.resolved_path))
        rows.append({'image_id': row.image_id, 'label': row.label, 'probability': p,
                     'logit': float(np.log(np.clip(p, 1e-12, 1-1e-12) / np.clip(1-p, 1e-12, 1))),
                     'logit_kind': 'logit_of_aggregate_probability', 'member_logits': json.dumps(predictor.last_logits),
                     'experiment_id': eid, **({'patient_id': row.patient_id} if hasattr(row, 'patient_id') else {})})
    prediction = pd.DataFrame(rows); prediction.to_csv(out / 'validation.csv', index=False)
    m = metrics(prediction.label, prediction.probability, c['threshold'])
    write_json(out / 'metrics.json', m); write_json(out / 'config.json', c)
    registry.update(eid, status='completed', AUROC=m['auroc'], AUPRC=m['auprc'],
        sensitivity_at_90_specificity=m['sensitivity_at_90_specificity'], checkpoint_path=str(bundle),
        backbone='+'.join(child.config['model']['name'] for child in predictor.children),
        input_size=','.join(str(child.config['image_size']) for child in predictor.children),
        model_size_mb=sum(f['bytes'] for f in predictor.metadata['inference_files']) / 1e6)
    measured = local_benchmark(bundle, c, out / 'latency')
    registry.update(eid, local_latency_ms=measured['mean_ms'])
    analyze(registry, str(registry.path.parent / 'analysis'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', default='outputs/experiment_manager/registry.sqlite')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run'); r.add_argument('--config', required=True); r.add_argument('--authorize')
    b = sub.add_parser('bind'); b.add_argument('--config', required=True); b.add_argument('--candidates', nargs='+', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--config', required=True)
    a = sub.add_parser('analyze'); a.add_argument('--shortlist', nargs='+'); a.add_argument('--bootstrap-repeats', type=int, default=2000)
    sub.add_parser('status')
    args = p.parse_args(); registry = Registry(args.registry)
    if args.command == 'run': run(args.config, registry, args.authorize)
    elif args.command == 'bind': bind(args.config, args.candidates, registry)
    elif args.command == 'evaluate': evaluate(args.config, registry)
    elif args.command == 'analyze': analyze(registry, str(registry.path.parent / 'analysis'), args.shortlist, args.bootstrap_repeats)
    else:
        registry.export()
        print(pd.DataFrame([{k: r.get(k) for k in ['experiment_id', 'status', 'AUROC', 'reason']} for r in registry.rows()]).to_string(index=False))


if __name__ == '__main__': main()
