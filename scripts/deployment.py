import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import local_benchmark
from src.inference import Predictor
from src.data import read_manifest, open_rgb
from src.metrics import metrics
from src.registry import Registry
from src.utils import load_config, sha256, write_json


def import_t4(registry, eid, report_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    row = registry.get(eid)
    if (not report.get('official_protocol') or not report.get('passed') or 'T4' not in (report.get('gpu') or '')
        or report.get('batch_size') != 1 or len(report.get('sample_ids', [])) != 100
        or len(report.get('latency_ms_by_round', [])) != 3 or any(report.get('mismatches_by_round', [1]))
        or report.get('mean_ms', float('inf')) > 100):
        raise ValueError('Passing official T4 100-image/3-round consistency evidence required')
    if report['checkpoint_sha256'] != sha256(row['checkpoint_path']): raise ValueError('T4 artifact SHA256 mismatch')
    if not str(report.get('python', '')).startswith('3.12.') or report.get('cuda') != '12.8' or not str(report.get('torch', '')).startswith('2.10.0'):
        raise ValueError('T4 evidence must use official Python 3.12 / PyTorch 2.10.0 / CUDA 12.8 environment')
    registry.update(eid, t4_latency_ms=report['mean_ms'], t4_report=str(Path(report_path).resolve()), t4_precision=report['precision'])
    registry.export()


def package(registry, eid, destination):
    row = registry.get(eid)
    if row.get('t4_latency_ms') is None or not row.get('t4_report'):
        raise ValueError('No T4 evidence; final package blocked')
    import_t4(registry, eid, row['t4_report'])
    predictor = Predictor(row['checkpoint_path'], 'cpu', 'fp32')
    out = Path(destination)
    if out.exists(): raise FileExistsError(out)
    files = predictor.metadata['inference_files']
    for entry in files:
        if sha256(entry['path']) != entry['sha256']: raise ValueError('Inference file changed')
    if sum(f['bytes'] for f in files) > 500_000_000: raise ValueError('Model package exceeds 500MB')
    children = predictor.children or [predictor]
    for child in children:
        provenance = child.provenance
        if provenance.get('status') != 'approved' or provenance.get('medical_pretraining') is not False:
            raise ValueError('Missing approved model provenance')
        for path, digest in [('license_file', 'license_sha256'), ('path', 'sha256'), ('model_card_file', 'model_card_sha256')]:
            if not provenance.get(path) or sha256(provenance[path]) != provenance.get(digest):
                raise ValueError(f'Missing/tampered provenance evidence: {path}')
    out.mkdir(parents=True)
    model_dir = out / 'model'; model_dir.mkdir()
    mapping = {}
    for i, entry in enumerate(files):
        if Path(entry['path']).suffix != '.json':
            name = f'{i:02d}_{Path(entry["path"]).name}'
            shutil.copy2(entry['path'], model_dir / name); mapping[str(Path(entry['path']).resolve())] = name
    if predictor.children:
        spec = json.loads(Path(row['checkpoint_path']).read_text(encoding='utf-8'))
        spec['members'] = [mapping[str(c.checkpoint)] for c in predictor.children]
        write_json(model_dir / 'inference.json', spec)
    for i, child in enumerate(children):
        evidence = out / 'compliance' / f'member_{i}'; evidence.mkdir(parents=True)
        write_json(evidence / 'provenance.json', child.provenance)
        for key in ['license_file', 'model_card_file', 'path']:
            shutil.copy2(child.provenance[key], evidence / Path(child.provenance[key]).name)
        (evidence / 'MODIFICATIONS.txt').write_text('Classifier replaced with a binary head; fine-tuned only on official training images. Original Apache-2.0 notices retained. Validation held out.\n', encoding='utf-8')
        teacher = child.provenance.get('teacher')
        depth = 0
        while teacher:
            from src.model import verified_weights
            proof = verified_weights(teacher['pretraining_provenance'])
            teacher_dir = evidence / f'teacher_{depth}'; teacher_dir.mkdir()
            write_json(teacher_dir / 'teacher_declaration.json', teacher)
            for key in ['license_file', 'model_card_file']:
                shutil.copy2(proof[key], teacher_dir / Path(proof[key]).name)
            teacher = proof.get('teacher'); depth += 1
    shutil.copy2(row['t4_report'], out / 't4_report.json')
    source = out / 'code'; source.mkdir()
    shutil.copytree('src', source / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('scripts', source / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    write_json(out / 'training_config.json', row['config'])
    for name in ['predict.py', 'cost.py', 'requirements.txt']: shutil.copy2(name, source / name)
    inventory = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in model_dir.iterdir()]
    if sum(f['bytes'] for f in inventory) > 500_000_000: raise ValueError('Relocated inference files exceed 500MB')
    write_json(out / 'inference_files.json', inventory)
    write_json(out / 'declaration.json', {'experiment_id': eid, 'source_checkpoint_sha256': sha256(row['checkpoint_path']),
        'inference_bytes': sum(f['bytes'] for f in inventory), 'precision': row['t4_precision'],
        'note': 'Relocated bundle paths change config hash; rerun cost.py on packaged artifact before submission. submission.csv must be generated from this fixed artifact; package does not invent predictions.'})
    from scripts.compliance import xlsx
    checklist = [['项目', '内容'], ['实验ID', eid], ['推理精度', row['t4_precision']],
                 ['实际推理文件字节数', sum(f['bytes'] for f in inventory)],
                 ['验证集用途', '仅验证/早停/模型选择，未并入训练'], ['测试集用途', '仅最终固定推理，不拟合参数'],
                 ['T4', '源候选已通过；重定位包仍需最终predict/cost复核']]
    for i, child in enumerate(children):
        for key in ['model', 'source_url', 'license_url', 'training_description_url', 'sha256']:
            checklist.append([f'member_{i}.{key}', child.provenance[key]])
    xlsx(out / '模型核验材料与数据声明清单.xlsx', checklist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', default='outputs/experiment_manager/registry.sqlite')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('benchmark'); b.add_argument('--config', required=True)
    t = sub.add_parser('import-t4'); t.add_argument('--experiment', required=True); t.add_argument('--report', required=True)
    k = sub.add_parser('package'); k.add_argument('--experiment'); k.add_argument('--config'); k.add_argument('--output', required=True)
    a = p.parse_args(); registry = Registry(a.registry)
    if a.command == 'import-t4': import_t4(registry, a.experiment, a.report)
    elif a.command == 'package':
        if a.config:
            c = load_config(a.config)
            if c.get('deployment', {}).get('method') != 'package' or len(c.get('bound_candidates', [])) != 1:
                raise ValueError('Use a bound D03 configuration')
            registry.register(c)
            package(registry, c['bound_candidates'][0], a.output)
            registry.update(c['experiment_id'], status='packaged_pending_recheck', output=a.output,
                reason='Source passed T4; relocated package requires final predict/cost rerun', t4_latency_ms=None)
            registry.export()
        elif a.experiment: package(registry, a.experiment, a.output)
        else: raise ValueError('Provide --config or --experiment')
    else:
        c = load_config(a.config)
        if c.get('deployment', {}).get('method') == 'package': raise ValueError('D03 uses package, not benchmark')
        if not c.get('bound_members') or len(c['bound_members']) != 1: raise ValueError('Bind one final candidate first')
        registry.register(c)
        if not registry.claim(c['experiment_id']): return
        out = Path(c['output']); out.mkdir(parents=True, exist_ok=False)
        checkpoint = c['bound_members'][0]
        if c['deployment']['method'] == 'channels_last':
            bundle = out / 'inference.json'
            write_json(bundle, {'members': [os.path.relpath(Path(checkpoint).resolve(), out.resolve())],
                'weights': [1.], 'tta': ['identity'], 'channels_last': True, 'threshold': c['threshold']})
            checkpoint = str(bundle)
        results = [local_benchmark(checkpoint, c, out, precision) for precision in c['deployment']['precisions']]
        write_json(out / 'precision_comparison.json', results)
        frame = read_manifest(c['data']['val_csv'], c['data']['root'])
        validation_metrics = {}
        for precision in c['deployment']['precisions']:
            predictor = Predictor(checkpoint, c['device'], precision)
            predictions = []
            for row in frame.itertuples():
                probability = predictor.predict_image(open_rgb(row.resolved_path))
                predictions.append({'image_id': row.image_id, 'label': row.label, 'probability': probability,
                    'logit': predictor.last_logits[0] if len(predictor.last_logits) == 1 else float(np.log(np.clip(probability, 1e-12, 1-1e-12) / np.clip(1-probability, 1e-12, 1))),
                    'member_logits': json.dumps(predictor.last_logits), 'experiment_id': c['experiment_id'], 'precision': precision})
            df = pd.DataFrame(predictions)
            df.to_csv(out / f'validation_{precision}.csv', index=False)
            validation_metrics[precision] = metrics(df.label, df.probability, c['threshold'])
        write_json(out / 'validation_precision_metrics.json', validation_metrics)
        registry.update(c['experiment_id'], status='completed', checkpoint_path=checkpoint,
            local_latency_ms=results[0]['mean_ms'], model_size_mb=sum(x['bytes'] for x in results[0]['inference_files']) / 1e6,
            reason='Local precision screening only; T4 and full validation consistency still required')
        registry.export()


if __name__ == '__main__': main()
