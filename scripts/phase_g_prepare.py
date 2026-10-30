"""Lock Phase F evidence, compare predefined fusions and make portable FP32 candidates."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.bootstrap import paired_bootstrap
from src.data import read_manifest
from src.model import verified_weights
from src.metrics import metrics
from src.phase_f import fuse, validate_predictions
from src.phase_g import strict_pair_comparison
from src.utils import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    root = Path(root)
    cfg_path = root / 'configs/phase_g.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    out = root / 'outputs/phase_g'
    if out.exists():
        raise FileExistsError('Phase G output already exists; frozen evidence cannot be overwritten')
    phase_f = root / 'outputs/phase_f'
    evidence = json.loads((phase_f / 'analysis/model_evidence.json').read_text(encoding='utf-8'))
    evidence = {v['model']: v for v in evidence}
    official = read_manifest(root / 'data/official/manifests/valid.csv', root / 'data/official')
    if len(official) != 5849:
        raise ValueError('Official Val must contain 5849 samples')
    frames = {}
    for model in ('B02', 'G01', 'G02'):
        e = evidence[model]
        checkpoint = root / e['checkpoint']
        if sha256(checkpoint) != e['checkpoint_sha256']:
            raise ValueError(f'{model} checkpoint changed')
        pred = root / ('outputs/runs/' + model + '/validation.csv') if model != 'G02' else root / 'results/G02/seed_42/fold_none/validation.csv'
        # Use the exact file identified by the frozen Phase F evidence.
        if not pred.exists() or sha256(pred) != e['prediction_sha256']:
            matches = [p for p in root.glob(f'**/validation.csv') if sha256(p) == e['prediction_sha256']]
            if len(matches) != 1:
                raise ValueError(f'{model} frozen prediction could not be identified uniquely')
            pred = matches[0]
        frames[model] = validate_predictions(pd.read_csv(pred), official, 5849)
        p = dict(e['pretraining'])
        for field in ('path', 'license_file', 'model_card_file', 'upstream_license_file'):
            if p.get(field):
                p[field] = str(root / p[field].replace('\\', '/'))
        verified_weights(p, p['model'])
    fusions = {}
    for name, spec in cfg['fusions'].items():
        frozen_path = phase_f / 'analysis' / f'{name}_validation.csv'
        frozen = validate_predictions(pd.read_csv(frozen_path), official, 5849)
        recomputed = fuse(frames[spec['members'][0]], frames[spec['members'][1]], spec['method'])
        if frozen.experiment_id.ne(name).any() or not np.allclose(frozen.probability, recomputed.probability, atol=1e-12, rtol=0):
            raise ValueError(f'{name} does not match locked 0.5/0.5 fusion')
        fusions[name] = frozen
    f01, f04 = fusions['F01'], fusions['F04']
    comp = paired_bootstrap(f01, f04, cfg['bootstrap']['replicates'], cfg['bootstrap']['seed'])
    pair = strict_pair_comparison(f01, f04)
    result = {name: {key: metrics(f.label, f.probability)[key] for key in ('auroc', 'auprc')}
              for name, f in fusions.items()}
    result.update({'paired_bootstrap_F04_minus_F01': comp, 'positive_negative_ranking': pair,
                   'interpretation': 'Exploratory reuse of official Val; image bootstrap cannot establish patient independence or stable generalization.'})
    out.mkdir(parents=True)
    lock = out / 'freeze'
    lock.mkdir()
    for name in ('F01', 'F04'):
        shutil.copyfile(phase_f / 'analysis' / f'{name}_validation.csv', lock / f'{name}_validation.csv')
    for name in ('B02', 'G01'):
        shutil.copyfile(root / f'outputs/runs/{name}/validation.csv', lock / f'{name}_validation.csv')
    sample = official.sample(n=105, random_state=cfg['t4_validation']['sample_seed']).sort_index()
    sample[['image_id', 'path', 'label']].to_csv(lock / 't4_validation_105.csv', index=False)
    source_files = [cfg_path, root / 'data/official/manifests/valid.csv',
                    phase_f / 'analysis/model_evidence.json', phase_f / 'final_verification.json',
                    phase_f / 'analysis/F01_validation.csv', phase_f / 'analysis/F04_validation.csv']
    source_files += [root / evidence[m]['checkpoint'] for m in ('B02', 'G01', 'G02')]
    source_files += [root / f'outputs/runs/{m}/validation.csv' for m in ('B02', 'G01')]
    write_json(lock / 'source_hashes.json', {str(p.relative_to(root)): sha256(p) for p in source_files})
    write_json(lock / 'fusion_definitions.json', cfg['fusions'])
    write_json(out / 'f04_vs_f01.json', result)
    pd.DataFrame([{'candidate': 'F04', 'reference': 'F01', **result['F04'], **comp, **pair}]).to_csv(out / 'comparison.csv', index=False)
    for candidate, members in {'F01': ('G01', 'B02'), 'B02': ('B02',), 'G01': ('G01',)}.items():
        folder = out / 'candidates' / candidate
        folder.mkdir(parents=True)
        copies = []
        for member in members:
            source = root / evidence[member]['checkpoint']
            target = folder / f'{member}{source.suffix}'
            shutil.copyfile(source, target)
            if sha256(target) != evidence[member]['checkpoint_sha256']:
                raise ValueError(f'{member} package checkpoint mismatch')
            copies.append(target)
        compliance = folder / 'compliance'
        compliance.mkdir()
        compliance_files = []
        for member in members:
            provenance = evidence[member]['pretraining']
            for key in ('license_file', 'model_card_file', 'upstream_license_file'):
                if provenance.get(key):
                    source = root / provenance[key].replace('\\', '/')
                    target = compliance / f'{member}_{source.name}'
                    shutil.copyfile(source, target)
                    digest_key = {'license_file': 'license_sha256', 'model_card_file': 'model_card_sha256',
                                  'upstream_license_file': 'upstream_license_sha256'}[key]
                    if sha256(target) != provenance[digest_key]:
                        raise ValueError(f'{member} packaged compliance material mismatch')
                    compliance_files.append({'name': str(target.relative_to(folder)), 'sha256': sha256(target)})
        if len(copies) == 2:
            bundle = folder / 'inference.json'
            bundle.write_text(json.dumps({'members': [p.name for p in copies], 'weights': [.5, .5],
                'threshold': .5, 'tta': ['identity']}, indent=2) + '\n', encoding='utf-8')
            inference = bundle
        else:
            inference = copies[0]
        files = [inference] + ([] if inference in copies else copies)
        size = sum(p.stat().st_size for p in files)
        manifest = {'candidate': candidate, 'status': 'UNVERIFIED', 'precision': 'fp32',
            'checkpoint': inference.name, 'loaded_files': [{'name': p.name, 'bytes': p.stat().st_size,
                'sha256': sha256(p)} for p in files], 'loaded_bytes': size,
            'loaded_limit_pass': size <= cfg['t4_validation']['loaded_files_limit_bytes'],
            't4_mean_ms': None, 't4_status': 'UNVERIFIED',
            'compliance_files': compliance_files,
            'members': [{'id': m, 'config_hash': evidence[m]['config_hash'],
                'checkpoint_sha256': evidence[m]['checkpoint_sha256'],
                'pretraining': evidence[m]['pretraining']} for m in members],
            'frozen_validation_sha256': sha256(lock / f'{candidate}_validation.csv'),
            'phase_g_config_sha256': sha256(cfg_path)}
        write_json(folder / 'candidate_manifest.json', manifest)
    write_json(out / 'status.json', {'F01': 'UNVERIFIED', 'F04': 'UNVERIFIED',
        'B02': 'UNVERIFIED', 'G01': 'UNVERIFIED', 'reason': 'No actual NVIDIA T4 run'})
    return result


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
