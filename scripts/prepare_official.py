"""Connect a user-supplied official ZIP without modifying the source archive."""
import argparse
import io
import shutil
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml

from src.audit import audit
from src.utils import load_config, sha256, write_json


def safe_member(name, root):
    relative = PurePosixPath(name)
    if '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe ZIP member: {name}')
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'ZIP member escapes destination: {name}')
    return target


def manifest(z, split):
    prefix = 'glaucoma_first/'
    original = pd.read_csv(io.BytesIO(z.read(f'{prefix}{split}.csv')), dtype=str, keep_default_na=False)
    required = {'Image_name'} | ({'True_Label'} if split != 'test' else set())
    if not required.issubset(original.columns) or original.empty:
        raise ValueError(f'Invalid {split} schema')
    names = set(z.namelist())
    folded = {}
    for name in names:
        key = name.casefold()
        if key in folded and folded[key] != name:
            raise ValueError('Ambiguous archive case mapping')
        folded[key] = name
    requested = [f'{prefix}{split}/{x}' for x in original.Image_name]
    if any(p.casefold() not in folded for p in requested):
        raise ValueError(f'{split}: CSV references images missing from archive')
    paths = [folded[p.casefold()] for p in requested]
    result = pd.DataFrame({'image_id': original.Image_name, 'path': paths})
    if result.image_id.eq('').any() or result.image_id.duplicated().any():
        raise ValueError(f'{split}: invalid IDs')
    if split != 'test':
        if not original.True_Label.isin(['0', '1']).all():
            raise ValueError(f'{split}: invalid labels')
        result['label'] = original.True_Label
    if 'patient_id' in original:
        result['patient_id'] = original.patient_id
    return result, list(original.columns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--destination', default='data/official')
    p.add_argument('--report-dir', default='outputs/official_ingest')
    p.add_argument('--config', default='configs/official.local.yaml')
    p.add_argument('--resume', action='store_true', help='Verify and resume an interrupted extraction')
    p.add_argument('--audit-workers', type=int, default=8)
    a = p.parse_args()
    archive = Path(a.archive).resolve(strict=True)
    root = Path(a.destination).resolve()
    reports = Path(a.report_dir)
    if (root.exists() or reports.exists() or Path(a.config).exists()) and not a.resume:
        raise FileExistsError('Destination/report/config already exists; use new paths to avoid overwrites')
    if a.resume:
        import json
        prior = json.loads((reports / 'ingest.json').read_text(encoding='utf-8'))
        if prior['archive'] != str(archive) or prior['archive_bytes'] != archive.stat().st_size:
            raise ValueError('Resume source does not match previous ingest')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        frames, schemas = {}, {}
        for split in ['train', 'valid', 'test']:
            frames[split], schemas[split] = manifest(z, split)
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError('Duplicate archive member names')
        targets = [safe_member(i.filename, root) for i in infos]
        if len({str(p).casefold() for p in targets}) != len(targets):
            raise ValueError('Case-insensitive archive path collision')
        for i in infos:
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks forbidden')
        needed = sum(i.file_size for i in infos)
        remaining = sum(i.file_size for i, t in zip(infos, targets) if not t.is_file())
        if shutil.disk_usage(root.parent if root.parent.exists() else Path.cwd()).free < remaining + 5_000_000_000:
            raise ValueError('Insufficient free disk space plus 5GB safety margin')
        root.mkdir(parents=True, exist_ok=a.resume)
        reports.mkdir(parents=True, exist_ok=a.resume)
        status = {'archive': str(archive), 'archive_bytes': archive.stat().st_size,
            'uncompressed_bytes': needed, 'source': 'User supplied as official competition dataset; download provenance not independently verified',
            'schemas': schemas, 'rows': {s: len(d) for s, d in frames.items()},
            'filename_case_policy': 'Keep CSV Image_name unchanged; resolve actual ZIP paths using unique case-insensitive mapping',
            'patient_id_available': all('patient_id' in d for s, d in frames.items() if s != 'test'),
            'test_usage': 'Schema/file mapping only; no test image statistics or model selection',
            'formal_training_started': False, 'status': 'hashing_archive'}
        write_json(reports / 'ingest.json', status)
        print('Computing source archive SHA256...', flush=True)
        status['archive_sha256'] = sha256(archive)
        if a.resume and status['archive_sha256'] != prior['archive_sha256']:
            raise ValueError('Resume archive hash differs')
        status['status'] = 'extracting'
        write_json(reports / 'ingest.json', status)
        for n, (info, target) in enumerate(zip(infos, targets), 1):
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                valid = False
                if a.resume and target.exists() and target.stat().st_size == info.file_size:
                    crc = 0
                    with target.open('rb') as f:
                        for block in iter(lambda: f.read(1024 * 1024), b''):
                            crc = zlib.crc32(block, crc)
                    valid = crc == info.CRC
                if not valid:
                    with z.open(info) as src, target.open('wb' if a.resume else 'xb') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
            if n % 2000 == 0:
                print(f'Extracted {n}/{len(infos)} entries', flush=True)
                status['extracted_entries'] = n
                write_json(reports / 'ingest.json', status)
    status['extracted_entries'] = len(infos)
    manifests = root / 'manifests'
    manifests.mkdir(exist_ok=a.resume)
    for s, df in frames.items():
        df.to_csv(manifests / f'{s}.csv', index=False, encoding='utf-8')
    config = load_config('configs/baseline.yaml')
    config['data'].update(root=str(root), train_csv=str(manifests / 'train.csv'), val_csv=str(manifests / 'valid.csv'))
    config['output'] = 'outputs/official_baseline'
    Path(a.config).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    status.update(status='auditing_train_validation', config=a.config,
        manifest_sha256={s: sha256(manifests / f'{s}.csv') for s in frames})
    write_json(reports / 'ingest.json', status)
    print('Auditing train/validation decoded images; no training...', flush=True)
    report = audit(manifests / 'train.csv', manifests / 'valid.csv', root, workers=a.audit_workers)
    write_json(reports / 'audit.json', report)
    status.update(status='completed' if report['passed'] else 'audit_blocked', audit_passed=report['passed'])
    write_json(reports / 'ingest.json', status)
    print(f"Complete: audit_passed={report['passed']}, patient_check={report['patient_check']['status']}", flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
