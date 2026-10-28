import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .data import read_manifest, open_rgb
from .utils import sha256, write_json


def inspect_image(row):
    try:
        im = open_rgb(row.resolved_path)
        h = hashlib.sha256(str(im.size).encode())
        h.update(im.tobytes())
        return row.image_id, f'{im.width}x{im.height}', h.hexdigest(), None
    except FileNotFoundError:
        return row.image_id, None, None, 'missing'
    except (OSError, ValueError) as e:
        return row.image_id, None, None, str(e)


def inspect_rows(df, workers):
    rows = list(df.itertuples())
    # Bound queued work and decoded-image memory; preserve deterministic row order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), 256):
            yield from pool.map(inspect_image, rows[start:start + 256])


def audit(train_csv, val_csv, root, workers=4):
    if not 1 <= workers <= 32:
        raise ValueError('Audit workers must be between 1 and 32')
    frames = {s: read_manifest(p, root) for s, p in [('train', train_csv), ('val', val_csv)]}
    report = {'manifest_sha256': {'train': sha256(train_csv), 'val': sha256(val_csv)}, 'splits': {}}
    hashes = defaultdict(list)
    fatal = []
    for split, df in frames.items():
        sizes, missing, corrupt = Counter(), [], []
        for n, (image_id, size, digest, error) in enumerate(inspect_rows(df, workers), 1):
            if error == 'missing':
                missing.append(image_id)
            elif error is not None:
                corrupt.append({'image_id': image_id, 'error': error})
            else:
                sizes[size] += 1
                hashes[digest].append({'split': split, 'image_id': image_id})
            if n % 2000 == 0:
                print(f'Audit {split}: {n}/{len(df)} images decoded', flush=True)
        counts = {str(k): int(v) for k, v in df.label.value_counts().items()}
        report['splits'][split] = {'count': len(df), 'class_counts': counts,
            'positive_ratio': float(df.label.mean()), 'dimensions': dict(sizes), 'missing': missing, 'corrupt': corrupt}
        if missing or corrupt or df.label.nunique() != 2:
            fatal.append(f'{split}: missing/corrupt images or single class')
    report['duplicate_groups'] = [v for v in hashes.values() if len(v) > 1]
    report['cross_split_duplicate_groups'] = [v for v in hashes.values() if len({r['split'] for r in v}) > 1]
    if report['cross_split_duplicate_groups']:
        fatal.append('Train/validation identical decoded images')
    a, b = frames.values()
    if all('patient_id' in d for d in (a, b)) and all(d.patient_id.ne('').all() for d in (a, b)):
        overlap = sorted(set(a.patient_id) & set(b.patient_id))
        report['patient_check'] = {'status': 'checked', 'overlap': overlap}
        if overlap:
            fatal.append('Patient leakage')
    else:
        report['patient_check'] = {'status': 'unavailable_or_incomplete', 'overlap': None}
    report['fatal_errors'] = fatal
    report['passed'] = not fatal
    return report


def main():
    p = argparse.ArgumentParser()
    for x in ['train-csv', 'val-csv', 'root', 'output']:
        p.add_argument('--' + x, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    r = audit(a.train_csv, a.val_csv, a.root, workers=a.workers)
    write_json(a.output, r)
    if not r['passed']:
        raise SystemExit('Audit failed; see report')


if __name__ == '__main__':
    main()
