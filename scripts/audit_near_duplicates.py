"""Read-only train/validation perceptual audit; never accesses test images."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

from src.data import read_manifest
from src.utils import sha256, write_json


def phash(im):
    a = np.asarray(im.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    v = dctn(a, type=2)[:8, :8].flatten()
    return int.from_bytes(np.packbits(v > np.median(v)).tobytes(), 'big')


def inspect(item):
    split, row = item
    try:
        digest = sha256(row.resolved_path)
        with Image.open(row.resolved_path) as source:
            width, height = source.size
            source.draft('RGB', (256, 256))
            im = ImageOps.exif_transpose(source).convert('RGB')
            im.thumbnail((256, 256), Image.Resampling.LANCZOS)
            return dict(split=split, image_id=row.image_id, label=int(row.label), path=row.resolved_path,
                        sha256=digest, width=width, height=height,
                        phash=f'{phash(im):016x}', mirror_phash=f'{phash(ImageOps.mirror(im)):016x}', error='')
    except Exception as exc:
        return dict(split=split, image_id=row.image_id, label=int(row.label), path=row.resolved_path,
                    sha256='', width=None, height=None, phash='', mirror_phash='', error=str(exc))


def candidates(rows, radius=6):
    # Seven disjoint bands: any pair with <=6 changed bits shares at least one band.
    widths = [10, 9, 9, 9, 9, 9, 9]
    bands, hashes = [defaultdict(list) for _ in widths], []
    for i, row in enumerate(rows):
        h = int(row['phash'], 16)
        queries = [h, int(row['mirror_phash'], 16)]
        neighbors = set()
        for q in queries:
            offset = 0
            for b, width in zip(bands, widths):
                neighbors.update(b[(q >> offset) & ((1 << width) - 1)])
                offset += width
        for j in sorted(neighbors):
            d = [(q ^ hashes[j]).bit_count() for q in queries]
            if min(d) <= radius:
                yield j, i, min(d), 'mirror' if d[1] < d[0] else 'identity'
        offset = 0
        for b, width in zip(bands, widths):
            b[(h >> offset) & ((1 << width) - 1)].append(i)
            offset += width
        hashes.append(h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/phase2_g01/near_duplicate_audit')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    frames = {s: read_manifest(f'data/official/manifests/{f}.csv', 'data/official')
              for s, f in [('train', 'train'), ('val', 'valid')]}
    items = [(s, row) for s, frame in frames.items() for row in frame.itertuples()]
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for start in range(0, len(items), 256):
            rows.extend(pool.map(inspect, items[start:start + 256]))
            if start % 2048 == 0:
                print(f'Perceptual audit {len(rows)}/{len(items)}', flush=True)
    with (out / 'image_hashes.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    errors = [r for r in rows if r['error']]
    good = [r for r in rows if not r['error']]
    pairs = []
    for j, i, distance, orientation in candidates(good):
        x, y = good[j], good[i]
        pairs.append(dict(left_id=x['image_id'], right_id=y['image_id'], left_split=x['split'],
            right_split=y['split'], left_label=x['label'], right_label=y['label'],
            left_path=x['path'], right_path=y['path'], distance=distance, orientation=orientation,
            exact_bytes=x['sha256'] == y['sha256'], cross_split=x['split'] != y['split'],
            review='pending'))
    pairs.sort(key=lambda r: (not r['cross_split'], r['distance'], r['left_id'], r['right_id']))
    columns = list(pairs[0]) if pairs else ['left_id','right_id','distance','cross_split','review']
    with (out / 'suspicious_pairs.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(pairs)
    cross = [r for r in pairs if r['cross_split']]
    # Render all cross-split candidates in bounded sheets for review. No raw images modified.
    for start in range(0, min(len(cross), 120), 12):
        canvas = Image.new('RGB', (768, 12 * 150), 'white'); draw = ImageDraw.Draw(canvas)
        for k, r in enumerate(cross[start:start + 12]):
            for side, x in [('left', 0), ('right', 384)]:
                with Image.open(r[side + '_path']) as im:
                    im.draft('RGB', (256, 256)); im = ImageOps.exif_transpose(im).convert('RGB')
                    im.thumbnail((185, 125)); canvas.paste(im, (x, k * 150 + 24))
                draw.text((x, k * 150), f"{start+k}: {r[side+'_id']} y={r[side+'_label']} d={r['distance']}", fill='black')
        canvas.save(out / f'cross_pairs_{start//12:04d}.jpg')
    write_json(out / 'report.json', {'images': len(rows), 'errors': errors, 'pairs': len(pairs),
        'cross_split_pairs': len(cross), 'cross_split_exact_bytes': sum(r['exact_bytes'] for r in cross),
        'rendered_cross_pairs': min(len(cross), 120),
        'cross_split_distance_le2': sum(r['distance'] <= 2 for r in cross),
        'status': 'review_required' if cross or errors else 'no_cross_split_candidates',
        'training_gate': 'blocked_pending_review' if cross or errors else 'pass_with_patient_id_limitation',
        'manifest_sha256': {s: sha256(f'data/official/manifests/{f}.csv') for s,f in [('train','train'),('val','valid')]},
        'algorithm': '64-bit DCT pHash radius6, identity and horizontal mirror; JPEG draft256; seven-band exhaustive radius search',
        'limitations': ['No patient IDs: patient-level leakage cannot be excluded',
                       'Perceptual matches are candidates, not proof of same patient/image',
                       'Not exhaustive for rotations, arbitrary crops or radius>6'],
        'patient_check': 'unavailable', 'test_used': False})
    print(json.dumps({'images':len(rows), 'pairs':len(pairs), 'cross':len(cross), 'errors':len(errors)}), flush=True)


if __name__ == '__main__':
    main()
