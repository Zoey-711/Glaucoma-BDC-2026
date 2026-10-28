"""Save the observed audit blocker without running models or touching source data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter

from src.utils import sha256, write_json


def thumbnail(path):
    with Image.open(path) as im:
        im.draft('RGB', (512, 512))
        im = ImageOps.exif_transpose(im).convert('RGB')
        return im.resize((256, 256), Image.Resampling.LANCZOS)


def main():
    out = Path('outputs/phase2_g01/near_duplicate_audit')
    f = pd.read_csv(out / 'image_hashes.csv', keep_default_na=False)
    assert len(f) == 57245 and not f.error.ne('').any()
    train, val = f[f.split == 'train'], f[f.split == 'val']
    x = train.merge(val, on='phash', suffixes=('_train', '_val'))
    reviewed = {('train_00331.jpg','valid_02826.jpg'), ('train_00972.jpg','valid_00188.jpg')}
    scores = []
    for r in x.itertuples():
        a, b = [np.asarray(thumbnail(p).convert('L'), dtype=float)[24:-24,24:-24]
                for p in [r.path_train, r.path_val]]
        edge_a, edge_b = a - gaussian_filter(a, 3), b - gaussian_filter(b, 3)
        scores.append(float(np.corrcoef(edge_a.ravel(), edge_b.ravel())[0,1]))
    x['edge_correlation'] = scores
    x['visual_review'] = ['suspected_same_eye_or_near_duplicate' if (r.image_id_train,r.image_id_val) in reviewed
                          else 'not_individually_reviewed' for r in x.itertuples()]
    x.to_csv(out / 'cross_split_identical_phash.csv', index=False)
    selected = x[x.visual_review != 'not_individually_reviewed']
    canvas = Image.new('RGB', (1024, 2 * 440), 'white'); draw = ImageDraw.Draw(canvas)
    for k, r in enumerate(selected.itertuples()):
        for side, offset in [('train',0),('val',512)]:
            with Image.open(getattr(r,'path_'+side)) as im:
                im.draft('RGB',(512,512)); im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((500,400))
                canvas.paste(im, (offset,k*440+30))
            draw.text((offset,k*440),getattr(r,'image_id_'+side),fill='black')
    canvas.save(out / 'reviewed_suspects.jpg')
    freeze = json.loads(Path('outputs/migration_review/frozen_inventory.json').read_text(encoding='utf-8'))
    assert all(sha256(p) == h for p,h in freeze.items())
    source = json.loads(Path('outputs/migration_review/source_inventory.json').read_text(encoding='utf-8'))
    assert all(sha256(Path(source['root']) / p) == h for p,h in source['files'].items())
    report = {'status': 'blocked_unexplained_cross_split_similarity', 'training_gate': 'blocked',
        'experiment_id': 'G01', 'images_scanned': len(f), 'read_errors': 0,
        'byte_duplicate_images': int(f.sha256.duplicated().sum()),
        'cross_split_identical_phash_pairs': len(x), 'unique_train_images_in_pairs': x.image_id_train.nunique(),
        'unique_val_images_in_pairs': x.image_id_val.nunique(),
        'val_positive_images_in_pairs': x.loc[x.label_val == 1, 'image_id_val'].nunique(),
        'label_disagreement_pairs': int((x.label_train != x.label_val).sum()),
        'visually_reviewed_suspects': selected[['image_id_train','image_id_val','label_train','label_val','edge_correlation','visual_review']].to_dict('records'),
        'patient_id_available': False, 'all_candidates_confirmed': False,
        'radius6_status': 'Broad candidate export retained; rendering interrupted after serious-risk stop. Completeness not certified.',
        'reason': 'Unexplained cross-split suspected same-eye repeated captures/near duplicates, without patient IDs to establish independence.',
        'training_started': False, 'smoke_training_started': False, 'weights_downloaded': False,
        'test_images_used': False, 'original_data_modified': False,
        'frozen_files_verified_unchanged': len(freeze), 'old_source_files_verified_unchanged': len(source['files']),
        'limits': ['Same pHash is a screening signal, not proof of image identity or patient identity.',
                   'Edge correlation is descriptive, not a validated identity classifier.',
                   'Rotations/arbitrary crops and larger hash distances not exhaustively audited.',
                   'No inference about whether this explains the official score gap.']}
    write_json(out / 'review_report.json', report)
    write_json('outputs/phase2_g01/status.json', {'experiment_id': 'G01', 'status': 'blocked_data_audit',
        'report': str(out / 'review_report.json'), 'best_epoch': None, 'AUROC': None, 'AUPRC': None,
        'delta_auroc': None, 'bootstrap_ci95': None, 'train_duration': None, 'peak_memory_mib': None,
        'model_size_mb': None, 'local_latency_ms': None, 'error_complementarity': None})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
