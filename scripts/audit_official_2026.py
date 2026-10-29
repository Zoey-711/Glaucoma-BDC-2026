"""Read-only audit of the official CSV schemas and class balance."""

import json
import math
from pathlib import Path

import pandas as pd

from src.utils import sha256


RAW = Path('data/official/glaucoma_first')
MANIFESTS = Path('data/official/manifests')
SPLITS = {'train': 51396, 'valid': 5849, 'test': 5855}


def audit():
    report = {'splits': {}, 'source': str(RAW), 'read_only': True}
    for split, expected in SPLITS.items():
        raw_path = RAW / f'{split}.csv'
        manifest_path = MANIFESTS / f'{split}.csv'
        raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        age = pd.to_numeric(raw['Age'], errors='coerce') if 'Age' in raw else None
        row = {
            'raw_columns': list(raw.columns),
            'manifest_columns': list(manifest.columns),
            'raw_rows': len(raw), 'manifest_rows': len(manifest), 'expected_rows': expected,
            'raw_csv_sha256': sha256(raw_path), 'manifest_sha256': sha256(manifest_path),
            'unique_image_names': int(raw.Image_name.nunique()),
            'raw_manifest_id_order_equal': raw.Image_name.tolist() == manifest.image_id.tolist(),
            'age_available': age is not None,
            'age_numeric_count': int(age.notna().sum()) if age is not None else None,
            'age_missing_count': int(age.isna().sum()) if age is not None else None,
            'age_min': float(age.min()) if age is not None else None,
            'age_max': float(age.max()) if age is not None else None,
            'age_zero_count': int((age == 0).sum()) if age is not None else None,
            'age_over_100_count': int((age > 100).sum()) if age is not None else None,
        }
        if 'True_Label' in raw:
            labels = raw.True_Label
            row.update(positive_count=int((labels == '1').sum()),
                       negative_count=int((labels == '0').sum()),
                       invalid_labels=sorted(set(labels) - {'0', '1'}),
                       positive_rate=float((labels == '1').mean()),
                       raw_manifest_labels_equal=labels.tolist() == manifest.label.tolist())
        report['splits'][split] = row
        assert len(raw) == len(manifest) == expected
        assert row['unique_image_names'] == expected and row['raw_manifest_id_order_equal']
        if 'True_Label' in raw:
            assert not row['invalid_labels'] and row['raw_manifest_labels_equal']
    train = report['splits']['train']
    n, positive, negative = train['raw_rows'], train['positive_count'], train['negative_count']
    batch = 32
    # Sequential shuffled batches without replacement: exact single-batch probability.
    no_positive = math.exp(math.lgamma(negative + 1) - math.lgamma(negative - batch + 1)
                           - math.lgamma(n + 1) + math.lgamma(n - batch + 1))
    no_negative = math.exp(math.lgamma(positive + 1) - math.lgamma(positive - batch + 1)
                           - math.lgamma(n + 1) + math.lgamma(n - batch + 1))
    report['train_batch_32'] = {
        'sampler': 'shuffle_without_replacement',
        'p_no_positive_full_batch': no_positive,
        'p_no_negative_full_batch': no_negative,
        'expected_no_positive_full_batches_approx': math.floor(n / batch) * no_positive,
        'train_pos_weight_neg_over_pos': negative / positive,
        'train_sqrt_pos_weight': math.sqrt(negative / positive),
        'note': 'Last partial batch is excluded from the full-batch approximation.'
    }
    return report


if __name__ == '__main__':
    result = audit()
    target = Path('outputs/official_2026_audit/schema_balance_v2.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
