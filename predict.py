import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from src.data import read_manifest
from src.inference import Predictor
from src.submission import COLUMNS, validate_submission
from src.utils import sha256, write_json


def submission_path(output=None, round_name=None):
    if output is not None:
        return Path(output)
    name = round_name or 'round_01'
    if not re.fullmatch(r'round_[0-9]{2,}(?:_[A-Za-z0-9_-]+)?', name):
        raise ValueError('Round must look like round_01 or round_01_B02')
    return Path('submission') / name / 'submission.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--root', required=True)
    destination = p.add_mutually_exclusive_group()
    destination.add_argument('--output', help='Explicit output path (legacy callers)')
    destination.add_argument('--round', dest='round_name', help='e.g. round_01_B02; default round_01')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--allow-smoke', action='store_true')
    a = p.parse_args()
    import torch
    if a.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    output = submission_path(a.output, a.round_name)
    partial = output.with_suffix('.csv.partial')
    metadata_path = Path(str(output) + '.metadata.json')
    if output.exists() or partial.exists() or metadata_path.exists():
        raise FileExistsError(f'Refusing to overwrite an existing submission: {output}')
    df = read_manifest(a.csv, a.root, labeled=False)
    predictor = Predictor(a.checkpoint, a.device, a.precision, a.allow_smoke)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for index, row in enumerate(df.itertuples(), 1):
            predictor.predict_one(row.resolved_path, row.image_id, w, f)
            if index % 500 == 0:
                print(f'Predicted {index}/{len(df)}', flush=True)
    validate_submission(partial, df.image_id, predictor.threshold)
    metadata = dict(predictor.metadata)
    metadata.update(round=output.parent.name, created_at=datetime.now(timezone.utc).isoformat(),
                    manifest_path=str(Path(a.csv).resolve()), manifest_sha256=sha256(a.csv),
                    submission_sha256=sha256(partial), rows=len(df), threads=a.threads,
                    uploaded=False)
    write_json(metadata_path, metadata)
    partial.rename(output)
    print(f'Validated {len(df)} rows: {output}', flush=True)


if __name__ == '__main__':
    main()
