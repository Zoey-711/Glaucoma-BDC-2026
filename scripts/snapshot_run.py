"""Preserve matching source evidence for an existing run without changing training."""
import argparse
import json
import shutil
from pathlib import Path

from src.utils import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True)
    args = p.parse_args(); run = Path(args.run)
    recorded = json.loads((run / 'code_sha256.json').read_text(encoding='utf-8'))
    critical = ['src/experiment_training.py', 'src/model.py', 'src/data.py', 'src/cache.py', 'src/losses.py', 'src/metrics.py', 'src/utils.py', 'src/inference.py']
    normalized = {Path(k).as_posix(): v for k, v in recorded.items()}
    for path in critical:
        if sha256(path) != normalized[path]: raise ValueError(f'Core training source changed: {path}')
    out = run / 'source_snapshot'; out.mkdir(exist_ok=False)
    for folder in ['src', 'scripts']:
        shutil.copytree(folder, out / folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['predict.py', 'cost.py', 'requirements.txt', 'requirements-training.lock.txt']:
        shutil.copy2(name, out / name)
    shutil.copy2(run / 'config.json', out / 'config.json')
    write_json(out / 'snapshot_evidence.json', {'core_training_hashes_match_run_start': True,
        'non_core_files_changed_since_start': [k for k, v in recorded.items() if sha256(k) != v],
        'note': 'Snapshot taken during the run; core training/preprocessing/model/metrics/inference files verified byte-for-byte against the hashes recorded before epoch 1.',
        'files': {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}})


if __name__ == '__main__': main()
