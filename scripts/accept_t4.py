"""Preflight the frozen final CSV, then run the existing official cost.py protocol.

Preflight-only never loads a model or performs inference. A PASS requires a real T4
run of cost.py with the organizer's 100 measured and 5 separate warmup images.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.submission import validate_submission
from src.utils import sha256


def preflight(submission, checkpoint, test_manifest):
    submission, checkpoint = Path(submission), Path(checkpoint)
    meta = json.loads(Path(str(submission)+'.metadata.json').read_text(encoding='utf-8'))
    expected = pd.read_csv(test_manifest, dtype={'image_id':str}, keep_default_na=False)
    if len(expected) != 5855 or expected.image_id.duplicated().any():
        raise ValueError('Need complete official Test manifest with 5,855 unique IDs')
    df = validate_submission(submission, expected.image_id, meta['threshold'])
    if meta['manifest_sha256'] != sha256(test_manifest):
        raise ValueError('Test manifest differs from submission generation')
    if meta['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Checkpoint differs from final CSV metadata')
    if meta['submission_sha256'] != sha256(submission):
        raise ValueError('Final CSV changed since generation')
    if meta['precision'] not in {'fp32','fp16'}:
        raise ValueError('Undeclared inference precision')
    return {'status':'UNVERIFIED','rows':len(df),'checkpoint_sha256':meta['checkpoint_sha256'],
            'submission_sha256':meta['submission_sha256'],'declared_precision':meta['precision'],
            'reason':'Preflight passed; official T4 benchmark has not run'}


def main():
    p=argparse.ArgumentParser(description='Official T4 acceptance; preflight alone is never PASS')
    p.add_argument('submission');p.add_argument('pictures_100');p.add_argument('pictures_5')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--test-manifest',default='data/official/manifests/test.csv')
    p.add_argument('--output',required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    result=preflight(a.submission,a.checkpoint,a.test_manifest)
    print(json.dumps(result),flush=True)
    if a.preflight_only:return
    # cost.py enforces actual T4/Python3.12/torch2.10/torchvision0.25/CUDA12.8,
    # exactly 100+5 images, batch=1, three synchronized full-pipeline rounds,
    # rowwise label/probability/order consistency with final submission.
    subprocess.run([sys.executable,'cost.py',a.submission,a.pictures_100,a.pictures_5,
                    '--checkpoint',a.checkpoint,'--precision',result['declared_precision'],
                    '--output',a.output],check=True)
    report=json.loads((Path(a.output)/'report.json').read_text(encoding='utf-8'))
    if not report['passed'] or len(report['sample_ids'])!=100:
        raise SystemExit('Official T4 acceptance did not pass')


if __name__=='__main__':main()
