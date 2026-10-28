"""Retrieve a pinned public ImageNet ConvNeXt weight and retain license evidence."""
import json
import urllib.request
from pathlib import Path

from scripts.fetch_b02_weights import fetch
from src.utils import sha256, write_json


def main():
    repo='timm/convnext_tiny.fb_in1k'; out=Path('data/pretrained/convnext_tiny.fb_in1k')
    out.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=60) as r:
        info=json.load(r)
    if info.get('gated') or info.get('private') or info['cardData']['license']!='apache-2.0':
        raise ValueError('Expected public Apache-2.0 weights')
    revision=info['sha']; base=f'https://huggingface.co/{repo}/resolve/{revision}'
    write_json(out/'hub_metadata.json',info)
    fetch(base+'/README.md',out/'MODEL_CARD.md')
    card=(out/'MODEL_CARD.md').read_text(encoding='utf-8')
    if 'license: apache-2.0' not in card or 'imagenet-1k' not in card.lower():
        raise ValueError('Missing exact model-card license or training dataset')
    license_url='https://www.apache.org/licenses/LICENSE-2.0.txt'
    fetch(license_url,out/'LICENSE-2.0.txt')
    fetch('https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/LICENSE',out/'UPSTREAM_LICENSE.txt')
    entry=next(s for s in info['siblings'] if s['rfilename']=='model.safetensors')
    expected=entry['lfs']['sha256']; weight=out/'model.safetensors'
    if not weight.exists() or sha256(weight)!=expected: fetch(base+'/model.safetensors',weight)
    if sha256(weight)!=expected: raise ValueError('LFS SHA256 mismatch')
    m={'status':'approved','model':'convnext_tiny','backend':'timm','architecture':'convnext_tiny.fb_in1k',
       'dataset':'ImageNet-1K','medical_pretraining':False,'source_url':base+'/model.safetensors',
       'training_description_url':base+'/README.md','license_url':license_url,
       'license_file':str(out/'LICENSE-2.0.txt'),'license_sha256':sha256(out/'LICENSE-2.0.txt'),
       'model_card_file':str(out/'MODEL_CARD.md'),'model_card_sha256':sha256(out/'MODEL_CARD.md'),
       'upstream_license_file':str(out/'UPSTREAM_LICENSE.txt'),'upstream_license_sha256':sha256(out/'UPSTREAM_LICENSE.txt'),
       'allows_competition_and_redistribution':True,'non_commercial_or_gated':False,
       'reviewer':'Codex: exact weight card declares ImageNet-1K/Apache-2.0; public download; upstream MIT notice retained',
       'review_date':'2026-10-28','revision':revision,'path':str(weight),'sha256':expected,
       'review_basis':'Apache-2.0 sections 2/4 allow use and redistribution subject to notices. Retain license, attribution, modification notice; no CXR checkpoint.'}
    write_json('configs/weights_g01.local.json',m)
    print(json.dumps({'revision':revision,'sha256':expected,'status':'approved'}),flush=True)


if __name__=='__main__': main()
