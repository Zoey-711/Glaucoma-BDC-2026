"""Download only public, model-card Apache-2.0 ImageNet weights; retain evidence."""
import json
import urllib.request
from pathlib import Path

from src.utils import sha256, write_json


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=120) as r, Path(path).open('wb') as f:
        while block := r.read(1024 * 1024):
            f.write(block)


def main():
    repo = 'timm/efficientnet_b0.ra_in1k'
    out = Path('data/pretrained/efficientnet_b0.ra_in1k')
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60) as r:
        metadata = json.load(r)
    if metadata.get('gated') or metadata.get('private') or metadata['cardData']['license'] != 'apache-2.0':
        raise ValueError('Model is gated/private or lacks the reviewed Apache-2.0 declaration')
    revision = metadata['sha']
    base = f'https://huggingface.co/{repo}/resolve/{revision}'
    write_json(out / 'hub_metadata.json', metadata)
    fetch(base + '/README.md', out / 'MODEL_CARD.md')
    card = (out / 'MODEL_CARD.md').read_text(encoding='utf-8')
    if 'imagenet-1k' not in card.lower() or 'license: apache-2.0' not in card:
        raise ValueError('Missing reviewed training/weight license evidence')
    license_url = 'https://www.apache.org/licenses/LICENSE-2.0.txt'
    fetch(license_url, out / 'LICENSE-2.0.txt')
    weight = out / 'model.safetensors'
    entry = next(x for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    expected = entry['lfs']['sha256']
    if not weight.exists() or sha256(weight) != expected:
        fetch(base + '/model.safetensors', weight)
    if sha256(weight) != expected:
        raise ValueError('Upstream LFS SHA256 mismatch')
    m = {'status': 'approved', 'model': 'efficientnet_b0', 'backend': 'timm',
        'architecture': 'efficientnet_b0.ra_in1k', 'dataset': 'ImageNet-1K',
        'medical_pretraining': False, 'source_url': base + '/model.safetensors',
        'training_description_url': base + '/README.md', 'license_url': license_url,
        'license_file': str(out / 'LICENSE-2.0.txt'), 'license_sha256': sha256(out / 'LICENSE-2.0.txt'),
        'model_card_file': str(out / 'MODEL_CARD.md'), 'model_card_sha256': sha256(out / 'MODEL_CARD.md'),
        'allows_competition_and_redistribution': True, 'non_commercial_or_gated': False,
        'reviewer': 'Codex: public model-card declaration + Apache-2.0 sections 2/4 + upstream SHA256',
        'review_date': '2026-10-28', 'revision': revision, 'path': str(weight), 'sha256': expected,
        'review_basis': 'Specific model card declares Apache-2.0 and ImageNet-1k; public ungated download verified. Retain license, attribution and modification notice. Final organizer review remains applicable.'}
    write_json('configs/weights_b02.local.json', m)
    print(json.dumps({'revision': revision, 'sha256': expected, 'status': m['status']}))


if __name__ == '__main__':
    main()
