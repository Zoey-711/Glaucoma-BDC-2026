"""Lossless deterministic preprocessing cache, train/validation only."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from torchvision.transforms import Resize

from .data import BlackBorderCrop, open_rgb
from .utils import sha256, write_json


def build_cache(frame, manifest, size, preprocess, split, workers=8, root='data/cache'):
    if split not in ['train', 'val'] or preprocess not in ['resize', 'black_border', 'safe_fov_v1']:
        raise ValueError('Only train/val deterministic preprocessing can be cached')
    source_stats = []
    for path in frame.resolved_path:
        st = Path(path).stat()
        source_stats.append([path, st.st_size, st.st_mtime_ns])
    identity = {'manifest_sha256': sha256(manifest), 'size': size, 'preprocess': preprocess,
                'source_stats_sha256': hashlib.sha256(json.dumps(source_stats).encode()).hexdigest(),
                'version': 'pil-rgb-exif-bilinear-v1'}
    if preprocess == 'safe_fov_v1':
        identity['crop_code_sha256'] = sha256(Path(__file__).with_name('fov.py'))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(root) / f'{split}_{key}'
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    if marker.exists() and all((folder / f'{i:06d}.png').is_file() for i in range(len(frame))):
        return folder
    resize = Resize((size, size))
    crop = BlackBorderCrop()
    def convert(item):
        i, path = item
        dest = folder / f'{i:06d}.png'
        if dest.exists(): return
        im = open_rgb(path)
        if preprocess == 'black_border': im = crop(im)
        if preprocess == 'safe_fov_v1':
            from .fov import crop_fov
            im = crop_fov(im)[0]
        im = resize(im)
        tmp = dest.with_suffix('.tmp')
        im.save(tmp, format='PNG')
        tmp.replace(dest)
    items = list(enumerate(frame.resolved_path))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), 512):
            list(pool.map(convert, items[start:start + 512]))
            if start % 4096 == 0:
                print(f'Cache {split}: {min(start + 512, len(items))}/{len(items)}', flush=True)
    write_json(marker, identity)
    return folder
