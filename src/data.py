from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def read_manifest(csv, root, labeled=True):
    if not csv or not root:
        raise ValueError('Explicit CSV and image root required; no official paths have been supplied')
    root = Path(root).resolve(strict=True)
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    required = {'image_id', 'path'} | ({'label'} if labeled else set())
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f'Nonempty manifest requires columns {required}')
    if df.image_id.eq('').any() or df.image_id.duplicated().any() or df.path.eq('').any():
        raise ValueError('Empty/duplicate image IDs or empty image paths')
    if labeled:
        if not df.label.isin(['0', '1']).all():
            raise ValueError('Labels must be 0 or 1')
        df['label'] = df.label.astype(int)
    resolved = []
    for value in df.path:
        p = (root / value).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f'Image outside root: {value}')
        resolved.append(str(p))
    df['resolved_path'] = resolved
    return df


def open_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


class BlackBorderCrop:
    def __call__(self, im):
        import numpy as np
        # Fixed per-image threshold; no cross-image/test-derived fitting.
        a = np.asarray(im)
        mask = a.max(axis=2) > 10
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return im
        return im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def transform(size, train=False, augmentation='basic', preprocess='resize', preprocessed=False):
    from torchvision import transforms as T
    if preprocess not in ['resize', 'black_border']:
        raise ValueError('Reliable optic-disc ROI unavailable; ROI/fusion experiments must remain skipped')
    ops = []
    if not preprocessed:
        if preprocess == 'black_border': ops.append(BlackBorderCrop())
        ops.append(T.Resize((size, size)))
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
        if augmentation == 'color':
            ops.append(T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02))
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class FundusDataset:
    def __init__(self, frame, size, train=False, augmentation='basic', preprocess='resize', cache_dir=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform(size, train, augmentation, preprocess, preprocessed=cache_dir is not None)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        import torch
        row = self.frame.iloc[idx]
        path = self.cache_dir / f'{idx:06d}.png' if self.cache_dir else row.resolved_path
        return self.transform(open_rgb(path)), torch.tensor(float(row.label))
