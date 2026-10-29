import numpy as np
from PIL import Image,ImageDraw

from src.fov_v2 import crop_fov_v2
from scripts.s01_v2_qc import independent_safety


def image_with_dim_periphery():
    im=Image.new('RGB',(600,400))
    d=ImageDraw.Draw(im);d.ellipse((60,10,540,390),fill=(2,1,1));d.ellipse((140,50,380,350),fill=(80,30,10))
    return im


def test_extremely_dim_periphery_forces_fallback():
    im=image_with_dim_periphery();out,m=crop_fov_v2(im)
    assert m['fallback'] and m['native_signal_outside_proposal']
    assert np.array_equal(im,out)


def test_one_native_pixel_cannot_be_removed_by_downsampling():
    im=Image.new('RGB',(3000,2000));ImageDraw.Draw(im).ellipse((800,300,2200,1700),fill=(80,30,10))
    im.putpixel((3,1000),(1,0,0));out,m=crop_fov_v2(im)
    assert m['fallback'] and np.array_equal(im,out)


def test_black_margin_can_be_cropped_without_signal_loss():
    im=Image.new('RGB',(600,400));ImageDraw.Draw(im).ellipse((150,50,450,350),fill=(80,30,10))
    out,m=crop_fov_v2(im)
    assert m['applied'] and independent_safety(im,out,m['box'],True)['safety_verified']
    assert m==crop_fov_v2(im)[1]


def test_no_confident_foreground_falls_back():
    im=Image.new('RGB',(100,100));out,m=crop_fov_v2(im)
    assert m['fallback'] and out.size==im.size


def test_independent_check_detects_known_type_of_damage():
    im=image_with_dim_periphery();box=[140,0,400,400]
    assert not independent_safety(im,im.crop(box),box,True)['safety_verified']


def test_cache_and_online_are_identical(tmp_path):
    import pandas as pd
    import torch
    from src.cache import build_cache
    from src.data import transform
    im=image_with_dim_periphery();path=tmp_path/'source.png';im.save(path)
    frame=pd.DataFrame({'resolved_path':[str(path)]});manifest=tmp_path/'train.csv';frame.to_csv(manifest,index=False)
    cache=build_cache(frame,manifest,224,'safe_fov_v2','train',root=tmp_path/'cache')
    live=transform(224,preprocess='safe_fov_v2')(im)
    cached=transform(224,preprocess='safe_fov_v2',preprocessed=True)(Image.open(cache/'000000.png'))
    assert torch.equal(live,cached)


def test_official_train_05402_permanent_regression():
    from pathlib import Path
    import pytest
    from scripts.s01_v2_qc import source,KNOWN_ID,KNOWN_SHA
    from src.data import open_rgb
    from src.utils import sha256
    if not Path('data/official/manifests/train.csv').exists():
        pytest.skip('Official data unavailable; formal QC gate requires this regression separately')
    frame,_=source('train')
    path=frame.loc[frame.image_id==KNOWN_ID,'resolved_path'].iloc[0]
    assert sha256(path)==KNOWN_SHA
    original=open_rgb(path);out,meta=crop_fov_v2(original)
    assert meta['fallback'] and meta['box']==[0,0,*original.size]
    assert independent_safety(original,out,meta['box'],True)['safety_verified']
