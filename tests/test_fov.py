import numpy as np
import torch
from PIL import Image, ImageDraw

from src.fov import crop_fov
from src.data import transform
from src.cache import build_cache


def field():
    im=Image.new('RGB',(600,400));ImageDraw.Draw(im).ellipse((150,50,450,350),fill=(80,30,15))
    return im


def test_preserve_tissue_and_deterministic():
    im=field();a,m=crop_fov(im);b,n=crop_fov(im)
    assert m==n and np.array_equal(a,b) and m['applied']
    x,y,r,t=m['box'];assert x<150 and y<50 and r>450 and t>350
    assert np.array_equal(np.asarray(im)[y:t,x:r],np.asarray(a))


def test_black_fallback():
    im=Image.new('RGB',(128,128));a,m=crop_fov(im)
    assert a.size==im.size and m['fallback']


def test_small_or_fragmented_fallback():
    im=Image.new('RGB',(512,512));ImageDraw.Draw(im).ellipse((10,10,80,80),fill='red')
    assert crop_fov(im)[1]['fallback']


def test_full_image_no_crop():
    im=Image.new('RGB',(256,256),(80,30,20))
    assert crop_fov(im)[1]['reason']=='already_full_field'


def test_dim_field_safety():
    im=Image.new('RGB',(600,400));ImageDraw.Draw(im).ellipse((150,50,450,350),fill=(9,3,2))
    _,m=crop_fov(im)
    assert m['box'][0]<150 and m['box'][2]>450


def test_separate_visible_tissue_fallback():
    im=field();ImageDraw.Draw(im).rectangle((10,80,50,200),fill=(60,20,10))
    out,m=crop_fov(im)
    assert m['fallback'] and out.size==im.size


def test_resize_control_unchanged():
    from torchvision import transforms as T
    im=field()
    expected=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(im)
    assert torch.equal(expected,transform(224,preprocess='resize')(im))


def test_cache_matches_live_transform(tmp_path):
    import pandas as pd
    im=field();p=tmp_path/'im.png';im.save(p)
    df=pd.DataFrame({'resolved_path':[str(p)]});csv=tmp_path/'train.csv';df.to_csv(csv,index=False)
    cached=build_cache(df,csv,224,'safe_fov_v1','train',root=tmp_path/'cache')
    live=transform(224,preprocess='safe_fov_v1')(im)
    loaded=transform(224,preprocess='safe_fov_v1',preprocessed=True)(Image.open(cached/'000000.png'))
    assert torch.equal(live,loaded)
