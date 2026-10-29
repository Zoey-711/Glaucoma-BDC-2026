import json
from pathlib import Path

import torch
from PIL import Image

from src.data import transform


def test_r01b_train_only_color_and_binary_tensor():
    image=Image.new('RGB',(320,240),(120,60,30))
    train=transform(224,True,'r01b_light_color')
    val=transform(224,False,'r01b_light_color')
    assert train.transforms[0].__class__.__name__=='Resize'
    assert [x.__class__.__name__ for x in train.transforms[1:4]]==[
        'RandomHorizontalFlip','RandomRotation','RandomApply']
    apply=train.transforms[3]
    assert apply.p==.6
    jitter=apply.transforms[0]
    assert jitter.brightness==(.9,1.1)
    assert jitter.contrast==(.9,1.1)
    assert jitter.saturation==(.92,1.08)
    assert jitter.hue==(-.02,.02)
    assert all(x.__class__.__name__ not in {'RandomApply','ColorJitter'} for x in val.transforms)
    assert torch.equal(val(image),val(image))
    assert train(image).shape==(3,224,224)


def test_g01_actual_augmentation_is_basic():
    path=Path('outputs/runs/G01/config.json')
    if not path.exists():return
    c=json.loads(path.read_text(encoding='utf-8'))
    assert c['augmentation']=='basic' and c['seed']==42 and c['image_size']==224
