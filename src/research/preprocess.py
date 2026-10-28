import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms as T


def fov_crop(im, threshold=12, margin=.04, min_fraction=.15):
    rgb=np.asarray(im.convert('RGB'))
    thumb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
    mask=(thumb.max(axis=2)>threshold).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<=1: return im, {'applied':False,'reason':'no_component'}
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);x,y,w,h,area=map(int,stats[idx])
    if area/(512*512)<min_fraction:
        return im,{'applied':False,'reason':'small_component','fraction':area/(512*512)}
    pad=int(margin*max(w,h));x0=max(0,x-pad);y0=max(0,y-pad);x1=min(512,x+w+pad);y1=min(512,y+h+pad)
    box=(round(x0*im.width/512),round(y0*im.height/512),round(x1*im.width/512),round(y1*im.height/512))
    return im.crop(box), {'applied':True,'box':box,'fraction':area/(512*512)}


def lab_clahe(im, clip_limit=1.5):
    rgb=np.asarray(im.convert('RGB'))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=(8,8)).apply(lab[:,:,0])
    return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))


def color_enhance(im, gamma=1., contrast=1., clahe=False):
    if clahe: im=lab_clahe(im)
    if gamma != 1:
        lut=np.array([round(255*(i/255)**gamma) for i in range(256)],dtype=np.uint8)
        im=Image.fromarray(lut[np.asarray(im.convert('RGB'))])
    if contrast!=1: im=ImageEnhance.Contrast(im).enhance(contrast)
    return im


def jpeg_compress(im,quality):
    bio=io.BytesIO();im.save(bio,format='JPEG',quality=quality);bio.seek(0)
    with Image.open(bio) as saved:return saved.convert('RGB')


class OpticDiscLocator:
    def locate(self, image):
        """Return (x, y, confidence) in source-image pixels; implementation must be reviewed."""
        raise NotImplementedError


def crop_disc(im, locator, confidence=.8, scale=.35):
    found=locator.locate(im)
    if found is None or found[2]<confidence:
        return im,{'applied':False,'reason':'low_confidence_or_missing','confidence':None if found is None else found[2]}
    x,y,conf=found
    if not (0<=x<im.width and 0<=y<im.height):
        return im,{'applied':False,'reason':'invalid_coordinate','confidence':conf}
    radius=round(scale*min(im.size)); box=(max(0,x-radius),max(0,y-radius),min(im.width,x+radius),min(im.height,y+radius))
    return im.crop(box),{'applied':True,'box':box,'confidence':conf}


def transform_image(im,cfg,train=False,rng=None,locator=None,return_qc=False):
    rng=np.random.default_rng() if rng is None else rng
    im=ImageOps.exif_transpose(im).convert('RGB');qc={'crop_policy':cfg.crop_policy}
    if cfg.crop_policy=='fov': im,qc['crop']=fov_crop(im,**cfg.extra.get('fov',{}))
    elif cfg.crop_policy=='optic_disc':
        if locator is None: raise ValueError('Reviewed optic disc locator required')
        im,qc['crop']=crop_disc(im,locator,**cfg.extra.get('roi',{}))
    elif cfg.crop_policy!='resize': raise ValueError('Unknown crop policy')
    if cfg.extra.get('enhancement')=='lab_clahe_gamma':
        im=color_enhance(im,**cfg.extra.get('enhancement_params',{}))
    im=im.resize((cfg.image_size,cfg.image_size),Image.Resampling.BILINEAR)
    if train:
        aug=cfg.augmentation
        if aug.get('horizontal_flip',True) and rng.random()<.5:im=ImageOps.mirror(im)
        if aug.get('rotation',10):im=im.rotate(float(rng.uniform(-aug['rotation'],aug['rotation'])))
        jitter=aug.get('color_jitter',0.)
        if jitter:
            im=ImageEnhance.Color(im).enhance(float(rng.uniform(1-jitter,1+jitter)))
            im=ImageEnhance.Brightness(im).enhance(float(rng.uniform(1-jitter,1+jitter)))
        if aug.get('blur_sigma',0) and rng.random()<.5:im=im.filter(ImageFilter.GaussianBlur(aug['blur_sigma']))
        if aug.get('jpeg_quality') and rng.random()<.5:im=jpeg_compress(im,aug['jpeg_quality'])
    tensor=T.Normalize([.485,.456,.406],[.229,.224,.225])(T.ToTensor()(im))
    return (tensor,qc,im) if return_qc else tensor
