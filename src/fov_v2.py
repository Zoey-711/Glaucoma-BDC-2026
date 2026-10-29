"""Frozen per-image FOV proposal with native-resolution lossless-signal guard.

This is not vessel segmentation. Weak texture/connected edges flag uncertainty.
Only exactly black decoded RGB pixels may be removed; any nonzero signal outside
the proposed safe box forces an unchanged full-image output.
"""
import math

import cv2
import numpy as np
from PIL import Image


POLICY = dict(version='safe_fov_v2', thumbnail_long_side=512,
              core_value=8, core_saturation=20, achromatic_value=20,
              closing_kernel=3, min_component_fraction=.15,
              min_component_fill=.45, min_axis_ratio=.55,
              max_secondary_fraction=.01, margin_fraction=.10,
              margin_thumbnail_pixels=2, min_removed_area_fraction=.02,
              native_outside_max_value=0, weak_value=1,
              texture_gradient=1., connected_weak_min_pixels=3)


def contains(outer, inner):
    return inner is None or (outer[0]<=inner[0] and outer[1]<=inner[1]
                             and outer[2]>=inner[2] and outer[3]>=inner[3])


def crop_fov_v2(image, diagnostics=False):
    im=image.convert('RGB');W,H=im.size
    scale=min(1.,POLICY['thumbnail_long_side']/max(W,H))
    w,h=max(1,round(W*scale)),max(1,round(H*scale))
    rgb=np.asarray(im.resize((w,h),Image.Resampling.BOX))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);v,s=hsv[:,:,2],hsv[:,:,1]
    core=((v>=POLICY['core_value']) & ((s>=POLICY['core_saturation']) | (v>=POLICY['achromatic_value']))).astype('uint8')
    k=POLICY['closing_kernel'];core=cv2.morphologyEx(core,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    weak=v>=POLICY['weak_value']
    gx=cv2.Sobel(v,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(v,cv2.CV_32F,0,1,ksize=3)
    texture=(np.abs(gx)+np.abs(gy))>=POLICY['texture_gradient']
    native_support=im.getbbox()  # exact union of every nonzero decoded RGB channel
    full=[0,0,W,H]
    meta=dict(width=W,height=H,box=full,proposed_box=full,applied=False,fallback=True,
              area_ratio=1.,brightness=float(v.mean()),black_fraction=float((v<8).mean()),
              eccentricity=0.,foreground_fraction=float(core.mean()),component_fraction=0.,
              native_signal_box=list(native_support) if native_support else None,
              boundary_risk=False,weak_pixels_outside_proposal=0,texture_pixels_outside_proposal=0,
              continuous_structure_candidate=False,native_signal_outside_proposal=False)
    masks=dict(core=core*255,weak=weak.astype('uint8')*255,texture=texture.astype('uint8')*255)
    def result(reason,box=None):
        m={**meta,'reason':reason}
        if box is not None:
            m.update(box=box,applied=True,fallback=False,area_ratio=(box[2]-box[0])*(box[3]-box[1])/(W*H))
        out=im if box is None else im.crop(box)
        return (out,m,masks) if diagnostics else (out,m)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(core,8)
    if n<=1:return result('no_confident_foreground')
    idx=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,bw,bh,area=map(int,stats[idx])
    meta['component_fraction']=area/(w*h)
    meta['eccentricity']=float(np.linalg.norm((centers[idx]-[w/2,h/2])/[w,h]))
    if area/(w*h)<POLICY['min_component_fraction'] or area/(bw*bh)<POLICY['min_component_fill']:
        return result('small_or_fragmented_field')
    if min(bw,bh)/max(bw,bh)<POLICY['min_axis_ratio']:
        return result('uncertain_geometry')
    if (int(core.sum())-area)/max(area,1)>POLICY['max_secondary_fraction']:
        return result('separate_foreground')
    pad=math.ceil(POLICY['margin_fraction']*max(bw,bh))+POLICY['margin_thumbnail_pixels']
    x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
    proposed=[math.floor(x0*W/w),math.floor(y0*H/h),math.ceil(x1*W/w),math.ceil(y1*H/h)]
    meta['proposed_box']=proposed
    outside=np.ones((h,w),bool);outside[y0:y1,x0:x1]=False
    weak_out=weak & outside
    meta['weak_pixels_outside_proposal']=int(weak_out.sum())
    meta['texture_pixels_outside_proposal']=int((texture & outside).sum())
    cc,_,cs,_=cv2.connectedComponentsWithStats(weak_out.astype('uint8'),8)
    meta['continuous_structure_candidate']=bool(cc>1 and cs[1:,cv2.CC_STAT_AREA].max()>=POLICY['connected_weak_min_pixels'])
    meta['native_signal_outside_proposal']=not contains(proposed,native_support)
    meta['boundary_risk']=bool(meta['native_signal_outside_proposal'] or meta['weak_pixels_outside_proposal'] or meta['texture_pixels_outside_proposal'])
    if meta['boundary_risk']:return result('weak_signal_or_texture_outside')
    if (proposed[2]-proposed[0])*(proposed[3]-proposed[1])/(W*H)>1-POLICY['min_removed_area_fraction']:
        meta['fallback']=False
        return result('already_full_field')
    return result('cropped',proposed)


class SafeFOVCropV2:
    def __call__(self,image):return crop_fov_v2(image)[0]
