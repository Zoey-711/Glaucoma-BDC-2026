"""Conservative deterministic field-of-view crop v1; no fitted data statistics."""
import math

import cv2
import numpy as np
from PIL import Image


def crop_fov(image):
    im = image.convert('RGB')
    scale = min(1., 512 / max(im.size))
    w, h = max(1, round(im.width*scale)), max(1, round(im.height*scale))
    a = np.asarray(im.resize((w, h), Image.Resampling.BOX))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    # Include dim coloured peripheral tissue; achromatic bright structures also count.
    mask = ((v >= 8) & ((s >= 20) | (v >= 20))).astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    meta = dict(width=im.width, height=im.height, box=[0, 0, im.width, im.height],
                applied=False, fallback=True, area_ratio=1., brightness=float(v.mean()),
                black_fraction=float((v < 8).mean()), eccentricity=0., foreground_fraction=float(mask.mean()))
    def fallback(reason):
        return im, {**meta, 'reason': reason}
    if n <= 1:
        return fallback('no_foreground')
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = map(int, stats[idx])
    meta['eccentricity'] = float(np.linalg.norm((centers[idx]-[w/2,h/2])/[w,h]))
    meta['component_fraction'] = area/(w*h)
    if area/(w*h) < .15 or area/(bw*bh) < .45:
        return fallback('small_or_fragmented_field')
    if min(bw,bh)/max(bw,bh) < .55:
        return fallback('non_fundus_geometry')
    if (mask.sum()-area)/max(area, 1) > .01:
        return fallback('separate_foreground')
    # 6% of largest diameter plus two thumbnail pixels protects dim boundary tissue.
    pad = math.ceil(.06*max(bw,bh))+2
    x0,y0,x1,y1 = max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
    outside = np.ones((h,w), bool); outside[y0:y1,x0:x1] = False
    # Reject a proposed crop if it would remove visible dim tissue, including colourless tissue.
    if np.count_nonzero((v >= 8) & outside) > .0001*w*h:
        return fallback('visible_signal_outside_box')
    box = [math.floor(x0*im.width/w),math.floor(y0*im.height/h),
           math.ceil(x1*im.width/w),math.ceil(y1*im.height/h)]
    ratio = (box[2]-box[0])*(box[3]-box[1])/(im.width*im.height)
    if ratio > .98:
        return im, {**meta, 'fallback':False, 'reason':'already_full_field'}
    return im.crop(box), {**meta, 'box':box, 'area_ratio':ratio, 'applied':True,
                          'fallback':False, 'reason':'cropped'}


class SafeFOVCrop:
    def __call__(self, image):
        return crop_fov(image)[0]
