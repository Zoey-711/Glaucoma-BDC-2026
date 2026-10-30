"""Conservative Train-only optic-disc proposal. Never a reviewed locator by itself."""
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DiscProposal:
    x: int | None
    y: int | None
    confidence: float
    box: tuple[int,int,int,int] | None
    applied: bool
    reason: str
    latency_ms: float
    metrics: dict


class ConservativeDiscLocator:
    """Deterministic heuristic for QC candidates, with no formal-use approval."""
    reviewed=False

    def __init__(self, confidence_threshold=.85):
        self.confidence_threshold=float(confidence_threshold)
        if not .5<=self.confidence_threshold<=1:
            raise ValueError('Invalid confidence threshold')

    def locate(self, image):
        proposal=self.propose(image)
        return (proposal.x,proposal.y,proposal.confidence) if proposal.applied else None

    def propose(self, image):
        start=time.perf_counter_ns()
        rgb=np.asarray(image.convert('RGB'))
        height,width=rgb.shape[:2]
        def finish(x=None,y=None,confidence=0.,box=None,reason='uncertain',metrics=None):
            applied=box is not None and confidence>=self.confidence_threshold
            return DiscProposal(x,y,float(confidence),box if applied else None,applied,
                'accepted_proposal_requires_human_qc' if applied else reason,
                (time.perf_counter_ns()-start)/1e6,metrics or {})
        if min(width,height)<128:return finish(reason='small_image')
        thumb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(thumb,cv2.COLOR_RGB2HSV)
        gray=cv2.cvtColor(thumb,cv2.COLOR_RGB2GRAY)
        field=((gray>12)|(hsv[:,:,1]>22)).astype(np.uint8)
        n,components,stats,_=cv2.connectedComponentsWithStats(field,8)
        if n<2:return finish(reason='no_field')
        chosen=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        valid=components==chosen
        coverage=float(valid.mean())
        if not .25<=coverage<=.92:return finish(reason='field_geometry',metrics={'field_coverage':coverage})
        values=gray[valid]
        mean=float(values.mean())
        blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        if mean<22:return finish(reason='very_dark',metrics={'field_mean':mean,'field_coverage':coverage})
        if blur<8:return finish(reason='blurred',metrics={'laplacian_variance':blur})
        green=cv2.GaussianBlur(thumb[:,:,1],(0,0),5).astype(np.float32)
        threshold=float(np.percentile(green[valid],97))
        bright=((green>=threshold)&valid).astype(np.uint8)
        bright=cv2.morphologyEx(bright,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,labels,regions,centers=cv2.connectedComponentsWithStats(bright,8)
        if count<2:return finish(reason='no_bright_component')
        candidates=[]
        for idx in range(1,count):
            area=int(regions[idx,cv2.CC_STAT_AREA]);cx,cy=centers[idx]
            if 30<=area<=3500 and 18<cx<238 and 18<cy<238:
                candidates.append((area,idx,cx,cy))
        if not candidates:return finish(reason='ambiguous_bright_region')
        candidates.sort(reverse=True)
        area,idx,cx,cy=candidates[0]
        contrast=float(green[labels==idx].mean()-np.median(green[valid]))
        separation=float(area/candidates[1][0]) if len(candidates)>1 else 10.
        confidence=float(np.clip(min(contrast/70,1)*min(separation/2,1),0,1))
        x=round(cx*width/256);y=round(cy*height/256)
        radius=round(.18*min(width,height))
        box=(x-radius,y-radius,x+radius,y+radius)
        metrics={'field_coverage':coverage,'field_mean':mean,'laplacian_variance':blur,
            'bright_area':area,'contrast':contrast,'second_component_ratio':separation}
        if min(box[0],box[1])<0 or box[2]>width or box[3]>height:
            return finish(x,y,confidence,reason='roi_boundary',metrics=metrics)
        if confidence<self.confidence_threshold:
            return finish(x,y,confidence,reason='low_confidence',metrics=metrics)
        return finish(x,y,confidence,box,metrics=metrics)

    def crop_or_fallback(self,image):
        proposal=self.propose(image)
        return image.crop(proposal.box) if proposal.applied else image,proposal
