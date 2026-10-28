"""Fixed-threshold image linkage diagnostics; no training or split modification."""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score

from src.data import open_rgb
from src.metrics import metrics
from src.utils import sha256, write_json

ENHANCED = '--enhanced' in sys.argv
OUT = Path('outputs/phase2_diagnostics_v3' if ENHANCED else 'outputs/phase2_diagnostics_v2')
POLICY = {'analysis_size': 512, 'sift_features': 2500, 'ratio_test': .7,
          'ransac_pixels': 3., 'near_ssim': .97, 'near_rmse': .03,
          'near_min_inliers': 40, 'near_inlier_fraction': .6, 'near_coverage': .2,
          'same_eye_min_inliers': 20, 'same_eye_inlier_fraction': .5, 'same_eye_coverage': .15,
          'seed': 42, 'bootstrap_draws': 2000,
          'purpose': 'Prespecified diagnostic triage, not validated patient identification or data cleaning'}
POLICY['feature_preprocess'] = '1024 green CLAHE, SIFT contrast .01' if ENHANCED else '512 grayscale default SIFT'


@lru_cache(maxsize=32)
def read(path):
    im = open_rgb(path)
    pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    size = im.size
    rgb = np.asarray(im.resize((512,512), Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (rgb.max(axis=2) > 15).astype(np.uint8) * 255
    if ENHANCED:
        green = np.asarray(im.resize((1024,1024),Image.Resampling.LANCZOS))[:,:,1]
        feature = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(green)
        key, desc = cv2.SIFT_create(nfeatures=2500,contrastThreshold=.01).detectAndCompute(feature,cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST))
        key = [cv2.KeyPoint(k.pt[0]/2,k.pt[1]/2,k.size/2,k.angle,k.response,k.octave,k.class_id) for k in key]
    else:
        key, desc = cv2.SIFT_create(nfeatures=2500).detectAndCompute(gray, mask)
    return rgb, gray, mask, key, desc, pixel_hash, size


def masked_ssim(a, b, mask):
    inner = binary_erosion(mask, iterations=6)
    if inner.sum() < 100: return None, None
    _, score = structural_similarity(a, b, data_range=255, gaussian_weights=True,
                                     sigma=1.5, use_sample_covariance=False, full=True)
    return float(score[inner].mean()), float(np.sqrt(np.mean(((a.astype(float)-b)/255)[inner]**2)))


def classify(r):
    if r['same_file_sha256']: return 'identical_file'
    if r['same_decoded_pixels']: return 'identical_decoded_image'
    if (r['inliers'] >= 40 and r['inlier_fraction'] >= .6 and r['coverage'] >= .2
        and r['aligned_ssim'] is not None and r['aligned_ssim'] >= .97 and r['aligned_rmse'] <= .03):
        return 'visual_near_duplicate'
    if r['inliers'] >= 20 and r['inlier_fraction'] >= .5 and r['coverage'] >= .15:
        return 'suspected_same_eye_different_capture'
    return 'uncertain'


def compare(row, index):
    cv2.setRNGSeed(42)
    a, ag, am, ak, ad, ah, az = read(row.path_train)
    b, bg, bm, bk, bd, bh, bz = read(row.path_val)
    before, _ = masked_ssim(ag, bg, (am > 0) & (bm > 0))
    matches = []
    if ad is not None and bd is not None and len(ad)>1 and len(bd)>1:
        matcher = cv2.BFMatcher()
        forward = [m for m,n in matcher.knnMatch(ad,bd,k=2) if m.distance < .7*n.distance]
        reverse = {(m.trainIdx,m.queryIdx) for m,n in matcher.knnMatch(bd,ad,k=2) if m.distance < .7*n.distance}
        matches = [m for m in forward if (m.queryIdx,m.trainIdx) in reverse]
    inliers, fraction, coverage, aligned, rmse, transform = 0, 0., 0., None, None, None
    valid = np.zeros(len(matches), dtype=bool)
    if len(matches) >= 4:
        p = np.float32([ak[m.queryIdx].pt for m in matches])
        q = np.float32([bk[m.trainIdx].pt for m in matches])
        transform, mask = cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=5000,confidence=.999)
        if transform is not None and mask is not None:
            valid = mask.ravel().astype(bool); inliers = int(valid.sum()); fraction=inliers/len(matches)
            if inliers >= 3:
                coverage = min(cv2.contourArea(cv2.convexHull(p[valid])), cv2.contourArea(cv2.convexHull(q[valid]))) / (512*512)
            warped = cv2.warpAffine(ag,transform,(512,512))
            wm = cv2.warpAffine(am,transform,(512,512),flags=cv2.INTER_NEAREST)
            aligned,rmse = masked_ssim(warped,bg,(wm>0)&(bm>0))
    r = {'pair_id':index, 'train_id':row.image_id_train, 'val_id':row.image_id_val,
         'train_path':row.path_train, 'val_path':row.path_val,
         'train_label':int(row.label_train),'val_label':int(row.label_val),
         'train_dimensions':list(az),'val_dimensions':list(bz),
         'train_sha256':sha256(row.path_train),'val_sha256':sha256(row.path_val),
         'train_decoded_sha256':ah,'val_decoded_sha256':bh,
         'same_file_sha256':sha256(row.path_train)==sha256(row.path_val), 'same_decoded_pixels':ah==bh,
         'label_disagreement':bool(row.label_train!=row.label_val),
         'raw_ssim':before,'aligned_ssim':aligned,'aligned_rmse':rmse,
         'mutual_sift_matches':len(matches),'inliers':inliers,'inlier_fraction':fraction,'coverage':coverage,
         'affine_matrix':transform.tolist() if transform is not None else None}
    r['category'] = classify(r)
    r['high_risk'] = r['category'] != 'uncertain'
    if r['high_risk']:
        evidence = cv2.drawMatches(a,ak,b,bk,matches,None,matchesMask=valid.astype(int).tolist(),flags=2)
        Image.fromarray(evidence).save(OUT / 'matches' / f'pair_{index:04d}.jpg')
    return r


def bootstrap(pred, excluded, groups, repeats=2000):
    y=pred.label.to_numpy(); p=pred.probability.to_numpy(); keep=~pred.image_id.isin(excluded).to_numpy()
    rng=np.random.default_rng(42); rows=[]; attempts=0
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    while len(rows)<repeats and attempts<repeats*20:
        attempts+=1
        idx=(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) if groups is not None
             else np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
        clean=idx[keep[idx]]
        if len(np.unique(y[idx]))<2 or len(np.unique(y[clean]))<2: continue
        full=roc_auc_score(y[idx],p[idx]); diagnostic=roc_auc_score(y[clean],p[clean])
        rows.append((full,diagnostic,diagnostic-full))
    if len(rows)!=repeats: raise ValueError('Insufficient bootstrap draws')
    return {'replicates':repeats,'seed':42,
            **{name:dict(zip(['low','high'],np.quantile(np.asarray(rows)[:,i],[.025,.975]).tolist()))
               for i,name in enumerate(['full_ci95','diagnostic_ci95','diagnostic_minus_full_ci95'])}}


def diagnostics(pairs):
    pred=pd.read_csv('outputs/runs/B02/validation.csv')
    official=pd.read_csv('data/official/manifests/valid.csv')
    assert list(pred.image_id)==list(official.image_id) and np.array_equal(pred.label,official.label)
    graph=nx.Graph(); graph.add_nodes_from('v:'+x for x in pred.image_id)
    high=pairs[pairs.high_risk]
    # Conservative sensitivity analysis: connect all screened candidates, not asserted patients.
    for r in pairs.itertuples(): graph.add_edge('t:'+r.train_id,'v:'+r.val_id)
    lookup={x:i for i,x in enumerate(pred.image_id)}
    groups=[np.array([lookup[n[2:]] for n in g if n.startswith('v:')],dtype=int) for g in nx.connected_components(graph)]
    groups=[g for g in groups if len(g)]
    scopes={'high_risk':set(high.val_id), 'strict_duplicate':set(pairs[pairs.category.isin(['identical_file','identical_decoded_image','visual_near_duplicate'])].val_id),
            'all_phash_candidates':set(pairs.val_id)}
    report={'official_full':metrics(pred.label,pred.probability),'n':len(pred),'diagnostic_only':True,
            'label_blind_risk_policy':POLICY,'scenarios':{},'patient_id_available':False,
            'cluster_basis':'Connected components of all 430 pHash candidates; NOT verified patient groups',
            'validation_components':len(groups),'largest_validation_component':max(map(len,groups))}
    for name,excluded in scopes.items():
        keep=~pred.image_id.isin(excluded); clean=pred[keep]
        counts={str(label):{'total':int((pred.label==label).sum()),
            'affected':int(((pred.label==label)&~keep).sum()),
            'affected_fraction':float(((pred.label==label)&~keep).sum()/(pred.label==label).sum())} for label in [0,1]}
        result={'excluded_count':len(excluded),'retained':len(clean),'class_impact':counts,
                'metrics':metrics(clean.label,clean.probability),
                'image_stratified_bootstrap':bootstrap(pred,excluded,None),
                'candidate_component_bootstrap':bootstrap(pred,excluded,groups)}
        report['scenarios'][name]=result
        print(f'Diagnostics {name}: {result["metrics"]["auroc"]:.9f}',flush=True)
    membership=pd.DataFrame({'image_id':pred.image_id,**{name:pred.image_id.isin(ids) for name,ids in scopes.items()}})
    membership.to_csv(OUT/'diagnostic_membership.csv',index=False)
    write_json(OUT/'b02_diagnostics.json',report)


def main():
    if (OUT/'pairs.jsonl').exists(): raise FileExistsError('Refusing to overwrite completed/partial review')
    OUT.mkdir(exist_ok=True); (OUT/'matches').mkdir(exist_ok=True); (OUT/'contact_sheets').mkdir(exist_ok=True)
    cv2.setNumThreads(2); write_json(OUT/'policy.json',POLICY)
    frame=pd.read_csv('outputs/phase2_g01/near_duplicate_audit/cross_split_identical_phash.csv')
    rows=[]
    with (OUT/'pairs.jsonl').open('x',encoding='utf-8') as f:
        for i,row in enumerate(frame.itertuples()):
            result=compare(row,i); rows.append(result); f.write(json.dumps(result)+'\n');f.flush()
            if (i+1)%25==0: print(f'Reviewed {i+1}/{len(frame)}',flush=True)
    pairs=pd.DataFrame(rows); pairs.to_csv(OUT/'pair_review.csv',index=False)
    risk=pairs[pairs.high_risk]
    risk.to_csv(OUT/'high_risk_contact_table.csv',index=False)
    records=[]
    for start in range(0,len(risk),6):
        sheet=Image.new('RGB',(1024,6*270),'white');draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(risk.iloc[start:start+6].itertuples()):
            for x,path in [(0,r.train_path),(512,r.val_path)]:
                im=Image.fromarray(read(path)[0]);im.thumbnail((500,240));sheet.paste(im,(x,k*270+30))
            draw.text((0,k*270),f'{r.pair_id}: {r.train_id} / {r.val_id} {r.category} labels={r.train_label}/{r.val_label}',fill='black')
            records.append({'pair_id':r.pair_id,'sheet':f'contact_sheets/sheet_{start//6:03d}.jpg','row':k,'matches':f'matches/pair_{r.pair_id:04d}.jpg'})
        sheet.save(OUT/'contact_sheets'/f'sheet_{start//6:03d}.jpg')
    pd.DataFrame(records).to_csv(OUT/'contact_index.csv',index=False)
    write_json(OUT/'summary.json',{'pairs':len(pairs),'categories':pairs.category.value_counts().to_dict(),
        'high_risk_pairs':len(risk),'high_risk_val_images':risk.val_id.nunique(),
        'high_risk_label_disagreements':int(risk.label_disagreement.sum()),'training_gate':'pending_evidence_review',
        'data_modified':False,'test_used':False,'identity_inference':'provisional, no patient IDs'})
    diagnostics(pairs)
    print('Multilevel audit and diagnostics complete',flush=True)


if __name__=='__main__': main()
