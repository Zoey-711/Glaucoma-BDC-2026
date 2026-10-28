import numpy as np

from scripts.phase2_multilevel_audit import classify, masked_ssim


def test_ssim_identity_and_mask_excludes_background():
    a=np.random.default_rng(42).integers(20,255,(64,64),dtype=np.uint8)
    b=a.copy(); b[:8]=0
    mask=np.zeros((64,64),bool);mask[16:48,16:48]=True
    s,r=masked_ssim(a,b,mask)
    assert s==1 and r==0


def test_same_eye_not_automatically_exact_duplicate():
    r={'same_file_sha256':False,'same_decoded_pixels':False,'inliers':60,
       'inlier_fraction':.9,'coverage':.5,'aligned_ssim':.8,'aligned_rmse':.1}
    assert classify(r)=='suspected_same_eye_different_capture'
    r.update(aligned_ssim=.98,aligned_rmse=.02)
    assert classify(r)=='visual_near_duplicate'
    r['inliers']=2
    assert classify(r)=='uncertain'
    r['same_decoded_pixels']=True
    assert classify(r)=='identical_decoded_image'
    r['same_file_sha256']=True
    assert classify(r)=='identical_file'
