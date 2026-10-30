import numpy as np
import pandas as pd
import pytest
from src.phase_g import compare_frozen_to_inference,strict_pair_comparison


def frame(scores):
    return pd.DataFrame({'image_id':['a','b','c','d'],'label':[1,1,0,0],
        'probability':scores})


def test_pair_comparison_counts_and_alignment():
    a=frame([.9,.2,.8,.1]);b=frame([.9,.7,.3,.1])
    v=strict_pair_comparison(a,b)
    assert v['positive_negative_pairs']==4 and v['F04_fixes_F01_pairs']==1
    assert v['F04_breaks_F01_pairs']==0 and v['net_pair_loss_reduction']==1
    with pytest.raises(ValueError):strict_pair_comparison(a,b.iloc[::-1])


def test_locked_inference_consistency_detects_small_rank_flip():
    frozen=frame([.50001,.7,.1,.2]);actual=frozen[['image_id','probability']].copy()
    assert compare_frozen_to_inference(frozen,actual)['passed']
    actual.loc[0,'probability']=.49999
    result=compare_frozen_to_inference(frozen,actual)
    assert result['pred_label_mismatches']==1 and not result['passed']
    actual=frozen[['image_id','probability']].copy();actual.loc[2,'probability']=.21
    assert compare_frozen_to_inference(frozen,actual,tolerance=.2)['pairwise_order_mismatches']>=1
