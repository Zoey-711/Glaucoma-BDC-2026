import json
import pandas as pd
import pytest

from scripts.accept_t4 import preflight
from src.utils import sha256


def fixture(tmp_path):
    manifest=tmp_path/'test.csv';ids=[f'test_{i:05d}.jpg' for i in range(5855)]
    pd.DataFrame({'image_id':ids}).to_csv(manifest,index=False)
    submission=tmp_path/'submission.csv'
    pd.DataFrame({'Image_name':ids,'Probability':[.25]*5855,'Pred_Label':[0]*5855}).to_csv(submission,index=False)
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'example')
    meta={'manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),
          'submission_sha256':sha256(submission),'precision':'fp32','threshold':.5}
    (tmp_path/'submission.csv.metadata.json').write_text(json.dumps(meta))
    return submission,checkpoint,manifest


def test_preflight_is_not_t4_pass(tmp_path):
    submission,checkpoint,manifest=fixture(tmp_path)
    result=preflight(submission,checkpoint,manifest)
    assert result['status']=='UNVERIFIED' and result['rows']==5855


def test_wrong_checkpoint_rejected(tmp_path):
    submission,checkpoint,manifest=fixture(tmp_path)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checkpoint differs'):
        preflight(submission,checkpoint,manifest)
