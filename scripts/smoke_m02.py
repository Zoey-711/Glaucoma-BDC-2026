"""Small synthetic fusion-head training test; no official images or labels."""
from pathlib import Path
import numpy as np
import torch

from scripts.run_m02 import fit_head, head_logits
from src.multimodal import AgePreprocessor
from src.utils import write_json


def main():
    out = Path('outputs/M02/smoke'); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96,768)).astype(np.float32)
    y = np.tile([0,1],48)
    age = rng.uniform(20,90,96)
    p = AgePreprocessor.fit(age[:64], 'internal_train')
    a = p.apply(age)
    original = x.copy()
    head, chosen = fit_head(x[:64],a[:64],y[:64],.001,2,out/'history.csv', (x[64:],a[64:],y[64:]))
    z = head_logits(head,x[64:],a[64:])
    assert z.shape == (32,) and np.isfinite(z).all()
    assert np.array_equal(x,original)
    write_json(out/'report.json', {'status':'PASSED','synthetic_only':True,'image_features_unchanged':True,
               'binary_logits_shape':list(z.shape),'selected_epoch':chosen['best_epoch'],
               'gpu':torch.cuda.get_device_name(),'official_training_started':False})


if __name__ == '__main__':
    main()
