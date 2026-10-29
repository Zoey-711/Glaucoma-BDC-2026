"""Derive L03 configs from the frozen, actually executed G01 configuration."""
import copy
import json
from pathlib import Path
import yaml
from src.utils import sha256, write_json


def main():
    original=json.loads(Path('outputs/runs/G01/config.json').read_text(encoding='utf-8'))
    evidence={'reference':'G01','allowed_differences':['experiment_id','output','loss','plan'],'configs':{},
              'G01_training_code_sha256':sha256('outputs/phase2_g01_execution/source_snapshot/src/experiment_training.py'),
              'current_training_code_sha256':sha256('src/experiment_training.py')}
    assert evidence['G01_training_code_sha256']==evidence['current_training_code_sha256']
    for eid,weight in [('L03a',0.),('L03b',.05),('L03c',.2)]:
        config=copy.deepcopy(original)
        config.update(experiment_id=eid,output=f'results/{eid}/seed_42/fold_none',loss={'name':'pairwise_auc','lambda':weight},
                      plan={'stage':'L','kind':'train','reference':'G01','factor':'pairwise_lambda','initial_status':'planned',
                            'authorization':'Only L03b formal training authorized; L03a and L03c are configurations only'})
        path=Path(f'configs/experiments/{eid}.yaml')
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
        actual={k for k in set(config)|set(original) if config.get(k)!=original.get(k)}
        assert actual<=set(evidence['allowed_differences'])
        evidence['configs'][eid]={'sha256':sha256(path),'changed_fields':sorted(actual),'lambda':weight}
    write_json('outputs/M02d_L03/G01_protocol_equivalence.json',evidence)


if __name__=='__main__':main()
