from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap import paired_bootstrap
from src.metrics import metrics
from .results import validate_result


def analyze(result_dirs,official_val,output,reference='outputs/runs/B02/validation.csv'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    baseline=pd.read_csv(reference).rename(columns={'image_id':'Image_name','label':'Label','probability':'Probability'})
    rows=[];paired=[];latency=[];seed_groups={}
    for path in result_dirs:
        info=validate_result(path,official_val,freeze_inventory='outputs/migration_review/frozen_inventory.json')
        pred=pd.read_csv(Path(path)/'validation.csv')
        b=baseline.set_index('Image_name').loc[pred.Image_name].reset_index()
        if not np.array_equal(pred.Label,b.Label):raise ValueError('B02 labels differ')
        a=b.rename(columns={'Image_name':'image_id','Label':'label','Probability':'probability'})
        c=pred.rename(columns={'Image_name':'image_id','Label':'label','Probability':'probability'})
        compare=paired_bootstrap(a,c,repeats=2000,seed=42)
        paired.append({'experiment_id':info['experiment_id'],'seed':info['seed'],
                       **{k:v for k,v in compare.items() if k!='seed'},'bootstrap_seed':compare['seed']})
        latency_file=Path(path)/'latency_4090.json'
        local_ms=None
        if latency_file.exists():
            import json
            report=json.loads(latency_file.read_text(encoding='utf-8'))
            local_ms=report.get('mean_ms')
            latency.append({'experiment_id':info['experiment_id'],'gpu':report.get('gpu'),
                            'local_latency_ms':local_ms,'t4_latency_ms':None,'official_t4':False})
        rows.append({**info,'delta_auroc':compare['delta_auroc'],'ci95_low':compare['ci95_low'],
                     'ci95_high':compare['ci95_high'],'local_latency_ms':local_ms,'result_dir':str(path)})
        seed_groups.setdefault(info['experiment_id'],[]).append(info['AUROC'])
    board=pd.DataFrame(rows,columns=['experiment_id','seed','fold','AUROC','AUPRC','delta_auroc','ci95_low','ci95_high',
                                     'model_size_mb','local_latency_ms','config_hash','git_commit','status','result_dir'])
    if len(board):board=board.sort_values(['AUROC','AUPRC'],ascending=False)
    board.to_csv(out/'experiment_leaderboard.csv',index=False)
    pd.DataFrame(paired,columns=['experiment_id','seed','bootstrap_seed','delta_auroc','ci95_low','ci95_high','replicates','unit','interpretation']).to_csv(out/'paired_bootstrap.csv',index=False)
    pd.DataFrame(latency,columns=['experiment_id','gpu','local_latency_ms','t4_latency_ms','official_t4']).to_csv(out/'latency_report.csv',index=False)
    summary=['# Ablation report','','All scores use official full validation. B02 is frozen. Single-seed deltas are exploratory.','']
    for r in board.itertuples(index=False):
        summary.append(f'- {r.experiment_id} seed={r.seed}: AUROC={r.AUROC:.6f}, AUPRC={r.AUPRC:.6f}, ΔAUROC={r.delta_auroc:+.6f}, CI=[{r.ci95_low:+.6f},{r.ci95_high:+.6f}], size={r.model_size_mb:.2f}MB.')
    summary+=['','A confidence interval crossing zero does not establish a stable gain. Repeated seeds and patient-independent validation are separate requirements.','No 4090 latency is a T4 measurement.']
    (out/'ablation_report.md').write_text('\n'.join(summary),encoding='utf-8')
    return board
