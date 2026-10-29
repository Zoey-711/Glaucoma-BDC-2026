"""Independent G02/G01 paired comparison and training-curve artifact."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bootstrap import paired_bootstrap
from src.metrics import metrics
from src.research.results import validate_result
from src.utils import write_json


def main():
    result=Path('results/G02/seed_42/fold_none')
    output=Path('outputs/G02_analysis');output.mkdir(parents=True,exist_ok=True)
    checked=validate_result(result,'data/official/manifests/valid.csv','configs/research/G02.yaml',
                            freeze_inventory='outputs/migration_review/frozen_inventory.json')
    g01=pd.read_csv('outputs/runs/G01/validation.csv')
    g02=pd.read_csv(result/'validation.csv')
    reference=g01.rename(columns={'image_id':'image_id','label':'label','probability':'probability'})
    candidate=g02.rename(columns={'Image_name':'image_id','Label':'label','Probability':'probability'})
    if list(reference.image_id)!=list(candidate.image_id) or not np.array_equal(reference.label,candidate.label):
        raise ValueError('G01/G02 official validation IDs or labels differ')
    baseline=metrics(reference.label,reference.probability)
    comparison=paired_bootstrap(reference,candidate,repeats=2000,seed=42)
    history=pd.read_csv(result/'train_history.csv')
    run=json.loads((result/'run_manifest.json').read_text(encoding='utf-8'))
    best=json.loads((result/'metrics.json').read_text(encoding='utf-8'))
    latency_file=output/'latency_4060.json'
    latency=json.loads(latency_file.read_text(encoding='utf-8')) if latency_file.exists() else None
    g01_latency=json.loads(Path('outputs/runs/G01/latency/local_fp32.json').read_text(encoding='utf-8'))
    if latency and latency['sample_ids']!=g01_latency['sample_ids']:
        raise ValueError('G01/G02 local latency samples differ')
    elapsed=(datetime.fromisoformat(run['finished_utc'])-datetime.fromisoformat(run['started_utc'])).total_seconds()
    report={'experiment_id':'G02','reference_id':'G01','same_validation_images':len(g02),
            'G01_AUROC':baseline['auroc'],'G01_AUPRC':baseline['auprc'],
            'G02_AUROC':checked['AUROC'],'G02_AUPRC':checked['AUPRC'],
            'G02_best_epoch':best['best_epoch'],'G02_recall_at_0_5':best['recall'],
            'G02_specificity_at_0_5':best['specificity'],
            'paired_bootstrap':comparison,'epochs_completed':len(history),
            'train_seconds_sum':float(history.train_seconds.sum()),'run_wall_seconds':elapsed,
            'peak_allocated_mib':float(history.train_peak_allocated_mib.max()),
            'peak_reserved_mib':float(history.train_peak_reserved_mib.max()),
            'model_bytes':(result/'best.pth').stat().st_size,
            'local_latency_4060_ms':latency['mean_ms'] if latency else None,
            'local_latency_4060_p95_ms':latency['p95_ms'] if latency else None,
            'G01_local_latency_4060_ms':g01_latency['mean_ms'],
            'local_latency_delta_ms':latency['mean_ms']-g01_latency['mean_ms'] if latency else None,
            'T4_latency_ms':None,'git_commit':run['git_commit'],
            'data_manifest_hash':run['data_manifest_hash'],'config_hash':run['config_hash'],
            'single_seed':True,'patient_independence_unverified':True}
    write_json(output/'G02_comparison.json',report)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,column,title in zip(axes,['train_loss','auroc','auprc'],['Train BCE loss','Official Val AUROC','Official Val AUPRC']):
        ax.plot(history.epoch,history[column],marker='o',label='G02 384')
        if column=='auroc':ax.axhline(baseline['auroc'],color='tab:orange',ls='--',label='G01 224 best')
        if column=='auprc':ax.axhline(baseline['auprc'],color='tab:orange',ls='--',label='G01 224 best')
        ax.axvline(best['best_epoch'],color='gray',ls=':',alpha=.7)
        ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    lines=['# G02 ConvNeXt-Tiny-384 实验报告','',
           f"官方 Val 共 {len(g02)} 张；G02 最佳 epoch {best['best_epoch']}，共完成 {len(history)} 轮。",
           f"G02 AUROC {checked['AUROC']:.6f}，AUPRC {checked['AUPRC']:.6f}；G01 AUROC {baseline['auroc']:.6f}，AUPRC {baseline['auprc']:.6f}。",
           f"配对 ΔAUROC={comparison['delta_auroc']:+.6f}，图像分层 Bootstrap 95% CI [{comparison['ci95_low']:+.6f}, {comparison['ci95_high']:+.6f}]（{comparison['replicates']} 次）。",
           f"完整运行 {elapsed/3600:.2f} 小时，Train 累计 {history.train_seconds.sum()/3600:.2f} 小时；峰值显存分配 {report['peak_allocated_mib']:.0f} MiB、保留 {report['peak_reserved_mib']:.0f} MiB；模型 {(result/'best.pth').stat().st_size/1e6:.2f} MB。",
           f"RTX 4060 同样本端到端平均延迟：G02 {latency['mean_ms']:.2f} ms，G01 {g01_latency['mean_ms']:.2f} ms，增加 {latency['mean_ms']-g01_latency['mean_ms']:.2f} ms。" if latency else 'RTX 4060 本地延迟：尚未测得。',
           '', '结论：本次不优先执行 G04 512 或 G02 五折。384 未显示 AUROC/AUPRC 收益，4060 延迟已超过 100 ms；先做严格同管线、重复种子的分辨率复核，再考虑更高分辨率或五折。',
           '单种子、无患者 ID 的验证差值仅作探索性比较；4090/4060 延迟不能代替 T4 实测。',
           '训练器与冻结 G01 的图像缓存、增强随机序列存在实现差异，分辨率并非严格唯一变化；此限制须在复核中保留。']
    (output/'G02_experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
