import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .bootstrap import paired_bootstrap
from .registry import Registry, FIELDS, training_hash
from .utils import write_json


def table(df):
    if df.empty: return '暂无已完成实验。'
    fmt = lambda v: '' if pd.isna(v) else str(round(v, 6)) if isinstance(v, float) else str(v)
    return '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---'] * len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False, name=None))


def training_plot(output):
    out = Path(output)
    if not (out / 'history.json').exists(): return
    h = pd.read_json(out / 'history.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ['train_loss', 'val_loss']:
        if key in h: axes[0].plot(h.epoch, h[key], label=key)
    for key in ['auroc', 'auprc']:
        axes[1].plot(h.epoch, h[key], label=key)
    for ax in axes: ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=160); plt.close(fig)


def recommend(rows):
    b = next((r for r in rows if r['experiment_id'] == 'B02' and r['status'] == 'completed'), None)
    if b is None: return []
    recommendations = [
        {'experiment_id': 'I02', 'reason': '训练集阳性仅约3.2%；以训练计数计算Weighted BCE，检验召回与AUROC权衡。'},
        {'experiment_id': 'I03', 'reason': '固定其余设置，检验Focal对难例与极端不平衡的效果；不预先宣称增益。'},
        {'experiment_id': 'P02', 'reason': '单独检验黑边裁切能否提高有效视网膜区域比例。'},
        {'experiment_id': 'P03', 'reason': '单独提升到320，检验细节与本地速度的权衡，后续仍需T4实测。'},
        {'experiment_id': 'B03', 'reason': '提供ConvNeXt-Tiny架构对照；先完成其独立权重许可审核。'}]
    if b.get('local_latency_ms') is not None and b['local_latency_ms'] > 100:
        recommendations = [{'experiment_id': 'D01', 'reason': f"B02本地原图端到端均值{b['local_latency_ms']:.2f}ms，先在验证集比较FP32/FP16的精度与速度；RTX 4060结果不能替代T4。"}] + [r for r in recommendations if r['experiment_id'] != 'P03']
    return recommendations


def analyze(registry=None, out_dir='outputs/analysis', shortlist=None, repeats=2000):
    registry = registry or Registry()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = registry.rows(); registry.export()
    frame = pd.DataFrame([{k: r.get(k) for k in FIELDS} for r in rows], columns=FIELDS)
    frame.sort_values(['AUROC', 'AUPRC'], ascending=False, na_position='last').to_csv(out / 'experiment_leaderboard.csv', index=False)
    completed = [r for r in rows if r['status'] in ['completed', 'reused'] and not r['config'].get('smoke')]
    by_id = {r['experiment_id']: r for r in rows}
    ablations = []
    for r in completed:
        ref = by_id.get(r['config'].get('plan', {}).get('reference'))
        if ref and ref.get('AUROC') is None:
            ref = next((other for other in completed if training_hash(other['config']) == training_hash(ref['config'])), ref)
        if ref and ref.get('AUROC') is not None:
            ablations.append({'experiment_id': r['experiment_id'], 'reference': ref['experiment_id'],
                'factor': r['config']['plan'].get('factor'), 'delta_auroc': r['AUROC'] - ref['AUROC'],
                'delta_auprc': r['AUPRC'] - ref['AUPRC'], 'evidence': 'single_seed_official_validation_only'})
    abl = pd.DataFrame(ablations, columns=['experiment_id', 'reference', 'factor', 'delta_auroc', 'delta_auprc', 'evidence'])
    abl.to_csv(out / 'ablation_summary.csv', index=False)
    latency_cols = ['experiment_id', 'AUROC', 'model_size_mb', 'local_latency_ms', 't4_latency_ms', 'status']
    frame[latency_cols].to_csv(out / 'model_latency_report.csv', index=False)
    predictions = {}
    for r in completed:
        file = Path(r['output']) / 'validation.csv'
        if file.exists(): predictions[r['experiment_id']] = pd.read_csv(file, dtype={'image_id': str, 'patient_id': str})
        training_plot(r['output'])
    comparisons = []
    if shortlist is None and len(predictions) >= 2:
        ranked = sorted([r for r in completed if r['experiment_id'] in predictions], key=lambda r: (r['AUROC'], r['AUPRC']), reverse=True)
        unique, seen = [], set()
        for r in ranked:
            if r['checkpoint_path'] not in seen:
                seen.add(r['checkpoint_path']); unique.append(r['experiment_id'])
        reference = 'B02' if 'B02' in predictions else unique[0]
        shortlist = [reference] + [eid for eid in unique if eid != reference][:3]
    if shortlist:
        reference = shortlist[0]
        for eid in shortlist[1:]:
            if reference not in predictions or eid not in predictions: raise ValueError('Shortlist requires completed real validation predictions')
            comparisons.append({'reference': reference, 'candidate': eid, **paired_bootstrap(predictions[reference], predictions[eid], repeats)})
    bootstrap_path = out / 'bootstrap_comparison.csv'
    if comparisons or not bootstrap_path.exists():
        pd.DataFrame(comparisons, columns=['reference', 'candidate', 'delta_auroc', 'ci95_low', 'ci95_high', 'replicates', 'seed', 'unit', 'interpretation']).to_csv(bootstrap_path, index=False)
    seed_groups = {}
    for r in completed:
        if r['experiment_id'].startswith('R') and r['config'].get('bound_candidates'):
            candidate = r['config']['bound_candidates'][0]
            seed_groups.setdefault(candidate, {})[r['seed']] = r['AUROC']
    seed_rows = [{'candidate': key, 'distinct_seeds': len(values), 'auroc_mean': float(np.mean(list(values.values()))),
                  'auroc_std': float(np.std(list(values.values()), ddof=1)) if len(values) > 1 else None,
                  'three_seed_review_complete': len(values) >= 3} for key, values in seed_groups.items()]
    pd.DataFrame(seed_rows, columns=['candidate', 'distinct_seeds', 'auroc_mean', 'auroc_std', 'three_seed_review_complete']).to_csv(out / 'seed_replication_summary.csv', index=False)
    for kind in ['roc', 'pr']:
        fig, ax = plt.subplots(figsize=(6, 5))
        for eid, df in predictions.items():
            if kind == 'roc': x, y, _ = roc_curve(df.label, df.probability)
            else: y, x, _ = precision_recall_curve(df.label, df.probability)
            ax.plot(x, y, label=eid)
        if predictions: ax.legend(fontsize=8)
        else: ax.text(.5, .5, 'No completed real validation experiment', ha='center')
        ax.set(xlabel='FPR' if kind == 'roc' else 'Recall', ylabel='TPR' if kind == 'roc' else 'Precision', xlim=(0, 1), ylim=(0, 1))
        fig.tight_layout(); fig.savefig(out / f'{kind}_curves.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(abl): ax.bar(abl.experiment_id, abl.delta_auroc)
    else: ax.text(.5, .5, 'No measured paired ablation yet', ha='center')
    ax.set_ylabel('Delta AUROC (single seed; not established stable gain)')
    fig.tight_layout(); fig.savefig(out / 'experiment_gains.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in [('local_latency_ms', 'Local GPU (not T4 certification)'), ('t4_latency_ms', 'T4 measured')]:
        subset = frame.dropna(subset=[key, 'AUROC'])
        if len(subset):
            ax.scatter(subset[key], subset.AUROC, label=label)
            for r in subset.itertuples(): ax.annotate(r.experiment_id, (getattr(r, key), r.AUROC))
    if ax.collections: ax.legend()
    else: ax.text(.5, .5, 'No measured latency/AUROC pair', ha='center')
    ax.set(xlabel='End-to-end latency (ms)', ylabel='Validation AUROC')
    fig.tight_layout(); fig.savefig(out / 'latency_auroc.png', dpi=160); plt.close(fig)
    recommendations = recommend(rows)
    write_json(out / 'next_experiments.json', recommendations)
    measured = frame[frame.status.isin(['completed', 'reused'])]
    report = '# 分阶段实验报告\n\n所有数值来自真实记录；空白代表尚未测量。仅官方训练集训练、官方验证集选择模型；测试集不用于实验调整。\n\n'
    report += table(measured[['experiment_id', 'best_epoch', 'AUROC', 'AUPRC', 'sensitivity_at_90_specificity', 'model_size_mb', 'local_latency_ms', 't4_latency_ms']])
    for r in completed:
        if r['experiment_id'] == 'B02' and (Path(r['output']) / 'history.json').exists():
            history = json.loads((Path(r['output']) / 'history.json').read_text(encoding='utf-8'))
            best = next(h for h in history if h['epoch'] == r['best_epoch'])
            report += f"\n\nB02最佳epoch={r['best_epoch']}，固定阈值0.5：Recall={best['recall']:.6f}，Specificity={best['specificity']:.6f}。逐样本原始logit和概率见 `{r['output']}/validation.csv`，训练曲线见同目录 `training_curves.png`。\n"
    report += '\n\n单折或单种子结果不构成稳定收益。配对Bootstrap只量化当前验证集采样不确定性，不代替三随机种子复核，也不消除多重比较和反复模型选择的乐观偏差。\n'
    if seed_rows: report += '\n\n三种子复核（同一官方验证集）：\n\n' + table(pd.DataFrame(seed_rows)) + '\n'
    report += '\n患者ID缺失时仅做分层图像Bootstrap，不能假定同患者图像独立。T4字段为空表示未实测，RTX 4060延迟仅供初筛。AUPRC为PR曲线梯形积分；Sensitivity@90Specificity为经验ROC上满足Specificity≥0.9的最高TPR，不是独立测试集性能。\n'
    report += '\n默认将已完成、非重复权重的验证AUROC前三名作为探索性入围候选，与B02配对Bootstrap；可用--shortlist显式冻结对照与候选。候选不足两项时Bootstrap表为空。\n'
    report += '\n## 状态\n\n' + table(frame[['experiment_id', 'status']])
    report += '\n\n## 下一批候选（需要用户确认，未自动启动）\n\n'
    report += '\n'.join(f"- {r['experiment_id']}：{r['reason']}" for r in recommendations) or '等待 B02 完成后生成建议。'
    report += '\n\n图表：roc_curves.png、pr_curves.png、experiment_gains.png、latency_auroc.png；各运行目录保存 training_curves.png。\n'
    (out / 'experiment_report.md').write_text(report, encoding='utf-8')
    return frame
