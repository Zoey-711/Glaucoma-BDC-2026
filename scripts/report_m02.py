"""Independently validate M02 artifacts and render the final exploratory comparison."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from scripts.run_m02 import C, OUT, verify_frozen
from src.metrics import metrics
from src.utils import sha256, write_json


def main():
    manifest = json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'COMPLETED':
        raise ValueError('M02 is not complete')
    official = pd.read_csv('data/official/glaucoma_first/valid.csv')
    original = torch.load(C['checkpoint'], map_location='cpu', weights_only=True)
    fused = torch.load('results/M02c/seed_42/fold_none/best.pth', map_location='cpu', weights_only=True)
    assert original['state_dict'].keys() == fused['backbone_state_dict'].keys()
    assert all(torch.equal(v, fused['backbone_state_dict'][k]) for k,v in original['state_dict'].items())
    rows = []; preds = {}; checks = {'frozen_files_verified': verify_frozen(), 'backbone_tensors_unchanged': True, 'results': {}}
    for eid in ['M02a','M02b','M02c']:
        directory = Path(f'results/{eid}/seed_42/fold_none')
        lines = (directory/'checksums.sha256').read_text().splitlines()
        assert all(sha256(directory/line.split('  ',1)[1]) == line.split('  ',1)[0] for line in lines)
        frame = pd.read_csv(directory/'validation.csv')
        assert list(frame.image_id) == list(official.Image_name) and not frame.image_id.duplicated().any()
        assert np.array_equal(frame.label, official.True_Label)
        assert np.isfinite(frame.logit).all() and frame.probability.between(0,1).all()
        assert np.allclose(torch.sigmoid(torch.tensor(frame.logit.to_numpy())).numpy(),frame.probability,atol=1e-6)
        score = metrics(frame.label, frame.probability)
        reported = json.loads((directory/'metrics.json').read_text())
        assert all(abs(score[k]-reported[k]) < 1e-10 for k in ['auroc','auprc','recall','specificity'])
        latency = json.loads((directory/'latency_4060.json').read_text())
        ci = reported['paired_bootstrap']
        rows.append({'experiment_id':eid,'AUROC':score['auroc'],'AUPRC':score['auprc'],
                     'Recall_0_5':score['recall'],'Specificity_0_5':score['specificity'],
                     'delta_AUROC':ci['delta_auroc'],'ci95_low':ci['ci95_low'],'ci95_high':ci['ci95_high'],
                     'model_MB':latency['model_bytes']/1e6,'latency_4060_ms':latency['mean_ms']})
        preds[eid] = frame
        checks['results'][eid] = {'rows':len(frame),'metrics_recomputed':True,'checksums_passed':True,
                                  'max_deployed_probability_difference':latency['max_probability_difference_vs_cached_val']}
    pd.DataFrame(rows).to_csv(OUT/'experiment_comparison.csv',index=False)
    write_json(OUT/'verification.json',checks)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for eid, frame in preds.items():
        fpr,tpr,_ = roc_curve(frame.label,frame.probability)
        precision,recall,_ = precision_recall_curve(frame.label,frame.probability)
        axes[0].plot(fpr,tpr,label=eid);axes[1].plot(recall,precision,label=eid)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Official Val ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Official Val precision-recall')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'roc_pr.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for file in (OUT/'training').glob('lr_*.csv'):
        history = pd.read_csv(file)
        axes[0].plot(history.epoch,history.internal_auroc,marker='.',label=file.stem)
        axes[1].plot(history.epoch,history.train_bce,label=file.stem)
    axes[0].set(xlabel='Epoch',ylabel='Internal holdout AUROC',title='Head selection within Train')
    axes[1].set(xlabel='Epoch',ylabel='Train BCE',title='Head learning curves')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'head_selection_curves.png',dpi=160);plt.close(fig)
    audit = json.loads((OUT/'age_audit.json').read_text())
    selection = json.loads((OUT/'training/selection_locked.json').read_text())
    robustness = json.loads((OUT/'robustness.json').read_text())
    selected = selection['selected']; fused_score = json.loads(Path('results/M02c/seed_42/fold_none/metrics.json').read_text())
    ci = fused_score['paired_bootstrap']
    seconds = (datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds()
    evidence = ('本次观察到正向差值且探索性配对区间不跨零，值得先做同训练头的去年龄对照及多种子复核。'
                if ci['delta_auroc']>0 and ci['ci95_low']>0 else
                '本次没有获得优于冻结 G01 的充分证据，暂不建议直接扩展到 8×4090 多折多模态训练。')
    lines = ['# M02 年龄与冻结 G01 图像特征融合实验', '',
             f'完成时间：{manifest["finished_utc"]}；RTX 4060 Laptop；总运行 {seconds/60:.1f} 分钟；seed=42。', '',
             '## 结果', '', '| 实验 | AUROC | AUPRC | ΔAUROC 对 G01 | 95% 配对 CI | Recall@0.5 | 模型 MB | 4060 ms |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["experiment_id"]} | {row["AUROC"]:.6f} | {row["AUPRC"]:.6f} | {row["delta_AUROC"]:+.6f} | [{row["ci95_low"]:+.6f}, {row["ci95_high"]:+.6f}] | {row["Recall_0_5"]:.6f} | {row["model_MB"]:.4f} | {row["latency_4060_ms"]:.2f} |')
    lines += ['', evidence, '',
              'M02a 直接复用冻结 G01 的逐样本概率；M02b 仅使用标准化年龄和缺失指示变量；M02c 使用 G01 的 768 维池化特征与这两个年龄变量训练指定 MLP，backbone 权重逐张量核验完全不变。',
              f'内部候选学习率固定为 {C["head"]["learning_rates"]}，weight_decay={C["head"]["weight_decay"]}；Train 内部 80/20 分层划分选中 lr={selected["lr"]}、epoch={selected["best_epoch"]}。随后重新初始化融合头，在全部 Train 训练固定轮次，再首次评估官方 Val。年龄 Logistic Regression 固定 C=1。',
              '年龄预处理仅在 Train 拟合；内部头选择时另在内部训练子集拟合预处理。最终 Train 年龄有效范围为 17–96 岁，缺失或范围外值以 Train 中位数 57 填补并置缺失标志；均值 56.738754、总体标准差 10.185559。Test 只做年龄质量描述，未提取 Test 图像特征或生成 Test 预测。', '',
              '## 年龄质量与标签分布', '', '| Split | 缺失 | 非正年龄 | Train 范围外 | 中位数 | P25 / P75 | 最小 / 最大 |', '|---|---:|---:|---:|---:|---|---|']
    for split in ['train','valid','test']:
        x=audit['splits'][split]['all'];q=x['quantiles']
        lines.append(f'| {split} | {x["missing_nonfinite"]} | {x["nonpositive"]} | {x["outside_train_positive_envelope"]} | {q["median"]} | {q["p25"]} / {q["p75"]} | {q["min"]} / {q["max"]} |')
    lines += ['', '完整分位数、按标签的年龄分布与年龄段阳性比例保存在 `outputs/M02/age_audit.json`；预处理参数在 `age_preprocessing.yaml`。', '',
              '## 排序与鲁棒性', '',
              f'M02c 与 G01 在 {fused_score["ranking"]["positive_negative_pairs"]:,} 个 Val 正负配对中，'
              f'改变了 {fused_score["ranking"]["changed_pair_orders_including_ties"]:,} 个排序关系（含平局变化），'
              f'纠正 {fused_score["ranking"]["corrected_pair_orders"]:,} 个，新增错误/平局 {fused_score["ranking"]["newly_wrong_pair_orders"]:,} 个。'
              '最终以 AUROC 及其配对区间判断排序改善，不能只依据 Recall 或输出概率变大。', '',
              '| 诊断扰动 | M02b AUROC | M02c AUROC | M02c AUPRC |', '|---|---:|---:|---:|']
    for name,r in robustness.items():
        lines.append(f'| {name} | {r["M02b"]["auroc"]:.6f} | {r["M02c"]["auroc"]:.6f} | {r["M02c"]["auprc"]:.6f} |')
    lines += ['', '这些 Val 扰动只用于报告鲁棒性，没有用于选模型。全部缺失和全部异常必须走相同回退；训练中只有 14 条缺失/异常记录，因此不能从小样本声称缺失鲁棒性已可靠建立。', '',
              'M02b 在年龄全部缺失时输出常数，AUROC=0.5。现有梯形 PR 积分在全部分数相同的极端情形会因端点插值得到约 0.5163，不能据此声称有预测能力；此时 average precision 等于 Val 阳性率 0.032655。主要结论以 AUROC、正常输入的 AUPRC 和配对结果为准。', '',
              '| Val 年龄层 | 阳性 / 阴性 | M02a AUROC | M02b AUROC | M02c AUROC |', '|---|---|---:|---:|---:|']
    all_scores = {eid:json.loads(Path(f'results/{eid}/seed_42/fold_none/metrics.json').read_text()) for eid in preds}
    for group in C['age_strata']:
        r=all_scores['M02a']['age_strata'][group]
        values=[all_scores[eid]['age_strata'][group]['auroc'] for eid in preds]
        formatted=['样本不足' if v is None else f'{v:.6f}' for v in values]
        lines.append(f'| {group} | {r["positive"]} / {r["negative"]} | '+ ' | '.join(formatted)+' |')
    lines += ['', '分层 AUROC 只在正、负样本各至少 20 张时计算；这些分层未用于拟合。', '', '## 可追溯性及结论边界', '',
              '58 项单元测试通过；GPU 合成数据 Smoke Test 通过。真实运行第一次因 Windows 沙箱阻止 DataLoader 管道而在首 batch 前失败，失败记录保留在 `outputs/M02_attempt01_permission_failed/`，获准后使用同一协议重新启动。',
              f'最终冻结核验覆盖 {checks["frozen_files_verified"]} 个文件；模型、预测及校验和保存在 `results/M02a`、`results/M02b`、`results/M02c`。`outputs/M02/` 保存内部划分、候选与最终训练历史、源文件快照、配置哈希、年龄审计、特征缓存及最终报告。',
              'G01 已在全部 Train 训练且曾用官方 Val 选择 checkpoint；这次内部 holdout 只能隔离融合头拟合，不是独立的 backbone 验证。没有患者 ID，也不能保证患者独立性。Bootstrap 为按图像、按类别分层的 2,000 次配对重采样，不能消除选模偏差或证明多种子稳定性。',
              '首次报告检查发现 YAML 将未加引号的年龄段 40_59、60_79 解析成整数，导致这两组的摘要为空。已补充回归测试并仅重算年龄分层摘要；修正前摘要、校验和及更正记录保存在 `outputs/M02/strata_serialization_correction/`。训练参数、整体指标、模型和逐样本预测未改变，原执行配置快照保留。',
              'M02c 同时更换了原 G01 线性分类头，因此 M02c 对 M02a 的差值不能完全归因于年龄。年龄置空/打乱诊断有助于检查依赖，但仍不能替代同一 MLP 的图像单独训练对照；本次没有额外启动该训练。',
              '本地延迟覆盖原图读取/解码、年龄查询与预处理、冻结 backbone、融合头、CSV 写入和 flush；5 张预热、100 张 Val、连续 3 轮、batch=1。M02b 为统一计时范围也读取图像，虽然分类器不使用图像。M02a 是本次会话重新测速，不能覆盖或替代冻结 G01 历史 99.16 ms；跨会话负载、缓存、功耗状态可影响时延，不能把这次小幅差异认定为融合更快。这里均为 RTX 4060 实测，T4 尚未测试。年龄模型的正式部署还需将官方 Age 输入和统一 predict.py/cost.py 接口接入并核验；当前专用 M02Predictor 用于实验本地测试。',
              '下一步应先由证据决定是否做同头去年龄对照和三种子复核；患者/重复分组不可信时，不宜直接宣称五折多模态结果稳定。未修改原始数据、G01/B02/G02/Round01，未生成或提交新的官方测试结果。']
    text='\n'.join(lines)+'\n'
    (OUT/'M02_experiment_report.md').write_text(text,encoding='utf-8')
    Path('docs/M02_experiment_report.md').write_text(text,encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(evidence)


if __name__ == '__main__':
    main()
