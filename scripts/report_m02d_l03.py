"""Read-only experiment validation and reports; never launches training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiment_guard import freeze_or_verify
from src.metrics import metrics
from src.utils import sha256, write_json


def verify(directory):
    path=Path(directory)
    for line in (path/'checksums.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert sha256(path/name)==digest, name
    pred=pd.read_csv(path/'validation.csv')
    official=pd.read_csv('data/official/manifests/valid.csv')
    assert list(pred.image_id)==list(official.image_id) and not pred.image_id.duplicated().any()
    assert np.array_equal(pred.label,official.label)
    assert np.isfinite(pred.logit).all() and pred.probability.between(0,1).all()
    assert np.allclose(torch.sigmoid(torch.tensor(pred.logit.to_numpy())).numpy(),pred.probability,atol=1e-6)
    score=metrics(pred.label,pred.probability)
    saved=json.loads((path/'metrics.json').read_text())
    assert all(abs(score[k]-saved[k])<1e-10 for k in ['auroc','auprc','recall','specificity'])
    return score


def m02d():
    path=Path('results/M02d/seed_42/fold_none')
    score=verify(path)
    obj=torch.load(path/'best.pth',map_location='cpu',weights_only=True)
    original=torch.load('outputs/runs/G01/best.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,obj['backbone_state_dict'][k]) for k,v in original['state_dict'].items())
    assert obj['age_input']=='zeros' and 'age_preprocessing' not in obj
    saved=json.loads((path/'metrics.json').read_text());ci=saved['M02c_minus_M02d']
    selection=json.loads((path/'selection_locked.json').read_text())['selected']
    latency=json.loads((path/'latency_4060.json').read_text())
    proof={'status':'PASSED','frozen_files_verified':freeze_or_verify(),'backbone_tensors_identical':True,
           'placeholder':[0.,0.],'age_fields_not_read':True,'val_rows':5849,'metrics_recomputed':True,
           'checkpoint_sha256':sha256(path/'best.pth'),'prediction_sha256':sha256(path/'validation.csv')}
    write_json('outputs/M02d_L03/M02d_verification.json',proof)
    text=f'''# M02d 同结构零年龄占位消融

M02d 已完成。年龄与缺失标志均恒为 0.0；不读取 Age 列。复用 M02c 的 G01 冻结特征、768+2 输入 MLP、seed 42、原内部 80/20 索引、AdamW、候选学习率与早停协议。特征缓存校验和匹配，G01 backbone 参数逐张量未改变。

| 实验 | AUROC | AUPRC |
|---|---:|---:|
| G01 | 0.983562 | 0.804615 |
| M02c 真实年龄 | 0.982882 | 0.802841 |
| M02d 零占位 | {score['auroc']:.6f} | {score['auprc']:.6f} |

真实年龄相对占位的 ΔAUROC={ci['delta_auroc']:+.6f}，2,000 次类别分层配对 Bootstrap 95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]。区间跨零；真实年龄并未获得可靠排序收益，AUPRC 也没有优于零占位。

同一 Train 内部选择协议为 M02d 选出 lr={selection['lr']}、{selection['best_epoch']} 轮；M02c 为 lr=0.001、27 轮。因此这是相同选择流程的消融，最终轮次并非完全固定一致。未额外用 Val 选择轮次或训练固定 27 轮候选。

模型 {latency['model_bytes']/1e6:.3f} MB；本地 RTX 4060 FP32 batch=1 端到端 {latency['mean_ms']:.2f} ms，含原始图片解码、预处理、冻结 backbone、MLP、CSV 写入/flush，5 张预热、100 张 Val、3 轮。未在 T4 实测。

结果：`results/M02d/seed_42/fold_none/`。配置、内部候选训练历史、最终训练历史、checkpoint、逐样本预测、配对区间和 SHA256 均保留。日志：`outputs/M02d_L03/M02d.log`。验收：`outputs/M02d_L03/M02d_verification.json`。

当前不建议扩展年龄多模态五折；应优先等待 L03b 和其他受控图像模型结果。无患者 ID，内部 holdout 的图像已被 G01 backbone 用于训练；单种子和官方 Val 反复使用的限制继续存在。没有改动 M02a/M02b/M02c 或生成 Test 提交。
'''
    Path('docs/M02d_ablation_report.md').write_text(text,encoding='utf-8')
    print(text)


def l03():
    path=Path('results/L03b/seed_42/fold_none')
    manifest=json.loads((path/'run_manifest.json').read_text())
    assert manifest['status']=='COMPLETED'
    score=verify(path);report=json.loads((path/'comparison.json').read_text())
    history=pd.read_csv(path/'train_history.csv');ci=report['paired_bootstrap']
    assert (history.positive_draws==1657).all() and (history.negative_draws==49739).all()
    assert np.allclose(history.train_loss,history.bce_loss+history.weighted_pairwise_loss,atol=1e-5)
    artifact=json.loads((path/'artifact.json').read_text())
    g01=json.loads(Path('outputs/runs/G01/comparison.json').read_text())['metrics']
    result={'status':'PASSED','frozen_files_verified':freeze_or_verify(),'val_rows':5849,
            'epoch_class_counts_match_train':True,'loss_decomposition_matches':True,
            'checkpoint_sha256':sha256(path/'best.pt'),'validation_sha256':sha256(path/'validation.csv')}
    write_json('outputs/M02d_L03/L03b_verification.json',result)
    conclusion=('观察到正向差值且配对区间未跨零，但仍需三种子复核。' if ci['ci95_low']>0 else
                '未获得显著优于 G01 的证据，不能将 Recall 或单种子差异解释为稳定排序收益。')
    text=f'''# L03b BCE + Pairwise AUC Loss 实验报告

{conclusion}

| 实验 | AUROC | AUPRC | Recall@0.5 |
|---|---:|---:|---:|
| G01 | {g01['auroc']:.6f} | {g01['auprc']:.6f} | {g01['recall']:.6f} |
| L03b λ=0.05 | {score['auroc']:.6f} | {score['auprc']:.6f} | {score['recall']:.6f} |

ΔAUROC={ci['delta_auroc']:+.6f}，配对 Bootstrap 95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]，2,000 次类别分层图像重采样。最佳 epoch={report['best_epoch']}，完成 {report['epochs_completed']} 轮；训练与验证共 {artifact['train_duration']/60:.1f} 分钟。

训练直接调用与 G01 冻结快照 SHA256 一致的 `src/experiment_training.py`。从已核验的通用非医学 ImageNet 权重重新初始化，未载入 G01 checkpoint。除了实验标识、输出目录、Loss 和说明，配置逐字段保持一致：224、batch32、seed42、AdamW lr=0.0003、weight_decay=0.0001、Cosine、AMP、30 轮上限、patience5、原随机打乱与六进程增强管线。不改变采样、梯度裁剪或有效 batch。

排序项为当前 batch 所有正负 logits 的 `mean(softplus(negative-positive))`；单类别 batch 返回原 BCE。λ=0 分支直接返回 BCE，单元测试证明数值、梯度和 AdamW 更新一致。L03a、L03c 仅创建配置，未启动；G01 用作原 BCE 协议对照，此证据并不保证跨硬件/运行环境逐比特重现完整训练。

每轮 BCE、Pairwise、加权 Pairwise/BCE 比例、有效 batch 比例、正负抽样数和配对数均保存于 `train_history.csv`。有效 batch 比例范围 {history.valid_batch_fraction.min():.3%}–{history.valid_batch_fraction.max():.3%}，加权 Pairwise/BCE 比例范围 {history.weighted_pairwise_over_bce.min():.4f}–{history.weighted_pairwise_over_bce.max():.4f}；每轮准确覆盖 1,657 阳性、49,739 阴性，无额外重采样。

排序比较：{json.dumps(report['ranking'],ensure_ascii=False)}。这些统计直接比较正负样本概率次序，结论不依赖固定阈值下 Recall。

模型 {report['model_bytes']/1e6:.3f} MB；峰值分配显存 {report['peak_allocated_mib']:.0f} MiB，保留显存 {report['peak_reserved_mib']:.0f} MiB；4060 FP32 batch1 端到端均值 {report['latency_4060_ms']:.2f} ms。尚无 T4 实测。测速含原始图片解码、预处理、模型和 CSV 写入/flush，100 张 Val、5 张预热、3 轮；历史会话时延不能直接作为速度收益证据。

结果：`results/L03b/seed_42/fold_none/`；执行源码快照、Smoke Test、配置与数据哈希：`outputs/L03b_execution/`。完整 Val 5,849 张与 G01 顺序/标签一致；指标独立复算、文件哈希和冻结资产核验通过。

下一批建议：若成本允许，先对最佳候选做 G01/L03b 配对多种子复核；再考虑 S01 确定性视野裁切（先 QC）和已合规权重的其他骨干。年龄消融没有支持扩大多模态规模；不自动启动 L03c 或五折。患者独立性未证实，Bootstrap 不纠正反复选模偏差。
'''
    Path('docs/L03b_experiment_report.md').write_text(text,encoding='utf-8');print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['M02d','L03b']);a=p.parse_args()
    {'M02d':m02d,'L03b':l03}[a.experiment]()
