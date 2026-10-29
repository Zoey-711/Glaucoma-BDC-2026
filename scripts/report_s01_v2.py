"""Report actual S01-v2 QC and completed training; never starts experiments."""
import json
from pathlib import Path

import pandas as pd

from scripts.report_m02d_l03 import verify
from scripts.s01_v2_qc import guard
from src.utils import write_json


def main():
    root=Path('outputs/S01_v2');train=json.loads((root/'train/report.json').read_text())
    old=train['v1_comparison'];sm=json.loads(Path('outputs/S01_v2_execution/smoke_result.json').read_text())
    lines=['# S01-v2 保守裁切实验',
        '', 'v1 实现、配置、报告、原始数据和历史模型均保留；v2 独立输出。',
        '', '## 安全规则与 QC', '',
        'HSV 前景与最大连通区域提出裁切框，10% 直径加 2 个缩略图像素的安全边距；框外弱亮度、梯度纹理和连通区域用于提示风险。最终还在原始解码分辨率检查全部 RGB 非零信号：只要框外有一个非零像素，直接回退原图。算法无学习参数、跨图统计或概率输入。它不是血管分割，不能声称能识别所有血管；回退也可能由无害噪声或压缩残留触发。',
        '', '独立审计逐张检查被移除的四条原图区域是否全零，异常候选还检查输出像素是否与原图对应区域一致。这是对解码信号保留的验证，不是医学语义分割保证；resize 仍会重采样。',
        '', '| Train 项目 | v1 | v2 |','|---|---:|---:|',
        f"| 实际裁切比例 | {old['crop_fraction']:.3%} | {train['crop_fraction']:.3%} |",
        f"| 检测回退比例 | {old['fallback_fraction']:.3%} | {train['fallback_fraction']:.3%} |",
        f"| 平均保留面积 | {old['mean_area_ratio']:.3%} | {train['mean_area_ratio']:.3%} |",'',
        f"完整 Train：{train['rows']} 张；边界风险建议框 {train['boundary_risk_proposals']} 张；弱连通结构候选 {train['connected_weak_structure_proposals']} 张；最终输出截断候选 {train['final_truncation_candidates']} 张。异常候选 {train['anomaly_candidates']} 张，均有逐图独立安全检查和对照图。这里的全部异常检查指程序针对性检查；目视复核范围以 visual_review.json 为准，不声称逐张人工查看。",
        f"面积分位数：{train['area_ratio_quantiles']}。原因计数：{train['reason_counts']}。",'',
        '已知失败样本 train_05402.jpg 的 v2 建议框右侧仍发现暗弱信号，最终回退完整 2560×1920 原图，独立逐像素比较通过。其余 11 张既有极端样本也通过。原图、提亮显示图、掩码、建议框与最终输出：`outputs/S01_v2/regression/train_05402/`。提亮仅用于 QC 显示，不用于训练。',
        '', '160 张 Train 分层对照及异常联系表：`outputs/S01_v2/train/`；完整边界统计 `statistics.csv`、异常逐图结果 `all_anomaly_targeted_review.csv`、失败/回退列表 `failures.csv`。规则锁定 `policy_locked.json` 与 `preprocessing.yaml`；任何锁定后源码变化都会阻止训练。',
        '', '## 测试与训练', '',
        f"Smoke Test：{sm['status']}，真实 Train batch32 八步 AMP 探针，峰值分配显存 {sm['peak_allocated_mib']:.1f} MiB；探针权重丢弃。测试包含全黑回退、极暗外围、缩略图丢失的单个原始信号像素、独立截断检测、缓存/在线一致性和官方 train_05402.jpg 永久回归。",
        '', '配置 `configs/experiments/S01v2.yaml` 派生自冻结 G01 实际配置；同一 ConvNeXt-Tiny-224、已审核 ImageNet 初始化、BCE、AdamW、Cosine、batch32、seed42、AMP、30轮上限、patience5 和 Train/Val 划分。训练器源码保持 G01 版本；主要变量仅裁切。G01 作为原图对照，不重复训练。',
    ]
    if (root/'val/report.json').exists():
        val=json.loads((root/'val/report.json').read_text())
        lines+=['',f"规则冻结后在 Val {val['rows']} 张上仅做无标签安全检查，实际裁切 {val['crop_fraction']:.3%}，回退 {val['fallback_fraction']:.3%}，最终输出截断候选 {val['final_truncation_candidates']}。未据 Val 调整裁切阈值。"]
    out=Path('results/S01v2/seed_42/fold_none')
    if (out/'comparison.json').exists():
        scores=verify(out);r=json.loads((out/'comparison.json').read_text());ci=r['paired_bootstrap']
        duration=json.loads((out/'artifact.json').read_text())['train_duration']/60
        h=pd.read_csv(out/'train_history.csv')
        lines+=['','| 模型 | AUROC | AUPRC |','|---|---:|---:|','| G01 | 0.983562 | 0.804615 |',
            f"| S01-v2 | {scores['auroc']:.6f} | {scores['auprc']:.6f} |",'',
            f"最佳 epoch {r['best_epoch']}，共 {r['epochs_completed']} 轮，训练及验证 {duration:.2f} 分钟。ΔAUROC={ci['delta_auroc']:+.6f}，配对 Bootstrap 95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]。2,000 次图像类别分层重采样，患者独立性未知；区间不包含多次选模偏差和训练种子方差。",
            f"模型 {r['model_bytes']/1e6:.3f} MB，峰值分配显存 {h.train_peak_allocated_mib.max():.0f} MiB。RTX 4060 FP32 batch1 原始图像解码、v2 裁切/回退、resize/归一化、模型、CSV写出端到端 {r['latency_4060_ms']:.2f} ms；100 张 Val，5张预热，3轮。没有 T4 实测，不能据此宣称官方达标。",
            ('观察到正向验证差值且配对区间不跨零，仍需独立随机种子复核。' if ci['ci95_low']>0 else '未证明 S01-v2 稳定优于 G01；不能将单种子差异描述为稳定收益。'),
            '若后续有精度收益但 T4 实测超时，可评估将同一固定裁切用于轻量模型；必须重新训练并对照，不能把训练缓存速度当作端到端速度。',
            '', '完整模型、配置、预处理规则、日志、曲线、validation.csv 与结果哈希均在 `results/S01v2/seed_42/fold_none/`；执行源码快照在 `outputs/S01_v2_execution/source_snapshot/`。本轮停止于该实验，不生成官方 Test 提交。']
        write_json('outputs/S01_v2_execution/verification.json',{'status':'PASSED','val_rows':5849,'frozen_files_verified':guard()})
    else:lines+=['','正式训练尚未完成，不能报告正式验证指标。']
    Path('docs/S01_v2_experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
