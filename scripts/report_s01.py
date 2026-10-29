"""Read-only S01 verification and report generation."""
import json
from pathlib import Path

from scripts.report_m02d_l03 import verify
from scripts.s01_qc import guard
from src.utils import write_json


def main():
    qc=json.loads(Path('outputs/S01_qc/audit.json').read_text())
    visual=json.loads(Path('outputs/S01_qc/visual_review.json').read_text())
    text=f'''# S01 有效视野裁切实验

当前 QC 状态：**{visual['status']}**。本次完成预处理审计和 Smoke Test；仅在 QC 通过后才允许正式训练。

## Train 预处理审计

官方 Train {qc['rows']:,} 张，未使用 Val/Test 确定参数。规则为 `safe_fov_v1`：长边不超过 512 的等比例缩略图，HSV 亮度/饱和度前景，最大连通域，6% 直径加 2 像素安全边距；小区域、异常形状、分离区域、框外可见信号回退原图。接近全图时不裁切。不保证自动掩码等同于医学分割。

- 实际裁切比例：{qc['crop_fraction']:.3%}。
- 检测回退比例：{qc['fallback_fraction']:.3%}；“already_full_field”不算检测失败。
- 面积比例分位数：{qc['area_ratio_quantiles']}。
- 原因计数：{qc['reason_counts']}。
- 120 张对照：`outputs/S01_qc/pairs/`；12 张联系表：`outputs/S01_qc/sheet_*.jpg`。
- 全量边界/尺寸：`train_crop_statistics.csv`；异常清单：`anomaly_candidates.csv`；回退和无需裁切清单：`fallback_and_uncropped.csv`。
- 视觉复核：{visual['status']}。{visual['notes']}

## 对照与实现

S01a 复用 G01，原 resize 路径的数值一致性测试通过。S01b 从 G01 实际配置派生，只改变裁切策略；224、BCE、shuffle、batch32、seed42、AdamW、Cosine、AMP、30 轮上限和 patience5 均相同。训练器源码与 G01 快照相同。使用同一已核验通用 ImageNet 初始化，禁止加载 G01 训练 checkpoint 初始化。

裁切函数同时用于训练/验证缓存与在线 Predictor，predict.py/cost.py 共享逻辑。缓存身份包含裁切源码 SHA256。原始数据不修改，结果不覆盖。合成训练和真实 Train 八个 batch 的 Smoke Test 通过（首个合成样本重复被审计拦截，修正后独立重跑）。
'''
    out=Path('results/S01b/seed_42/fold_none')
    if (out/'comparison.json').exists():
        score=verify(out);r=json.loads((out/'comparison.json').read_text());ci=r['paired_bootstrap']
        duration=json.loads((out/'artifact.json').read_text())['train_duration']/60
        text+=f'''
## 实测结果

| 模型 | AUROC | AUPRC |
|---|---:|---:|
| G01 | 0.983562 | 0.804615 |
| S01b | {score['auroc']:.6f} | {score['auprc']:.6f} |

最佳 epoch {r['best_epoch']}，共 {r['epochs_completed']} 轮，训练+验证 {duration:.1f} 分钟。ΔAUROC={ci['delta_auroc']:+.6f}，配对 Bootstrap 95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]。2,000 次类别分层图像重采样，不能证明种子稳定性或患者独立性，也不纠正反复选模偏差。

模型 {r['model_bytes']/1e6:.3f} MB，4060 FP32 batch1 原图解码→裁切→resize/归一化→前向→CSV 写入端到端 {r['latency_4060_ms']:.2f} ms（100 张 Val，5 次预热，3 轮）。未在 T4 测速，不能据此宣称官方延迟达标。Val 固定应用规则，回退 {r['val_fallback_fraction']:.3%}、平均保留面积 {r['val_mean_area_ratio']:.3%}。

模型、完整 Val 逐样本 logits/概率、训练日志、曲线、配置、源码快照、数据/权重追溯及哈希保存在 `results/S01b/seed_42/fold_none/` 与 `outputs/S01_execution/`。完整 Val 的 ID/标签/概率和独立指标复算通过。
'''
        write_json('outputs/S01_execution/verification.json',{'status':'PASSED','val_rows':5849,'frozen_files_verified':guard()})
    else:
        text+='\n正式训练未启动，不提供 AUROC、AUPRC、Bootstrap 或正式模型延迟。当前候选在极暗图 train_05402.jpg 上存在外周截断风险：原图 2560×1920，建议裁切框 [20, 0, 2085, 1920]，右侧框外仍有暗弱信号。证据为 `outputs/S01_qc/extreme_05402_brightness_review.jpg`（仅显示提亮，未修改原图或训练图）。应先改进暗弱信号边界保护或保守回退，再重跑完整 QC 和 Smoke Test；禁止直接将 visual_review.json 改为 PASSED 绕过检查。\n\n验收：73 项测试通过；真实 Train batch32 八步 AMP 探针通过，峰值分配显存约 2364 MiB。253 个历史冻结文件核验通过。保留 S01a/S01b 配置，未启动其他训练、未生成 Test 提交。\n'
    Path('docs/S01_experiment_report.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
