# G01 ConvNeXt-Tiny-224 实验结果

日期：2026-10-28。G01 使用官方 Train 51,396 张训练、官方 Val 5,849 张验证；分类头单输出，BCEWithLogitsLoss、AdamW、Cosine LR、训练 AMP、seed=42。224×224 RGB 缩放、增强、划分、阈值 0.5、batch=32、早停与 B02 保持一致。模型采用独立核验的通用 ImageNet-1K 预训练权重，未加载胸片 checkpoint。完整配置见 [`configs/experiments/G01.yaml`](../configs/experiments/G01.yaml)，权重核验记录见本地 `configs/weights_g01.local.json`。

## 结果

G01 在第12个 epoch 触发早停；最佳 checkpoint 是**第7个 epoch**。所有指标来自相同的完整官方验证集，逐样本 ID 与标签已和 B02 对齐。

| 指标 | 冻结 B02 | G01 | 说明 |
|---|---:|---:|---|
| Val AUROC | 0.983059709 | **0.983562171** | G01−B02 = **+0.000502462** |
| Val AUPRC（PR 梯形积分） | 0.745762044 | **0.804615323** | 同一定义 |
| Recall @ 0.5 | 0.680628 | **0.712042** | 191 张阳性 |
| Specificity @ 0.5 | 0.993107 | 0.992223 | 5658 张阴性 |
| Sensitivity @ 90% specificity | 0.952880 | 0.963351 | 经验 ROC |

配对分层图像 Bootstrap 2,000 次（seed=42）的 ΔAUROC 95% CI：**[−0.003950, +0.004962]**。区间跨过 0；这个单种子实验未证明稳定优于 B02。患者 ID 不可用，因此图像重采样不保证患者独立，也没有校正模型选择或多重比较。

训练及验证实测 **2,838.28 秒（47分18秒）**，不含预处理缓存准备。训练峰值显存 allocated **2,363.35 MiB**、reserved **2,872.00 MiB**；验证 allocated 峰值 **916.38 MiB**。最佳 checkpoint **111,331,231 字节（111.33 MB）**，SHA256 `9220f2afcdf338b24c77755407797dbe606d6a55c06b2a244d2976a73c3c0452`。

本地 RTX 4060 Laptop、FP32、batch=1、原图读取至 CSV 写入/flush，固定100张验证图、5张预热、3轮均值：**99.16 ms/图**。这是本地初筛，**T4 尚未实测**，不能据此声称满足官方100ms。B02以前在同机测得101.01ms，但测量时间和系统负载不同，不能将两数之差当成精确的模型速度优势。

## 错误互补性

按冻结阈值0.5统计：两模型均错误62张；仅B02错误、G01正确38张；仅G01错误、B02正确37张。阳性191张中，G01纠正了B02的19张漏判，同时新增13张漏判；阴性中纠正19张B02误判，同时新增24张误判。逐样本比较见 [`error_complementarity.csv`](../outputs/runs/G01/error_complementarity.csv)。这说明预测存在互补样本，但**不是启动融合的充分证据**；本轮未训练融合或五折模型。

## 工件与验证

- [`comparison.json`](../outputs/runs/G01/comparison.json)：全部指标、配对区间、资源占用、错误分解。
- [`validation.csv`](../outputs/runs/G01/validation.csv)：5849行原始logit、概率、标签、实验ID。
- [`history.json`](../outputs/runs/G01/history.json) 与 [`training_curves.png`](../outputs/runs/G01/training_curves.png)：逐轮训练/验证记录与曲线。
- [`best.pt`](../outputs/runs/G01/best.pt)、[`artifact.json`](../outputs/runs/G01/artifact.json)：模型与哈希/大小/训练耗时。
- [`local_fp32.json`](../outputs/runs/G01/latency/local_fp32.json)：原始本地测速记录。
- [`Phase 2 审计报告`](phase2_audit_report.md)：430对候选复核和B02诊断子集。

GPU smoke 先通过，正式训练在独立输出目录运行一次；未启动其他正式模型、测试集推理或第二轮提交。B02 与 Round01 的53个冻结文件、旧胸片工程12个已登记源码/配置文件已通过 SHA256 复查。旧胸片项目未被写入。
