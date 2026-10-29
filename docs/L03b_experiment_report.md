# L03b BCE + Pairwise AUC Loss 实验报告

未获得显著优于 G01 的证据，不能将 Recall 或单种子差异解释为稳定排序收益。

| 实验 | AUROC | AUPRC | Recall@0.5 |
|---|---:|---:|---:|
| G01 | 0.983562 | 0.804615 | 0.712042 |
| L03b λ=0.05 | 0.983426 | 0.767908 | 0.638743 |

ΔAUROC=-0.000136，配对 Bootstrap 95% CI [-0.003779, +0.004021]，2,000 次类别分层图像重采样。最佳 epoch=7，完成 12 轮；训练与验证共 44.6 分钟。

训练直接调用与 G01 冻结快照 SHA256 一致的 `src/experiment_training.py`。从已核验的通用非医学 ImageNet 权重重新初始化，未载入 G01 checkpoint。除了实验标识、输出目录、Loss 和说明，配置逐字段保持一致：224、batch32、seed42、AdamW lr=0.0003、weight_decay=0.0001、Cosine、AMP、30 轮上限、patience5、原随机打乱与六进程增强管线。不改变采样、梯度裁剪或有效 batch。

排序项为当前 batch 所有正负 logits 的 `mean(softplus(negative-positive))`；单类别 batch 返回原 BCE。λ=0 分支直接返回 BCE，单元测试证明数值、梯度和 AdamW 更新一致。L03a、L03c 仅创建配置，未启动；G01 用作原 BCE 协议对照，此证据并不保证跨硬件/运行环境逐比特重现完整训练。

每轮 BCE、Pairwise、加权 Pairwise/BCE 比例、有效 batch 比例、正负抽样数和配对数均保存于 `train_history.csv`。有效 batch 比例范围 63.970%–66.584%，加权 Pairwise/BCE 比例范围 0.0145–0.0642；每轮准确覆盖 1,657 阳性、49,739 阴性，无额外重采样。

排序比较：{"positive_negative_pairs": 1080678, "changed_pair_orders_including_ties": 15539, "corrected_pair_orders": 7696, "newly_wrong_pair_orders": 7843}。这些统计直接比较正负样本概率次序，结论不依赖固定阈值下 Recall。

模型 111.331 MB；峰值分配显存 2363 MiB，保留显存 2872 MiB；4060 FP32 batch1 端到端均值 71.08 ms。尚无 T4 实测。测速含原始图片解码、预处理、模型和 CSV 写入/flush，100 张 Val、5 张预热、3 轮；历史会话时延不能直接作为速度收益证据。

结果：`results/L03b/seed_42/fold_none/`；执行源码快照、Smoke Test、配置与数据哈希：`outputs/L03b_execution/`。完整 Val 5,849 张与 G01 顺序/标签一致；指标独立复算、文件哈希和冻结资产核验通过。

下一批建议：若成本允许，先对最佳候选做 G01/L03b 配对多种子复核；再考虑 S01 确定性视野裁切（先 QC）和已合规权重的其他骨干。年龄消融没有支持扩大多模态规模；不自动启动 L03c 或五折。患者独立性未证实，Bootstrap 不纠正反复选模偏差。
