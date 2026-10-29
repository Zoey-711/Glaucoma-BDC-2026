# 朋友服务器任务单（2026-10-29 修订）

本轮仅审查、修复和发布代码，不授权新批量训练。参赛身份、服务器数据访问及第三方协作仍为 NEEDS_ORGANIZER_CONFIRMATION；没有主办方依据时只运行合成 Smoke。执行步骤见 FRIEND_TRAINING_GUIDE.md，交付契约见 RESULT_CONTRACT.md。

## 不要再跑的任务

旧 phase3-platform-v0.3.0、first_batch_8gpu.yaml、after_local_g02.yaml 不再作为待执行任务单。G02 已完成；停止 G04/G09 分辨率扩展与 L03 参数搜索。旧 research S01 使用另一套未通过安全 QC 的裁切，入口已强制阻止。本地 S01-v2 已完成且未证明收益，不移交重复训练。所有历史结果保留。

## 新候选队列

使用 configs/research/server_reviewed.yaml，正式启动前由负责人确认任务范围与增强 QC。GPU6/7 留空，不为占满八卡运行低价值任务。

| GPU | ID | 目的及依赖 |
|---:|---|---|
| 0 | B02_REPRO | 建立同一服务器 Trainer 的 B0-224 对照，独立输出 |
| 1 | G01_REPRO | 建立同一服务器 Trainer 的 ConvNeXt-224 对照，独立输出 |
| 2 | G03 | 等 B02_REPRO 完成，比较 B0-384；此前不是启动许可 |
| 3 | R01 | 等 G01_REPRO 完成，并先通过颜色增强 QC |
| 4 | G05 | 等 G03；V2-S 权重审查未完成，BLOCKED |
| 5 | G07 | 等 G01_REPRO；Swin 权重审查未完成，BLOCKED |

服务器 Trainer 的增强随机数、梯度裁剪、DataLoader 重建方式不同于历史 G01，不能声称逐步等价。新 Loss/增强应与同一服务器 Trainer 的 G01_REPRO 比较；历史 G01 只作背景参考。相同 seed 跨机器不保证逐位复现。

## 必须交付

每项保存 results/{experiment_id}/seed_42/fold_none/，交付自动生成的完整 tar.gz 及 SHA256：

- best.pth：最佳模型，不能只交截图或一个指标。
- validation.csv：完整官方 Val，包含 Image_name、Label、logit、Probability，ID/标签/顺序与清单一致。
- metrics.json、train_history.csv、training.log。
- config.yaml、run_manifest.json、environment.txt、checksums.sha256。
- 保留 last.pth 供同代码/配置/数据断点恢复；五折 OOF 须对应最佳模型，并与官方 Val 分开。

回报最佳 epoch、AUROC/AUPRC、时长、显存、模型大小、实际本机延迟、失败原因。未测值留空，4090 不可称 T4。结果包只经获授权渠道传递，不进 GitHub。

```bash
python -m scripts.validate_results results/G01_REPRO/seed_42/fold_none --config configs/research/G01_REPRO.yaml
```

负责人还须使用事先约定的发布 commit 验收：将该 SHA 填入验收配置 git_commit，不能只接受结果包自报版本。已运行旧代码的结果保留原样，不在运行中 git pull，不擅自重跑。
