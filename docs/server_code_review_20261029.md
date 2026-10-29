# 协作训练代码复核（2026-10-29）

审查对象为本仓库朋友任务单、research Trainer、队列及结果验收器；没有连接朋友服务器，未声称其当前运行版本或结果已验收。本轮不启动远端训练。

## 已修复

| 问题 | 修复位置与行为 |
|---|---|
| 旧 S01 可绕过失败 QC，旧任务单重复推荐 L03/G02 | `ExperimentConfig.assert_execution_policy` 同时用于 Trainer 与队列；旧 `fov` 一律阻止，退休实验禁止正式执行。独立的本地 S01-v2 不由该入口启动 |
| 断点恢复引用未导入的 ExperimentConfig | `src/research/trainer.py` 补齐导入；合成训练模拟打包中断后成功恢复 |
| 平衡采样每 epoch 使用同一随机序列 | loader 将 dataset epoch 传入 sampler；测试验证不同 epoch 改变、相同 epoch 可复现 |
| 最后不足累计步的训练 loss 日志放大 | 使用实际累计组大小恢复原始 loss，不再一律乘固定累计步数 |
| OOF 被每轮覆盖，可能不匹配 best checkpoint | 仅在保存最佳权重时保存该轮 OOF；官方 Val 文件仍单独保存 |
| 队列状态可跨配置/代码错误复用 | 状态绑定矩阵文件 SHA256、配置 identity、Git commit；不匹配拒绝，保留旧现场 |
| dry-run 未检查工作区修改 | 与正式入口一致核查 commit 和 tracked 工作区 |
| NaN 指标可绕过差值比较 | 验收要求 reported 指标有限；显式固定的期望 Git commit 必须匹配 |

## 对照与执行范围

服务器 Trainer 与历史 G01 不是逐步等价：随机增强种子、DataLoader 生命周期及梯度裁剪不同。保留原始历史配置，不冒充严格等价；新矩阵先要求 B02_REPRO/G01_REPRO，再运行其依赖对照。新候选矩阵为 `configs/research/server_reviewed.yaml`，不代表授权执行。R01 仍需要增强 QC；G05/G07 缺少权重审核。旧状态文件不得删除后悄悄重跑。

朋友赛事身份和数据访问依然 NEEDS_ORGANIZER_CONFIRMATION。文档门禁不构成主办方证明；队列 READY 不是许可。没有对外发送官方数据或权重。

## 测试与边界

本机完整 `python -m pytest -q`：85 passed。包含八 GPU 模拟队列并发/互斥测试；新增临时合成数据 CPU Smoke 覆盖训练、失败后恢复、打包、导入与 NaN 指标拒绝。CPU pin_memory 提示不影响测试。未在 Linux/8×4090 实测，朋友须按指南运行独立合成 Smoke。

测试不是对所有 26 组实验的全面认证。尤其五折模型仍按官方 Val 选择 best，因此 OOF 只能描述 Train 折外预测，不能声称对官方 Val 选模过程完全独立；没有患者 ID，不称患者级五折。未授权五折训练。

## 同期本地 S01-v2 结果

独立结果目录 `results/S01v2/seed_42/fold_none/`，代码来源 `42880513c372b538cc669d2102d9385042701235`，先于本次服务器修复。最佳 epoch 5，10 轮早停；AUROC 0.981368178、AUPRC 0.789361406，相对 G01 −0.002193993，配对 95% CI [−0.007119905, +0.002506806]。未证明收益。模型 111.331 MB，4060 原图端到端 102.982ms；没有 T4 实测。细节与冻结证据见 S01_v2_experiment_report.md。

全部历史 444 个受保护文件再次核验通过。只发布代码、配置及聚合报告；完整模型、逐样本预测、数据和 QC 原图不提交 GitHub。
