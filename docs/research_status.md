# Phase 3 实现与运行状态（2026-10-29）

本表中的 `TESTED` 指代码经合成输入、单元测试或抽样 QC 验证；它不是正式训练成绩。`READY` 指代码具备启动条件，仍需固定 Git 版本、合法持有官方数据、通过权重和服务器前置检查。`BLOCKED` 不会被调度器启动。G02 已在本机完成正式训练；其余新实验没有官方 Val 成绩。目前没有 T4 延迟或 4090 正式训练显存数值。

| ID | 实现状态 | 测试证据 | 正式运行依赖或阻塞 |
|---|---|---|---|
| G02 ConvNeXt-Tiny-384 | IMPLEMENTED, TESTED, COMPLETED | RTX 4060 batch32 完整 8 epoch；Val AUROC 0.982912、AUPRC 0.781187；[报告](G02_experiment_report.md) | 已完成；禁止同配置重复训练 |
| G03 EfficientNet-B0-384 | IMPLEMENTED, TESTED | 同上 | 已核验 B02 通用权重；batch 复核 |
| G04 ConvNeXt-Tiny-512 | IMPLEMENTED, TESTED | 512 输入及反向传播 | 16×2 有效 batch；与 G02 比较前须增加匹配的 16×2 对照 |
| G05 EfficientNetV2-S-384 | IMPLEMENTED, TESTED, BLOCKED | 随机初始化合成反向传播 | 独立非医学权重、许可证和 SHA256 尚未核验 |
| G06 ConvNeXt-Small-384 | IMPLEMENTED, TESTED, BLOCKED | 同上 | 同上 |
| G07 Swin-Tiny-384 | IMPLEMENTED, TESTED, BLOCKED | 同上 | 同上 |
| G08 DenseNet-121-384 | IMPLEMENTED, TESTED, BLOCKED | 同上；只复用架构 | 同上；禁止胸片 checkpoint |
| G09 EfficientNetV2-S-512 | IMPLEMENTED, TESTED, BLOCKED | 同上 | 同上，且需匹配 16×2 微批量对照 |
| L01 Weighted BCE | IMPLEMENTED, TESTED | ratio、sqrt、BCE 配置及训练集计数计算 | 合法数据/权重和实际训练 |
| L02 Focal | IMPLEMENTED, TESTED | gamma 1/2、稳定 logits 实现 | 同上 |
| L03 BCE+Pairwise AUC | IMPLEMENTED, TESTED | λ=0/.05/.2；单类别 batch 安全跳过 | 同上 |
| L04 Balanced Batch | IMPLEMENTED, TESTED | 每 batch 两类、重复数/覆盖率单测 | 同上；Val 不重采样 |
| S01 FOV 裁切 | IMPLEMENTED, TESTED | 官方 Train 6 图联系表、回退代码 | 扩大 QC 后方可视为正式可用 |
| S02 LAB-L/CLAHE | IMPLEMENTED, TESTED | 官方 Train 6 图增强联系表 | 扩大病灶伪影 QC 后方可正式比较 |
| S03 视盘 ROI | 接口 IMPLEMENTED, BLOCKED | 坐标/置信度/失败回退代码 | 无经可靠性验证且合规的定位器；禁止构造伪 ROI |
| S04 全图+局部双分支 | IMPLEMENTED, TESTED, BLOCKED | 三模式单 logit/反向传播、双输入确定性验证单测；统一 Trainer 与 Predictor/cost 接线 | S03 定位器缺失；未经过真实 ROI/正式训练/T4 验收 |
| R01 Color Jitter | IMPLEMENTED, TESTED | 轻/中/无增强配置、Train 6 图联系表 | 扩大 QC 和真实训练 |
| R02 Blur/JPEG | IMPLEMENTED, TESTED | 模糊/JPEG/无增强配置、Train QC、固定扰动脚本 | 扩大 QC 和压力测试 |
| R03 MixUp | IMPLEMENTED, TESTED | α=.2/.4、软标签 BCE、Train QC | 真实训练 |
| R04 SimSiam | IMPLEMENTED, TESTED, BLOCKED | Train-only 数据门槛、合成 1 epoch encoder smoke | 正式 Train-only 来源审核；三个初始化对照尚无训练结果 |
| V01 三种子 | IMPLEMENTED, BLOCKED | 复核脚本和三 seed 配置 | 入围模型尚未选定 |
| V02 五折 OOF | IMPLEMENTED, BLOCKED | Train-only 分组五折、覆盖校验代码 | 缺经人工确认的 Train 重复组；不能声称患者级五折 |
| F01 等权概率融合 | IMPLEMENTED, TESTED, BLOCKED | 标签/ID 校验、等权单测 | 入围成员完整 checkpoint 与 Val 预测 |
| F02 等权 logit 融合 | IMPLEMENTED, TESTED, BLOCKED | 同一融合模块双模式 | 同上；总模型大小与 T4 延迟待测 |
| D01 Bernoulli KL 蒸馏 | IMPLEMENTED, TESTED, BLOCKED | KL 梯度单测、Train-only 教师门槛 | 经合规审核的教师及其官方 Train 预测 |
| D02 FP32/FP16/T4 | IMPLEMENTED, TESTED, BLOCKED | 精度/本地延迟代码、伪 4090 报告拒绝单测 | 最终候选及真实 NVIDIA T4 cost.py 报告 |

代码与合成验收：`python -m pytest -q`；`outputs/research_smoke_v3/report.json`；`outputs/research_preflight/backbones.json`；`outputs/research_ssl_smoke/history.json`；`outputs/research_qc/{S01,S02,R01,R02,R03}/`。S01/S02 的六图目视抽样未发现明显裁切或增强伪影，不能替代全数据 QC。4060 batch1 显存值仅是架构冒烟值，不能外推 4090 配置 batch。

## 发布检查清单

1. `git status --short`，确认没有冻结资产修改；对 `outputs/migration_review/frozen_inventory.json` 的 53 项逐项复算 SHA256，另复算旧胸片源文件清单。
2. `git add` 后查看 `git diff --cached --name-only` 与 `git diff --cached --stat`；禁止数据、权重、密钥、Round 01、B02/G01 结果进入 commit。
3. 固定 commit 与 tag；朋友只 checkout 指定 tag，且验证 commit SHA 一致。**本机不会自动 push。**
4. 朋友身份及官方数据远程访问许可未获证实前，只能在服务器运行合成 smoke；不能传输官方训练/验证/测试图像、标签或压缩包。
5. 朋友自行合法取得官方 ZIP 后，核验 archive 与语义 Train/Val manifest hash；分别取得已核验通用权重，复制 `configs/weights_templates/{g01,b02}.json` 为相应 `configs/weights_*.local.json` 并验证所有 SHA256。
6. 正式执行前在 4090 机器以配置微批量跑架构预检，再 dry-run 8 GPU 队列；BLOCKED 项保持不启动。结果按 [RESULT_CONTRACT.md](../RESULT_CONTRACT.md) 验收。T4 只接受 `cost.py` 实测报告。

旧 B02/G01 单种子差异不显著；跨划分疑似关联及患者 ID 缺失详见 [Phase 2 审计](phase2_audit_report.md)。任何新单种子增益只作探索性结果。
