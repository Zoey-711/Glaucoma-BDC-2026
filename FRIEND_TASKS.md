# 给 8×RTX 4090 训练协作者的任务单

请先阅读 [服务器执行指南](FRIEND_TRAINING_GUIDE.md)、[结果交付契约](RESULT_CONTRACT.md) 与 [协作及合规边界](COLLABORATION_RULES.md)。本项目的目标是用同一官方 Train/Val 划分比较模型，主指标为完整官方 Val AUROC，次指标为 AUPRC。B02、G01 与 Round 01 已冻结；请不要重跑或覆盖它们的目录。

## 你先要做什么

1. 从 `git@github.com:Zoey-711/Glaucoma-BDC-2026.git` 检出固定 tag `phase3-platform-v0.2.0`，确认 `git status --porcelain` 为空。把 `git rev-parse HEAD` 写进每份结果的 `run_manifest.json`。
2. 在你参与赛事及服务器访问官方数据的资格、许可获得明确确认前，只做合成数据测试。**不要通过 Git、聊天或结果包交换官方图像、标签、ZIP、测试预测。**
3. 在服务器安装训练环境，运行 `python -m pytest -q` 和一次 `python -m scripts.smoke_research`。Smoke 使用随机合成图像，其 AUROC 没有医学意义。
4. 有数据许可后，从你自己的合法渠道取得官方 ZIP；按指南接入并核对 ZIP SHA256、Train/Val 清单哈希。获取已审核的通用 ImageNet 权重与许可证材料，核对 `configs/weights_templates/` 给出的 SHA256。未通过门槛的实验不得正式训练。
5. 用 `python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml --archive "$OFFICIAL_ARCHIVE"` 看 dry-run。只执行显示 READY 的任务；BLOCKED 原因连同日志回报。

## 第一批 8 卡任务

每张卡最多一个进程。每项使用 seed 42、官方 Train 训练、完整官方 Val 评估；不读取 Test 标签或用 Test 预测调参。训练优先级如下：

| GPU | ID | 你要验证的问题 | 当前前置状态 |
|---:|---|---|---|
| 0 | G02 | ConvNeXt-Tiny 从 G01 的 224 提到 384 是否提高 AUROC | G01 通用权重已审；检查 384 配置 batch 显存 |
| 1 | G03 | EfficientNet-B0 从 B02 的 224 提到 384 是否提高 AUROC | B02 通用权重已审；检查 batch 显存 |
| 2 | G05 | 同为 384，EfficientNetV2-S 是否优于 G03 | **BLOCKED**：V2-S 权重来源、许可证、再分发、SHA256 未审完 |
| 3 | G07 | Swin-Tiny-384 是否与 G02 互补 | **BLOCKED**：Swin 权重审查未完成 |
| 4 | L03 | G01-224 加固定 λ=.05 的 batch 内成对排序项 | 记录有效正负 pair 数；与 G01 受控比较 |
| 5 | S01 | G01-224 加确定性眼底视野裁切 | 先检查裁切 QC、失败回退；与无裁切 G01 比较 |
| 6 | R01 | G01-224 加轻度 Color Jitter | 先检查增强 QC 是否破坏病理结构 |
| 7 | G01_REPRO | 独立复现冻结 G01 的配置和指标波动 | 写到独立结果目录，绝不覆盖原 G01 |

第一批队列命令（**满足许可并确认 READY 后才执行**）：

```bash
export OFFICIAL_ARCHIVE='<你合法持有的官方ZIP绝对路径>'
python -m scripts.preflight_backbones --cases G02 G03 --use-config-batch
python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml --archive "$OFFICIAL_ARCHIVE"
python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml --archive "$OFFICIAL_ARCHIVE" --execute
```

调度器会跳过 G05/G07，直到权重审核完成。若只有部分任务 READY，请把 dry-run 报告发给项目负责人，由负责人确定是否启动已 READY 的任务。显存不足或 OOM 时记录真实峰值与失败日志，再用新实验 ID 做公平的批量对照，不私改已发布配置。成功任务无需重复跑；失败任务保留现场，确认版本一致后使用 `--retry-failed`。

## 每次训练要交付什么

每个实验在 `results/{experiment_id}/seed_42/fold_none/` 保存完整文件，并交付自动生成的 `.tar.gz` 结果包。验收的九个必需文件是：

| 文件 | 用途 |
|---|---|
| `best.pth` | 按官方 Val AUROC 选择的最佳权重，必须能加载 |
| `validation.csv` | 官方 Val 每张图的 `Image_name,Label,logit,Probability`，不得缺失、重复或重排 |
| `metrics.json` | AUROC、AUPRC、Recall、Specificity、最佳 epoch；可从 CSV 复算 |
| `config.yaml` | 实际运行的完整配置与配置哈希 |
| `train_history.csv` | 每 epoch 训练曲线、时长、显存、LR 等 |
| `training.log` | 训练过程及异常记录 |
| `run_manifest.json` | Git commit、数据/权重/配置 SHA256、GPU、seed、起止时间、状态 |
| `environment.txt` | Python、PyTorch、CUDA、GPU 环境 |
| `checksums.sha256` | 结果目录所有其他文件的哈希 |

同时保留 `last.pth` 供断点恢复；推荐 `training_curves.png`、QC 联系表、`latency_4090.json`。4090 延迟只标 `local_latency_ms`，不要填写 `t4_latency_ms`。请先执行验收：

```bash
python -m scripts.validate_results results/G02/seed_42/fold_none --config configs/research/G02.yaml
```

训练结束后向项目负责人报告：完成和失败的实验 ID、每项最佳 epoch 与官方 Val AUROC/AUPRC、相对冻结 B02 的探索性 ΔAUROC、训练时间/显存峰值、模型大小、4090 本地延迟、结果包文件名和 SHA256、失败/阻塞原因。不要用单种子提升声称稳定收益。结果包只通过双方确认获授权的渠道传递，**不要提交到 GitHub**。

G04/G06/G08/G09、其他 Loss/增强、V01/V02、F01/F02、D01/D02 是后续批次；只有第一批分析与依赖审核完成后再由负责人指定。S03/S04 在可靠视盘定位器审核通过前保持 BLOCKED；V02 没有患者 ID 时只能称为重复组约束五折，不能称为患者级五折。最终 T4 速度由 `cost.py` 在 NVIDIA T4 上独立核验。
