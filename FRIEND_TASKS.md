# 朋友服务器实验任务单

**代码分支：** `phase2-platform`。开始前请使用负责人本次发送的完整发布 commit SHA，不要仅凭分支名运行。请先确认自己已获赛事参赛/协作授权，且官方数据可以在该服务器上使用。授权未确认前，只能运行合成 Smoke Test；不要复制、上传或转发官方图像和标注。

## 你需要做什么

当前优先任务是运行 **H03 ConvNeXt-Tiny 多尺度对照**。三项使用同一官方 Train/Val、224 输入、seed 42、BCE、优化器和训练预算，只比较分类头：

| 实验 ID | 训练头 | 目的 |
| --- | --- | --- |
| `H03_G01_HEAD` | G01 原始单层分类头 | 本轮同一服务器训练器的内部基线 |
| `H03_MATCHED_MLP` | 仅最后 stage 特征 + 参数量匹配 MLP | 判断提升是否来自更大的头 |
| `H03_MULTISCALE` | stage 2 与最后 stage 特征池化后融合 | 验证多尺度表征是否改善 AUROC |

三个实验必须各自使用独立 GPU 进程、配置和结果目录。先检查服务器上所有队列和 GPU 进程；**不要中止正在运行的任务，也不要在正在训练的工作目录执行 pull**。有空闲卡才启动。建议先跑 `H03_G01_HEAD`，确认训练流程和输出正常后，再分别运行另外两个对照。不要因为有 8 张卡就启动无关实验。

## 启动前检查

在服务器新建独立 checkout，检出上面的完整 commit；不要复用正在训练的工作目录。检查官方数据版本、预训练权重和服务器任务状态：

```bash
RELEASE_COMMIT='<负责人提供的本次完整 commit SHA>'
git fetch origin
git checkout --detach "$RELEASE_COMMIT"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git status --short
nvidia-smi
python -m scripts.h_gpu_task status
```

确认已获授权后，安装仓库要求的 Python/CUDA 环境并运行代码说明中的 Smoke Test。`H03` 使用 G01 已审核的通用 ConvNeXt ImageNet 权重；核对服务器上的 `configs/weights_g01.local.json` 指向正确权重文件，来源、许可证、比赛使用/再分发权限和 SHA256 均通过检查。这个本地清单被 `.gitignore` 忽略，切勿把权重或私有清单提交 GitHub。若检查失败，停止 H03 并报告具体错误。

官方数据必须是现有冻结版本，归档 SHA256 必须匹配配置。若服务器归档或 Train/Val manifest 不匹配，停止并联系负责人，不要修改 YAML 哈希绕过检查。逐卡确认 GPU 空闲后再确定 `--gpu` 编号。以下示例的 0、1、2 只是示意，不能覆盖或抢占现有任务：

```bash
python -m scripts.h_gpu_task launch --id H03_G01_HEAD --gpu 0 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task status
python -m scripts.h_gpu_task launch --id H03_MATCHED_MLP --gpu 1 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task launch --id H03_MULTISCALE --gpu 2 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task status
```

每条 `launch` 会独立绑定一张 GPU，并异步启动单卡进程；不使用 DDP。结束后用 `status` 查看结果。任务失败时保留日志和 checkpoint，先把错误报告给负责人，不要改配置后复用同一结果目录。已结束任务的 GPU 锁可用以下命令释放，结果不会删除：

```bash
python -m scripts.h_gpu_task release --id H03_G01_HEAD
```

## 暂时不要启动

- **H04_RANDOM / H04_BALANCED：BLOCKED。** EfficientNetV2-S 的确切权重来源、许可证和文件 SHA256 尚未完成核验。不得自行下载后宣称合规、手动解除 BLOCKED 或改用医学权重。核验完成后负责人会发布新的配置版本和明确启动通知。
- **H01 ROI 分类或双分支训练：BLOCKED。** 目前只是定位候选与 QC 工具，没有通过人工质量审核的视盘定位器。若负责人明确要求，可在获得数据使用授权后运行 Train-only QC；其中必须包含 `train_05402.jpg`，QC 结果仍需人工审核。不得跳过失败样本，也不能用 Val/Test 调定位规则。
- **G05/G06/G07/G08：暂不重复启动。** 这些模型和配置已在旧实验工程中实现；当前权重核验未完成，服务器正在运行的状态也未核实。不要按旧的 8-GPU 队列盲目重跑。
- 不要重跑已完成的 G01/B02/G02/M02/L03/R01、旧分辨率搜索或 S01。不要使用 Test 训练、伪标签、统计拟合、调参或选模型。

## 每个 H03 实验必须交付

完整目录为 `results/{experiment_id}/seed_42/fold_none/`，训练脚本会自动生成压缩包。请交回完整压缩包及其 SHA256，不要只发截图、AUROC 数字或单独的 checkpoint。必须包含：

- `best.pth`、`validation.csv`、`metrics.json`
- `config.yaml`、`train_history.csv`、`training.log`
- `run_manifest.json`、`environment.txt`、`checksums.sha256`
- `last.pth`（用于同版本断点恢复）
- H03 另有 `model_cost_estimate.json`，其中 FLOPs 是估算值，推理延迟未实测

`validation.csv` 必须是完整官方 Val 顺序，包含 `Image_name, True_Label, logit, Probability`；共 5849 个样本。不要编辑 CSV 或结果包。`run_manifest.json` 应包含 commit、配置/数据/权重哈希、seed、GPU 型号及起止状态。报告最佳 epoch、AUROC、AUPRC、训练时长、峰值显存、模型大小和实际测速（没测就留空）；4090 延迟不能写成 T4 成绩。

结果包和模型权重通过双方确认获授权的私密渠道交回负责人，**不要上传到 GitHub 仓库**。负责人会在本地运行完整文件哈希、样本/标签、指标复算和冻结资产校验；通过验收前不能用于融合或提交。

## 参考文档

- [Linux 环境与固定版本说明](FRIEND_TRAINING_GUIDE.md)
- [结果文件交付及验收格式](RESULT_CONTRACT.md)
- [Phase H 运行、门禁和结果导入细节](docs/phase_h_remote_execution.md)
