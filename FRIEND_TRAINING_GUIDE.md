# 8×RTX 4090 训练机执行指南

每组实验的实现、测试和阻塞状态见 [research_status.md](docs/research_status.md)。本机只有 RTX 4060，尚未在 8×4090 或 T4 上执行正式实验。

已公开审查过的 G01/B02 权重来源模板位于 `configs/weights_templates/`。朋友须自行合法取得对应非医学 ImageNet 权重和许可证、模型卡文件，核对模板中的每个 SHA256，并把模板复制为 `configs/weights_g01.local.json`、`configs/weights_b02.local.json`。这些本地文件与权重均不进入 Git；缺失或哈希不符时队列保持 BLOCKED。

本仓库只发布代码和配置，不包含比赛数据、训练权重或提交 CSV。必须先确认朋友本人符合赛事参与身份，并取得主办方允许其访问、存放和使用官方数据的依据；**未确认前仅运行合成数据 smoke，不传输官方图像或标签**。服务器 GPU 4090 的延迟只作本地初筛，最终 T4 验收另行进行。

## 固定版本与环境

管理员在本地创建受审 Git commit/tag 并记录 SHA256；朋友只检出指定 tag/commit，不运行工作区未提交的修改。部署前运行 `git status --porcelain`，应为空。Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8 是官方 T4 环境目标；4090 训练机应记录自己的驱动和 torch CUDA 版本。

```bash
git clone git@github.com:Zoey-711/Glaucoma-BDC-2026.git glaucoma-bdc-2026
cd glaucoma-bdc-2026
git checkout phase3-platform-v0.2.0
git rev-parse HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-research.txt
python -m pytest -q
python -m scripts.preflight_backbones --cases G02 G03 --use-config-batch
```

首次**只用合成数据**测试服务器；`scripts/smoke_research.py`会在 `outputs/research_smoke_v3/` 和 `results/SMOKE_RESEARCH_V3/` 写入随机模型及结果包，目录已存在时拒绝覆盖。完成一次即可，不重复运行同配置。可先运行队列 dry-run，查看缺少许可的任务为何阻塞：

```bash
python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml
```

## 官方数据授权后的接入

朋友应通过自己的合法渠道取得官方 ZIP，设置其真实本地路径。不要从本项目聊天或 Git 传输官方数据。接入脚本只在该机器本地解压并审计；同名目标存在时拒绝覆盖。

```bash
export OFFICIAL_ARCHIVE='<朋友合法持有的官方ZIP绝对路径>'
python -m scripts.prepare_official --archive "$OFFICIAL_ARCHIVE" --destination data/official --report-dir outputs/official_ingest --config configs/official.local.yaml
python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml --archive "$OFFICIAL_ARCHIVE"
```

配置中的 Train/Val 哈希采用按列名和行顺序规范化的内容哈希，不受 Windows/Linux 换行差异影响；ZIP SHA256 必须严格相同。所有训练必须使用原官方划分。新骨干的配置保持 `BLOCKED`，须先完成独立通用非医学权重的来源、许可证、再分发与SHA256审查；不能直接使用旧胸片 checkpoint。

## 8 GPU 第一批执行

当 dry-run 仅对已满足条件任务显示 READY，且用户明确批准所选任务后运行：

```bash
python -m scripts.gpu_queue --matrix configs/research/first_batch_8gpu.yaml --archive "$OFFICIAL_ARCHIVE" --execute
```

调度器为 GPU0–GPU7 各启动最多一个独立进程，设置对应 `CUDA_VISIBLE_DEVICES`，并在 `outputs/research_queue/status.json` 记录状态。每个任务写入唯一 `results/<实验ID>/seed_42/fold_none/`；已有完整或部分结果不覆盖，只有带 `last.pth` 且配置/数据/代码哈希完全一致时可恢复。失败任务与其他 GPU 隔离。默认队列：GPU0 G02、GPU1 G03、GPU2 G05、GPU3 G07、GPU4 L03、GPU5 S01、GPU6 R01、GPU7 G01_REPRO。**G05/G07 在新权重审查完成前会显示 BLOCKED，绝不会启动。**

单独任务命令（仅获得授权且前置核验通过时执行）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_research --config configs/research/G02.yaml --archive "$OFFICIAL_ARCHIVE"
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_research --config configs/research/G02.yaml --archive "$OFFICIAL_ARCHIVE" --resume
```

完成后按 [RESULT_CONTRACT.md](RESULT_CONTRACT.md) 校验并打包结果。只回传经过授权允许共享的结果包；不通过 Git 提交数据、模型或原图。机器上独立保存 CUDA/驱动环境和 4090 本地测速，不能写成 T4 认证。
