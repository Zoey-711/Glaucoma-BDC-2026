# Linux 服务器执行指南

任务范围见 FRIEND_TASKS.md。本次只修复发布工程，不自动授权正式训练。主办方对协作者身份和数据访问未确认前仅做合成测试。比赛图像、标注、权重和结果包不进入 Git。

## 固定版本

不要继续用旧 phase3-platform-v0.3.0。负责人提供本次发布的完整 RELEASE_COMMIT；等旧任务结束后在独立 checkout 检出，不能更新正在训练的工作树。

```bash
git fetch origin
git checkout --detach "$RELEASE_COMMIT"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain)"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-research.txt
python -m pytest tests/test_server_review.py -q
python -m scripts.smoke_research
```

RELEASE_COMMIT 必须由负责人给出。Smoke 已有输出拒绝覆盖，首次执行一次即可。test_server_review.py 用临时合成图像验证训练、恢复、打包、导入和篡改拒绝，不依赖历史本地资产。全套历史测试有本地资产依赖，新服务器不能直接复现全部测试。

## 授权后接入

从自己的合法渠道取得官方 ZIP。configs/weights_templates/ 给出已审核 B0/ConvNeXt 的通用 ImageNet 来源；自行下载核验全部 SHA256、许可证及模型卡，创建 configs/weights_b02.local.json 与 configs/weights_g01.local.json。G05/G07 权重未审完，不修改状态绕过。

```bash
export OFFICIAL_ARCHIVE='<合法官方 ZIP 实际绝对路径>'
python -m scripts.prepare_official --archive "$OFFICIAL_ARCHIVE" --destination data/official --report-dir outputs/official_ingest --config configs/official.local.yaml
python -m scripts.gpu_queue --matrix configs/research/server_reviewed.yaml --archive "$OFFICIAL_ARCHIVE"
```

READY 只代表工程检查满足，不代表赛事或负责人授权。清单内容哈希、原始 ZIP SHA256 必须一致，不修改官方 Train/Val。任务许可与增强 QC 确认后才执行：

```bash
python -m scripts.gpu_queue --matrix configs/research/server_reviewed.yaml --archive "$OFFICIAL_ARCHIVE" --status-file outputs/research_queue/server_reviewed.json --execute
```

一 GPU 一个进程，依赖未完成不启动。状态文件绑定矩阵、配置及代码版本；版本变化时保留旧状态并使用新文件。恢复必须有 last.pth，且配置/数据/代码一致；COMPLETED 不可重跑。单独训练入口同样有退休实验与旧 FOV 门禁。

按任务单验收完整结果包。Linux/8×4090 尚待朋友实际 Smoke，不能将本机 Windows 测试冒充远端实测。4090 延迟仅作初筛，最终需官方 T4 环境验收。
