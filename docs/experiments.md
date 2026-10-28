# 分阶段实验管理

目标：以官方验证 AUROC 优先、AUPRC 次优选择模型，并独立核验模型实际加载文件 ≤500MB、官方 T4 端到端均值 ≤100ms 及预训练/数据合规。当前仅授权运行 B02；其他正式训练必须由用户明确选择后执行。

## 实验矩阵与控制变量

`configs/experiments/` 含 30 份完整 YAML：

| 组 | 实验 |
| --- | --- |
| B01–B05 | ResNet-50 / EfficientNet-B0 / ConvNeXt-Tiny / EfficientNet-B2 / Swin-Tiny |
| I01–I06 | BCE / Weighted BCE / Focal / ASL / Weighted Sampling+BCE / Effective-number Class-balanced BCE |
| P01–P06 | 基础缩放 / 黑边裁切 / P03=320、P03_384=384 / 颜色增强 / 视盘ROI / 全图+局部 |
| F01–F06 | 最佳单模型 / 双模型均值 / 三种子均值 / 三模型均值 / 固定水平翻转TTA / 合规教师蒸馏 |
| R01–R03 | 同一冻结候选，seed=42/2026/3407；42已有等价实验则复用 |
| D01–D03 | FP32/FP16本地对照 / channels-last速度优化 / 最终材料打包 |

初筛统一 seed=42、BCE 基准、batch=32、224输入、AdamW lr=3e-4 / weight_decay=1e-4、Cosine、最多30 epoch、patience=5、训练AMP、验证FP32、阈值0.5。I/P组均对照B02，不把多种优化叠加后冒充单因素消融。P03两个分辨率分别独立运行，P04仅改变颜色增强。更改batch/LR等必须创建新版本配置，不能静默改动现有对照。

骨干组保持微调设置一致，但公开预训练权重的训练配方不同，因此结论是“骨干+其公开预训练权重”的对照，不能归因成纯架构效应。除B02外，其余骨干的权重来源尚待各自审核；不存在的权重登记文件会阻止启动。

P05/P06：缺少可靠视盘标注或经验证定位器，状态为 skipped；不使用任意中心裁剪冒充视盘ROI。F/R/D在候选未确定前为 blocked，必须先绑定真实已完成实验，生成新的 `_bound.yaml` 快照。绑定不启动训练。

## 注册、去重与失败处理

SQLite 注册表：`outputs/experiment_manager/registry.sqlite`；导出 CSV：`experiment_registry.csv`。保留用户要求的全部字段；所有未测量值为 SQL NULL / CSV空白。train_duration 单位秒，排除缓存构建、包含训练和验证；model_size_mb 使用十进制 MB。精度及设备细节见各运行目录的 latency JSON。

原子事务领取任务，配置哈希不同且实验ID相同会拒绝；相同训练配置已完成时复用，运行中时拒绝并发重复。I01/P01与B02等价，因此以后执行时复用B02。复用不是独立随机种子证据。失败运行保留目录和原因，不自动覆盖或重新训练；需检查故障后显式创建版本化实验ID。

数据只允许已登记的官方 train/valid CSV（SHA256一致）、指定原图根目录。沿用已完成全量审计，要求原始数据保持不可变；如果修改图片，必须重新运行全量审计，不能继续引用旧报告。训练和验证的确定性缩放结果可缓存为无损PNG，cache键包含原图路径/大小/mtime、清单哈希及预处理配置。测试图片不进入缓存、实验选择、Bootstrap或本地速度初筛。

## 已授权的 B02

采用 `timm/efficientnet_b0.ra_in1k`，上游固定版本 `1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a`。该模型自己的模型卡明确声明 ImageNet-1K 和 Apache-2.0，权重公开、非 gated；Apache-2.0允许按条款使用、修改及再分发，材料保留许可证、来源、修改说明。不是以源码许可代替权重许可。

- [固定版本模型卡](https://huggingface.co/timm/efficientnet_b0.ra_in1k/blob/1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a/README.md)
- [Apache-2.0 全文](https://www.apache.org/licenses/LICENSE-2.0)
- 本地证据：`data/pretrained/efficientnet_b0.ra_in1k/`、`configs/weights_b02.local.json`。
- 权重 SHA256：`d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd`（与上游 LFS 哈希一致）。最终仍以主办方核验为准。

本机 CUDA PyTorch 2.10.0+cu128 / torchvision 0.25.0+cu128，GPU RTX 4060 Laptop 8GB，Python 3.11。本地结果不能冒充官方 Python3.12/T4 复测。

```powershell
# 查看状态，不启动任何任务
.\.venv\Scripts\python.exe -m scripts.experiments status
# 生成/刷新分析，不启动任何任务
.\.venv\Scripts\python.exe -m scripts.experiments analyze
# B02（已运行/完成时不要重复启动；系统也会拦截重复）
.\.venv\Scripts\python.exe -m scripts.experiments run --config configs/experiments/B02.yaml
```

B02产物位于 `outputs/runs/B02/`：config、audit、provenance、environment、完整依赖、code_sha256、best.pt、history.json、training_curves.png、validation.csv、artifact.json、latency/local_fp32.json。validation.csv保存原始logit、概率、标签、image_id、experiment_id（若提供患者ID则保留）。每个epoch更新注册表和曲线；训练完成后自动本地原图端到端测速、生成总报告和下一批5个实验建议，**不会自动启动建议实验**。

## 候选绑定及分析

下面是待用户明确选择后的操作示例，当前未执行：

```powershell
# 指定某个额外训练的明确授权（仅示例）
.\.venv\Scripts\python.exe -m scripts.experiments run --config configs/experiments/I02.yaml --authorize I02
# 使用已完成验证预测做配对Bootstrap，第一项是reference
.\.venv\Scripts\python.exe -m scripts.experiments analyze --shortlist B02 I02 --bootstrap-repeats 2000
# 为最佳候选冻结三种子配置；以下仅绑定，不训练
.\.venv\Scripts\python.exe -m scripts.experiments bind --config configs/experiments/R01.yaml --candidates B02
# 冻结TTA或融合后用统一Predictor评估
.\.venv\Scripts\python.exe -m scripts.experiments bind --config configs/experiments/F05.yaml --candidates B02
.\.venv\Scripts\python.exe -m scripts.experiments evaluate --config configs/experiments/F05_bound.yaml
```

F02/F04的候选数必须为2/3；F03必须是相同训练设置、不同随机种子的三个模型。融合权重默认等权并冻结；不做测试集拟合。融合/TTA保存每个成员/视图的原始logits以及聚合概率，聚合的logit明确标为由聚合概率计算。蒸馏仅用官方训练集，师生都必须非医学合规初始化；当前实现要求师生输入大小/预处理一致，禁止医学教师间接引入知识。

分析目录为 `outputs/experiment_manager/analysis/`：experiment_leaderboard.csv、ablation_summary.csv、bootstrap_comparison.csv、model_latency_report.csv、experiment_report.md、seed_replication_summary.csv，以及ROC/PR、实验增益、延迟-AUROC图。没有数据时绘制明确的空状态，不制造曲线。Bootstrap默认2000次、配对同一image_id、95%百分位CI；有完整患者ID时按患者聚类重采样，否则按类别分层图像重采样。缺失患者ID、多重比较、验证集反复选择以及单种子限制均在报告中说明。

当已有至少两个可比较实验时，默认对验证AUROC前三名的非重复模型与B02做探索性配对Bootstrap；也可用--shortlist显式冻结候选。当前仅B02完成，Bootstrap表为空，不能编造比较CI。

## 部署与最终复测

所有单模型、融合、TTA均通过 `src.inference.Predictor`，predict.py与cost.py共用同一个原图读取至CSV写出函数。本地benchmark仅抽取固定官方验证图，5图预热、100图、三轮、batch=1，明确 official_protocol=false。部署D01测试FP32/FP16，D02测试冻结的channels-last设置；精度改变后需重新验证AUROC及样本一致性，不能仅凭速度选择最终提交。

```powershell
.\.venv\Scripts\python.exe -m scripts.experiments bind --config configs/experiments/D01.yaml --candidates B02
.\.venv\Scripts\python.exe -m scripts.deployment benchmark --config configs/experiments/D01_bound.yaml
# 官方T4上使用原有 predict.py / cost.py 命令完成官方100/5图复测后导入证据
.\.venv\Scripts\python.exe -m scripts.deployment import-t4 --experiment B02 --report <真实T4报告>
.\.venv\Scripts\python.exe -m scripts.experiments bind --config configs/experiments/D03.yaml --candidates B02
.\.venv\Scripts\python.exe -m scripts.deployment package --config configs/experiments/D03_bound.yaml --output outputs/final_package
```

没有真实通过的T4报告则拒绝最终打包；证据核对硬件、环境、100图/3轮、batch=1、结果一致性及模型SHA256。打包去重计算实际加载权重和配置，复制源权重、许可证、模型卡、修改说明及代码。路径重定位可能改变bundle配置哈希，打包状态为 packaged_pending_recheck，最终提交前必须用包内固定模型重新predict/cost；测试submission只在最终推理时生成，不能拿未训练结果占位。

规则复核日期2026-10-28：官网链接仍指向20261023参赛规则及20261027补充说明，详见 docs/compliance.md。官方另行提供的100图和5张预热图未在当前包中独立标识，因此本地验证图测速不作为官方测速替代。
