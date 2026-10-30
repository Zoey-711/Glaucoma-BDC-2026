# glaucoma-bdc-2026

Phase F 已完成：固定等权概率融合 F01（G01+B02）Val AUROC **0.986582**、AUPRC **0.823281**，ΔAUROC **+0.003019**，未校正配对 95% CI **[+0.000347,+0.006123]**；F04（G01+G02）AUROC **0.987077**，但尚未测融合延迟。F01 实际加载 **127.63MB**，4060 FP32 独立测时 **82.34ms**，交错复核 **83.26ms**，T4 仍为 **UNVERIFIED**。五项比较是探索性结果，不能宣称稳定泛化收益。详见 [融合分析](docs/phase_f_fusion_report.md)、[部署预检](docs/phase_f_deployment_report.md)与 [Train 质量压力测试](docs/phase_f_robustness_report.md)。本阶段没有新训练或官方提交。

R01b 轻度颜色增强已完成：G01 同配置下仅增加训练期 ColorJitter，Val AUROC **0.983430**、AUPRC **0.796771**，相对 G01 ΔAUROC **−0.000132**，配对 95% CI **[−0.004268, +0.003473]**，未证明收益。最佳 epoch 9，4060 端到端 **102.87ms**；详见 [R01b 实验报告](docs/R01b_experiment_report.md)。没有 T4 实测，也未启动 R01c/R02。

S01-v2 已完成并验收：最佳 epoch 5，Val AUROC **0.981368**、AUPRC **0.789361**，相对 G01 的 ΔAUROC **−0.002194**，95% CI **[−0.007120, +0.002507]**，未证明收益。4060 端到端 **102.98ms**，不是 T4 成绩。Train 51,396 张 QC、160 张分层对照及已知暗弱外周回归通过；详细结果见 [S01-v2 报告](docs/S01_v2_experiment_report.md)。下方 S01 BLOCKED 状态特指保留的 v1。朋友服务器请使用更新后的 [任务单](FRIEND_TASKS.md)，旧队列不再作为待执行授权。

正式赛规审计见[审计报告](docs/final_compliance_audit.md)和[逐项证据表](docs/final_compliance_evidence.md)。当前尚无官方 T4 实测；旧版合规 XLSX 与实际 B02 权重不符，旧 S01 多 GPU 队列未继承已失败的 QC 门禁，提交前须处理。

S01 裁切 QC 已完成，**当前 BLOCKED，未启动正式训练**：审计官方 Train 51,396 张，生成 120 张前后对照并额外复核 12 张极端候选。55.65% 可裁切、1.18% 检测回退；但极暗图 `train_05402.jpg` 存在外周截断风险，不能仅凭普通联系表判定安全。73 项测试和真实 Train Smoke Test 通过，仍需先修正规则、重新通过 QC。详见 [S01 审计报告](docs/S01_experiment_report.md)。

M02d 与 L03b 已完成并验收。M02d 零年龄占位消融 AUROC **0.982597**、AUPRC **0.803863**；真实年龄相对占位的 ΔAUROC **+0.000285**，95% CI **[−0.000256, +0.000813]**，未证明年龄有效。L03b（BCE + 0.05 × Pairwise）最佳 epoch **7**，AUROC **0.983426**、AUPRC **0.767908**；相对 G01 的 ΔAUROC **−0.000136**，95% CI **[−0.003779, +0.004021]**，未证明排序收益。详见 [M02d 消融报告](docs/M02d_ablation_report.md)与 [L03b 实验报告](docs/L03b_experiment_report.md)。65 项测试通过，完整 Val 指标独立复算、结果哈希及 127 个冻结文件核验通过。L03c、五折及新官方提交均未启动。

最新 M02 年龄融合实验已完成：冻结 G01 图像特征 + 年龄 MLP 的 Val AUROC **0.982882**、AUPRC **0.802841**；相对 G01 的 ΔAUROC **−0.000680**，配对 95% CI **[−0.002415, +0.001001]**。本次没有证明融合优于 G01，暂不扩大到多折多模态训练。年龄质量审计、Train 内部选型、分层指标、鲁棒性与 4060 测速见 [M02 实验报告](docs/M02_experiment_report.md)。原有模型与 Round 01 保持冻结。

Phase 3 研究工程：20 组主实验配置、6 组复核/部署配置、统一 Trainer、8 GPU 队列、结果包验收与合成 Smoke Test 已接入。G02 ConvNeXt-Tiny-384 已在本机完成，官方 Val AUROC **0.982912**、AUPRC **0.781187**，详见 [G02 完整实验报告](docs/G02_experiment_report.md)。逐组状态见 [研究状态表](docs/research_status.md)，服务器命令见 [朋友训练指南](FRIEND_TRAINING_GUIDE.md)，交付文件见 [结果契约](RESULT_CONTRACT.md)。目前没有 T4 实测。

协作者请先读 [第一批 8 GPU 任务单](FRIEND_TASKS.md)：逐卡说明需要训练什么、何时不能运行、每个实验需要交付哪些文件。

Phase 2 最新结果：G01 ConvNeXt-Tiny-224 已完成，Val AUROC 0.983562、AUPRC 0.804615，相对 B02 的 ΔAUROC +0.000502，配对 Bootstrap 95% CI 跨0；详见 [G01实验报告](docs/G01_experiment_report.md)及[Train/Val关联审计](docs/phase2_audit_report.md)。未生成 Round02 提交。

早期感知哈希筛查的阻塞记录保留在 [G01历史阻塞报告](docs/G01_execution_status.md)；多层复核及用户授权的双轨推进结论以本轮审计报告为准。[胸片组件迁移方案](docs/cxr_migration_plan.md)记录此前只读迁移阶段，8份禁用候选草案仍在 `configs/experiments/migration_drafts/`。

2026 第八届全国高校计算机能力挑战赛 · 大数据挑战赛 · 眼底图像青光眼阳性概率预测。
分阶段实验工程：官方数据已完成接入和审计，原实验配置及注册/分析系统已建立。B02、G01、G02 和 M02 已完成；M02 只训练年龄分类器与冻结图像特征融合头。所用 ImageNet 预训练权重完成来源、许可证及 SHA256 核验；患者级划分独立性仍未获证实。

## 分轮提交

提交文件统一放入 [`submission/`](submission/README.md)，按本地提交轮次归档，例如 `submission/round_01_B02/submission.csv`。同目录保存模型与文件 SHA256、精度及环境元数据；禁止覆盖已有提交。合成 smoke 文件保留在 `outputs/`。

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint outputs/runs/B02/best.pt --csv data/official/manifests/test.csv --root data/official --round round_01_B02 --device cuda --precision fp32
```

命令只生成本地文件，不自动上传；已生成后不要重复执行同轮次。

B02 最佳 epoch 为 5（共训练 10 个 epoch 后早停），官方验证集 AUROC **0.983060**、AUPRC **0.745762**、Recall **0.680628**、Specificity **0.993107**（阈值 0.5）。模型 16.30 MB，本地 RTX 4060 FP32 端到端平均 101.01 ms；**T4 尚未实测**。单种子结果不代表稳定收益。曲线和逐样本预测见 `outputs/runs/B02/`。

## 阶段二：实验管理入口

完整说明见 [分阶段实验管理](docs/experiments.md)。

```powershell
.\.venv\Scripts\python.exe -m scripts.experiments status
.\.venv\Scripts\python.exe -m scripts.experiments analyze
```

- 配置：`configs/experiments/`（30份；P03含320/384两份）。
- 注册表：`outputs/experiment_manager/experiment_registry.csv`，SQLite负责原子去重。
- 总报告：`outputs/experiment_manager/analysis/experiment_report.md`。
- B02训练/验证曲线、逐样本logits和概率、最佳checkpoint及本地原图测速：`outputs/runs/B02/`。
- 后台日志位置：`outputs/experiment_manager/B02_process.json`。训练结束自动刷新报告及下一批建议，不会自动运行其他实验。
- 当前没有T4实测，相关字段保持空白。下面保留第一阶段工程和基础命令说明；阶段二正式实验统一从实验管理入口执行。

## 目录

```text
configs/       baseline 配置、权重合规登记模板
src/           数据、审计、模型、训练、指标、统一推理、提交校验
scripts/       合成数据 smoke、提交校验、合规材料导出
tests/         自动化回归测试
data/          本地官方数据（不入版本控制）
outputs/       实验登记、checkpoint、审计、预测、测速和合规报告
docs/          数据接入、规则依据、合规、复现和风险说明
predict.py     提交预测入口
cost.py        官方协议测速入口
requirements.txt
requirements-smoke.lock.txt  本机 CPU smoke 的完整依赖锁定
```

## 快速复现（PowerShell，项目根目录）

当前已建立 `.venv`（Python 3.11，本机 CPU 测试环境）。官方环境为 Python 3.12、PyTorch 2.10.0 / torchvision 0.25.0、CUDA 12.8；本机环境不是官方测速环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke --output outputs/smoke_new
```

smoke 为 4 张训练、4 张验证、2 张推理、1 张预热的**随机像素合成图**；随机初始化 EfficientNet-B0，64×64、CPU、1 epoch。它只证明流程可运行，不代表医学性能或 T4 速度。每次使用新 output 目录，已有实验不覆盖。

## 官方数据接入与下一步

详见 [数据接入](docs/data.md)。用户提供数据包 `F:\计挑大数据\glaucoma_first.zip`，接入脚本生成 `configs/official.local.yaml`，原始 `configs/baseline.yaml` 仍保持空路径模板。审计状态以 `outputs/official_ingest/ingest.json` 及 `audit.json` 为准。

1. 将官方图像保存在受控本地目录，将原始标注映射为 `image_id,path,label,patient_id` CSV；patient_id 仅在官方确实提供时填写。保留官方划分。
2. 复制 baseline 配置为本地配置，填写实际 root / train_csv / val_csv。所有相对路径相对于运行时项目根目录，不相对于 YAML。
3. 完成非医学 ImageNet 权重的来源与**权重本身再分发许可**核验，填写完整 SHA256、许可证全文与人工审核信息；未经审核的模板无法训练。
4. 审计通过后，再单独决定启动正式训练。本阶段未执行下面正式训练命令。

```powershell
# 用真实路径替换占位符，不能原样执行
.\.venv\Scripts\python.exe -m src.audit --train-csv <真实训练CSV> --val-csv <真实验证CSV> --root <真实图像根目录> --output outputs/official_audit.json
.\.venv\Scripts\python.exe -m scripts.experiments run --config configs/experiments/B02.yaml
.\.venv\Scripts\python.exe predict.py --checkpoint outputs/baseline/best.pt --csv <真实测试清单CSV> --root <真实图像根目录> --output outputs/submission.csv --device cuda --precision fp32
.\.venv\Scripts\python.exe -m scripts.validate_submission --submission outputs/submission.csv --sample <官方示例提交CSV>
.\.venv\Scripts\python.exe cost.py outputs/submission.csv <官方100张目录> <官方5张预热目录> --checkpoint outputs/baseline/best.pt --device cuda --precision fp32 --output outputs/cost_t4
.\.venv\Scripts\python.exe -m scripts.compliance --manifest configs/weights_manifest.local.json --output outputs/compliance_final
```

T4 环境需安装 CUDA 12.8 版 PyTorch：`python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128`。应在官方 Python 3.12 环境重新锁定完整依赖，勿将 CPU lock 用作 GPU 环境锁。

## 实现与限制

- BCEWithLogitsLoss、AdamW、CosineAnnealingLR、CUDA AMP、固定种子与确定性算法；验证 FP32；早停按 AUROC，AUPRC 用于并列时择优。
- AUPRC 使用 PR 曲线梯形积分（非 average precision）；Recall / Specificity 阈值默认 0.5，固化于 checkpoint。极端类别不平衡下应结合阳性召回分析，但禁止利用测试集选择阈值。
- 保存 best.pt、完整配置、来源信息、数据清单哈希、环境依赖、历史指标、验证预测和 checkpoint SHA256；完成实验登记于 `outputs/experiments.csv`。随机性跨设备/依赖版本不保证逐 bit 一致。
- 审计逐图统计尺寸/比例、缺失/损坏图、解码像素 SHA256 重复以及患者交集；患者 ID 不完整时明确显示无法完整检查，不推测患者 ID。近重复、不同裁剪、左右眼关联需要额外人工审查。
- 推理 eval + inference_mode、固定归一化、无随机增强、无梯度更新、无跨样本自适应。predict/cost 共用 Predictor 和 predict_one。
- cost 默认强制 T4、100/5 张和 3 轮；读文件到写 CSV+flush，包含预处理、传输、前向、后处理；加载/预热不计时，CUDA 同步。系统文件缓存未清空，flush 不等同物理磁盘 fsync。保存逐图逐轮耗时、均值/P95及一致性结果。
- cost 核对三轮概率误差 ≤1e-3、二值标签、完整概率排序（含并列）。测速目录文件名按唯一的大小写无关映射对应官方 Image_name，保留提交 ID 原值，重名/歧义拒绝。模型及必要 config 均嵌入唯一 checkpoint，计入大小；新增引擎/外置 config 时必须扩展清单和大小检查。
- B02 使用的独立权重声明已通过来源、Apache-2.0许可证与SHA256核验；通用权重模板及其他骨干仍需各自审核。训练数据来源已按用户提供的官方压缩包登记，代码无法代替官方来源证明。

规则来源、风险及材料清单见 [合规报告](docs/compliance.md)。
