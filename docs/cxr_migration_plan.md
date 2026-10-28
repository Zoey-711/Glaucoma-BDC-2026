# 胸片组件迁移与青光眼泛化实验方案

日期：2026-10-28。当前阶段为源码检查、已有结果复核和配置设计；**停止于方案，不迁入运行时代码、不下载权重、不启动训练或测试推理**。

## 当前结论与冻结边界

旧工程已定位到 `E:\计挑赛\cxr-multilabel-challenge`，只读检查，未加载其中任何 checkpoint。冻结 `outputs/runs/B02/` 与 `submission/round_01_B02/` 下全部文件；本次操作前后 SHA256 清单见 `outputs/migration_review/frozen_inventory.json` 与 `verification.json`。新材料均写入独立目录，不修改原实验注册记录。

B02 官方验证 AUROC 0.9830597088124308、AUPRC 0.7457620442511624；用户反馈官方测试 AUROC 0.973200、排名约600多/700多，作为用户报告单独登记，未独立核验。两集合分数差约0.009860，不是同一数据集上的配对差，不能由此判断泄漏、统计显著退化或稳定泛化能力；也不能依据单次榜单反推模型参数。后续选择依据固定验证协议与重复实验，测试预测及榜单不用于调参。

策略由 Loss 扫描转为：先完成审计，再做分辨率和骨干对照，最后才讨论多种子、5-fold与融合。暂缓 I02/I03 等 Loss 实验；不会改写历史自动建议文件，新的优先级以本方案为准。

## 已检查组件及迁移决策

所有下列路径相对于旧工程根目录；已读源码SHA256见 `outputs/migration_review/source_inventory.json`。

| 来源 | 可复用部分 | 必须隔离或修改 |
|---|---|---|
| `src/cxr/models/factory.py` | torchvision DenseNet121、ConvNeXt-Tiny、Swin-Tiny、EfficientNetV2-S 的架构构建及分类头定位 | 删除10类硬限制，改1输出；`weights=None`初始化结构，只加载独立核验的通用权重；禁止`pretrained=True`自动下载；不能导入旧训练权重 |
| `scripts/train.py` | AMP溢出计数、实际optimizer步计数、峰值allocated/reserved显存记录、按有效样本数归一化的梯度累积 | 去掉10标签mask、study采样/聚合、胸片增强、EMA和差分学习率等默认策略；第一批不改变B02单lr、按epoch cosine、patience5 |
| `src/cxr/utils/training_resume.py` | 主进程Python/NumPy/torch及DataLoader生成器状态保存/恢复 | 仅后续同一青光眼运行可恢复；要求配置/数据/权重来源一致，拒绝胸片恢复文件；persistent worker内部RNG未覆盖，不能宣称任意断点逐位复现 |
| `src/cxr/engine/early_stopping.py` | 明确状态机、非法值检查、状态序列化思路 | 不直接替换当前逻辑：旧版study macro-AUC、不同patience/min_delta，缺少当前AUROC并列时AUPRC择优逻辑 |
| `src/cxr/batch/plan.py` | 任务ID唯一、配置/代码/划分哈希冻结、依赖图与禁用任务检查 | 原schema、GPU组、有效batch16与胸片fold路径均不适用，须改为青光眼任务契约 |
| `src/cxr/batch/delivery.py` | 分离结果/模型归档、SHA256校验、回传包完整性检查 | 不执行原打包器，不复制任何胸片数据/模型/预测；新包只白名单收录本赛道批准的文件 |
| `src/cxr/batch/results.py` | 按ID对齐、重复ID拒绝、成对比较的设计 | 禁用10类macro-AUC/AP、subject/study聚合和旧OOF；沿用本工程`src/metrics.py`与`src/bootstrap.py` |
| `src/cxr/utils/cloud_paths.py` | 本机根目录映射思想 | 不能继承旧缓存/输出目录，训练身份哈希与机器路径映射分离 |

DenseNet替换`classifier`，ConvNeXt替换`classifier[-1]`，Swin替换`head`，EfficientNetV2替换`classifier[-1]`，均为`Linear(in_features,1)`；训练直接将logits送入`BCEWithLogitsLoss`，验证/推理才sigmoid。

当前青光眼`src/model.py`仅torchvision B0分支完整；其他现有骨干走timm。迁移时须新增独立torchvision分支，扩展DenseNet/EfficientNetV2允许名单，并让`Predictor`按checkpoint内backend恢复。旧工厂和当前timm同名骨干可能存在实现/权重差异，不能无验证互换state_dict。本次不实施这些运行时代码修改。

## B02 审计复核与待办

本次实际结果见 `outputs/migration_review/audit_review.json`：

- 训练51396张（1657阳性），验证5849张（191阳性）。逐行核对原始`True_Label`与标准化`label`、ID及顺序；重新检查全部训练/验证文件存在性；清单哈希与既有全量审计一致。
- 官方定义1为青光眼阳性（确诊/疑似），0为阴性；不翻转标签、不将年龄或文件序号推断为患者ID。
- 复算保存的5849行验证预测指标，检查logit-sigmoid概率一致与标签对齐；与最佳epoch5日志吻合。AUPRC保持当前梯形PR积分，不能将旧胸片average precision替代成同名指标。官方精确AUPRC实现未披露时需标记定义差异。
- 历史全量解码审计未发现缺失、损坏、像素完全重复或跨划分完全重复；**本次没有重新解码全部图片**，也未做感知近重复扫描。因此这不是新的全量内容无变更证明。
- 原始清单仅`Image_name,Age,True_Label`，无患者ID。患者级泄漏**无法排除**；不能声称已通过患者独立性审计。
- RGB+EXIF方向、直接双线性缩放、ImageNet归一化一致；验证与统一推理关闭随机增强。此次从训练/验证各固定抽16张，原图与B02缓存预处理张量完全相等；这是32张抽检，不代表全缓存检查。
- 缓存身份依赖源文件路径/大小/mtime，历史审计复用只检查清单哈希；正式迁移前需新增文件内容哈希清单和预处理代码/库版本哈希，不能把相同CSV等同于图像内容未变。

后续执行顺序：重新全量解码/像素哈希审计 → train/valid内部和跨集合感知近重复候选检索（缩放/裁切变体、人工复核，不自动删图或改官方划分）→ train/valid按图像尺寸、年龄缺失/年龄段等官方元数据作固定分层统计，标明阳性样本数与不确定性 → 完成来源/代码/数据契约再训练。近重复不是患者身份，不凭相似图片伪造患者ID。测试集不参与这些统计或预处理拟合。

## 首批独立配置

`configs/experiments/migration_drafts/`包含8份完整YAML草案，`plan.kind=migration_draft`、`initial_status=blocked`、`migration.enabled=false`。现有执行器拒绝训练此类型。它们不是可立即运行的正式配置，未登记进正式SQLite，也没有新模型成绩。

| ID | 架构/分辨率 | 对照与目的 |
|---|---|---|
| B02-320 / B02-384 / B02-448 | 原timm B0，320/384/448 | 与冻结B02仅改变分辨率；224缓存不能放大后冒充高分辨率原图 |
| ConvNeXt-Tiny-224 | 迁移torchvision，224 | 同尺寸骨干与通用权重方案对照；不声称纯架构因果效应 |
| **ConvNeXt-Tiny-384** | 迁移torchvision，384 | **第一优先实现**；vs B02为组合升级，vs ConvNeXt224为尺寸对照，vs B02-384为同尺寸骨干对照 |
| EfficientNetV2-S-384 | 迁移torchvision，384 | 显存与许可证通过后再考虑 |
| DenseNet-121-384 | 迁移torchvision，384 | 架构接口候选，第二梯队，不自动排队 |
| Swin-Tiny-384 | 迁移torchvision，384 | 架构接口候选，第二梯队；验证尺寸/window padding与推理一致性 |

共同固定：官方划分、seed42、BCE、shuffle、RGB缩放/basic增强、AdamW lr0.0003/wd0.0001、30epochs、cosine、patience5、训练AMP、验证FP32、阈值0.5、batch32、workers6。不引入旧工程的warmup、EMA、不同增强、差分学习率或训练checkpoint。

所有新骨干的权重声明保持null/pending；需核对权重本身的公开来源、主数据集、许可证全文、比赛使用及再分发权利、非NC/非gated、固定版本和SHA256。不能把torchvision代码许可证或旧赛道审批当成此赛道权重许可证审批。未下载、未核验的新权重不得标approved。

先用合成图小型资源/接口smoke检查显存与输出，不把资源smoke当正式成绩。batch32不足时暂停该严格对照；可交由显存足够的另一台机器。若必须降micro-batch并累积到有效batch32，建立新的配套对照ID，尤其B0/V2/DenseNet含BN，不能认为累积等价原batch32。验证FP32也需单独检查峰值显存。

## 两台电脑协作设计（待实施）

采用独立实验任务分派，而非跨机同步DDP；双方只使用各自合规取得的同版官方数据，不自动传输数据。`outputs/migration_review/collaboration_contract.json`冻结当前档案哈希、划分哈希、种子、预处理/指标/训练/推理代码哈希、依赖锁和候选配置哈希。新训练器实现完成后应生成v2契约，不能沿用旧代码哈希宣称相同实现。

机器A建议负责B0分辨率组；机器B建议负责ConvNeXt组，优先384。这是建议分工，机器规格与任务领取均未绑定，没有自动启动。V2-S等待显存检查。每台机器维护独立SQLite和输出目录，不在网络盘共享SQLite；中央领取表记录唯一experiment_id、config_hash、machine_id、时间和状态，避免重复启动。输出包先暂存并核验再导入，冲突拒绝覆盖。

身份分为可移植实验哈希（相对路径、数据内容/划分、超参数、代码/权重哈希）与机器执行清单（实际根路径、GPU/驱动/OS、依赖版本、workers、精度）。现有注册哈希含绝对路径，须新增规范化逻辑后才可跨机去重。机器路径可以不同，算法参数不得静默改变；异构GPU不保证逐位相同。

统一回传：完整解析配置、环境、权重合规证据、数据契约、checkpoint及SHA256、history、validation.csv（image_id/logit/probability/label/experiment_id，官方提供时含patient_id）、报告、模型字节数、分阶段峰值显存、延迟原始记录。中央只凭ID对齐并重算指标，不信任手填成绩；同配置已完成则复用。回传程序只白名单收录青光眼运行文件，路径穿越、错误标签/缺失ID、哈希不符均拒绝。

## 分析协议与后续批准条件

每个新实验记录AUROC、同定义AUPRC、Recall/Specificity、相对冻结B02的ΔAUROC、配对Bootstrap 95%CI（2000次、seed42）、模型实际文件大小、GPU型号/精度/端到端均值与P95、训练和推理峰值allocated/reserved MiB。无实测均留空，候选登记草案见 `candidate_plan.csv`。B02旧运行未记录峰值显存，不回填猜测值。

Bootstrap固定对齐同一5849样本；存在可靠患者ID时按患者cluster重采样，否则使用图像分层重采样并明确可能低估关联导致的不确定性。该CI只表示当前验证集条件下的不确定性，不覆盖选模型、多重比较或种子变异。新骨干与B02的比较同时涉及权重来源差异。

本地延迟统一从原始图片读取到CSV写入flush，batch1、固定100样本/5预热/3轮，使用共同推理接口；各机器延迟分列，不能直接当架构速度公平对照。所有入围模型最终在同一T4环境实测，文件合计≤500MB、端到端均值≤100ms并验证结果一致；4060不能替代T4。

先检查单模型正向ΔAUROC、AUPRC和成本。CI跨0不能称稳定提升；CI下限>0仍需多种子复核与分层审计。确认有价值后另行批准3种子，再决定5-fold或融合；5-fold仅切官方训练集、官方验证仍独立，患者信息不足时不能称患者独立交叉验证。严禁现在自动启动5-fold、融合或新的提交。

## 实施清单与停止点

1. 完成全量/近重复审计、文件内容哈希；保留本次初步复核作为独立记录。
2. 在青光眼工程适配单输出torchvision工厂、经审查权重加载、显存记录和协作协议；不修改旧工程。扩展测试：输出[N,1]、BCE有限梯度、禁止胸片checkpoint、predict/cost一致、篡改/去重/冻结保护。
3. 完成ConvNeXt-Tiny权重审查及384资源smoke，冻结正式候选配置，提交给用户确认。
4. 确认后只启动获批任务；后续每个实验独立目录，结果导入再分析；Round01永不覆盖。

**本次已停在第0阶段：源码检查、只读结果复核、8份禁用配置草案和协作契约设计。上述运行时迁移与正式训练均未执行。**

规则依据（2026-10-28核对）：[官方赛题与标签定义](https://www.ncccu.org.cn/index/Paper/case2.html?way=1)、[10月27日补充说明](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)。补充说明要求非医学可再分发权重及来源材料，并禁止医学预训练、外部医学续训和测试信息参与模型选择；predict/cost必须使用相同加载和推理逻辑。
