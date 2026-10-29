# 2026 大数据挑战赛正式赛规合规性审计

审计日期：2026-10-29（北京时间）。本次为**只读赛规审计**：没有启动训练、改动冻结 checkpoint/提交 CSV 或生成新的 Test 预测。逐项状态、证据位置、缺口和修复建议见[合规证据表](final_compliance_evidence.md)；可复算的本地文件哈希、字节数、Git 扫描及提交校验结果见 `outputs/final_compliance_audit/local_evidence.json`。其中 `PASS` 只代表本地列明的证据通过，主办方最终核验仍独立进行。

## 依据和结论边界

用户提供的《比赛参赛规则》PDF：`F:\计挑赛\大数据规则\aa80bececc2ee40e69cde5f8f2a0eaec.pdf`，SHA256 `ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6`；2026-10-27《补充说明》PDF：`F:\计挑赛\大数据规则\b23d6a1f19602da2a9bd4048108175d0.pdf`，SHA256 `836c4f5d3202f2dc65830fd919ca195797c12b7507d750121786a21672dc7a44`。外部公开规则链接见项目已有[规则依据](compliance.md)。本报告以用户给定 PDF 的书面条款为准；未将赛题 FAQ 或以往口头说明作为额外许可。

目前**不具备最终合规提交的全部证据**：没有官方 T4 100+5 实测、最终材料包及主办方确认的朋友服务器授权。还有两个确定的工程缺口：旧 S01 多 GPU 队列绕过了已失败的裁切 QC；旧版 `outputs/compliance/` 核验表记录了与真实 B02 不一致的预训练来源，不能递交。M02 年龄模型未接入正式 `predict.py`/`cost.py`。这些是现存代码/材料问题，不能以本次未运行训练来掩盖。

## 1. 预训练权重

| 本地模型 | 实际初始化权重与主数据集 | 固定公开权重链接与文件 SHA256 | 许可证证据与状态 |
|---|---|---|---|
| B02 EfficientNet-B0-224 | `timm/efficientnet_b0.ra_in1k`，ImageNet-1K | [固定 revision 权重](https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a/model.safetensors)，`d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd` | [固定模型卡](https://huggingface.co/timm/efficientnet_b0.ra_in1k/blob/1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a/README.md)标识 ImageNet-1K、Apache-2.0；本地全文 `data/pretrained/efficientnet_b0.ra_in1k/LICENSE-2.0.txt`，SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`。 |
| G01/G02 ConvNeXt-Tiny，M02c/d 图像骨干 | `timm/convnext_tiny.fb_in1k`，ImageNet-1K | [固定 revision 权重](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors)，`08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276` | [固定模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/blob/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)标识 ImageNet-1K、Apache-2.0；本地 Apache-2.0 全文及上游 ConvNeXt MIT 文本均存档且 SHA256 匹配。 |
| M02b Age-only | 无视觉预训练权重 | 仅 Train Age 拟合的逻辑回归参数 | 不把逻辑回归结果虚构成外部预训练。 |

> 审计原则：`timm` 库的代码许可证与具体权重的授权分开检查。项目本机 `timm 1.0.22` 包元数据声明代码 Apache-2.0；权重许可依据是两个**具体权重**的模型卡和同版本归档许可证。Apache-2.0 全文允许附条件使用和再分发，须带许可证、著作权及必要修改声明；ConvNeXt 上游代码还保留 MIT 文本。两组本地权重/许可证/模型卡 SHA256 均经 `src/model.py:verified_weights` 实际复算。公开模型页未要求登录或 NC 授权。最终接受权归主办方。

G05 EfficientNetV2-S、G06 ConvNeXt-Small、G07 Swin-Tiny、G08 DenseNet-121、G09 EfficientNetV2-S 等只有架构/配置，缺精确权重的公开来源、主数据集、许可证全文及文件哈希，状态均为 `UNVERIFIED`，不能正式运行。D01 教师与 R04 自监督没有正式权重/训练产物；若启用，须依补充说明逐师生登记初始化、软标签数据范围、蒸馏关系和哈希。旧胸片 checkpoint 未见于上述已完成模型的初始化清单。

## 2. 数据和模型选择

官方原始 CSV 字段为 Train/Val `Image_name, Age, True_Label`，Test `Image_name, Age`；本地读取行数为 51,396/5,849/5,855。B02/G01/G02 使用图像，训练集和验证集分别读取。训练器按 Val AUROC、AUPRC 选 checkpoint；并未把 Val 合并到已完成模型的训练集。M02b/c 只用官方 Train 标签拟合 Age-only 模型与融合头，M02c 使用冻结 G01 特征；内部 LR/轮次选择用 Train 内 80/20，随后在全 Train 重训固定轮次。G01 backbone 曾由官方 Val 选择，因此 M02 内部 holdout **不是整个系统的独立验证**。

年龄参数见 `outputs/M02/age_preprocessing.yaml`：仅由 51,396 行官方 Train 得出有效年龄下界 17、上界 96、中位数 57、填补后均值 56.738754、标准差 10.185559。Train 原始 Age 有 10 个非数值缺失和 4 个 0；按既定规则共 14 个被填补。Val/Test 只套用固定规则，没有用 Test Age 分布拟合。图像归一化采用固定 ImageNet 常数，验证和推理无随机增强。S01 裁切参数只在 Train 审计，但其 QC 已确认极暗图截断风险，当前训练门禁为失败。

本地代码及产物没有显示 Test 图像用于训练、伪标签、早停、归一化统计、类别先验或阈值拟合。代码检查无法证明朋友服务器或任何未留档进程的完整历史；此项整体保留 `UNVERIFIED`。已有 Train/Val 关联审计未发现精确重复，但记录 1 对疑似同眼不同拍摄；无患者 ID，不能把验证指标称作患者独立结果。若以后合并 Val 训练，须在合并前锁定所有选择规则并留存时间顺序证据，否则不得用并入后的 Val 再早停或调参。

**确定缺口：**`configs/research/S01.yaml` 仍为 `PENDING`、没有依赖，`after_local_g02.yaml` 仍安排 GPU5 的 S01；该路径使用 `src/research/preprocess.py` 的旧裁切算法，不受 `outputs/S01_qc/visual_review.json` 的 `BLOCKED` 限制。应先在队列中阻止该任务，改进边界保护并完成新一轮 QC，才能重新讨论训练。

## 3. 推理、大小与 T4 验收

`predict.py` 和 `cost.py` 均调用 `src/inference.py:Predictor.predict_one()`，包含磁盘解码、固定预处理、H2D、前向、概率/标签生成以及 CSV 写入/flush；`Predictor` 在加载后执行 `eval()` 并冻结参数。`cost.py` 正式模式强制 NVIDIA T4、Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8，100 测时图与独立 5 预热图、batch=1 顺序执行 3 轮，CUDA 同步后按每图三轮均值再求 100 图均值。逐样本核对概率绝对差 ≤1e-3、二值标签和全对排序；只在实测满足约束时由脚本给出结果。新建 `scripts/accept_t4.py` 先核验完整 Test CSV、模型/CSV 哈希，再调用同一 `cost.py`；`--preflight-only` 固定输出 `UNVERIFIED`。

| 已完成模型 | 实际单文件字节 | 十进制 MB | 当前本地记录 |
|---|---:|---:|---|
| B02 | 16,295,141 | 16.295 | 4060 FP32，101.01 ms；并非 T4 |
| G01 | 111,331,231 | 111.331 | 4060 FP32，99.16 ms；并非 T4 |
| G02 | 111,329,503 | 111.330 | 4060 FP32，123.11 ms；并非 T4 |
| M02b / M02c / M02d | 377 / 111,734,977 / 111,734,785 | 0.000377 / 111.735 / 111.735 | 仅本地年龄实验 |
| L03b | 111,331,295 | 111.331 | 4060 FP32，71.08 ms；并非 T4 |

以上只统计**当前模型自身的实际文件**，未把训练恢复用 `last.pth`、未加载的预训练原始权重计入推理；最终若组合模型、使用引擎文件或外部配置，必须按实际加载文件去重后重新计算解压后总量。当前各单模型文件低于 500,000,000 字节，但没有最终部署包，故最终大小状态为 `UNVERIFIED`。FP16 全样本预测一致性和正式 T4 延迟亦为 `UNVERIFIED`。旧单图重放证据中 B02 相对其冻结 Val 概率差 `2.53e-5`，虽低于补充说明的 `1e-3`，仍应在最终精度和官方样本上核对排序。

M02b/c/d 有单独的 `src/multimodal.py:M02Predictor`，当前正式 `predict.py`/`cost.py` 未接入该年龄接口；若选 M02 作最终模型，此路径检查为 `FAIL`，不能直接提交。若最终采用纯图像 B02/G01/G02，此缺口不影响其现有统一接口。

## 4. Round 01 CSV 和最终材料清单

`submission/round_01_B02/submission.csv`：5,855 行；UTF-8 无 BOM；列名与顺序恰为 `Image_name, Probability, Pred_Label`；图像名与**原始** Test CSV 完全同序；无重复 ID，概率范围 `6.50063e-7`–`0.996598`，二值标签为固定 0.5 阈值；本地 CSV SHA256 `7ab6106b170e9fee679a3135139e1cd4723473e9952d57ff32f542da2d51414c`，元数据中的 B02 checkpoint SHA256 与现存权重一致。但元数据 `uploaded:false`，且未见平台最高分回执，故不能证明此 CSV 就是历史最高分对应文件。

| 官方材料项 | 当前情况 | 提交前动作 |
|---|---|---|
| `code/`，含 `train.py`、`predict.py`、`model.py`、`utils.py`、依赖表 | 仓库有相应代码与 `requirements*.txt`，尚无最终整理目录 | 固定 commit，复制**实际运行版本**和 Python3.12/T4 可复现依赖；证明脚本入口。 |
| `model/` | B02/G01/G02/M02 等本地 checkpoint 存在，未选定最终模型/打包 | 仅放最终推理实际加载的文件，生成解压后字节清单和 SHA256。 |
| `pretrain_model/` | 本地有两个公开 ImageNet 原始权重；最终包未整理 | 按最终使用模型提供权重或按官方材料要求说明其公开来源，并保留完整哈希。 |
| `licenses/` | 两个 Apache-2.0 全文在本地；ConvNeXt MIT 文本也在本地 | 复制适用全文和著作权/修改通知，逐文件校验。 |
| `cost.py` | 代码具备；T4 实测缺失 | 在指定环境和官方 100+5 图执行 `scripts/accept_t4.py`，留存3轮逐图时长与零不一致证明。 |
| `模型核验材料与数据声明清单.xlsx` | `outputs/compliance/` 有**旧版模板**，但记录 torchvision rwightman 链接、许可证/哈希空、`formal_training_started:false`，与真实 B02 不符 | **禁止直接提交旧表**；依据最终模型和本次证据重建中文表，填全 A1–A6、数据/Val/精度/师生与 T4 声明。 |
| `submission.csv` | 本地 Round 01 格式通过；最高分归属未核实 | 从平台回执核对最高分 CSV 精确 SHA256，再复制进最终包，与模型和 cost.py 一致。 |
| `队伍名_最终提交材料.zip` | 不存在 | 全部项目齐备、解压后重验文件/证据，再由参赛账号提交。 |

## 5. GitHub 与协作边界

本地 `git ls-files` 检查 226 个已跟踪文件，`git rev-list --objects --all` 检查本地可见历史：未见官方原始数据、权重、Round 01 CSV 或常见私钥文件路径；常见私钥头和 GitHub token 前缀内容扫描无命中。`.gitignore` 覆盖 `data/*`、`outputs/*`、`results/`、`submission/round_*/`、权重与密钥扩展名。此项不构成对**远端所有分支、GitHub 缓存或未列出的秘密格式**的完全证明，发布前仍需远端复核。

朋友服务器的参赛身份、其访问/存放官方数据的权利、第三方协作范围没有主办方书面证明，状态为 `NEEDS_ORGANIZER_CONFIRMATION`。在证明前，服务器仅运行合成数据；不得传输官方 Train/Val/Test、标签或以朋友账号访问赛方数据。若未来由朋友训练，结果包须附固定 Git commit、数据版本、初始化权重、全部训练配置和日志，先独立验收再纳入最终材料。

## 本次核验与可执行命令

- 冻结资产 SHA256：253 个文件匹配。
- `scripts.final_compliance_audit` 只读复算两个精确权重和许可证哈希、模型文件大小、CSV/元数据、Git 路径与常见秘密标志，结果在 `outputs/final_compliance_audit/local_evidence.json`。
- `scripts.accept_t4 --preflight-only` 对现有 Round 01 给出 `UNVERIFIED`，5,855 行及模型/CSV/清单哈希匹配；没有 T4 设备/官方图片，因此没有运行正式测速。

最终实测命令（仅在真正的官方 T4 环境、官方 100 张与独立 5 张图像可用时）：

```bash
python -m scripts.accept_t4 submission/round_01_B02/submission.csv <官方pictures_100目录> <官方pictures_5目录> --checkpoint outputs/runs/B02/best.pt --test-manifest data/official/manifests/test.csv --output outputs/cost_t4_round01
```

该命令核验的是当前 Round 01 模型与 CSV；若最终最高分另有文件，必须替换为**真实最高分的冻结模型和 CSV**并重新执行，不能复用本次预检作为通过证明。
