# 2026 大数据挑战赛实现核验（2026-10-29）

本报告仅核验本地官方数据、冻结的 B02/G01/G02 产物和当前代码。未启动训练，未生成新的官方 Test 预测，未修改冻结结果。

## 核验依据

- [官方赛题页面](https://www.ncccu.org.cn/index/Paper/case2.html?way=1)：标签 1 为青光眼阳性；Train/Val/Test 分别为 51,396/5,849/5,855；AUROC 为主指标，AUPRC 为辅助指标；提交 CSV 采用 `Image_name,Probability,Pred_Label`、UTF-8 和默认 0.5 阈值。
- [官方参赛规则，2026-10-23](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)：T4 16GB 上按固定 100 张、另 5 张预热、batch=1、连续 3 轮计入从读图到写出结果的端到端时间；实际加载模型文件及必要配套文件解压后合计不超过 500 MB。
- [官方补充说明，2026-10-27](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)：`cost.py` 必须调用与 `predict.py` 相同的推理逻辑，100 张样本的标签判定及概率排序一致，概率绝对差容许 `1e-3`；以书面规则及正式补充说明为准。

## 1. 官方数据与字段

逐行读取 `data/official/glaucoma_first/{train,valid,test}.csv` 与项目规范化清单 `data/official/manifests/{train,valid,test}.csv`。三个原始 CSV 的图像 ID 与清单行序完全一致，原始 Train/Val 标签与清单完全一致，均无重复 ID；行数与官网一致。

| Split | 原始列 | 行数 | 阳性 | 阴性 | 阳性比例 | Age 数值缺失 | Age 范围 |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | `Image_name, Age, True_Label` | 51,396 | 1,657 | 49,739 | 3.223986% | 10 | 0–96 |
| Val | `Image_name, Age, True_Label` | 5,849 | 191 | 5,658 | 3.265515% | 0 | 24–137 |
| Test | `Image_name, Age` | 5,855 | 无公开标签 | 无公开标签 | 不可计算 | 0 | 23–92 |

原始文件只有上述列，没有患者 ID 或其他辅助临床列。Test 确实有 `Age`，但现有规范化清单只保存图像 ID、路径和 Train/Val 标签；B02/G01/G02 均为纯图像模型，未使用年龄。Train 有 4 个年龄值为 0，Val 有 2 个年龄值超过 100；这些值不应被当成已核实的临床年龄。当前不据此修改模型或官方划分。患者独立性仍无法验证。

机器可读核验结果位于 `outputs/official_2026_audit/schema_balance_v2.json`，包含三个原始 CSV 与规范化清单的 SHA256。

## 2. 类别不平衡与采样

当前 B02/G01/G02 使用训练集随机打乱、无放回抽取，batch=32，没有类别均衡采样。按 Train 的 1,657 阳性与 49,739 阴性计算，一个完整 batch 没有阳性的精确超几何概率约 **35.03%**；每轮约 1,606 个完整 batch 中，约 **563 个**会没有阳性（期望值近似）。因此 batch 内 Pairwise AUC 项经常没有有效正负对，必须退化为 BCE 并记录有效对数。没有阴性的概率约 `1.39e-48`。

若研究类别加权，Train 独立给出的 `pos_weight = 49739 / 1657 = 30.0175`、平方根权重 `5.47882`；Val/Test 标签与比例不得用于权重或采样器拟合。更换采样器会同时改变每批有效配对数及重复样本覆盖率，须另立对照，不能归因于 Loss。

## 3. 提交格式和模型选择

冻结的 `submission/round_01_B02/submission.csv` 已用项目校验器对照**原始官方 Test CSV**完成检查：5,855 行；UTF-8 可解码且无 BOM；列顺序准确；图像 ID 与 Test 文件顺序完全一致；无重复；概率范围 `6.50063e-7` 至 `0.996598`；`Pred_Label` 为 `Probability >= 0.5`，值仅为 0/1。文件 SHA256 为 `7ab6106b170e9fee679a3135139e1cd4723473e9952d57ff32f542da2d51414c`。

`src/train.py`、`src/experiment_training.py`、`src/research/trainer.py` 均以 `(Val AUROC, Val AUPRC)` 选择最佳 checkpoint；阈值 0.5 仅用于 Recall/Specificity 与提交标签。排序依赖连续概率，不依赖固定阈值 Accuracy。当前 `src/metrics.py` 的 AUPRC 使用 PR 曲线梯形积分；官网文字未指定数值积分定义，若官方采用 average precision，辅助指标可能存在细微差异，需在提交前按官方实现说明复核。

## 4. 推理与部署核验

`predict.py` 和 `cost.py` 均调用 `src.inference.Predictor.predict_one()`；此函数包含原始图像解码、确定性预处理、H2D、模型前向、概率/标签生成及 CSV 写入与 flush。`cost.py` 排除加载及预热时间，要求正式运行提供 100+5 张图、3 轮、T4 和指定 Python/PyTorch/torchvision/CUDA 版本，并逐张检查概率、标签及样本排序。合成数据端到端 Smoke Test：4 张计时图、5 张预热图，3 轮预测差异数均为 0；该 24.61 ms 的 CPU 结果没有官方测速意义。

| 模型 | 实际加载权重字节 | 500 MB 文件门槛 | 单张 Val 推理重放与保存概率绝对差 | 当前 4060 本地延迟 |
|---|---:|---|---:|---:|
| B02 | 16,295,141 | 通过 | `2.53e-5` | 见冻结本地报告 |
| G01 | 111,331,231 | 通过 | `2.18e-9` | 99.16 ms |
| G02 | 111,329,503 | 通过 | `4.49e-10` | 123.11 ms |

三模型均以 `eval()` 载入，参数 `requires_grad=False`。B02 单图重放与冻结 Val CSV 有 `2.53e-5` 差异；当前证据不能确定原因。冻结 Round 01 提交是用当前推理接口生成的，最终仍须在指定 T4 上对完整提交执行 `cost.py` 一致性和测速。单图重放不能替代全量验证或正式 100 张一致性检查。G02 在 4060 的 123.11 ms 不能视作 T4 结果，也不能据此宣布 T4 达标或不达标。

本机环境为 Python 3.11.9、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、CUDA 12.8、RTX 4060 Laptop。官方目标为 Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8、NVIDIA T4 16GB；本地未能完成官方硬件认证。

## 需要修改或补充的代码清单

1. **本次已修复：`cost.py` 环境门禁。** 正式模式现在核验 Python 主次版本 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8；原有 T4 GPU 检查与驱动/cuDNN 元数据留存继续执行。新测试验证本地 Python 3.11 不会被误报为官方环境。
2. **P1：`scripts/prepare_official.py`、`src/data.py`。** 若未来新增年龄分支，必须同时维护 Train/Val/Test 的 `Age` 字段，规定缺失/异常值处理，所有填补和标准化参数仅由 Train 拟合；目前纯图像 B02/G01/G02 无需改变。
3. **P1：`src/research/losses.py`、`src/research/trainer.py`。** L03 正式训练前补齐每 epoch 的正/负样本数、有效正负配对数、BCE 与 Pairwise 单项及加权比例日志，并对单类别 batch 做数值/梯度测试。当前实现已用 `softplus(negative-positive)` 且无有效配对时返回 BCE，但只记录配对数，尚不足以满足 L03 审计要求。
4. **P2：`src/metrics.py`。** 取得官方 AUPRC 具体数值定义后，将官方算法与本地梯形积分做一致性测试；在官方定义未公开前保留当前方法并注明口径。
5. **部署验收：** 在真正的 Python 3.12/T4 容器中，以主办方 100 张计时图、5 张预热图和最终 `submission.csv` 执行 `python cost.py submission.csv pictures_100 pictures_5 --checkpoint <最终权重>`；把 3 轮逐样本结果、加载文件清单、环境和零不一致记录留存。当前状态为 **待 T4 实测**。

## 本次测试

- `python -m scripts.validate_submission --submission submission/round_01_B02/submission.csv --sample data/official/glaucoma_first/test.csv`：通过。
- `python -m pytest -q tests/test_pipeline.py tests/test_submission_rounds.py tests/test_research_platform.py`：29 项通过。
- 合成数据 `predict.py` → `cost.py --smoke`：3 轮均 0 不一致。
- B02/G01/G02 冻结 checkpoint 各加载并重放一张 Val：通过；结果见 `outputs/official_2026_audit/inference_check.json`。
- `scripts/check_frozen_assets.py`：53 个冻结文件未变化；G02 提交版本为 `a406ab8`。
