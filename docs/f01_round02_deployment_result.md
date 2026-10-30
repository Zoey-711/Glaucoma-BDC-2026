# F01 Round 02：部署路径重验证与本地 CSV 结果

部署代码 commit：`92303b7bda09c45f35e52934c3af07848ad239aa`。本报告保留 [原始门禁报告](f01_round02_submission_gate.md) 的 `BLOCKED` 结论：该结论准确记录了历史 batch=32 Val 预测与部署 batch=1 预测的排序差异。原 `submission/round_02_F01/manifest.json`、`validation_report.md`、`checksums.sha256` 未修改；新结果写入独立的 `manifest_v2.json`、`validation_report_v2.md`、`checksums_v2.sha256`。

## 原因审计

F01 仍为 G01 ConvNeXt-Tiny-224 与 B02 EfficientNet-B0-224 的 FP32 概率等权平均，即 `0.5 × sigmoid(G01 logit) + 0.5 × sigmoid(B02 logit)`。checkpoint、成员顺序、原始预训练来源、配置哈希、224×224 双线性缩放、ImageNet 归一化、eval 状态与原始冻结证据匹配。固定 105 张 Val 图像的即时预处理与冻结 PNG 缓存张量逐元素一致。两个已知排序差异样本在原 batch=32 路径下可复现冻结成员概率。部署 FP32 路径的 matmul TF32 关闭、cuDNN TF32 开启、cuDNN benchmark 关闭、确定性算法开启；没有依据 Val 分数尝试不同数值路径。差异由 batch=32 与 batch=1 的 GPU 运算形状变化解释，未发现实质性预处理或融合公式错误。审计证据保存在本地 `outputs/f01_deployment_v1/audit.json`。

## 完整 Val 重验证

两次独立 `predict.py` batch=1 FP32 推理均覆盖官方完整 5,849 张 Val。两次 CSV SHA256 同为 `631e56175da59dbab04fecaadde76a3b50a055650bf5931cbef1293c3895a05f`；逐样本概率最大差、Pred_Label 变化及全部样本对排序变化均为 **0**。冻结部署版 `validation.csv` SHA256 为 `5d89c8db46568de9220f56817d4ed4417bb1857b9e9ed843dbabf575b5149ab2`。

| 指标 | 历史 batch=32 | 部署 batch=1 | 差值 |
|---|---:|---:|---:|
| AUROC | 0.986581571939 | **0.986580646594** | −0.000000925345 |
| AUPRC（项目既有 PR 曲线积分定义） | 0.823281300295 | **0.823303134326** | +0.000021834032 |

两种 batch 路径有 11 个正负样本对排序变化，其中 5 对改善、6 对变差；净损失 1 个正确排序对。在 1,080,678 个正负样本对上，AUROC 改变量恰为 `−1/1,080,678 = −0.000000925345`。此外有 3,763 对同类样本排序变化；这些不进入 AUROC 分母。没有根据这些结果更改模型、融合权重、阈值、精度或推理算法。`metrics.json` 与逐样本预测在本机 `outputs/f01_deployment_v1/`。

## 正式 Test CSV 与一致性

最终文件：`F:\glaucoma-bdc-2026\submission\round_02_F01\submission.csv`；SHA256 **`47467d62a5df27ae335e430eac9915f5cc45b09fc119838f9ea4954955de6308`**。CSV 为 UTF-8、**5,855 行**，仅含按序 `Image_name,Probability,Pred_Label`；文件名与官方 Test manifest 完全同序、无重复，概率有限且在 `[0,1]`，标签恰为 `Probability >= 0.5`。最终文件与 `predict.py` 暂存输出字节一致。

`cost.py --smoke` 对最终 CSV 的固定 100 张 Test 图像连续三轮复算，逐样本概率、阈值标签和完整排序均 **0 个不匹配**。该检查与正式 `cost.py` 调用同一 `Predictor.predict_one`。本地 RTX 4060 Laptop 的端到端均值为 130.839 ms/图，属于本地诊断；硬件和 Python 版本（本机 3.11.9）不同于官方 T4/Python 3.12。**T4 延迟验收仍为 UNVERIFIED，不能标记为满足 100 ms 限制。**

Test 只用于最终冻结方案前向与固定路径复算；未使用 Test 标签、分布统计、伪标签、训练或参数选择。模型实际加载文件合计 127,626,524 字节，低于 500 MB。比赛官网尚未上传。提交前仍需在指定 T4 16GB、Python 3.12、PyTorch 2.10、torchvision 0.25、CUDA 12.8 环境按官方 5 张预热、100 张计时、batch=1、连续 3 轮执行性能验收，且须再次核对概率和排序。

状态：**CSV_VALIDATED_T4_UNVERIFIED**。代码、摘要报告可进入 Git；原始比赛数据、Test 预测 CSV、权重与逐样本诊断保留在被 `.gitignore` 排除的本机目录。
