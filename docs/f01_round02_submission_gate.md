# F01 Round 02：正式提交生成门禁

状态：**BLOCKED**。`submission/round_02_F01/submission.csv` 未生成；未运行正式 Test 图像推理，未上传比赛官网。此次门禁仅使用官方 Val 图像与冻结预测。Test 清单只核对过表头及前两项文件名，没有读取 Test 图像或标签，也没有用于模型选择。

## 冻结方案核查

F01 定义为 `0.5 × sigmoid(G01_logit) + 0.5 × sigmoid(B02_logit)`，以此顺序 FP32 推理，TTA 仅 identity，固定阈值 0.5。冻结 Val 文件逐行满足该概率融合公式，最大计算差 `1.11×10⁻¹⁶`。成员配置均为 224×224、确定性 bilinear 缩放、`ToTensor`、ImageNet 均值和标准差；模型处于 eval 模式，参数不参与梯度计算，预测函数由 `torch.inference_mode()` 包裹。G01/B02 原始权重 SHA256 与 Phase F 证据一致：

| 文件 | SHA256 |
|---|---|
| F01 `inference.json` | `8d957a1b1f8b637640836966d9b91f801921b712ebdcaaeeeb42f9e0a7475cfc` |
| G01 checkpoint | `9220f2afcdf338b24c77755407797dbe606d6a55c06b2a244d2976a73c3c0452` |
| B02 checkpoint | `66761ca674e84d878fb530481089dd3bd510037ec4f791425347c7306e9a5182` |

三份实际加载文件合计 **127,626,524 B**。冻结 Val 预测 SHA256 为 `eb4aff71490e67b3df68f81d6394640113d5738582192813f36227cce51015a8`。

## 一致性诊断

| 项目 | 结果 |
|---|---:|
| 官方 Val 图像 | 5,849 |
| 冻结 F01 AUROC | 0.986581571939 |
| 固定 batch=1 FP32 路径 AUROC | 0.986580646594 |
| 最大概率绝对差 | 0.000742942 |
| Pred_Label 变化 | 0 |
| 全部样本对排序变化 | **3,774** |
| 其中正负样本对 | **11** |
| 其中同类样本对 | **3,763** |

在 100 张固定 Val 诊断图像上，`predict.py` 与冻结 F01 参考有 1 对排序变化：`valid_04165.jpg` 与 `valid_04956.jpg`，两者都是阴性。`cost.py --smoke` 使用相同 bundle、FP32 和 `Predictor.predict_one`，连续三轮与 `predict.py` 结果的差异数为 **0、0、0**。两个脚本自身的加载、预处理、融合和 CSV 写出路径一致。本地 RTX 4060 的 `cost.py` 完整端到端均值为 **124.773 ms/图**；这是本地诊断，T4 状态仍为 **UNVERIFIED**。

两张已知图像从原始 JPEG 即时缩放与训练时冻结 PNG 缓存读取所得张量逐元素相同；用冻结验证的 batch=32 路径重新前向，G01/B02 概率分别与冻结成员预测相差低于 `1×10⁻¹⁶`。这强烈支持 batch=32 与部署 batch=1 的 GPU 数值路径差异，不能通过调整权重、阈值或单张概率修正。即使跨类别排序变化只有 11 对，用户规定存在任何概率排序变化就不得生成正式 CSV，故门禁阻止 Test 推理。

本机证据：`outputs/f01_submission_gate/gate_report.json`、`predict_val_100.csv`、`cost_val_smoke/report.json`。Round 02 目录只含 `manifest.json`、`validation_report.md` 和 `checksums.sha256`，其状态明确为 `BLOCKED`，无 `submission.csv`。代码和本报告可提交 Git；官方图像、诊断预测和模型权重保持在忽略目录。

后续需要先制定并冻结适用于 batch=1 的验证参考或取得主办方对数值容差及排序差异的明确认可，然后重新运行门禁。不得覆盖原冻结 Val 预测或借 Test 结果决定处理方式。
