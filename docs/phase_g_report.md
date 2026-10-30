# Phase G：冻结融合候选与部署预检

日期：2026-10-30。基线代码：`549e8396f9a5cede9dd361d661d5979aa0436dfb`。本阶段没有训练、读取 Test、生成提交或覆盖 Phase F 及更早冻结产物。

## 1. 冻结证据

`configs/phase_g.yaml` 固定 F01 = G01+B02、F04 = G01+G02，均为各 0.5 的**概率**平均，单视图，阈值 0.5；没有搜索融合权重。`outputs/phase_g/freeze/` 存放两份完整 5,849 行 Val 预测的只新增副本、融合定义、原始文件 SHA256 及固定的 105 张 Val 测速名单。`outputs/phase_g/f04_vs_f01.json` 和 `comparison.csv` 是可重算的比较结果。生成时重新从成员原始预测合成 F01/F04，逐行与 Phase F 冻结预测核对；原始 checkpoint、预测、官方 Val manifest、许可证和模型卡均逐文件核验 SHA256。患者 ID 缺失，以下 Bootstrap 以图像为单位，不能推断患者独立性。

| 指标 | F01 | F04 | F04 − F01 |
|---|---:|---:|---:|
| AUROC | 0.986581572 | 0.987076632 | +0.000495060 |
| AUPRC（项目原有 PR 曲线梯形积分定义） | 0.823281300 | 0.826678390 | +0.003397090 |

配对分层图像 Bootstrap：2,000 次，seed 42，AUROC 差 95% CI **[-0.002493111, +0.004044359]**。共 1,080,678 个正负样本对；F01 严格误排序 14,501 对，F04 误排序 13,966 对。F04 修正 F01 的 5,010 对，同时新增 4,475 对误排序，净减少 535 对。CI 跨零，且两者都是反复观察同一个 Val 后的探索性比较；这不是稳定泛化收益证据。F04 值得保留为研究候选，现阶段不取代 F01 的首选部署顺序。

## 2. FP32 部署状态

| 候选 | 实际加载文件合计 | 冻结来源 | 本地 4060 参考 | T4 状态 |
|---|---:|---|---:|---|
| F01 | 127,626,524 B（127.63 MB） | G01 + B02 | Phase F 交错测速 83.26 ms | **UNVERIFIED** |
| B02 后备 | 16,295,141 B（16.30 MB） | B02 | Phase F 交错测速 55.29 ms | **UNVERIFIED** |
| G01 后备 | 111,331,231 B（111.33 MB） | G01 | Phase F 交错测速 60.99 ms | **UNVERIFIED** |

三个候选均低于 500,000,000 B。Phase F 的 4060 时间不能代替 T4。旧 F01 bundle 含本机绝对路径；Phase G 新包 `outputs/phase_g/candidates/{F01,B02,G01}/` 使用相对路径且包含逐文件 SHA256、模型配置哈希、原始预训练来源、许可证及模型卡副本。新包已在本机用 `src.inference.Predictor` 加载并做单图 FP32 推理。`predict.py` 与 `cost.py` 均调用 `Predictor.predict_one`。`scripts/phase_g_t4_val.py` 也调用该函数，包含磁盘读取、解码、预处理、数据传输、模型前向、融合、后处理、CSV 写入及 flush，并同步 CUDA 计时。脚本锁定 Val 的 5 张预热图和另外 100 张正式计时图，batch=1、连续 3 轮；逐样本比对冻结 Val 概率（误差 ≤0.001）、阈值标签和全部 100 张之间的概率排序。它要求实际 16GB NVIDIA T4 与 Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8，并核对加载文件哈希。当前机器是 RTX 4060，脚本按预期拒绝运行，**没有 T4 PASS**。

部署默认 FP32。Phase F 的 FP16 对照出现概率误差及排序差异，尤其 F01 有 2 张样本超出 0.001、7 个样本对排序变化；没有对 FP16 给出验收通过结论。冻结 Val 的早期批量预测与 batch=1 推理也可能有微小数值差异，T4 脚本会明示并阻止不一致的 PASS，不能跳过该检查。

## 3. 预训练权重与合规门禁

| 成员 | 非医学上游 | 原始权重 SHA256 | 许可与材料 |
|---|---|---|---|
| G01 / G02 | `timm/convnext_tiny.fb_in1k`，ImageNet-1K | `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276` | 模型卡声明 Apache-2.0；保留完整许可证与上游 MIT 通知 |
| B02 | `timm/efficientnet_b0.ra_in1k`，ImageNet-1K | `d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd` | 模型卡声明 Apache-2.0；保留完整许可证 |

公开模型卡：[ConvNeXt-Tiny](https://huggingface.co/timm/convnext_tiny.fb_in1k)、[EfficientNet-B0](https://huggingface.co/timm/efficientnet_b0.ra_in1k)；[Apache-2.0 全文](https://www.apache.org/licenses/LICENSE-2.0)。固定 revision 下载链接、权重哈希、许可证和模型卡哈希见 `outputs/phase_f/analysis/model_evidence.json` 与各候选 `candidate_manifest.json`。Phase G 重新对本地原始文件执行 `verified_weights` 核验，并区分模型权重许可证与代码许可证。未使用医学 checkpoint。正式提交材料仍须按主办方要求保留归属、许可证和修改说明，并由主办方完成最终规则认定。

## 4. 结论与 T4 命令

**F01 尚未满足全部部署验收条件**：文件大小与现有来源审计通过，T4 延迟和冻结预测一致性仍未验证。**F04 保留为探索性候选**：Val 数值略高，但 CI 跨零，且未做 T4 验收。**当前没有可以宣称完成正式 T4 合规验收的候选**；F01 优先，B02/G01 为单模型后备。正式 Test 预测和主办方脚本核验须待最终方案冻结之后进行。

在已获合法官方 Val 数据的 T4 机器、相同代码版本与完整 `outputs/phase_g` 候选包上，依次执行：

```bash
python -m scripts.phase_g_t4_val --candidate outputs/phase_g/candidates/F01 --output outputs/phase_g/t4_F01
python -m scripts.phase_g_t4_val --candidate outputs/phase_g/candidates/B02 --output outputs/phase_g/t4_B02
python -m scripts.phase_g_t4_val --candidate outputs/phase_g/candidates/G01 --output outputs/phase_g/t4_G01
```

每次输出 `report.json`、5 张预热和 3 轮逐样本 CSV。若硬件、版本、文件哈希、概率/标签/排序一致性或平均延迟失败，状态为 FAIL 或直接拒绝；不把预检冒充主办方最终认证。当前本地完整测试：91 passed；未启动训练。
