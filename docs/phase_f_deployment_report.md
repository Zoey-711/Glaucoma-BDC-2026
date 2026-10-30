# Phase F：部署预检

本轮在 RTX 4060 Laptop 本机执行推理；**官方 T4 验收仍为 UNVERIFIED**。本机 Python 3.11 不是官方 Python 3.12 环境。没有生成官方提交、没有调用 Test。

## 权重来源与文件大小

五个模型的权重来源、模型卡及完整许可证文件 SHA256 再次验证通过。模型卡声明的权重许可证为 Apache-2.0，训练数据为通用 ImageNet-1K；ConvNeXt 另外保留上游 MIT notice。依据的是具体权重模型卡声明和完整许可条款，不能仅因 timm 代码是开源就推断权重许可。使用和再分发仍须保留许可证、归属说明及修改声明，最终主办方材料核验不能由本地审核代替。

- **B02**：efficientnet_b0.ra_in1k，ImageNet-1K；[固定 revision 原始下载](https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a/model.safetensors)；[权重模型卡](https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a/README.md)；权重 SHA256 `d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd`；许可证全文 `data/pretrained/efficientnet_b0.ra_in1k/LICENSE-2.0.txt`（SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。
- **G01**：convnext_tiny.fb_in1k，ImageNet-1K；[固定 revision 原始下载](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors)；[权重模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)；权重 SHA256 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`；许可证全文 `data/pretrained/convnext_tiny.fb_in1k/LICENSE-2.0.txt`（SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。
- **G02**：convnext_tiny.fb_in1k，ImageNet-1K；[固定 revision 原始下载](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors)；[权重模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)；权重 SHA256 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`；许可证全文 `data/pretrained/convnext_tiny.fb_in1k/LICENSE-2.0.txt`（SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。
- **L03b**：convnext_tiny.fb_in1k，ImageNet-1K；[固定 revision 原始下载](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors)；[权重模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)；权重 SHA256 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`；许可证全文 `data/pretrained/convnext_tiny.fb_in1k/LICENSE-2.0.txt`（SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。
- **R01b**：convnext_tiny.fb_in1k，ImageNet-1K；[固定 revision 原始下载](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors)；[权重模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)；权重 SHA256 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`；许可证全文 `data/pretrained/convnext_tiny.fb_in1k/LICENSE-2.0.txt`（SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。

单模型所需配置嵌入 checkpoint；F01 实际加载 G01、B02 checkpoint 和 bundle JSON。逐文件解压后字节数与 SHA256 见 `outputs/phase_f/deployment_run/loaded_files.json`。不把原始预训练文件重复计入推理加载大小，它属于额外合规材料。其余融合当前是离线分析，未宣称已经打包完成。

| model | loaded_bytes | model_size_mb | size_status |
|---|---|---|---|
| B02 | 16295141 | 16.2951 | PASS |
| G01 | 111331231 | 111.3312 | PASS |
| G02 | 111329503 | 111.3295 | PASS |
| L03b | 111331295 | 111.3313 | PASS |
| R01b | 111331103 | 111.3311 | PASS |
| F01 | 127626610 | 127.6266 | PASS |

## 同一推理接口与数值一致性

predict.py 与 cost.py 均使用 src.inference.Predictor.predict_one：原图读取/解码、固定 resize/ImageNet 常数、传输、eval 模型前向、概率和标签、CSV 写出与 flush。模型使用 eval、requires_grad_(False) 与 inference_mode。本轮无 TTA、无随机推理增强。保存接口源码 hash 供追溯。

固定 105 张 Train 的 FP32/FP16 对照如下；0.001 为预先锁定的概率容差，同时单独统计阈值翻转与完整两两排序差异。**局部数值通过不代表全 Val 或官方核验通过**。同精度三轮计时的稳定性见各 timing.json。

| model | max_probability_error | over_tolerance | threshold_flips | pairwise_order_disagreements | probability_threshold_consistent | strict_order_consistent |
|---|---|---|---|---|---|---|
| B02 | 0.0022789 | 2 | 0 | 7 | False | False |
| G01 | 0.0004771 | 0 | 0 | 64 | True | False |
| G02 | 0.0003399 | 0 | 0 | 52 | True | False |
| L03b | 0.00041 | 0 | 0 | 62 | True | False |
| R01b | 0.0003743 | 0 | 0 | 52 | True | False |
| F01 | 0.0011393 | 2 | 0 | 7 | False | False |

B02 与 F01 各有 2/105 张超过 0.001 的跨精度概率容差，当前跨精度检查未通过；所有模型都有部分近邻排序变化。这不等同于官方“最终 CSV 与同精度推理一致性”违规，而是说明不能直接将 FP16 替换进已有 FP32 结果。若将来选择 FP16，必须用该精度重新完成全 Val 和最终同精度验收。

另外每个模型抽查 32 张官方 Val 的 batch1 FP32 与冻结预测。B02 最初超过额外的 0.0001 检查：最大差 0.00017929。专项诊断证明原图与缓存输入张量完全相同，batch32 推理与历史预测一致（最大差约 9.2e-17），batch1 卷积计算路径发生数值变化，当前 cuDNN TF32 开启。保留失败诊断在 `outputs/phase_f/deployment/`，继续使用预先锁定的 0.001 容差，并在表中保留严格检查结果。不能把历史 batch32 分数直接声称为最终 batch1/FP16 分数；候选提交前需用实际部署精度完整复核 Val。

## 本机计时与 T4 准备

同一固定 Train 100 张、另 5 张预热、batch1、连续 3 轮，覆盖原图解码至 CSV flush，逐图 CUDA 同步。没有使用训练缓存计时，也没有清空操作系统页缓存。FP32/FP16 是两次独立测量，不是缩放推算。

| model | precision | model_size_mb | local_latency_ms | t4_status |
|---|---|---|---|---|
| B02 | fp32 | 16.2951 | 88.3309 | UNVERIFIED |
| B02 | fp16 | 16.2951 | 89.9609 | UNVERIFIED |
| G01 | fp32 | 111.3312 | 60.9066 | UNVERIFIED |
| G01 | fp16 | 111.3312 | 64.9529 | UNVERIFIED |
| F01 | fp32 | 127.6266 | 82.3379 | UNVERIFIED |
| F01 | fp16 | 127.6266 | 84.027 | UNVERIFIED |

B02 文件最小，但这份本机单独测量中反而比 G01 慢；不能用参数量直接推算端到端延迟。由于 F01 独立测时甚至低于 B02，额外按每张图轮换三模型执行顺序进行 FP32 交错核对（仍 100+5、每模型 3 轮、每次从原图解码）。核对均值：{'B02': 55.28945533333334, 'G01': 60.98530533333333, 'F01': 83.262068} ms。这项诊断用于说明执行顺序、GPU/系统状态和缓存敏感性，不替代正式连续测时协议；原始独立测量完整保留。

融合可共享一次图像解码，但仍执行两个模型前向，不能宣称融合比单模型更省计算，也不能把两项独立测速相加当实测。G01 与 F01 是否满足 100ms 的最终判断仍为 UNVERIFIED；4060 结果只说明值得带到实际 T4 做预检，没有转换比例可保证达标。

后续官方环境使用 `scripts/phase_f_t4.sh`，它调用现有 `scripts.accept_t4` 和同一 cost.py，检查 NVIDIA T4、Python3.12、PyTorch2.10.0、torchvision0.25.0、CUDA12.8，严格 100+5 张、batch1、3轮、<=100ms、<=500MB 和最终 CSV 一致性。须提供真实官方目录、获授权的最终冻结 CSV/metadata、checkpoint 及官方清单；本轮没有执行。迁移 F01 时要重写 bundle 的成员路径并重新核验其 SHA256，不能直接使用本机绝对路径。
