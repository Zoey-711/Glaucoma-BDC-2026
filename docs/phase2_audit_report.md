# Phase 2 Train/Val 关联审计

本报告仅使用官方训练集、官方验证集与冻结的 B02 验证预测；原始图像、标签及划分未更改。审计于 2026-10-28 完成。逐对结果见 [`outputs/phase2_diagnostics_v3/pair_review.csv`](../outputs/phase2_diagnostics_v3/pair_review.csv)，高风险联系表见 [`high_risk_contact_table.csv`](../outputs/phase2_diagnostics_v3/high_risk_contact_table.csv)，图像联系表见 [`contact_sheets/`](../outputs/phase2_diagnostics_v3/contact_sheets)。

## 核验方法与判定

先对全部 57,245 张 Train/Val 图像计算文件 SHA256、解码图像哈希与感知哈希。对跨划分 pHash 完全相同的 430 对逐一复核原始尺寸、原图 SSIM、配准后 SSIM、归一化 RMSE、局部 SIFT 双向匹配、RANSAC 内点数及空间覆盖率。第二轮使用 1024 像素绿色通道 CLAHE 提取局部特征；判据预先写入 [`policy.json`](../outputs/phase2_diagnostics_v3/policy.json)。方法参考 [scikit-image SSIM 文档](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html)与 [OpenCV 特征匹配/几何验证文档](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)。

| 分类 | 对数 | 解释 |
|---|---:|---|
| 文件 SHA256 完全相同 | 0 | 未见同字节重复 |
| 解码像素完全相同 | 0 | 未见相同像素重编码 |
| 达到严格视觉近重复判据 | 0 | 未确认精确视觉重复 |
| 疑似同眼不同拍摄 | 1 | `train_46024.jpg` ↔ `valid_05425.jpg`，标签均为 0 |
| 无法确定 | 429 | pHash 一致，但没有足够局部结构证据，不能将其等同于泄漏 |

高风险对有 199 个双向局部匹配、193 个几何内点，内点比例 0.970，覆盖图像约 15.7%；配准后 SSIM 0.922，尺寸同为 3456×2304，文件及解码像素哈希不同。对应的 [局部匹配图](../outputs/phase2_diagnostics_v3/matches/pair_0402.jpg)和 [联系表](../outputs/phase2_diagnostics_v3/contact_sheets/sheet_000.jpg)显示血管结构高度一致。它支持“疑似同一眼不同拍摄”，不能证明具体患者身份或采集时间。其余 11 对 pHash 候选标签不一致，但均未达到上述局部证据阈值，**不能断定真实标签冲突**。

审计检测器做了小型能力检查：4 张由原图人工缩放/重编码/轻微旋转或亮度变化产生的已知重复均被检出，20 对随机图未被标高风险。该小样本检查不能证明对任意裁切、拍摄条件和患者身份的灵敏度。原始图未被修改。此前在较小图上肉眼怀疑的两对，在增强局部特征复核后归为“无法确定”；旧报告保留为历史筛查记录，本报告为本轮多层复核结论。

## B02 验证指标与诊断子集

完整官方验证集 5,849 张：AUROC **0.983059709**，AUPRC **0.745762044**。这一结果仍是正式验证指标，以下排除只用于诊断，不改变模型选择、早停、训练或提交数据。详细数值与固定 2,000 次 Bootstrap 见 [`b02_diagnostics.json`](../outputs/phase2_diagnostics_v3/b02_diagnostics.json)。

| 仅诊断的范围 | 排除验证图 | 阳性受影响 | 阴性受影响 | 保留集 B02 AUROC | 相对完整集变化 | 变化的图像分层 Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 仅高风险疑似同眼 | 1 | 0/191（0%） | 1/5658（0.0177%） | 0.983058565 | -0.000001144 | [-0.000005959, 0.000001114] |
| 严格精确/视觉重复 | 0 | 0/191 | 0/5658 | 0.983059709 | 0 | [0, 0] |
| 所有 pHash 候选敏感性分析 | 221 | 3/191（1.57%） | 218/5658（3.85%） | 0.982628676 | -0.000431032 | [-0.000837008, -0.000117861] |

“全部 pHash 候选”包括大量无法确定的相似眼底图，**不是泄漏清洗后的成绩**。基于候选图连通分量的保守 Bootstrap 结果也存于 JSON；连通分量不是患者分组，患者级独立性仍无法验证。排除后分数变化不能解释官方测试分差；不同集合之间不能做配对差异检验。

## G01 训练闸门与剩余风险

本轮未确认完全相同文件、完全相同解码图像或符合严格判据的视觉近重复；高风险候选无有证据支持的标签冲突。用户明确授权双轨推进，因此 G01 使用**原始完整官方训练集与验证集**开展实验，并继续独立保留疑似同眼关联记录。结果不得标为患者独立验证；若主办方提供患者/眼别与采集信息，需要再次核验。G01 不使用测试集调参，也不据此生成新的官方提交。

权重证据：G01 使用 [timm ConvNeXt-Tiny ImageNet-1K 模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k/blob/b43a6303c9fcf176d2d707478a128c2c91e93528/README.md)所声明的 Apache-2.0 通用非医学权重，实际文件 SHA256 为 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`，上游原始 [ConvNeXt MIT 许可证](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE)一并留存。所有权重由固定 revision 下载并与上游 LFS SHA256 核对；未使用旧胸片 checkpoint。
