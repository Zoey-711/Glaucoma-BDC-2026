# 官方数据接入

用户已提供数据包 `F:\计挑大数据\glaucoma_first.zip`（33,243,465,372 bytes）。接入脚本保留原包，将内容解压到项目 `data/official/`；来源由用户指定为官方数据，原始下载凭据仍须自行留存。

原始结构为 `glaucoma_first/train/`、`valid/`、`test/` 及对应 CSV。训练 51,396 行、验证 5,849 行、测试 5,855 行；训练/验证列为 `Image_name,Age,True_Label`，测试列为 `Image_name,Age`。没有患者 ID；Age 不作患者身份，也不用于当前纯图像 baseline。

`scripts.prepare_official` 将 Image_name 映射为 image_id，True_Label 映射为 label，保持原始行顺序及划分。原始 CSV 的 `.jpg` 与实际 ZIP 图片的 `.JPG` 大小写不一致，接入时唯一映射到实际路径，提交 ID 仍保留 CSV 原值；无法唯一映射则报错。标准清单位于 `data/official/manifests/`，本地配置 `configs/official.local.yaml`，接入状态及 SHA256 见 `outputs/official_ingest/ingest.json`，完整训练/验证审计见同目录 `audit.json`。这些文件仅在对应阶段完成后生成；请以实际报告状态为准，`audit_blocked` 时禁止训练。

复用接入脚本命令（当前目标已存在时会拒绝覆盖）：

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_official --archive 'F:\计挑大数据\glaucoma_first.zip'
```

smoke 图像仅位于 outputs/smoke*/synthetic，是随机像素合成图，与官方数据隔离。当前未启动正式训练。

## 2026-10-28 实际审计结果

已完成解压、清单生成及全量训练/验证图像审计，状态为 completed，audit_passed=true。

| 划分 | 图像数 | 阴性 | 阳性 | 阳性比例 | 缺失/损坏 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 51,396 | 49,739 | 1,657 | 3.224% | 0 / 0 |
| valid | 5,849 | 5,658 | 191 | 3.266% | 0 / 0 |
| test | 5,855 | 未统计 | 未统计 | 未统计 | 仅核对 CSV 与压缩包文件映射，不解码审计 |

训练集 49 种尺寸，宽 1024–6000、高 682–4000；验证集 40 种尺寸，宽 320–6000、高 212–4000。宽高范围分别统计，不代表存在任意组合。训练与验证合并检查后，解码像素完全相同的重复组为 0，跨集精确重复组为 0。没有患者 ID，无法排除患者级泄漏；未进行近重复检查。

源包 SHA256：`9f55b6c8f5a893ed0bc77af98e3a3efbfd2ae98fd0d1dd3151d66c94f2c5f910`。

实际报告：`outputs/official_ingest/summary.json`（汇总）、`audit.json`（完整尺寸等统计）、`ingest.json`（来源及清单哈希）。合规清单附带实际数据接入信息，位于 `outputs/compliance_data/`，权重许可证审核仍为 pending_review。

正式训练尚未执行。完成预训练来源和许可证审核后，在 `configs/official.local.yaml` 填写 weights_manifest，再使用该配置训练。当前 weights_manifest=null 会阻止正式随机初始化训练。

训练/验证 CSV 必须为 UTF-8：

```csv
image_id,path,label,patient_id
```

这里只列字段，没有虚构任何样本或官方路径。`image_id` 应原样保留官方图像文件名；`path` 为相对于明确指定 root 的真实图像路径；label 为 0/1；patient_id 可省略。禁止从文件名擅自推断患者关系。ID 不得重复/为空，路径不得逃出 root。图像必须可以解码。

测试清单仅需 `image_id,path`，严格按官方 sample submission 行顺序构造，不读取或推断测试标签。`scripts.validate_submission` 使用官方 sample 的 Image_name 完整核对数量、集合及顺序。

官方训练和验证集固定分开。若审计发现跨集重复或患者交集，训练直接阻断；记录问题并向官方确认或建立独立的患者划分验证方案，不静默更改官方标签/划分。患者 ID 缺失时只能声明未能排除患者泄漏。像素哈希只查精确重复，不保证发现近重复。

数据审计仅处理训练与验证，不对测试集统计类别先验或归一化参数；测速严格使用赛方另行指定的图像。当前工程不实现验证集并入训练、自监督、蒸馏、伪标签或测试集调参。

官方数据不得上传公共代码库、第三方工具或分发。数据来源、获取时间、原始压缩包 SHA256、官方授权和标注映射说明应另行留存于受控本地材料，不能用 smoke 记录替代。
