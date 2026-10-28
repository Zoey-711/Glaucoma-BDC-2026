# 第一阶段合规报告

核验日期：2026-10-28（Asia/Shanghai）。已接入并审计用户确认的完整官方数据包，解压到 data/official。阶段二已授权启动B02，使用经独立模型卡许可证核验的timm EfficientNet-B0 ImageNet权重；证据与SHA256见 docs/experiments.md 和 configs/weights_b02.local.json。其他正式训练未启动，尚未获得T4性能认证。

## 规则依据

- [官方赛题页](https://www.ncccu.org.cn/index/Paper/case2.html?way=1)
- [比赛参赛规则，官网 20261023 附件](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)
- [2026-10-27 补充说明](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)

以规则及正式补充解释为准。医学预训练（含医学教师间接蒸馏）及外部医学数据禁止；测试数据不得训练、伪标签、拟合统计量或选择模型。官方训练集自监督及合规师生蒸馏允许，但须完整申报，本阶段未实现或使用。工程选择将验证集仅用于评估及模型选择。

推理应冻结参数和归一化统计，关闭随机增强。统一接口已落实。T4 batch=1 串行、5 图预热、100 图连续 3 轮，从读取至结果写出计时，先逐图取三轮均值再总平均；须对照提交概率、标签及排序，误差口径见正式补充说明。模型权重与必要配套文件合计不超过 500MB。最终以官方复测为准。

## 第一阶段旧权重候选审核记录（未启用）

候选为 TorchVision EfficientNet-B0 ImageNet-1K V1：

- [版本固定的官方模型说明](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.efficientnet_b0.html)
- [候选公开权重](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth)
- [TorchVision 0.25 源码许可证 BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE)
- [官方预训练模型条款说明](https://docs.pytorch.org/vision/0.25/models.html)

TorchVision 源码许可不能直接视为预训练权重及其数据条款的完整许可。官方文档要求使用者核查模型自身许可。本阶段不能据此确认赛事要求的再分发权利，因此 template 保持 pending_review，未提供的权重文件 SHA256、许可证全文及授权确认均为空。

使用前须人工核实：公开下载无 gated、非 NC、许可涵盖赛事和权重再分发、主预训练数据非医学，保留来源和上游训练说明。将许可证全文保存在本地，计算权重和许可证 SHA256 后填入 manifest；补齐审核人和日期后才能设 approved。程序检查声明和本地文件哈希，不自动作法律判断；声明必须真实。随机初始化只允许合成数据 smoke，不能冒充 ImageNet 预训练实验。

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath <已核验权重真实路径>
Get-FileHash -Algorithm SHA256 -LiteralPath <许可证全文真实路径>
```

## 最终材料与风险

`python -m scripts.compliance` 导出 JSON 报告及“模型核验材料与数据声明清单.xlsx”，未提供字段如实显示未提供，不能把模板当作最终已审核证明。

提交前须提供实际权重、许可证全文、来源及上游数据说明、全部 SHA256；最终训练配置和数据声明；环境锁定；训练记录；最终 submission；推理精度；T4 cost 报告；实际加载文件大小清单。若后续加入教师/自监督，须追加权重来源、哈希、许可证、数据范围、脚本和关系材料。

当前B02改用模型卡明确声明Apache-2.0的 `timm/efficientnet_b0.ra_in1k`，其来源、上游训练说明、许可证全文和权重哈希已留存。上述旧TorchVision模板不作为B02合规材料，其他骨干仍需独立核验。

尚存风险：官方标注未提供患者 ID，不能排除患者级泄漏；精确哈希无法覆盖近重复；其余候选权重仍待审核；B02性能以实际训练报告为准；本机RTX 4060/Python3.11与官方Python3.12/T4环境不同，100ms约束尚未获官方协议实测证明。数据包由用户提供并确认完整，原始下载凭据仍需留存。任何最终优化仅凭训练/验证集决定，不据测试图像或排行榜反馈调参。
