# 第一阶段验证记录

日期：2026-10-28。环境：Windows、Python 3.11.9、PyTorch 2.10.0+cpu / torchvision 0.25.0+cpu。完整依赖见 requirements-smoke.lock.txt。

执行 `python -m pytest -q`：13 passed（5.68s）。覆盖数据审计、跨集精确重复、患者交集、缺失/损坏文件、患者信息缺失、路径逃逸、标签合法性、指标、提交概率/标签/顺序、测速排序一致性、权重审核阻断、权重哈希篡改、XLSX XML 格式，以及推理确定性/参数和 BN 统计冻结。

执行 `python -m scripts.smoke --output outputs/smoke_final`：通过。使用随机像素图，4 张训练、4 张验证、2 张测试、1 张预热；CPU、64×64、batch=2 训练 1 epoch，推理 batch=1 连续 3 轮。完成数据审计、训练、验证、最优 checkpoint、预测、提交校验及共用接口测速。

训练 smoke 的 AUROC=0.5、AUPRC=0.75；这只是无医学意义的随机像素结果。三轮输出与提交一致性不一致数均为 0。CPU 诊断平均 13.744ms，不能证明官方 T4 达标；official_protocol=false、passed=false 表示未通过官方认证，smoke 进程本身成功。

smoke checkpoint：16,295,013 bytes（约 16.30 MB）；SHA256：`3eb48064d3add80e81c05dcda3564b376c33633c46e3b5a0f66c2afddde3bf24`。这是合成数据随机初始化实验权重，不能作为合规 ImageNet 权重材料或正式参赛模型。新运行的 checkpoint 序列化哈希可能不同，必须以实际文件计算结果为准。

产物：outputs/smoke_final/result.json、run/audit.json、run/history.json、run/artifact.json、run/code_sha256.json、submission.csv、submission.csv.metadata.json、cost/report.json。实验登记 outputs/experiments.csv。待审核的合规清单 outputs/compliance/模型核验材料与数据声明清单.xlsx。

`python -m compileall -q src scripts tests predict.py cost.py` 通过。

未执行：官方数据训练、任何医学数据下载、预训练权重下载、GPU AMP 实测、T4 100 图/5 预热图官方协议、正式提交。权重许可及真实性能仍待确认。

## 官方数据接入补充验证

用户提供 `F:\计挑大数据\glaucoma_first.zip` 后，已完成源包 SHA256、63,107 个 ZIP 条目解压及 CRC 校验、CSV 映射、全量 57,245 张训练/验证图像解码和像素哈希审计。未下载外部数据，原包保持不变；测试图片仅解压和核对文件映射，不参与图像统计或模型选择。

审计通过：缺失 0、损坏 0、精确重复组 0、跨训练/验证精确重复组 0。患者 ID 未提供，患者级检查状态 unavailable_or_incomplete，不作“无患者泄漏”的结论。详见 docs/data.md 及 outputs/official_ingest/。

新增安全 ZIP 路径检查、模式/缺失文件测试、CSV .jpg 与实际 .JPG 映射和测速 ID 映射测试。扩展测试共 20 项通过。正式训练仍未启动。
