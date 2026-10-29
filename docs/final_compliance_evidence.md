# 正式赛规合规证据表

审计范围：本地工作区、Git 当前可见历史、用户提供的两份 PDF、公开模型卡；无 T4 或朋友服务器访问。证据快照：`outputs/final_compliance_audit/local_evidence.json`。本表状态只使用 `PASS`、`FAIL`、`UNVERIFIED`、`NEEDS_ORGANIZER_CONFIRMATION`。`PASS` 表示所列局部检查成立，不等于主办方最终认可。

| 项目 | 状态 | 证据文件与代码位置 | 缺口及修复建议 |
|---|---|---|---|
| 规则版本与哈希 | PASS | 两份用户 PDF；`outputs/final_compliance_audit/local_evidence.json` | 主规则 SHA256 `ff6aa3ad...13fb6`；补充说明 SHA256 `836c4f5d...7a44`。保留原件随材料归档。 |
| B02 预训练来源、非医学主数据集、许可证全文与哈希 | PASS | `configs/weights_b02.local.json`；`data/pretrained/efficientnet_b0.ra_in1k/{model.safetensors,MODEL_CARD.md,LICENSE-2.0.txt}`；`src/model.py:12` | 当前使用的精确权重为公开 ImageNet-1K、Apache-2.0 模型卡声明，三文件 SHA256 与登记匹配；最终包需携带许可证全文和来源说明。 |
| G01/G02/M02 图像骨干预训练证据 | PASS | `configs/weights_g01.local.json`；`data/pretrained/convnext_tiny.fb_in1k/{model.safetensors,MODEL_CARD.md,LICENSE-2.0.txt,UPSTREAM_LICENSE.txt}`；`src/model.py:12` | 当前使用的精确权重为公开 ImageNet-1K、Apache-2.0 模型卡声明，上游 ConvNeXt 代码为 MIT；各文件哈希匹配。保留两份许可证及通知。 |
| 区分代码与权重许可证 | PASS | 上述模型卡、许可证全文；本机 `timm` 包元数据 `License: Apache-2.0`；`docs/final_compliance_audit.md` | `timm` 代码许可证不能单独证明权重权利；本项以具体模型卡的权重许可证声明为依据。 |
| G05/G06/G07/G08/G09 新骨干权重 | UNVERIFIED | `configs/research/G05.yaml` 等；`docs/research_status.md` | 来源、主数据集、许可证全文及文件 SHA256 未逐权重核验；保持不运行。S04 局部分支也需独立查清实际权重。 |
| D01 教师与 R04 自监督材料 | UNVERIFIED | `src/research/distill.py`、`src/research/ssl.py`；`configs/research/D01.yaml`、`R04*.yaml` | 只有代码与阻塞配置，未有正式教师/自监督权重或训练记录；若启用，追加每个初始化权重、教师关系、Train 来源和 SHA256。 |
| B02/G01/G02 图像与标签范围 | PASS | `src/experiment_training.py:19`、`src/research/data.py:74`；训练配置和结果 manifest | 本地训练入口读取官方 Train/Val；G02 使用固定官方清单；Test 不在本地训练读取路径。服务器历史执行未在本项覆盖。 |
| 外部医学数据或 Test 参与训练 | UNVERIFIED | 本地代码搜索、`outputs/M02/run_manifest.json`、`COLLABORATION_RULES.md` | 本地运行记录未显示外部医学数据/Test 训练；无法凭本机代码证明朋友服务器或未留存的所有运行。交付其数据来源和进程日志后复核。 |
| 年龄填补、标准化与异常界限 | PASS | `src/multimodal.py:18`、`scripts/run_m02.py:61`；`outputs/M02/age_preprocessing.yaml` | 51396 行 Train 拟合中位数 57、均值 56.738754、标准差 10.185559、有效年龄 17–96；Val/Test 只应用固定规则。内部选型另用 Train 内部拟合。 |
| 图像归一化和随机增强隔离 | PASS | `src/data.py:57`、`src/research/preprocess.py:61`、`src/inference.py:84` | 使用固定 ImageNet 常数；验证/推理不启用随机增强。S01 候选裁切 QC 另有失败记录，见下。 |
| Val 并入训练及提前冻结参数 | UNVERIFIED | `src/experiment_training.py:19`、`src/research/trainer.py:60`、`outputs/M02/training/selection_locked.json` | 已核查本地 B02/G01/G02/M02 为 Train 训练、Val 选择/比较；未见并入后继续选型的本地运行。后续若合并 Val，必须事前锁定早停、checkpoint 和超参，提供时间戳。 |
| 患者独立性与跨集关联 | UNVERIFIED | `docs/phase2_audit_report.md`；`outputs/phase2_diagnostics_v3/` | 无患者 ID；1 对疑似同眼不同拍摄，429 对 pHash 候选未确定。不得声称独立患者验证或清洗后成绩。 |
| S01 多 GPU 队列 QC 门禁 | FAIL | `configs/research/S01.yaml:39`、`configs/research/after_local_g02.yaml:9`、`src/research/preprocess.py:9`；`outputs/S01_qc/visual_review.json` | 旧队列仍为 `PENDING`、无 QC 依赖，使用未通过本轮极暗图检查的旧裁切。立即从执行矩阵移除或标记阻塞，修正算法后重跑 QC。 |
| B02/G01/G02 模型文件单项大小 | PASS | `outputs/runs/{B02,G01}/best.pt`、`results/G02/seed_42/fold_none/best.pth`；证据 JSON | 分别 16.295141、111.331231、111.329503 MB；G02 的 `last.pth` 属训练恢复文件，不参与当前单模型推理。最终组合仍须按实际加载文件复算。 |
| M02b/M02c/M02d 文件单项大小 | PASS | `results/M02{b,c,d}/seed_42/fold_none/`；证据 JSON | 逐文件低于 500 MB。M02c/d 的图像骨干嵌在 `best.pth`；若最终选用年龄分支，须同时核实官方 Age 输入和部署路径。 |
| 最终部署包解压后总大小 | UNVERIFIED | `src/inference.py:106`、`outputs/final_compliance_audit/local_evidence.json` | 尚无最终包。必须列出实际加载权重、配置、归一化文件及引擎，去重后合计 ≤500,000,000 字节。 |
| `predict.py` 与 `cost.py` 统一推理及确定性 | PASS | `predict.py:65`、`cost.py:75`、`src/inference.py:91`、`src/inference.py:124` | 两脚本使用 `Predictor.predict_one()`；加载后 `eval()` 并冻结参数。M02b/c/d 有独立 `M02Predictor`，当前未接入这两个正式 CLI。 |
| M02 作为最终候选的正式统一接口 | FAIL | `src/multimodal.py:80`、`predict.py:67`、`cost.py:74` | 当前 CLI 只认识通用 `Predictor` checkpoint，不能直接加载 M02 的年龄模型；若选择 M02，须先接入并通过完整 Test CSV/T4 一致性。 |
| FP32/FP16 预测一致性 | UNVERIFIED | `cost.py:46`、`src/inference.py:122`、`outputs/official_2026_audit/inference_check.json` | 本地 B02/G01/G02 只有 FP32 单图重放证据；B02 与保存 Val 概率有 `2.53e-5` 差异。最终精度和逐样本排序需实测。 |
| T4 100 测时图、5 预热图、batch1、3 轮 | UNVERIFIED | `cost.py:49`、`scripts/accept_t4.py`；本次 preflight 输出 | 协议脚本已实现并限制 T4/Python3.12/torch2.10/torchvision0.25/CUDA12.8；未得到真实 T4 及主办方样本，不能记 PASS。 |
| Round 01 CSV 格式、顺序、概率和标签 | PASS | `submission/round_01_B02/submission.csv`、同名 metadata；`src/submission.py:10`；证据 JSON | 5855 行，UTF-8 无 BOM，列顺序 `Image_name,Probability,Pred_Label`；严格匹配原始 Test 顺序，概率均在 [0,1]，标签符合固定阈值。 |
| Round 01 本地 checkpoint/CSV 哈希对应 | PASS | `submission/round_01_B02/submission.csv.metadata.json`；`outputs/runs/B02/best.pt` | 元数据中的模型和 CSV SHA256 均匹配现存文件；这是本地文件链，不是官方上传记录。 |
| 历史最高分 CSV 与官方可复现最终模型 | UNVERIFIED | Round 01 metadata `uploaded:false`；本地未找到平台成绩/最高分回执或最终材料包 | 需官方平台最高分提交记录、精确 CSV SHA256、模型哈希及复测日志互相对照；不能将本地 Round 01 自动认定为最高分。 |
| 旧版合规 XLSX 模板与实际 B02 来源一致 | FAIL | `outputs/compliance/report.json`、同目录 `模型核验材料与数据声明清单.xlsx`；`configs/weights_b02.local.json` | 旧模板仍写 torchvision rwightman 预训练链接、许可证/哈希为空、`formal_training_started:false`，与实际 timm B02 不符。禁止直接递交；用真实最终模型重新生成并逐项核验。 |
| Git 当前和可见历史敏感文件路径扫描 | PASS | `outputs/final_compliance_audit/local_evidence.json`；`.gitignore`；`git ls-files` 与 `git rev-list --objects --all` | 226 个当前跟踪文件；在本地可见历史中未见官方数据、权重、提交 CSV、密钥文件路径；常见私钥/token 文本标志扫描为 0。仍须检查远端所有分支与平台保密扫描。 |
| 朋友服务器身份、官方数据访问和第三方合作 | NEEDS_ORGANIZER_CONFIRMATION | `COLLABORATION_RULES.md:3`、`FRIEND_TRAINING_GUIDE.md:5` | 无主办方书面依据或朋友正式队伍身份/数据访问证明；确认前只允许合成测试，不向服务器传输官方数据。 |
| 最终 code/model/pretrain_model/licenses/cost.py/核验表/最高分 CSV 包 | FAIL | 主规则附录 A；本地 `cost.py`、`requirements*.txt`、Round 01 CSV、权重许可证；`outputs/compliance/` 为旧模板；证据 JSON | 尚无最终 ZIP、打包后的 `model/`/`pretrain_model/`/`licenses/`、经真实模型更新的中文数据声明表及 T4 报告。按本报告清单组装后再提交。 |
