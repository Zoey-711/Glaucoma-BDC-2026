# Phase F：冻结融合分析

读取五个模型完整官方 Val 5,849 行（191 阳性、5,658 阴性），校验 ID、顺序、标签、有限 logits、概率范围、sigmoid 对应关系、checkpoint SHA256、配置和冻结证据。G02 使用研究配置历史 hash，其余旧模型无统一 canonical config hash 时，本轮计算并标明为本轮记录；同时验证 checkpoint 内嵌配置与保存配置完全一致，不能把新 hash 冒充历史证据。详见 `outputs/phase_f/analysis/model_evidence.json`。

公式在 `configs/phase_f.yaml` 中固定并在计算前锁定。四组概率融合均为两个概率各 0.5；F01_LOGIT 为两个原始 logit 各 0.5 后 sigmoid。没有拟合权重、阈值或读取 Test。

| experiment_id | members | auroc | auprc | delta_auroc | ci95_low | ci95_high |
|---|---|---|---|---|---|---|
| F01 | G01+B02 | 0.98658157 | 0.82328130 | 0.00301940 | 0.00034691 | 0.00612273 |
| F02 | G01+L03b | 0.98565530 | 0.80540501 | 0.00209313 | -0.00040914 | 0.00490444 |
| F03 | G01+R01b | 0.98506493 | 0.82771440 | 0.00150276 | 0.00000072 | 0.00327396 |
| F04 | G01+G02 | 0.98707663 | 0.82667839 | 0.00351446 | 0.00122597 | 0.00647492 |
| F01_LOGIT | G01+B02 | 0.98658250 | 0.81696202 | 0.00302033 | 0.00088359 | 0.00548093 |

G01 基线 AUROC 0.98356217、AUPRC 0.80461532。Bootstrap 固定 seed 42、类别分层图像配对重采样 2,000 次。以上为 **5 项探索性比较的未校正 95% CI**，不能据此宣称跨种子稳定收益或测试集泛化保证。F03 下界只有约 0.00000072，接近零且受 Monte Carlo 精度、多重比较和重复使用 Val 影响，应视为非常脆弱的证据。患者 ID 缺失，不能称患者级独立区间。

## 互补性

下面“reference wrong / candidate correct”统计 G01 把阳性排到阴性以下、另一模型排对的样本对；反向列是另一模型新增的错误。所有正负对共 1,080,678 个，不能当作 1,080,678 个独立统计样本。严格错误定义为 p(阳性)<p(阴性)，平分 ties 计入 AUROC 损失；阈值错误使用固定 0.5。

| candidate | pearson_probability | spearman_probability | reference_wrong_candidate_correct | reference_correct_candidate_wrong | threshold_both_errors |
|---|---|---|---|---|---|
| B02 | 0.883374 | 0.660965 | 9755 | 10298 | 62 |
| G02 | 0.814669 | 0.600168 | 11172 | 11875 | 57 |
| L03b | 0.91832 | 0.744735 | 7696 | 7843 | 80 |
| R01b | 0.865172 | 0.728192 | 8737 | 8880 | 66 |

G01/B02 各有 17,764 / 18,307 个严格误排序对，共同错误 8,009 个；B02 修复 G01 的 9,755 对，同时新增 10,298 对。F01 的总误排序降为 14,501 对。G01 的阈值错误为 99 张，B02 为 100 张，共同错误 62 张。F01 Recall 0.685864 低于 G01 0.712042，而 AUROC 提高，因此本次融合收益体现为整体排序变化，不能用 Recall 上升解释。

F04 的 Val AUROC 最高，但包含 224 与 384 两个 ConvNeXt，部署成本需单独实测；本轮重点计时的是 G01、B02 和 F01。F01 概率与 logit 融合的 AUROC 相差不足 0.000001，概率融合 AUPRC 更高；不继续搜索混合权重。

完整结果：`outputs/phase_f/fusion_comparison.csv`；逐组完整 Val 预测、相关矩阵和误排序/阈值互补性在 `outputs/phase_f/analysis/`；区间图 `outputs/phase_f/fusion_delta_auroc.png`。所有分数来自冻结预测，不是新 FP16 部署验证成绩。
