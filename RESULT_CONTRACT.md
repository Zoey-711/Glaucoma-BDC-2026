# 结果交付契约 v1

每个训练运行目录必须是 `results/{experiment_id}/seed_{seed}/fold_{fold_or_none}/`，且至少包含：`best.pth`、`validation.csv`、`metrics.json`、`config.yaml`、`train_history.csv`、`training.log`、`run_manifest.json`、`environment.txt`、`checksums.sha256`。另保存可恢复的 `last.pth`。结果目录不可覆盖。

`validation.csv` 使用原始官方验证顺序，至少有 `Image_name,Label,logit,Probability`；保存实验ID和来源划分。`metrics.json` 的 AUROC/AUPRC/Recall/Specificity 必须能由 CSV 独立复算。`run_manifest.json` 含 schema、git_commit、配置/数据/预训练 SHA256、seed、fold、GPU、起止时间和训练状态。`checksums.sha256` 覆盖目录内除自身之外全部文件。完整度和哈希由 `scripts.validate_results` 验收，不能只交截图或分数。

```bash
python -m scripts.validate_results results/G02/seed_42/fold_none --config configs/research/G02.yaml
python -m scripts.package_results results/G02/seed_42/fold_none --output outputs/delivery/G02.tar.gz
```

本地导入在**已获准接收该结果**后执行：

```bash
python -m scripts.import_results outputs/delivery/G02.tar.gz --experiment-id G02
python -m scripts.analyze_research remote_results/G02
```

导入器验证归档路径安全、文件清单和 SHA256、checkpoint 可加载、配置/代码/数据/权重标识、验证 ID/标签/概率/logit、独立计算指标及 B02/Round01 冻结哈希；失败保留 staging 证据，不接受为正式结果。五折每折单独保存模型和验证预测，`oof_train.csv` 只覆盖相应 Train 折；合并 OOF 时拒绝重复、遗漏或标签不一致。官方验证预测与 OOF 必须分开。

F01/F02 固定等权融合产生分析工件，须保留每个成员的 checkpoint 和 SHA256；在完成统一推理、总大小/T4核验前不是最终可提交模型。无患者ID时 Bootstrap 只按图像/已审近重复组作探索性比较，不得称为患者级验证。
