# 分轮提交文件

每次准备提交使用一个独立目录，例如 `round_01_B02/`、`round_02_I02/`。
这里的轮次是本地提交版本号，不代表官方赛程阶段，也不代表已经上传。

- `submission.csv`：通过格式、ID 顺序、概率范围及标签一致性校验后生成的文件。
- `submission.csv.metadata.json`：模型哈希、推理环境、精度、测试清单哈希、提交文件哈希及生成时间。
- `.csv.partial`：未完成文件，不能提交；失败后应保留排查，并改用新轮次。

首轮使用已经完成的 B02 最佳 checkpoint，固定 FP32、阈值 0.5；只做推理，不依据测试预测调整模型或参数。T4 尚未实测，通过 CSV 校验不代表通过最终部署验收。

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint outputs/runs/B02/best.pt --csv data/official/manifests/test.csv --root data/official --round round_01_B02 --device cuda --precision fp32
```

已有文件禁止覆盖。下一次明确选择新的轮次名；默认输出 `submission/round_01/submission.csv`。旧脚本仍可通过 `--output` 显式指定路径。合成 smoke 文件保留在 `outputs/smoke*/`，不放入正式提交目录。
