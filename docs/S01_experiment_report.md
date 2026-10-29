# S01 有效视野裁切实验

当前 QC 状态：**BLOCKED**。本次完成预处理审计和 Smoke Test；仅在 QC 通过后才允许正式训练。

## Train 预处理审计

官方 Train 51,396 张，未使用 Val/Test 确定参数。规则为 `safe_fov_v1`：长边不超过 512 的等比例缩略图，HSV 亮度/饱和度前景，最大连通域，6% 直径加 2 像素安全边距；小区域、异常形状、分离区域、框外可见信号回退原图。接近全图时不裁切。不保证自动掩码等同于医学分割。

- 实际裁切比例：55.654%。
- 检测回退比例：1.179%；“already_full_field”不算检测失败。
- 面积比例分位数：{'0.0': 0.6524122807017544, '0.01': 0.6878654970760234, '0.25': 0.7305194805194806, '0.5': 0.9416152263374485, '0.75': 1.0, '0.99': 1.0, '1.0': 1.0}。
- 原因计数：{'cropped': 28604, 'already_full_field': 22186, 'visible_signal_outside_box': 556, 'separate_foreground': 49, 'small_or_fragmented_field': 1}。
- 120 张对照：`outputs/S01_qc/pairs/`；12 张联系表：`outputs/S01_qc/sheet_*.jpg`。
- 全量边界/尺寸：`train_crop_statistics.csv`；异常清单：`anomaly_candidates.csv`；回退和无需裁切清单：`fallback_and_uncropped.csv`。
- 视觉复核：BLOCKED。The initial 120-image review found no obvious new truncation. Among 12 additional extreme cases, train_05402.jpg shows very dark fundus signal outside the proposed right boundary after diagnostic-only gamma=0.35 display. QC failed: the intensity threshold may omit peripheral tissue. Formal training is blocked. Visual review is not medical segmentation validation.

## 对照与实现

S01a 复用 G01，原 resize 路径的数值一致性测试通过。S01b 从 G01 实际配置派生，只改变裁切策略；224、BCE、shuffle、batch32、seed42、AdamW、Cosine、AMP、30 轮上限和 patience5 均相同。训练器源码与 G01 快照相同。使用同一已核验通用 ImageNet 初始化，禁止加载 G01 训练 checkpoint 初始化。

裁切函数同时用于训练/验证缓存与在线 Predictor，predict.py/cost.py 共享逻辑。缓存身份包含裁切源码 SHA256。原始数据不修改，结果不覆盖。合成训练和真实 Train 八个 batch 的 Smoke Test 通过（首个合成样本重复被审计拦截，修正后独立重跑）。

正式训练未启动，不提供 AUROC、AUPRC、Bootstrap 或正式模型延迟。当前候选在极暗图 train_05402.jpg 上存在外周截断风险：原图 2560×1920，建议裁切框 [20, 0, 2085, 1920]，右侧框外仍有暗弱信号。证据为 `outputs/S01_qc/extreme_05402_brightness_review.jpg`（仅显示提亮，未修改原图或训练图）。应先改进暗弱信号边界保护或保守回退，再重跑完整 QC 和 Smoke Test；禁止直接将 visual_review.json 改为 PASSED 绕过检查。

验收：73 项测试通过；真实 Train batch32 八步 AMP 探针通过，峰值分配显存约 2364 MiB。253 个历史冻结文件核验通过。保留 S01a/S01b 配置，未启动其他训练、未生成 Test 提交。
