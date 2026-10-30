# Phase H: authorized Linux execution and result handoff

This branch adds code only. No Phase H training or official Test prediction has run. The remote operator must confirm competition authorization and obtain the official data through their own permitted channel. Never transfer official images or weights through the Git repository.

## Existing work and status

G05 (EfficientNetV2-S), G06 (ConvNeXt-Small), G07 (Swin-Tiny), and G08 (DenseNet-121) already have model entries and independent YAML in `configs/research/`. They remain `BLOCKED` until each exact nonmedical weight file, source, license and SHA256 is reviewed. This checkout cannot establish which tasks are running on the remote server. The operator must inspect their existing scheduler and GPU processes before launching anything. Historical B02, G01, G02, F01/F04, Round 01 and other frozen outputs are untouched.

| ID | Purpose | Code | Formal run gate |
| --- | --- | --- | --- |
| H03_G01_HEAD | Server trainer's original single logit head control | Implemented, synthetic AMP tested | Approved G01 ImageNet weights; official manifest hashes and free GPU |
| H03_MATCHED_MLP | Final stage with parameter matched MLP | Implemented, synthetic AMP tested | Same as above |
| H03_MULTISCALE | Stage 2 + final stage pooled features | Implemented, synthetic AMP tested | Same as above |
| H04_RANDOM | V2-S-384 random Train sampling | Implemented, synthetic AMP tested | **BLOCKED:** V2-S weight provenance, license and SHA256 |
| H04_BALANCED | V2-S-384 balanced Train batch sampling | Implemented, synthetic AMP tested | **BLOCKED:** same weight gate; inspect duplicate/coverage metrics |
| H01 | Conservative optic disc candidate QC | Implemented, synthetic tested | **BLOCKED:** no reviewed locator; no ROI model training |

H03 head widths are 192 and 128, giving 147,841 and 147,713 trainable head parameters. Backbone parameters are shared. The ~4.5 GMAC figure is a model card estimate at 224 px; it is not measured FLOPs or latency. Server trainer random number use and training loop differ from historical G01, so `H03_G01_HEAD` is the internal control. Seed 42 across machines does not guarantee bitwise equality. H04 pairs differ only by sampler. Validation loading is unchanged and never resampled.

## Pinned checkout and preflight

Use the full commit SHA reported with this delivery, check out that commit or its immutable tag, and verify a clean tracked worktree. Use Python 3.12 and an appropriate CUDA build of PyTorch, plus `requirements.txt`. Review `FRIEND_TRAINING_GUIDE.md` and `RESULT_CONTRACT.md`. Verify that the local ignored `configs/weights_g01.local.json` points to the exact approved G01 weight file and passes the repository's SHA/license checks. H04 must remain blocked until its own reviewed manifest and config fields are completed and recommitted as a new version. Do not edit a published YAML in place for a run.

```bash
git rev-parse HEAD
git status --short
python -m pytest tests/test_h_experiments.py -q
python -m scripts.h03_preflight --config configs/h/H03_MULTISCALE.yaml --device cuda
python -m scripts.h_gpu_task status
nvidia-smi
```

`status` reports only jobs launched through the Phase H launcher in this checkout. Check other running jobs independently. Each explicit launch binds one process to one GPU with `CUDA_VISIBLE_DEVICES`; it refuses occupied GPUs, blocked configs and existing output directories. The launch is asynchronous, so GPU slots can finish at different times. A failed job leaves its logs and checkpoint for inspection. No DDP is used.

Examples below are **operator commands only**, never executed by local development. Use the exact path of an authorized local copy of the official archive; it is verified by SHA256. The first three H03 controls should use separate free GPUs. The operator may choose different GPU numbers after checking current allocations.

For an otherwise idle eight GPU host, the provisional allocation is GPU 0/1/2 for the three H03 controls and GPU 3–7 reserved for existing or independently approved tasks. H04_RANDOM and H04_BALANCED can later use two free GPUs only after a separate reviewed weight/config commit. H01 runs QC without formal GPU training. This allocation does not assert that any remote GPU is currently idle.

```bash
python -m scripts.h_gpu_task launch --id H03_G01_HEAD --gpu 0 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task launch --id H03_MATCHED_MLP --gpu 1 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task launch --id H03_MULTISCALE --gpu 2 --archive /authorized/path/official.zip
python -m scripts.h_gpu_task status
```

After a task ends and its PID is gone, run `python -m scripts.h_gpu_task release --id H03_MULTISCALE` to free its launcher lock. This does not remove results. To resume a failed task, inspect it first and use the existing `scripts.run_research --resume` only after the config/code/data hash checks pass. The H launcher never silently resumes or overwrites.

## H01 quality control

The QC script reads only the official Train manifest and images, includes `train_05402.jpg` from the S01 regression, writes a contact sheet, per image proposals and fallback reasons. Its result remains `BLOCKED` until independent manual localization review; no ROI classifier is enabled by this script. Do not tune it using Val/Test performance.

```bash
python -m scripts.h01_roi_qc --train-manifest data/official/manifests/train.csv --root data/official --output outputs/h01_qc_manual --count 150
```

## Required result bundle and local acceptance

The trainer writes `results/{experiment_id}/seed_42/fold_none/` with `best.pth`, `validation.csv` (`Image_name`, `True_Label`, `logit`, `Probability`), `metrics.json`, `config.yaml`, `train_history.csv`, `training.log`, `run_manifest.json`, `environment.txt`, and `checksums.sha256`, then makes a `.tar.gz` package. The manifest stores the Git commit, config/data/weight hashes, ordered image and annotation list hashes, seed/fold, GPU model, start/end status and time; environment records Python, torch and CUDA. H04 history also contains Train draw proportions, duplicate rates, coverage, valid batch ratio and risk flag. `model_cost_estimate.json` is written for H03 nonstandard heads. T4 status remains unverified.

The friend returns the complete package without official images. Local acceptance stages the archive, rejects missing or altered files and checkpoints, checks config/data versions and frozen inventory, verifies the 5,849 official Val IDs and labels, recomputes metrics, and imports only accepted bundles. Then compare accepted predictions with G01 and F01 by paired Bootstrap and error complementarity. No candidate should enter fusion on the basis of a screenshot or AUROC number alone.

```powershell
.\.venv\Scripts\python.exe -m scripts.import_results <received-H03-bundle.tar.gz> --experiment-id H03_MULTISCALE
.\.venv\Scripts\python.exe -m scripts.validate_results remote_results/H03_MULTISCALE/seed_42/fold_none --config configs/h/H03_MULTISCALE.yaml
.\.venv\Scripts\python.exe -m scripts.analyze_h_results remote_results/H03_MULTISCALE/seed_42/fold_none --output outputs/h03_multiscale_analysis
```

The host must never use official Test images for fitting, tuning, pseudolabels, or selecting models. No formal submission or T4 acceptance is implied by this work.
