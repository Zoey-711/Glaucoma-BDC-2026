"""Summarize measured Phase F artifacts; no inference, fitting, or training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.phase_f import ROOT,verify_frozen
from src.utils import sha256,write_json


def table(df):
    columns=list(df.columns)
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    frozen=verify_frozen()
    comparison=pd.read_csv(ROOT/'fusion_comparison.csv')
    fusion=comparison[comparison.experiment_id.str.startswith('F')].copy()
    complements=pd.read_csv(ROOT/'analysis/complementarity.csv')
    latency=pd.read_csv(ROOT/'model_latency_report.csv')
    precision=pd.read_csv(ROOT/'deployment_run/precision_consistency.csv')
    robustness=pd.read_csv(ROOT/'stress/robustness_comparison.csv')
    evidence=json.loads((ROOT/'analysis/model_evidence.json').read_text())
    interleaved=json.loads((ROOT/'deployment_run/interleaved_fp32/report.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4))
    ax.errorbar(fusion.delta_auroc,fusion.experiment_id,
        xerr=[fusion.delta_auroc-fusion.ci95_low,fusion.ci95_high-fusion.delta_auroc],fmt='o',capsize=4)
    ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel('Delta AUROC vs G01; unadjusted paired 95% CI')
    fig.tight_layout();fig.savefig(ROOT/'fusion_delta_auroc.png',dpi=180);plt.close(fig)
    columns=['experiment_id','members','auroc','auprc','delta_auroc','ci95_low','ci95_high']
    ftext=table(fusion[columns].map(lambda x:f'{x:.8f}' if isinstance(x,float) else x))
    cx=complements[complements.candidate.isin(['B02','G02','L03b','R01b'])]
    ctext=table(cx[['candidate','pearson_probability','spearman_probability','reference_wrong_candidate_correct',
                   'reference_correct_candidate_wrong','threshold_both_errors']].round(6))
    text=f'''# Phase F：冻结融合分析

读取五个模型完整官方 Val 5,849 行（191 阳性、5,658 阴性），校验 ID、顺序、标签、有限 logits、概率范围、sigmoid 对应关系、checkpoint SHA256、配置和冻结证据。G02 使用研究配置历史 hash，其余旧模型无统一 canonical config hash 时，本轮计算并标明为本轮记录；同时验证 checkpoint 内嵌配置与保存配置完全一致，不能把新 hash 冒充历史证据。详见 `outputs/phase_f/analysis/model_evidence.json`。

公式在 `configs/phase_f.yaml` 中固定并在计算前锁定。四组概率融合均为两个概率各 0.5；F01_LOGIT 为两个原始 logit 各 0.5 后 sigmoid。没有拟合权重、阈值或读取 Test。

{ftext}

G01 基线 AUROC 0.98356217、AUPRC 0.80461532。Bootstrap 固定 seed 42、类别分层图像配对重采样 2,000 次。以上为 **5 项探索性比较的未校正 95% CI**，不能据此宣称跨种子稳定收益或测试集泛化保证。F03 下界只有约 0.00000072，接近零且受 Monte Carlo 精度、多重比较和重复使用 Val 影响，应视为非常脆弱的证据。患者 ID 缺失，不能称患者级独立区间。

## 互补性

下面“reference wrong / candidate correct”统计 G01 把阳性排到阴性以下、另一模型排对的样本对；反向列是另一模型新增的错误。所有正负对共 1,080,678 个，不能当作 1,080,678 个独立统计样本。严格错误定义为 p(阳性)<p(阴性)，平分 ties 计入 AUROC 损失；阈值错误使用固定 0.5。

{ctext}

G01/B02 各有 17,764 / 18,307 个严格误排序对，共同错误 8,009 个；B02 修复 G01 的 9,755 对，同时新增 10,298 对。F01 的总误排序降为 14,501 对。G01 的阈值错误为 99 张，B02 为 100 张，共同错误 62 张。F01 Recall 0.685864 低于 G01 0.712042，而 AUROC 提高，因此本次融合收益体现为整体排序变化，不能用 Recall 上升解释。

F04 的 Val AUROC 最高，但包含 224 与 384 两个 ConvNeXt，部署成本需单独实测；本轮重点计时的是 G01、B02 和 F01。F01 概率与 logit 融合的 AUROC 相差不足 0.000001，概率融合 AUPRC 更高；不继续搜索混合权重。

完整结果：`outputs/phase_f/fusion_comparison.csv`；逐组完整 Val 预测、相关矩阵和误排序/阈值互补性在 `outputs/phase_f/analysis/`；区间图 `outputs/phase_f/fusion_delta_auroc.png`。所有分数来自冻结预测，不是新 FP16 部署验证成绩。
'''
    Path('docs/phase_f_fusion_report.md').write_text(text,encoding='utf-8')
    timing=latency[latency.local_latency_ms.notna()]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(timing.model+' '+timing.precision,timing.local_latency_ms)
    ax.axhline(100,color='red',linestyle='--');ax.set_ylabel('RTX 4060 Laptop mean end-to-end ms (NOT T4)')
    fig.tight_layout();fig.savefig(ROOT/'latency_4060.png',dpi=180);plt.close(fig)
    ltext=table(timing[['model','precision','model_size_mb','local_latency_ms','t4_status']].round(4))
    ptext=table(precision[['model','max_probability_error','over_tolerance','threshold_flips','pairwise_order_disagreements',
                         'probability_threshold_consistent','strict_order_consistent']].round(7))
    sources=[]
    for e in evidence:
        m=e['pretraining']
        sources.append(f"- **{e['model']}**：{m['architecture']}，{m['dataset']}；[固定 revision 原始下载]({m['source_url']})；[权重模型卡]({m['training_description_url']})；权重 SHA256 `{m['sha256']}`；许可证全文 `{m['license_file']}`（SHA256 `{m['license_sha256']}`）。")
    text=f'''# Phase F：部署预检

本轮在 RTX 4060 Laptop 本机执行推理；**官方 T4 验收仍为 UNVERIFIED**。本机 Python 3.11 不是官方 Python 3.12 环境。没有生成官方提交、没有调用 Test。

## 权重来源与文件大小

五个模型的权重来源、模型卡及完整许可证文件 SHA256 再次验证通过。模型卡声明的权重许可证为 Apache-2.0，训练数据为通用 ImageNet-1K；ConvNeXt 另外保留上游 MIT notice。依据的是具体权重模型卡声明和完整许可条款，不能仅因 timm 代码是开源就推断权重许可。使用和再分发仍须保留许可证、归属说明及修改声明，最终主办方材料核验不能由本地审核代替。

{chr(10).join(sources)}

单模型所需配置嵌入 checkpoint；F01 实际加载 G01、B02 checkpoint 和 bundle JSON。逐文件解压后字节数与 SHA256 见 `outputs/phase_f/deployment_run/loaded_files.json`。不把原始预训练文件重复计入推理加载大小，它属于额外合规材料。其余融合当前是离线分析，未宣称已经打包完成。

{table(latency[latency.precision=='fp32'][['model','loaded_bytes','model_size_mb','size_status']].round(4))}

## 同一推理接口与数值一致性

predict.py 与 cost.py 均使用 src.inference.Predictor.predict_one：原图读取/解码、固定 resize/ImageNet 常数、传输、eval 模型前向、概率和标签、CSV 写出与 flush。模型使用 eval、requires_grad_(False) 与 inference_mode。本轮无 TTA、无随机推理增强。保存接口源码 hash 供追溯。

固定 105 张 Train 的 FP32/FP16 对照如下；0.001 为预先锁定的概率容差，同时单独统计阈值翻转与完整两两排序差异。**局部数值通过不代表全 Val 或官方核验通过**。同精度三轮计时的稳定性见各 timing.json。

{ptext}

B02 与 F01 各有 2/105 张超过 0.001 的跨精度概率容差，当前跨精度检查未通过；所有模型都有部分近邻排序变化。这不等同于官方“最终 CSV 与同精度推理一致性”违规，而是说明不能直接将 FP16 替换进已有 FP32 结果。若将来选择 FP16，必须用该精度重新完成全 Val 和最终同精度验收。

另外每个模型抽查 32 张官方 Val 的 batch1 FP32 与冻结预测。B02 最初超过额外的 0.0001 检查：最大差 0.00017929。专项诊断证明原图与缓存输入张量完全相同，batch32 推理与历史预测一致（最大差约 9.2e-17），batch1 卷积计算路径发生数值变化，当前 cuDNN TF32 开启。保留失败诊断在 `outputs/phase_f/deployment/`，继续使用预先锁定的 0.001 容差，并在表中保留严格检查结果。不能把历史 batch32 分数直接声称为最终 batch1/FP16 分数；候选提交前需用实际部署精度完整复核 Val。

## 本机计时与 T4 准备

同一固定 Train 100 张、另 5 张预热、batch1、连续 3 轮，覆盖原图解码至 CSV flush，逐图 CUDA 同步。没有使用训练缓存计时，也没有清空操作系统页缓存。FP32/FP16 是两次独立测量，不是缩放推算。

{ltext}

B02 文件最小，但这份本机单独测量中反而比 G01 慢；不能用参数量直接推算端到端延迟。由于 F01 独立测时甚至低于 B02，额外按每张图轮换三模型执行顺序进行 FP32 交错核对（仍 100+5、每模型 3 轮、每次从原图解码）。核对均值：{interleaved['mean_ms']} ms。这项诊断用于说明执行顺序、GPU/系统状态和缓存敏感性，不替代正式连续测时协议；原始独立测量完整保留。

融合可共享一次图像解码，但仍执行两个模型前向，不能宣称融合比单模型更省计算，也不能把两项独立测速相加当实测。G01 与 F01 是否满足 100ms 的最终判断仍为 UNVERIFIED；4060 结果只说明值得带到实际 T4 做预检，没有转换比例可保证达标。

后续官方环境使用 `scripts/phase_f_t4.sh`，它调用现有 `scripts.accept_t4` 和同一 cost.py，检查 NVIDIA T4、Python3.12、PyTorch2.10.0、torchvision0.25.0、CUDA12.8，严格 100+5 张、batch1、3轮、<=100ms、<=500MB 和最终 CSV 一致性。须提供真实官方目录、获授权的最终冻结 CSV/metadata、checkpoint 及官方清单；本轮没有执行。迁移 F01 时要重写 bundle 的成员路径并重新核验其 SHA256，不能直接使用本机绝对路径。
'''
    Path('docs/phase_f_deployment_report.md').write_text(text,encoding='utf-8')
    varying=robustness[robustness.condition!='clean']
    heat=varying.pivot(index='model',columns='condition',values='mean_abs_probability_change')
    fig,ax=plt.subplots(figsize=(11,4));plot=ax.imshow(heat.to_numpy(),aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(heat.columns)),heat.columns,rotation=30,ha='right');ax.set_yticks(range(len(heat.index)),heat.index)
    fig.colorbar(plot,ax=ax,label='Mean absolute probability drift (Train diagnostic)')
    fig.tight_layout();fig.savefig(ROOT/'stress/robustness_drift.png',dpi=180);plt.close(fig)
    text=f'''# Phase F：图像质量压力测试

只从官方 Train 固定抽取 256 张（64 阳性、192 阴性，seed 20261030），样本清单在首次推理前保存。**这些图像曾用于全部模型训练，不是真正未见的 holdout。** 患者 ID 缺失，不能证明患者独立。该诊断只衡量同一图像受扰动后的预测变化，不报告或替代官方 Val AUROC，不据此选择模型或调整增强强度。

固定操作：先将 RGB 图像双线性缩放到 384×384 作为共同质量画布；clean 对照采用同样画布。分别施加 brightness 0.8/1.2、contrast 0.8/1.2、Gaussian blur sigma 0.5/1.0 画布像素、JPEG quality 85/60，之后由各模型自己的确定性预处理缩放到所需输入。每次仅一种扰动，没有拟合。这样模糊强度在共同画布上可比较，但此模拟不覆盖真实设备/曝光/压缩的所有变化。

平均绝对概率变化：

{table(heat.reset_index().round(5))}

R01b 在 brightness 1.2 和 JPEG quality 60 下的平均概率变化低于 G01，但在 brightness 0.8、contrast 0.8 和两档模糊下未全面优于 G01。这是混合的扰动响应，不支持“ColorJitter 带来整体鲁棒优势”的结论，也不能推翻其官方 Val 未证明收益的结果。

最大阈值翻转率及平均 logit 变化（跨八种固定扰动，仅作描述性汇总，不作为选模分数）：

{table(varying.groupby('model').agg(max_flip_fraction=('threshold_flip_fraction','max'),max_p95_probability_change=('p95_abs_probability_change','max'),mean_abs_logit_change=('mean_abs_logit_change','mean')).reset_index().round(5))}

绝对概率更稳定不一定代表判别更好：接近 0 的饱和概率也会产生很小的变化。因此同时保存 logit 变化、Spearman 排序相关、固定 0.5 阈值翻转以及正/负组翻转率，不把单一 drift 指标当成鲁棒泛化排名。逐样本数据及全部条件统计在 `outputs/phase_f/stress/`，图为 `robustness_drift.png`。

本轮未启动新训练或官方提交，历史 {frozen} 个受保护文件核验一致。后续优先完成 G01/B02/F01 的实际部署精度完整 Val 复核与 T4 验收，再决定是否需要进一步实验；不能用压力测试替代独立评估。
'''
    Path('docs/phase_f_robustness_report.md').write_text(text,encoding='utf-8')
    files=[*ROOT.rglob('*.csv'),*ROOT.rglob('*.json'),*ROOT.rglob('*.png')]
    write_json(ROOT/'final_verification.json',{'status':'PASS','frozen_files_verified':frozen,
        'test_used':False,'training_started':False,'T4_status':'UNVERIFIED',
        'files':{str(p.relative_to(ROOT)):sha256(p) for p in files if p.name!='final_verification.json'}})
    print('Three Phase F reports and plots generated; frozen assets verified',flush=True)


if __name__=='__main__':main()
