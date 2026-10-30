"""Frozen F01 batch-1 deployment audit, repeatability gate, and Round 02 generation.

Run `audit`, then `validate`, then `submit` in order. Test is not opened by the
first two commands. Historical batch-32 predictions are retained as evidence.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from scripts.accept_t4 import preflight as t4_preflight
from scripts.f01_submission_preflight import locked_model, ranking_changes
from src.data import open_rgb, read_manifest, transform
from src.metrics import metrics
from src.phase_f import validate_predictions
from src.submission import COLUMNS, validate_submission
from src.utils import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = Path('outputs/f01_deployment_v1')
FROZEN = Path('outputs/phase_g2/candidates/F01/inference.json')
VAL = Path('data/official/manifests/valid.csv')
TEST = Path('data/official/manifests/test.csv')
ROUND = Path('submission/round_02_F01')
SOURCE = [Path(p) for p in ('predict.py', 'cost.py', 'src/inference.py',
    'src/data.py', 'src/submission.py', 'src/utils.py', 'src/metrics.py',
    'src/model.py', 'src/phase_f.py', 'src/phase_g.py',
    'scripts/f01_submission_preflight.py', 'scripts/f01_deployment.py',
    'configs/phase_g.yaml')]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git_commit(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def execute_predict(root, manifest, output):
    subprocess.run([sys.executable, str(root/'predict.py'),
        '--checkpoint', str(root/FROZEN), '--csv', str(manifest),
        '--root', str(root/'data/official'), '--output', str(output),
        '--device', 'cuda', '--precision', 'fp32', '--threads', '4'], cwd=root, check=True)


def strict_ranking_changes(a, b):
    """Compare every pair including equal scores, in bounded memory."""
    if len(a) != len(b):
        raise ValueError('Repeated inference has different row count')
    total = 0
    for i in range(len(a)-1):
        total += int(np.count_nonzero(np.sign(a[i]-a[i+1:]) != np.sign(b[i]-b[i+1:])))
    return total


def cross_class_pair_effect(y, old, new):
    y = np.asarray(y, dtype=int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d_old = old[pos,None] - old[neg][None,:]
    d_new = new[pos,None] - new[neg][None,:]
    credit_old = (d_old > 0).astype(float) + .5*(d_old == 0)
    credit_new = (d_new > 0).astype(float) + .5*(d_new == 0)
    improvement = credit_new - credit_old
    return {'total_positive_negative_pairs': int(improvement.size),
        'ranking_changes': int(np.count_nonzero(improvement)),
        'improved_pairs': int((improvement > 0).sum()),
        'worsened_pairs': int((improvement < 0).sum()),
        'net_auc_pair_credit': float(improvement.sum()),
        'delta_auroc_from_pairs': float(improvement.mean())}


def snapshot(root):
    paths = [*SOURCE, FROZEN, FROZEN.parent/'G01.pt', FROZEN.parent/'B02.pt',
        Path('outputs/phase_g2/candidates/F01/candidate_manifest.json'), VAL,
        Path('outputs/phase_g2/freeze/F01_validation.csv'),
        ROUND/'manifest.json', ROUND/'validation_report.md', ROUND/'checksums.sha256']
    return {p.as_posix(): sha256(root/p) for p in paths}


def assert_snapshot(root, audit):
    if snapshot(root) != audit['source_hashes']:
        raise ValueError('Code, F01 member, Val manifest or historical BLOCKED evidence changed')
    if git_commit(root) != audit['git_commit']:
        raise ValueError('Git commit changed during F01 deployment validation')


def audit(root=ROOT):
    root = Path(root)
    out = root/OUT
    out.mkdir(parents=True, exist_ok=False)
    if (root/ROUND/'submission.csv').exists():
        raise FileExistsError('Round 02 submission already exists')
    old_block = read_json(root/ROUND/'manifest.json')
    if old_block['status'] != 'BLOCKED' or old_block['submission_created'] is not False:
        raise ValueError('Historical Round 02 BLOCKED evidence differs')
    predictor, candidate, cfg = locked_model(root, root/FROZEN,
        root/FROZEN.parent/'candidate_manifest.json')
    flags = {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision()}
    expected_flags = {'matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'deterministic_algorithms': True, 'float32_matmul_precision': 'highest'}
    if flags != expected_flags:
        raise ValueError(f'FP32 math policy differs from frozen batch-1 path: {flags}')
    if any(p.dtype != torch.float32 for child in predictor.children for p in child.model.parameters()):
        raise ValueError('F01 model parameters must stay FP32')
    official = read_manifest(root/VAL, root/'data/official')
    if len(official) != 5849:
        raise ValueError('Official Val count changed')
    sample = pd.read_csv(root/'outputs/phase_g2/freeze/t4_validation_105.csv', dtype={'image_id': str})
    sample_ids = list(dict.fromkeys([*sample.image_id.tolist(),
        'valid_04165.jpg', 'valid_04956.jpg']))
    cache = root/'data/cache/val_e0969008fada996ce2b9'
    if not cache.is_dir():
        raise FileNotFoundError('Frozen batch-32 validation cache missing')
    tensor_mismatches = []
    for image_id in sample_ids:
        indices = official.index[official.image_id.eq(image_id)]
        if len(indices) != 1:
            raise ValueError(f'Val sample ID is not unique: {image_id}')
        i = int(indices[0])
        raw = transform(224)(open_rgb(official.iloc[i].resolved_path))
        cached = transform(224, preprocessed=True)(open_rgb(cache/f'{i:06d}.png'))
        if not torch.equal(raw, cached):
            tensor_mismatches.append(image_id)
    if tensor_mismatches:
        raise ValueError(f'Substantive preprocessing mismatch: {tensor_mismatches[:5]}')
    old = read_json(root/'outputs/f01_submission_gate/gate_report.json')
    if (old['status'] != 'BLOCKED' or old['frozen_formula_max_error'] > 1e-12
        or old['cached_batch_path_diagnostic']['status'] != 'PASS'
        or old['cost_py_local_smoke_mismatches_by_round'] != [0,0,0]):
        raise ValueError('Historical batch-32 explanation or shared inference evidence missing')
    # Existing validation used batch 32 and no autocast; both scripts use Predictor.
    if ('batch_size=c[\'batch_size\']' not in (root/'src/experiment_training.py').read_text(encoding='utf-8')
        or 'with torch.inference_mode():' not in (root/'src/experiment_training.py').read_text(encoding='utf-8')):
        raise ValueError('Historical validation batch/precision code needs manual review')
    report = {'status':'PASS','scope':'F01 frozen batch-1 FP32 deployment path; no Test access',
        'git_commit':git_commit(root),'source_hashes':snapshot(root),
        'historical_blocked_manifest_sha256':sha256(root/ROUND/'manifest.json'),
        'historical_blocked_report_sha256':sha256(root/ROUND/'validation_report.md'),
        'bundle_sha256':sha256(root/FROZEN), 'loaded_files':candidate['loaded_files'],
        'loaded_bytes':candidate['loaded_bytes'], 'member_configs':cfg,
        'backend':flags, 'tensor_dtype':'torch.float32',
        'model_eval_and_inference_mode':True,'formula':'0.5*sigmoid(G01_logit)+0.5*sigmoid(B02_logit)',
        'historical_validation_batch_size':32,'deployment_batch_size':1,
        'preprocessing_cache_comparison':{'sample_count':len(sample_ids),
            'tensor_mismatch_count':len(tensor_mismatches)},
        'historical_batch32_reproduction':old['cached_batch_path_diagnostic'],
        'interpretation':'Matched source weights, preprocessing tensors, FP32/eval flags and formula; batch shape changes GPU floating-point operation path. No alternate algorithm was selected by Val score.',
        'environment':{'gpu':predictor.metadata['gpu'],'python':platform.python_version(),
            'torch':torch.__version__,'torchvision':torchvision.__version__,
            'cuda':torch.version.cuda,'pillow':Image.__version__},
        'test_images_read':False,'t4_status':'UNVERIFIED'}
    write_json(out/'audit.json',report)
    return report


def validate(root=ROOT):
    root = Path(root)
    out = root/OUT
    prior = read_json(out/'audit.json')
    if prior['status'] != 'PASS':
        raise ValueError('Phase A inference audit did not pass')
    assert_snapshot(root, prior)
    official = read_manifest(root/VAL, root/'data/official')
    if len(official) != 5849:
        raise ValueError('Official Val count changed')
    results = []
    for run_number in (1, 2):
        target = out/f'validation_batch1_run_{run_number}.csv'
        execute_predict(root, root/VAL, target)
        result = validate_submission(target, official.image_id, .5)
        results.append(result)
    a, b = results
    pa = a.Probability.to_numpy(np.float64)
    pb = b.Probability.to_numpy(np.float64)
    pairwise = strict_ranking_changes(pa,pb)
    repeat = {'rows':len(a),'max_probability_absolute_difference':float(np.max(np.abs(pa-pb))),
        'different_probability_values':int(np.count_nonzero(pa != pb)),
        'pred_label_changes':int(np.count_nonzero(a.Pred_Label.to_numpy()!=b.Pred_Label.to_numpy())),
        'all_pair_ranking_changes':pairwise,
        'first_csv_sha256':sha256(out/'validation_batch1_run_1.csv'),
        'second_csv_sha256':sha256(out/'validation_batch1_run_2.csv'),
        'passed':bool(np.array_equal(pa,pb) and pairwise==0 and
                      np.array_equal(a.Pred_Label.to_numpy(),b.Pred_Label.to_numpy()))}
    write_json(out/'repeatability.json',repeat)
    if not repeat['passed']:
        write_json(out/'validation_gate.json',{'status':'BLOCKED','reason':'Same batch-1 FP32 deployment path is not repeatable',
            'repeatability':repeat,'test_images_read':False})
        return {'status':'BLOCKED','repeatability':repeat}
    frozen = validate_predictions(pd.read_csv(root/'outputs/phase_g2/freeze/F01_validation.csv'),official,5849)
    historical = frozen.probability.to_numpy(np.float64)
    from scripts.f01_submission_preflight import ranking_changes
    history_ranking = ranking_changes(historical,pa,official.label.to_numpy(),official.image_id.tolist())
    auc_pairs = cross_class_pair_effect(official.label.to_numpy(),historical,pa)
    old_scores = metrics(official.label,historical)
    new_scores = metrics(official.label,pa)
    auc_delta = new_scores['auroc']-old_scores['auroc']
    if not np.isclose(auc_delta,auc_pairs['delta_auroc_from_pairs'],rtol=0,atol=1e-12):
        raise ValueError('AUROC change is not explained by positive-negative pair changes')
    deployment = pd.DataFrame({'Image_name':official.image_id,
        'Label':official.label,'Probability':pa,
        'Pred_Label':a.Pred_Label.to_numpy(int),
        'experiment_id':'F01_batch1_fp32','source_split':'official_val'})
    deployment.to_csv(out/'validation.csv',index=False,float_format='%.17g')
    report = {'status':'PASS','selection_policy':'No model, weight, threshold or inference algorithm selected by Val scores',
        'official_val_rows':len(official),'reference_batch_size':1,'precision':'fp32',
        'current_metrics':new_scores,'historical_batch32_metrics':old_scores,
        'delta_auroc':float(auc_delta),'delta_auprc':float(new_scores['auprc']-old_scores['auprc']),
        'historical_vs_deployment_ranking':history_ranking,
        'cross_class_auc_effect':auc_pairs,
        'historical_to_deployment_max_probability_error':float(np.max(np.abs(historical-pa))),
        'historical_to_deployment_pred_label_changes':int(np.count_nonzero((historical>=.5)!=(pa>=.5))),
        'repeatability':repeat,'validation_sha256':sha256(out/'validation.csv'),
        'git_commit':prior['git_commit'],'source_hashes':prior['source_hashes'],
        'test_images_read':False,'t4_status':'UNVERIFIED'}
    write_json(out/'metrics.json',report)
    write_json(out/'validation_gate.json',{'status':'PASS','same_deployment_repeatability':repeat,
        'historical_batch32_difference_requires_revalidation':True,
        'deployment_metrics_sha256':sha256(out/'metrics.json'),
        'validation_sha256':sha256(out/'validation.csv'),
        'test_images_read':False,'t4_status':'UNVERIFIED'})
    return report


def submit(root=ROOT):
    root = Path(root)
    out = root/OUT
    phase_a = read_json(out/'audit.json')
    phase_b = read_json(out/'validation_gate.json')
    if phase_a['status'] != 'PASS' or phase_b['status'] != 'PASS':
        raise ValueError('F01 batch-1 audit and repeatability gate must both pass before Test')
    assert_snapshot(root,phase_a)
    if sha256(out/'metrics.json') != phase_b['deployment_metrics_sha256']:
        raise ValueError('Deployment metrics changed after validation')
    if sha256(out/'validation.csv') != phase_b['validation_sha256']:
        raise ValueError('Deployment Val predictions changed after validation')
    final = root/ROUND/'submission.csv'
    final_meta = Path(str(final)+'.metadata.json')
    if final.exists() or final_meta.exists() or (root/ROUND/'manifest_v2.json').exists():
        raise FileExistsError('Round 02 final inference already exists; refusing overwrite')
    columns = pd.read_csv(root/TEST,nrows=0).columns.tolist()
    if columns != ['image_id','path']:
        raise ValueError('Test manifest must be unlabeled and contain only image_id,path')
    test = read_manifest(root/TEST,root/'data/official',labeled=False)
    if len(test) != 5855 or not all(Path(p).is_file() for p in test.resolved_path):
        raise ValueError('Complete 5855-image official Test set required')
    staging = out/'staging_submission.csv'
    execute_predict(root,root/TEST,staging)
    staged = validate_submission(staging,test.image_id,.5)
    if (list(staged.columns) != COLUMNS or len(staged) != 5855
        or not staged.Pred_Label.isin([0,1]).all()
        or not staging.read_bytes().startswith(b'Image_name,Probability,Pred_Label')):
        raise ValueError('Staged official submission schema/encoding invalid')
    # Use a fixed, probability-independent selection for local cost consistency.
    sample = test.iloc[:105]
    for part,directory in ((sample.iloc[:5],out/'test_warmup_5'),
                           (sample.iloc[5:],out/'test_measured_100')):
        directory.mkdir(parents=True,exist_ok=False)
        for row in part.itertuples():
            shutil.copyfile(row.resolved_path,directory/Path(row.resolved_path).name)
    # Promote identical bytes into Round 02, then make cost.py check that exact file.
    shutil.copyfile(staging,final)
    stage_meta = read_json(str(staging)+'.metadata.json')
    stage_meta.update(round='round_02_F01',
        submission_sha256=sha256(final),code_commit=git_commit(root),uploaded=False)
    write_json(final_meta,stage_meta)
    try:
        if sha256(final) != sha256(staging):
            raise ValueError('Final submission bytes differ from predict.py staging output')
        preflight = t4_preflight(final,root/FROZEN,root/TEST)
        if preflight['rows'] != 5855 or preflight['status'] != 'UNVERIFIED':
            raise ValueError('Final official preflight failed')
        subprocess.run([sys.executable,str(root/'cost.py'),str(final),
            str(out/'test_measured_100'),str(out/'test_warmup_5'),
            '--checkpoint',str(root/FROZEN),'--device','cuda','--precision','fp32',
            '--output',str(out/'cost_final_local_4060'),'--smoke'],cwd=root,check=True)
        cost = read_json(out/'cost_final_local_4060/report.json')
        if cost['mismatches_by_round'] != [0,0,0] or cost['official_protocol']:
            raise ValueError('Final CSV and shared cost.py inference disagree')
        validated = validate_submission(final,test.image_id,.5)
        if len(validated)!=5855 or sha256(final)!=stage_meta['submission_sha256']:
            raise ValueError('Final CSV changed after cost.py consistency check')
    except Exception:
        # Only these two files were created by this invocation; preserve historical evidence.
        final.unlink(missing_ok=True)
        final_meta.unlink(missing_ok=True)
        raise
    scores = read_json(out/'metrics.json')
    manifest = {'status':'CSV_VALIDATED_T4_UNVERIFIED','round':'round_02_F01',
        'rows':len(validated),'columns':COLUMNS,'encoding':'utf-8','probability':'positive_class_1',
        'pred_label_rule':'Probability >= 0.5','submission_sha256':sha256(final),
        'manifest_sha256':sha256(root/TEST),'checkpoint_bundle_sha256':sha256(root/FROZEN),
        'loaded_files':phase_a['loaded_files'],'loaded_bytes':phase_a['loaded_bytes'],
        'precision':'fp32','batch_size':1,'formula':phase_a['formula'],
        'code_commit':git_commit(root),'code_file_sha256':phase_a['source_hashes'],
        'historical_blocked_manifest_sha256':phase_a['historical_blocked_manifest_sha256'],
        'historical_blocked_report_sha256':phase_a['historical_blocked_report_sha256'],
        'deployment_validation_sha256':sha256(out/'validation.csv'),
        'deployment_metrics_sha256':sha256(out/'metrics.json'),
        'repeatability':scores['repeatability'],'validation_metrics':scores['current_metrics'],
        'cost_local_4060':{'mean_ms':cost['mean_ms'],'mismatches_by_round':cost['mismatches_by_round'],
            'gpu':cost['gpu'],'official_protocol':False},
        'environment':{'python':cost['python'],'torch':cost['torch'],
            'torchvision':cost['torchvision'],'cuda':cost['cuda'],'gpu':cost['gpu']},
        't4_status':'UNVERIFIED','official_site_uploaded':False,
        'test_used_for_parameter_selection':False}
    write_json(root/ROUND/'manifest_v2.json',manifest)
    effect = scores['cross_class_auc_effect']
    report = f"""# Round 02 F01 部署复验

状态：**CSV_VALIDATED_T4_UNVERIFIED**。保留原 `manifest.json`、`validation_report.md` 和 `checksums.sha256` 所记录的历史 `BLOCKED` 结论；该结论针对原 batch=32 Val 与 batch=1 的直接排序比较。本次修订后，batch=1 是固定的唯一部署参考，未重新训练或选择模型。

冻结公式：`{phase_a['formula']}`。G01、B02 按 checkpoint 原配置做 224×224 FP32 推理。实际加载文件 {phase_a['loaded_bytes']:,} 字节。

完整 Val 两次 batch=1 推理：概率差 {scores['repeatability']['max_probability_absolute_difference']:.12g}，Pred_Label 变化 {scores['repeatability']['pred_label_changes']}，全对排序变化 {scores['repeatability']['all_pair_ranking_changes']}。部署版 AUROC {scores['current_metrics']['auroc']:.12f}、AUPRC {scores['current_metrics']['auprc']:.12f}；历史 batch=32 分别为 {scores['historical_batch32_metrics']['auroc']:.12f}、{scores['historical_batch32_metrics']['auprc']:.12f}。历史至部署版有 {effect['ranking_changes']} 对正负排序变化，改善 {effect['improved_pairs']} 对、变差 {effect['worsened_pairs']} 对，净 AUROC 差 {scores['delta_auroc']:+.12f}。这些差异仅用于重验证，没有驱动权重、阈值或算法选择。

最终 `submission.csv`：{len(validated)} 行，UTF-8，列 `Image_name,Probability,Pred_Label`，官方清单顺序一致，概率有限且处于 [0,1]，标签与 0.5 阈值一致。SHA256：`{manifest['submission_sha256']}`。`cost.py --smoke` 在固定 100 张 Test 图像上连续三轮与最终 CSV 的不匹配数为 {cost['mismatches_by_round']}；本地 {cost['gpu']} 完整端到端平均 {cost['mean_ms']:.3f} ms。**T4 性能验收仍为 UNVERIFIED**。

正式 Test 仅用于冻结模型 FP32 前向与提交文件一致性检查；未读取标签、拟合统计、伪标签、训练或依据 Test 概率调整参数。未上传比赛网站。
"""
    (root/ROUND/'validation_report_v2.md').write_text(report,encoding='utf-8')
    files=[final,final_meta,root/ROUND/'manifest_v2.json',root/ROUND/'validation_report_v2.md']
    (root/ROUND/'checksums_v2.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files),encoding='utf-8')
    write_json(out/'final_status.json',{'status':manifest['status'],
        'submission_path':str(final.resolve()),'submission_sha256':manifest['submission_sha256'],
        'rows':manifest['rows'],'t4_status':'UNVERIFIED'})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','validate','submit'])
    parser.add_argument('--root',default=str(ROOT))
    args=parser.parse_args()
    result={'audit':audit,'validate':validate,'submit':submit}[args.stage](args.root)
    print(json.dumps({key:result.get(key) for key in ('status','rows','delta_auroc',
        'validation_metrics','submission_sha256','t4_status') if key in result},indent=2),flush=True)
    if result['status']=='BLOCKED':
        raise SystemExit(2)


if __name__=='__main__':
    main()
