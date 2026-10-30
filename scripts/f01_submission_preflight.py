"""Fail closed on frozen F01 validation ranking drift before any Test inference."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.data import open_rgb, read_manifest, transform
from src.inference import Predictor
from src.phase_f import validate_predictions
from src.phase_g import compare_frozen_to_inference
from src.submission import validate_submission
from src.utils import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def ranking_changes(reference, current, labels, ids):
    """Count all exact pairwise ordering changes without an O(n²) allocation."""
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    if len(a) != len(b) or len(a) != len(y) or len(a) != len(ids):
        raise ValueError('Ranking comparison arrays are not aligned')
    total = cross_class = same_class = 0
    examples = []
    for i in range(len(a) - 1):
        changed = np.flatnonzero(np.sign(a[i] - a[i + 1:]) != np.sign(b[i] - b[i + 1:])) + i + 1
        total += len(changed)
        cross_class += int((y[changed] != y[i]).sum())
        same_class += int((y[changed] == y[i]).sum())
        for j in changed[:max(0, 20 - len(examples))]:
            examples.append({'first': str(ids[i]), 'second': str(ids[j]),
                'labels': [int(y[i]), int(y[j])],
                'frozen_probabilities': [float(a[i]), float(a[j])],
                'batch1_probabilities': [float(b[i]), float(b[j])]})
    return {'all_pair_order_changes': total, 'positive_negative_order_changes': cross_class,
            'same_class_order_changes': same_class, 'examples': examples}


def locked_model(root, bundle, model_manifest):
    manifest = json.loads(model_manifest.read_text(encoding='utf-8'))
    definition = json.loads(bundle.read_text(encoding='utf-8'))
    if (manifest['candidate'] != 'F01' or manifest['precision'] != 'fp32'
        or definition != {'members': ['G01.pt', 'B02.pt'], 'weights': [.5, .5],
                          'threshold': .5, 'tta': ['identity']}):
        raise ValueError('F01 definition, member order, precision or weights changed')
    if manifest['frozen_validation_sha256'] != sha256(root / 'outputs/phase_g2/freeze/F01_validation.csv'):
        raise ValueError('F01 frozen Val prediction changed')
    for entry in manifest['loaded_files']:
        path = bundle.parent / entry['name']
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f"Frozen inference file changed: {entry['name']}")
    evidence = {v['model']: v for v in json.loads((root / 'outputs/phase_f/analysis/model_evidence.json').read_text(encoding='utf-8'))}
    for member in manifest['members']:
        key = member['model']
        if key not in ('G01', 'B02') or member['checkpoint_sha256'] != evidence[key]['checkpoint_sha256']:
            raise ValueError(f'{key} original checkpoint differs from frozen evidence')
        if member['config_hash'] != evidence[key]['config_hash']:
            raise ValueError(f'{key} config hash differs from frozen evidence')
    torch.set_num_threads(4)
    predictor = Predictor(bundle, 'cuda', 'fp32')
    if len(predictor.children) != 2 or predictor.weights != [.5, .5] or predictor.tta != ['identity']:
        raise ValueError('Runtime F01 fusion differs from frozen definition')
    configs = []
    expected_names = ['convnext_tiny', 'efficientnet_b0']
    for i, child in enumerate(predictor.children):
        cfg = child.config
        ops = child.transform.transforms
        if (cfg['image_size'] != 224 or cfg.get('preprocess', 'resize') != 'resize'
            or cfg['model']['name'] != expected_names[i] or len(ops) != 3
            or not isinstance(ops[0], T.Resize) or tuple(ops[0].size) != (224, 224)
            or ops[0].interpolation != InterpolationMode.BILINEAR
            or not isinstance(ops[1], T.ToTensor) or not isinstance(ops[2], T.Normalize)
            or list(ops[2].mean) != [.485, .456, .406]
            or list(ops[2].std) != [.229, .224, .225]
            or child.model.training or any(p.requires_grad for p in child.model.parameters())):
            raise ValueError('Frozen member preprocessing, eval state or model config differs')
        configs.append({'member': ['G01', 'B02'][i], 'architecture': cfg['model']['architecture'],
            'image_size': cfg['image_size'], 'preprocess': cfg.get('preprocess', 'resize'),
            'resize_interpolation': str(ops[0].interpolation), 'normalization_mean': list(ops[2].mean),
            'normalization_std': list(ops[2].std),
            'checkpoint_sha256': manifest['members'][i]['checkpoint_sha256']})
    if not getattr(Predictor.predict_image, '__wrapped__', None):
        raise ValueError('Inference-mode wrapper missing from predictor')
    for script in ('predict.py', 'cost.py'):
        source = (root / script).read_text(encoding='utf-8')
        if 'Predictor(' not in source or 'predictor.predict_one(' not in source:
            raise ValueError(f'{script} does not use shared Predictor.predict_one path')
    return predictor, manifest, configs


def reproduce_cached_batch(root, predictor, official, ids):
    """Check the frozen batch-32 cache path for the two previously observed IDs."""
    result = {}
    cache = root / 'data/cache/val_e0969008fada996ce2b9'
    if not cache.is_dir():
        return {'status': 'UNVERIFIED', 'reason': 'Frozen validation PNG cache unavailable'}
    for image_id in ids:
        index = int(official.index[official.image_id.eq(image_id)][0])
        raw = open_rgb(official.iloc[index].resolved_path)
        cached = open_rgb(cache / f'{index:06d}.png')
        direct = transform(224)(raw)
        stored = transform(224, preprocessed=True)(cached)
        result[image_id] = {'index': index, 'raw_and_frozen_cache_tensor_equal': bool(torch.equal(direct, stored))}
    with torch.inference_mode():
        for member_name, child in zip(('G01', 'B02'), predictor.children):
            for image_id in ids:
                index = int(official.index[official.image_id.eq(image_id)][0])
                start = (index // 32) * 32
                xs = [transform(224, preprocessed=True)(open_rgb(cache / f'{k:06d}.png'))
                      for k in range(start, min(start + 32, len(official)))]
                logits = child.model(torch.stack(xs).to(child.device)).flatten().float()
                probability = float(logits[index - start].sigmoid().item())
                source = pd.read_csv(root / f'outputs/runs/{member_name}/validation.csv')
                frozen_probability = float(source.probability.iloc[index])
                result[image_id][f'{member_name}_batch32_probability'] = probability
                result[image_id][f'{member_name}_frozen_probability'] = frozen_probability
                result[image_id][f'{member_name}_batch32_absolute_error'] = abs(probability - frozen_probability)
    return {'status': 'PASS' if all(v['raw_and_frozen_cache_tensor_equal'] for v in result.values()) else 'FAIL',
            'samples': result}


def run(root=ROOT):
    root = Path(root)
    out = root / 'outputs/f01_submission_gate'
    out.mkdir(parents=True, exist_ok=False)
    if (root / 'submission/round_02_F01/submission.csv').exists():
        raise FileExistsError('Round 02 submission already exists; never overwrite it')
    bundle = root / 'outputs/phase_g2/candidates/F01/inference.json'
    predictor, model_manifest, configs = locked_model(root, bundle, bundle.parent / 'candidate_manifest.json')
    official = read_manifest(root / 'data/official/manifests/valid.csv', root / 'data/official')
    if len(official) != 5849:
        raise ValueError('Official validation manifest must contain 5849 samples')
    frozen_file = root / 'outputs/phase_g2/freeze/F01_validation.csv'
    frozen = validate_predictions(pd.read_csv(frozen_file), official, 5849)
    recomputed_file = root / 'outputs/phase_g2/F01_fullval_recomputed.csv'
    previous = json.loads((root / 'outputs/phase_g2/fullval_consistency.json').read_text(encoding='utf-8'))['results']['F01']
    if sha256(recomputed_file) != previous['recomputed_prediction_sha256']:
        raise ValueError('Phase G2 full-Val batch-1 diagnostic changed')
    recomputed = pd.read_csv(recomputed_file)
    if (recomputed.image_id.tolist() != official.image_id.tolist()
        or not np.array_equal(recomputed.label, official.label)):
        raise ValueError('Full-Val prediction order/labels disagree with official manifest')
    expected = frozen.probability.to_numpy(float)
    observed = recomputed.probability.to_numpy(float)
    ranks = ranking_changes(expected, observed, official.label.to_numpy(), official.image_id.tolist())
    # Check archived F01 exactly equals post-sigmoid probability averaging, not logit averaging.
    original_members = []
    for name in ('G01', 'B02'):
        member = validate_predictions(pd.read_csv(root / f'outputs/runs/{name}/validation.csv'), official, 5849)
        original_members.append(member)
    formula = .5 * original_members[0].probability.to_numpy(float) + .5 * original_members[1].probability.to_numpy(float)
    formula_error = float(np.max(np.abs(formula - expected)))
    if formula_error > 1e-12:
        raise ValueError('Frozen F01 is not the locked average of member probabilities')
    sample = pd.read_csv(root / 'outputs/phase_g2/freeze/t4_validation_105.csv', dtype={'image_id': str})
    measured, warm = sample.iloc[5:], sample.iloc[:5]
    measured[['image_id', 'path']].to_csv(out / 'val_measure_manifest.csv', index=False)
    for rows, target in ((measured, out/'images_100'), (warm, out/'images_5')):
        target.mkdir()
        for row in rows.itertuples():
            shutil.copyfile(root/'data/official'/row.path, target/Path(row.path).name)
    prediction_csv = out / 'predict_val_100.csv'
    subprocess.run([sys.executable, str(root/'predict.py'), '--checkpoint', str(bundle),
                    '--csv', str(out/'val_measure_manifest.csv'), '--root', str(root/'data/official'),
                    '--output', str(prediction_csv), '--device', 'cuda', '--precision', 'fp32',
                    '--threads', '4'], cwd=root, check=True)
    predicted = validate_submission(prediction_csv, measured.image_id, .5)
    predicted = predicted.rename(columns={'Image_name':'image_id','Probability':'probability'})
    subset_check = compare_frozen_to_inference(frozen, predicted, tolerance=1e-3, threshold=.5)
    subprocess.run([sys.executable, str(root/'cost.py'), str(prediction_csv),
                    str(out/'images_100'), str(out/'images_5'), '--checkpoint', str(bundle),
                    '--device', 'cuda', '--precision', 'fp32', '--output', str(out/'cost_val_smoke'),
                    '--smoke'], cwd=root, check=True)
    cost_report = json.loads((out/'cost_val_smoke/report.json').read_text(encoding='utf-8'))
    cache_check = reproduce_cached_batch(root, predictor, official,
        ['valid_04165.jpg', 'valid_04956.jpg'])
    result = {'status': 'BLOCKED' if ranks['all_pair_order_changes'] or not subset_check['passed']
              or any(cost_report['mismatches_by_round']) else 'PASS',
        'reason': 'Frozen Val reference and fixed batch-1 inference differ in probability ranking' if ranks['all_pair_order_changes'] else None,
        'test_images_or_labels_accessed_by_preflight': False, 'official_submission_generated': False,
        'source_git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'device': predictor.metadata['gpu'], 'precision': 'fp32',
        'frozen_bundle_sha256': sha256(bundle), 'loaded_files': predictor.metadata['inference_files'],
        'loaded_bytes': model_manifest['loaded_bytes'], 'model_config_and_preprocessing': configs,
        'formula': '0.5 * sigmoid(G01_logit) + 0.5 * sigmoid(B02_logit)',
        'frozen_formula_max_error': formula_error,
        'frozen_validation_sha256': sha256(frozen_file),
        'batch1_validation_sha256': sha256(recomputed_file),
        'validation_count': len(official),
        'frozen_auroc': float(roc_auc_score(official.label, expected)),
        'batch1_auroc': float(roc_auc_score(official.label, observed)),
        'max_probability_error': float(np.max(np.abs(expected-observed))),
        'pred_label_mismatches': int(np.sum((expected>=.5)!=(observed>=.5))),
        'ranking': ranks, 'predict_py_subset': subset_check,
        'predict_py_output_sha256': sha256(prediction_csv),
        'cost_py_local_smoke_mismatches_by_round': cost_report['mismatches_by_round'],
        'cost_py_local_smoke_official_protocol': cost_report['official_protocol'],
        'cached_batch_path_diagnostic': cache_check,
        't4_status': 'UNVERIFIED'}
    write_json(out/'gate_report.json', result)
    return result


def write_blocked_round(root, report):
    """Reserve Round 02 with an explicit failed gate; never create submission.csv."""
    root = Path(root)
    if report['status'] != 'BLOCKED':
        raise ValueError('This writer is only for a failed pre-submission gate')
    directory = root / 'submission/round_02_F01'
    directory.mkdir(parents=True, exist_ok=False)
    cost_report = json.loads((root/'outputs/f01_submission_gate/cost_val_smoke/report.json').read_text(encoding='utf-8'))
    manifest = {'status': 'BLOCKED', 'submission_created': False, 'submission_sha256': None,
        'submission_rows': None, 'reason': report['reason'],
        'code_commit_at_preflight': report['source_git_commit'],
        'gate_script_sha256': sha256(Path(__file__)),
        'frozen_bundle_sha256': report['frozen_bundle_sha256'],
        'loaded_files': report['loaded_files'], 'loaded_bytes': report['loaded_bytes'],
        'frozen_validation_sha256': report['frozen_validation_sha256'],
        'batch1_validation_sha256': report['batch1_validation_sha256'],
        'precision': 'fp32', 'threshold': .5,
        'environment': {'gpu': report['device'], 'python': cost_report['python'],
            'torch': cost_report['torch'], 'torchvision': cost_report['torchvision'],
            'cuda': cost_report['cuda']},
        't4_status': 'UNVERIFIED', 'official_test_inference_started': False,
        'gate_evidence_sha256': sha256(root/'outputs/f01_submission_gate/gate_report.json')}
    write_json(directory/'manifest.json', manifest)
    ranks = report['ranking']
    message = f"""# Round 02 F01：提交前一致性门禁

状态：**BLOCKED**。未生成 `submission.csv`，未运行正式 Test 推理，也未上传比赛网站。

冻结 F01 公式：`0.5 * sigmoid(G01_logit) + 0.5 * sigmoid(B02_logit)`；FP32，G01/B02 checkpoint SHA256 与冻结证据一致，模型加载文件合计 {report['loaded_bytes']:,} 字节。

官方完整 Val：{report['validation_count']:,} 张。冻结 AUROC {report['frozen_auroc']:.12f}；固定 batch=1 推理 AUROC {report['batch1_auroc']:.12f}。两者最大概率差 {report['max_probability_error']:.9f}，Pred_Label 变化 {report['pred_label_mismatches']}。精确概率排序变化 {ranks['all_pair_order_changes']:,} 对，其中正负类别 {ranks['positive_negative_order_changes']} 对、同类别 {ranks['same_class_order_changes']:,} 对。

`predict.py` 的 100 张 Val 诊断预测与冻结参考有 {report['predict_py_subset']['pairwise_order_mismatches']} 对排序变化；`cost.py --smoke` 在同一 100 张图像上三轮与 `predict.py` 的不匹配数为 {report['cost_py_local_smoke_mismatches_by_round']}。本地完整端到端平均 {cost_report['mean_ms']:.3f} ms，GPU 为 {report['device']}；这是 4060 诊断，不是 T4 验收。

两张已知排序变化图像在原始 JPEG 路径和冻结 PNG 缓存路径的预处理张量逐元素相同；按冻结 batch=32 方式推理可重现其原始成员概率。差异与验证批量形状有关，不能修改融合权重或单张概率去掩盖。

证据：`outputs/f01_submission_gate/gate_report.json`、`outputs/f01_submission_gate/predict_val_100.csv`、`outputs/f01_submission_gate/cost_val_smoke/report.json`。这些 Val 诊断数据只保留在本机。
"""
    (directory/'validation_report.md').write_text(message, encoding='utf-8')
    files = sorted(p for p in directory.iterdir() if p.is_file())
    (directory/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files), encoding='utf-8')
    if (directory/'submission.csv').exists():
        raise RuntimeError('A blocked round must not contain submission.csv')
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--finalize-existing-block', action='store_true',
        help='Write the blocked Round 02 evidence from an existing completed Val gate report')
    args = parser.parse_args()
    if args.finalize_existing_block:
        path = Path(args.root)/'outputs/f01_submission_gate/gate_report.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        report.pop('test_accessed', None)
        report['test_images_or_labels_accessed_by_preflight'] = False
        write_json(path, report)
    else:
        report = run(args.root)
    if report['status'] == 'BLOCKED':
        write_blocked_round(args.root, report)
    print(json.dumps({k: report[k] for k in ('status','reason','frozen_auroc','batch1_auroc',
        'max_probability_error','ranking','predict_py_subset','cost_py_local_smoke_mismatches_by_round')}, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(2)
