"""Check frozen checkpoint loading and one official Val prediction without training."""

import gc
import json
from pathlib import Path

import pandas as pd
import torch

from src.data import open_rgb, read_manifest
from src.inference import Predictor
from src.utils import sha256


MODELS = {
    'B02': ('outputs/runs/B02/best.pt', 'outputs/runs/B02/validation.csv'),
    'G01': ('outputs/runs/G01/best.pt', 'outputs/runs/G01/validation.csv'),
    'G02': ('results/G02/seed_42/fold_none/best.pth',
            'results/G02/seed_42/fold_none/validation.csv'),
}


def main():
    val = read_manifest('data/official/manifests/valid.csv', 'data/official')
    first = val.iloc[0]
    report = {'sample_id': first.image_id, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'models': {}, 't4_measured': False}
    for name, (checkpoint, prediction_csv) in MODELS.items():
        predictor = Predictor(checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
        probability = predictor.predict_image(open_rgb(first.resolved_path))
        saved = pd.read_csv(prediction_csv).iloc[0]
        saved_id = saved['Image_name'] if 'Image_name' in saved else saved['image_id']
        saved_probability = saved['Probability'] if 'Probability' in saved else saved['probability']
        if saved_id != first.image_id:
            raise ValueError(f'{name}: validation sample order changed')
        files = predictor.metadata['inference_files']
        total = sum(x['bytes'] for x in files)
        report['models'][name] = {
            'checkpoint_sha256': sha256(checkpoint), 'loaded_file_count': len(files),
            'loaded_model_bytes': total, 'within_500mb': total <= 500_000_000,
            'first_val_saved_probability': float(saved_probability),
            'first_val_replayed_probability': float(probability),
            'absolute_replay_difference': abs(float(probability) - float(saved_probability)),
            'precision': predictor.precision, 'model_eval': not predictor.model.training,
            'all_parameters_frozen': not any(p.requires_grad for p in predictor.model.parameters()),
            'image_size': predictor.config['image_size'],
        }
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    target = Path('outputs/official_2026_audit/inference_check.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
