import copy
from pathlib import Path

import yaml

from src.registry import Registry
from src.utils import load_config


def matrix():
    base = load_config('configs/official.local.yaml')
    base.update(seed=42, batch_size=32, num_workers=6, epochs=30, patience=5,
                loss={'name': 'bce'}, sampler='shuffle', augmentation='basic',
                preprocess='resize', cache=True, torch_threads=4, log_every=100,
                experiment_id='B02', output='outputs/runs/B02',
                plan={'stage': 'B', 'kind': 'train', 'reference': None, 'factor': 'backbone', 'initial_status': 'planned'})
    base['model'] = {'name': 'efficientnet_b0', 'backend': 'timm', 'architecture': 'efficientnet_b0.ra_in1k',
                     'weights_manifest': 'configs/weights_b02.local.json'}
    configs = []
    def add(eid, updates=None, **plan):
        c = copy.deepcopy(base)
        c.update(experiment_id=eid, output=f'outputs/runs/{eid}')
        if updates:
            c.update(updates)
        c['plan'].update(stage=eid[0], **plan)
        configs.append(c)
        return c
    models = [('B01', 'resnet50', 'resnet50.a1_in1k'), ('B02', 'efficientnet_b0', 'efficientnet_b0.ra_in1k'),
              ('B03', 'convnext_tiny', 'convnext_tiny.fb_in1k'), ('B04', 'efficientnet_b2', 'efficientnet_b2.ra_in1k'),
              ('B05', 'swin_tiny', 'swin_tiny_patch4_window7_224.ms_in1k')]
    for eid, name, arch in models:
        m = {'name': name, 'backend': 'timm', 'architecture': arch,
             'weights_manifest': f'configs/weights_{eid.lower()}.local.json'}
        add(eid, {'model': m}, reference=None if eid == 'B02' else 'B02')
    for eid, loss in [('I01', 'bce'), ('I02', 'weighted_bce'), ('I03', 'focal'), ('I04', 'asl'), ('I05', 'bce'), ('I06', 'class_balanced')]:
        params = {'name': loss}
        if loss == 'focal': params.update(gamma=2.0, alpha=.25)
        if loss == 'asl': params.update(gamma_pos=1.0, gamma_neg=4.0, clip=.05)
        if loss == 'class_balanced': params.update(beta=.9999)
        add(eid, {'loss': params, 'sampler': 'weighted' if eid == 'I05' else 'shuffle'},
            reference='B02', factor='sampler' if eid == 'I05' else 'loss')
    add('P01', reference='B02', factor='preprocess')
    add('P02', {'preprocess': 'black_border'}, reference='P01', factor='preprocess')
    add('P03', {'image_size': 320}, reference='P01', factor='image_size', resolution_variant=320)
    add('P03_384', {'image_size': 384}, reference='P01', factor='image_size', resolution_variant=384)
    add('P04', {'augmentation': 'color'}, reference='P01', factor='augmentation')
    for eid, method in [('P05', 'optic_disc_roi'), ('P06', 'global_local')]:
        add(eid, {'preprocess': method}, reference='P01', factor='preprocess', initial_status='skipped',
            reason='No reliable official-data optic-disc annotations/localizer; no fabricated ROI')
    for eid, method, n in [('F01', 'single', 1), ('F02', 'ensemble', 2), ('F03', 'multiseed', 3),
                           ('F04', 'ensemble', 3), ('F05', 'tta', 1), ('F06', 'distill', 1)]:
        add(eid, {'selection': {'method': method, 'members': [], 'required_members': n,
                               'tta': ['identity', 'hflip'] if method == 'tta' else ['identity']}},
            kind='distill' if method == 'distill' else 'evaluate', reference='F01' if eid != 'F01' else None,
            factor='inference_strategy', initial_status='blocked', reason='Bind completed validation-selected candidates first')
    for eid, seed in [('R01', 42), ('R02', 2026), ('R03', 3407)]:
        add(eid, {'seed': seed, 'candidate': None}, reference='R01' if eid != 'R01' else None,
            factor='seed', initial_status='blocked', reason='Freeze one best single-model candidate before seed replication')
    for eid, method in [('D01', 'precision'), ('D02', 'channels_last'), ('D03', 'package')]:
        add(eid, {'candidate': None, 'deployment': {'method': method, 'precisions': ['fp32', 'fp16'],
            'max_bytes': 500_000_000, 'max_t4_ms': 100}}, kind='deployment', reference='D01' if eid != 'D01' else None,
            factor='runtime', initial_status='blocked', reason='Requires fixed final candidate; T4 evidence mandatory for final packaging')
    return configs


def main():
    folder = Path('configs/experiments')
    folder.mkdir(parents=True, exist_ok=True)
    registry = Registry()
    for c in matrix():
        path = folder / (c['experiment_id'] + '.yaml')
        if path.exists() and load_config(path) != c:
            raise ValueError(f'Refusing to overwrite edited config {path}')
        if not path.exists():
            path.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding='utf-8')
        registry.register(c)
    registry.export()
    print(f'Created/verified {len(matrix())} independent configs; no training started')


if __name__ == '__main__':
    main()
