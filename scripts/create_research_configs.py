"""Create 20 controlled experiment groups plus V/F/D modules; never runs training."""
import copy
import dataclasses
import json
from pathlib import Path

import yaml

from src.research.config import ExperimentConfig,manifest_semantic_hash
from src.utils import sha256


def existing(name):
    p=Path(f'configs/weights_{name}.local.json')
    return json.loads(p.read_text(encoding='utf-8'))


def matrix():
    b02,g01=existing('b02'),existing('g01')
    summary=json.loads(Path('outputs/official_ingest/summary.json').read_text(encoding='utf-8'))
    hashes={'train':manifest_semantic_hash('data/official/manifests/train.csv'),'val':manifest_semantic_hash('data/official/manifests/valid.csv'),
            'archive':summary['archive_sha256']}
    base=dict(experiment_id='G02',parent_experiment_id='G01',backbone='convnext_tiny',
              pretrained_source=g01['source_url'],pretrained_sha256=g01['sha256'],license='Apache-2.0',
              weights_manifest='configs/weights_g01.local.json',image_size=224,crop_policy='resize',
              normalization='imagenet',optimizer='AdamW',backbone_lr=.0003,head_lr=.0003,weight_decay=.0001,
              scheduler='cosine',warmup=0,epochs=30,patience=5,loss={'name':'bce'},sampler='shuffle',
              augmentation={'horizontal_flip':True,'rotation':10},batch_size=32,gradient_accumulation=1,
              seed=42,fold=None,data_manifest_hash=hashes,output_dir='',config_hash=None,git_commit=None,
              status='PENDING',dependencies=[],extra={'num_workers':4,'torch_threads':4,'grad_clip_norm':1.})
    rows=[]
    def add(eid,**changes):
        c=copy.deepcopy(base);c.update(experiment_id=eid,output_dir=f'results/{eid}/seed_42/fold_none')
        for key,val in changes.items():c[key]=val
        if c['backbone'] not in {'convnext_tiny','efficientnet_b0'}:
            c.update(pretrained_source=None,pretrained_sha256=None,license=None,weights_manifest=None,status='BLOCKED')
            c['extra']['block_reason']='New backbone weight source/license/SHA256 not yet reviewed'
        rows.append(c)
    add('G02',image_size=384)
    add('G03',parent_experiment_id='B02',backbone='efficientnet_b0',image_size=384,
        pretrained_source=b02['source_url'],pretrained_sha256=b02['sha256'],weights_manifest='configs/weights_b02.local.json')
    add('G04',parent_experiment_id='G02',image_size=512,batch_size=16,gradient_accumulation=2,
        extra={**base['extra'],'batch_change_warning':'Effective batch 32; microbatch differs, controlled comparison needs matched G02 batch16x2'})
    add('G05',parent_experiment_id='G03',backbone='efficientnet_v2_s',image_size=384)
    add('G06',parent_experiment_id='G02',backbone='convnext_small',image_size=384)
    add('G07',parent_experiment_id='G02',backbone='swin_tiny',image_size=384)
    add('G08',parent_experiment_id='G02',backbone='densenet121',image_size=384)
    add('G09',parent_experiment_id='G05',backbone='efficientnet_v2_s',image_size=512,
        batch_size=16,gradient_accumulation=2)
    add('L01',loss={'name':'weighted_bce','weight_mode':'ratio'})
    add('L02',loss={'name':'focal','alpha':.25,'gamma':2})
    add('L03',loss={'name':'pairwise_auc','lambda':.05})
    add('L04',sampler='balanced')
    add('S01',crop_policy='fov',extra={**base['extra'],'fov':{'threshold':12,'margin':.04,'min_fraction':.15}})
    add('S02',extra={**base['extra'],'enhancement':'lab_clahe_gamma',
                     'enhancement_params':{'clahe':True,'gamma':.95,'contrast':1.05}})
    add('S03',crop_policy='optic_disc',status='BLOCKED',dependencies=['locator_reviewed'],
        extra={**base['extra'],'block_reason':'No reviewed optic disc locator or validated reliability'})
    add('S04',status='BLOCKED',dependencies=['S03'],
        extra={**base['extra'],'block_reason':'Dual branch requires S03 ROI and local EfficientNet weights',
               'modes':['global_only','local_only','dual_branch','probability_fusion'],
               'branch_mode':'dual_branch','local_weights_manifest':'configs/weights_b02.local.json',
               'require_roi':True})
    add('R01',augmentation={'horizontal_flip':True,'rotation':10,'color_jitter':.08})
    add('R02',augmentation={'horizontal_flip':True,'rotation':10,'blur_sigma':.6})
    add('R03',extra={**base['extra'],'mixup_alpha':.2})
    add('R04',status='BLOCKED',dependencies=['ssl_train_only_review'],
        extra={**base['extra'],'block_reason':'SSL runner implemented; requires separate official-Train-only source gate',
               'ssl_method':'SimSiam','initialization':'imagenet_then_ssl_then_finetune'})
    variants=[]
    def variant(parent,suffix,**changes):
        c=copy.deepcopy(next(r for r in rows if r['experiment_id']==parent));eid=f'{parent}_{suffix}'
        c.update(experiment_id=eid,output_dir=f'results/{eid}/seed_42/fold_none',parent_experiment_id=parent)
        c.update(changes);variants.append(c)
    variant('L01','BCE',loss={'name':'bce'})
    variant('L01','SQRT',loss={'name':'weighted_bce','weight_mode':'sqrt'})
    variant('L02','GAMMA1',loss={'name':'focal','alpha':.25,'gamma':1})
    variant('L03','LAMBDA0',loss={'name':'pairwise_auc','lambda':0})
    variant('L03','LAMBDA02',loss={'name':'pairwise_auc','lambda':.2})
    variant('R01','MODERATE',augmentation={'horizontal_flip':True,'rotation':10,'color_jitter':.16})
    variant('R01','NONE',augmentation={'horizontal_flip':False,'rotation':0})
    variant('R02','JPEG',augmentation={'horizontal_flip':True,'rotation':10,'jpeg_quality':75})
    variant('R02','NONE',augmentation={'horizontal_flip':False,'rotation':0})
    variant('R03','A04',extra={**base['extra'],'mixup_alpha':.4})
    for suffix,mode in [('GLOBAL','global_only'),('LOCAL','local_only')]:
        variant('S04',suffix,extra={**next(r for r in rows if r['experiment_id']=='S04')['extra'],
            'branch_mode':mode})
    for eid,initialization in [('R04_DIRECT','imagenet_direct'),('R04_RANDOM_SSL','random_then_ssl_then_finetune')]:
        variant('R04',eid.removeprefix('R04_'),extra={**base['extra'],'ssl_method':'SimSiam','initialization':initialization,
            'block_reason':'Requires official-Train-only SSL source gate'})
    random_variant=next(v for v in variants if v['experiment_id']=='R04_RANDOM_SSL')
    random_variant.update(pretrained_source=None,pretrained_sha256=None,license=None,weights_manifest=None)
    extras=[]
    def extra(eid,**changes):
        c=copy.deepcopy(base);c.update(experiment_id=eid,output_dir=f'results/{eid}/seed_42/fold_none',
            status='BLOCKED',parent_experiment_id=None,dependencies=[])
        c.update(changes)
        extras.append(c)
    extra('V01',dependencies=['shortlist_selected'],extra={**base['extra'],'seeds':[42,2026,3407],
          'block_reason':'Need selected candidate after first stage'})
    extra('V02',dependencies=['reviewed_duplicate_groups'],extra={**base['extra'],'folds':5,
          'block_reason':'No patient ID; reviewed duplicate groups not available'})
    extra('F01',dependencies=['V01'],extra={**base['extra'],'mode':'probability','weights':'equal',
          'block_reason':'Requires frozen completed member predictions'})
    extra('F02',dependencies=['V01'],extra={**base['extra'],'mode':'logit','weights':'equal',
          'block_reason':'Requires frozen completed member predictions'})
    extra('D01',dependencies=['approved_teacher'],extra={**base['extra'],'distill_alpha':.5,
          'temperature':2.,'block_reason':'Needs approved official-Train teacher'})
    extra('D02',dependencies=['final_candidate','T4'],extra={**base['extra'],'block_reason':'T4 device and final candidate not present'})
    references=[]
    for eid,parent,backbone,proof in [('G01_REPRO','G01','convnext_tiny',g01),('B02_REPRO','B02','efficientnet_b0',b02)]:
        c=copy.deepcopy(base)
        c.update(experiment_id=eid,parent_experiment_id=parent,backbone=backbone,
            pretrained_source=proof['source_url'],pretrained_sha256=proof['sha256'],
            weights_manifest=f'configs/weights_{parent.lower()}.local.json',
            output_dir=f'results/{eid}/seed_42/fold_none')
        references.append(c)
    return rows,variants,extras,references


def main():
    directory=Path('configs/research');directory.mkdir(exist_ok=True)
    rows,variants,extras,references=matrix()
    if len(rows)!=20 or len(extras)!=6:raise AssertionError('Matrix size mismatch')
    for c in rows+variants+extras+references:
        c['config_hash']=ExperimentConfig(**c).identity()
        path=directory/f"{c['experiment_id']}.yaml"
        if path.exists():
            old=yaml.safe_load(path.read_text(encoding='utf-8'))
            if old==c:continue
            if Path(c['output_dir']).exists():raise ValueError(f'Config has run; version its ID: {path}')
        path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(json.dumps({'groups':len(rows),'variants':len(variants),'review_and_deployment':len(extras),'runner_references':len(references)}))


if __name__=='__main__':main()
