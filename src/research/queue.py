"""One isolated process per GPU, fail-closed task queue and durable status events."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import yaml

from src.utils import write_json,sha256
from .config import ExperimentConfig,git_commit


STATES={'PENDING','RUNNING','COMPLETED','FAILED','BLOCKED'}


def utc():return datetime.now(timezone.utc).isoformat()


def load_matrix(path):
    data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    tasks=data['tasks'];ids=[t['experiment_id'] for t in tasks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate queue experiment IDs')
    outputs=[]
    for t in tasks:
        cfg=ExperimentConfig.load(t['config'])
        if cfg.experiment_id!=t['experiment_id']:raise ValueError('Task/config ID mismatch')
        if t['gpu'] not in range(8):raise ValueError('GPU ID must be 0..7')
        outputs.append(cfg.output_dir)
        if any(d not in ids for d in t.get('depends_on',[])):raise ValueError('Unknown dependency')
    if len(outputs)!=len(set(outputs)):raise ValueError('Output collision')
    return data


def preflight(task,completed,archive=None,allow_uncommitted=False):
    cfg=ExperimentConfig.load(task['config'])
    try:cfg.assert_execution_policy()
    except ValueError as exc:return f'BLOCKED: {exc}'
    if cfg.status=='BLOCKED':return f"BLOCKED: {cfg.extra.get('block_reason','config blocked')}"
    missing=[x for x in task.get('depends_on',[]) if x not in completed]
    if missing:return 'BLOCKED: dependencies not completed: '+','.join(missing)
    if (Path(cfg.output_dir)/'run_manifest.json').exists():
        previous=json.loads((Path(cfg.output_dir)/'run_manifest.json').read_text(encoding='utf-8'))
        if previous.get('status')=='COMPLETED':return 'BLOCKED: completed run already exists; never rerun or overwrite'
    if Path(cfg.output_dir).exists() and not (Path(cfg.output_dir)/'last.pth').exists():
        return 'BLOCKED: output exists without resume checkpoint'
    try:cfg.assert_pretraining()
    except Exception as exc:return f'BLOCKED: pretrained provenance: {exc}'
    if archive is None:return 'BLOCKED: official archive path/hash not supplied'
    try:cfg.assert_data('data/official/manifests/train.csv','data/official/manifests/valid.csv',archive)
    except Exception as exc:return f'BLOCKED: data version: {exc}'
    if not allow_uncommitted:
        from dataclasses import replace
        try:replace(cfg,git_commit=git_commit()).assert_code()
        except ValueError as exc:return f'BLOCKED: code version: {exc}'
    return None


def run_queue(matrix_path,archive=None,dry_run=True,status_file='outputs/research_queue/status.json',retry_failed=False):
    plan=load_matrix(matrix_path);tasks=list(plan['tasks']);status_path=Path(status_file)
    identity={'matrix_sha256':sha256(matrix_path),
              'configs':{t['experiment_id']:ExperimentConfig.load(t['config']).identity() for t in tasks},
              'git_commit':git_commit()}
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        if status['matrix']!=str(matrix_path):raise ValueError('Status belongs to another matrix')
        if status.get('identity')!=identity:raise ValueError('Queue matrix/config/code changed; preserve old status and use a new status file')
        if not dry_run:
            for eid,row in status['tasks'].items():
                if row['status']=='RUNNING':
                    row.update(status='BLOCKED',detail='Previous scheduler exited while task was running; inspect child PID and output before retry')
                elif retry_failed and row['status']=='FAILED':
                    row.update(status='PENDING',detail='Explicit retry requested; checkpoint and versions rechecked before launch')
    else:status={'matrix':str(matrix_path),'identity':identity,'created_utc':utc(),'tasks':{t['experiment_id']:{'status':'PENDING'} for t in tasks},'events':[]}
    running={};completed={eid for eid,row in status['tasks'].items() if row['status']=='COMPLETED'}
    def update(eid,state,detail=None,gpu=None,pid=None):
        if state not in STATES:raise ValueError('Invalid state')
        status['tasks'][eid]={'status':state,'detail':detail,'gpu':gpu,'pid':pid,'updated_utc':utc()}
        status['events'].append({'time':utc(),'experiment_id':eid,'status':state,'detail':detail})
        if not dry_run:write_json(status_path,status)
    if dry_run:
        report=[]
        for task in tasks:
            eid=task['experiment_id'];reason=preflight(task,completed,archive)
            report.append({'experiment_id':eid,'gpu':task['gpu'],
                           'status':'BLOCKED' if reason else 'READY','reason':reason,'config':task['config']})
        return report
    while True:
        for eid,(process,handle,gpu) in list(running.items()):
            code=process.poll()
            if code is None:continue
            handle.close();del running[eid]
            state='COMPLETED' if code==0 else 'FAILED'
            if state=='COMPLETED':completed.add(eid)
            update(eid,state,f'exit_code={code}',gpu,process.pid)
        launched=False
        for task in tasks:
            eid=task['experiment_id'];current=status['tasks'][eid]['status']
            if current!='PENDING':continue
            if task['gpu'] in [gpu for _,_,gpu in running.values()]:continue
            reason=preflight(task,completed,archive)
            if reason:
                if reason.startswith('BLOCKED: dependencies') and any(status['tasks'][d]['status'] in {'PENDING','RUNNING'} for d in task.get('depends_on',[])):
                    continue
                update(eid,'BLOCKED',reason,task['gpu']);continue
            cfg=ExperimentConfig.load(task['config'])
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(task['gpu'])
            log_path=status_path.parent/f'{eid}.scheduler.log';log_path.parent.mkdir(parents=True,exist_ok=True)
            handle=log_path.open('a',encoding='utf-8')
            args=[sys.executable,'-u','-m','scripts.run_research','--config',task['config'],'--archive',archive]
            if Path(cfg.output_dir).exists():args.append('--resume')
            process=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT)
            running[eid]=(process,handle,task['gpu']);update(eid,'RUNNING',None,task['gpu'],process.pid);launched=True
        if not running and not any(v['status']=='PENDING' for v in status['tasks'].values()):break
        if not running and not launched:
            for eid,row in status['tasks'].items():
                if row['status']=='PENDING':update(eid,'BLOCKED','Unmet dependency cycle or failed prerequisite')
            break
        time.sleep(2)
    return status
