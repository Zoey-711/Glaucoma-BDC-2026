"""Explicit one-process-per-GPU H03/H04 launcher; never auto-starts tasks."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.research.config import ExperimentConfig,git_commit
from src.utils import sha256,write_json


CONFIGS={name:f'configs/h/{name}.yaml' for name in
    ('H03_G01_HEAD','H03_MATCHED_MLP','H03_MULTISCALE','H04_RANDOM','H04_BALANCED')}
LOCK_DIR=Path('outputs/h_gpu_locks')


def gpu_state():
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in data.splitlines():
        items=[x.strip() for x in line.split(',')]
        rows.append({'index':int(items[0]),'name':items[1],
            'memory_used_mib':int(items[2]),'utilization_pct':int(items[3])})
    return rows


def status():
    result=[]
    for eid,path in CONFIGS.items():
        cfg=ExperimentConfig.load(path)
        directory=Path(cfg.output_dir)
        marker=LOCK_DIR/f'{eid}.json'
        state='PENDING';reason=None
        if (directory/'run_manifest.json').exists():
            data=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
            state={'COMPLETED':'DONE','RUNNING':'RUNNING','FAILED':'FAILED'}.get(data['status'],data['status'])
            if state=='RUNNING' and marker.exists():
                launch_data=json.loads(marker.read_text(encoding='utf-8'))
                if not _alive(launch_data.get('pid')):
                    state='FAILED';reason='Training process exited without completing manifest'
        elif marker.exists():
            data=json.loads(marker.read_text(encoding='utf-8'))
            state='RUNNING' if data.get('pid') and _alive(data['pid']) else 'FAILED'
            reason='Launcher exited; inspect log and checkpoint before any resume' if state=='FAILED' else None
        elif cfg.status=='BLOCKED':state='BLOCKED';reason=cfg.extra.get('block_reason')
        result.append({'experiment_id':eid,'state':state,'gpu':json.loads(marker.read_text(encoding='utf-8')).get('gpu') if marker.exists() else None,
            'result_dir':cfg.output_dir,'reason':reason})
    return result


def release(experiment_id):
    """Explicitly free a finished GPU lock; never kill or overwrite a task."""
    if experiment_id not in CONFIGS:raise ValueError('Unknown H experiment ID')
    marker=LOCK_DIR/f'{experiment_id}.json'
    if not marker.exists():raise FileNotFoundError(marker)
    data=json.loads(marker.read_text(encoding='utf-8'))
    if _alive(data.get('pid')):raise ValueError('Task process still alive')
    state=next(row['state'] for row in status() if row['experiment_id']==experiment_id)
    if state not in {'DONE','FAILED'}:raise ValueError('Task is not finished')
    gpu_lock=LOCK_DIR/f"gpu_{data['gpu']}.lock"
    if gpu_lock.read_text(encoding='utf-8')!=experiment_id:
        raise ValueError('GPU lock owner mismatch')
    gpu_lock.unlink();marker.unlink()
    return {'experiment_id':experiment_id,'state':state,'gpu_lock_released':True}


def _alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,ValueError,TypeError):return False


def launch(experiment_id,gpu,archive):
    if experiment_id not in CONFIGS:raise ValueError('Unknown H experiment ID')
    cfg=ExperimentConfig.load(CONFIGS[experiment_id])
    if cfg.status=='BLOCKED':raise ValueError('Config is BLOCKED; review legal weights before any run')
    cfg.assert_execution_policy();cfg.assert_pretraining()
    cfg.assert_data('data/official/manifests/train.csv','data/official/manifests/valid.csv',archive)
    from dataclasses import replace
    replace(cfg,git_commit=git_commit()).assert_code()
    if Path(cfg.output_dir).exists():raise FileExistsError('Result exists; never overwrite or silently resume')
    gpu=next((row for row in gpu_state() if row['index']==gpu),None)
    if not gpu:raise ValueError('Requested GPU is absent')
    if gpu['memory_used_mib']>1500 or gpu['utilization_pct']>15:
        raise ValueError(f"GPU appears occupied: {gpu}; inspect current server tasks first")
    LOCK_DIR.mkdir(parents=True,exist_ok=True)
    lock=LOCK_DIR/f'{experiment_id}.json'
    gpu_lock=LOCK_DIR/f'gpu_{gpu["index"]}.lock'
    with gpu_lock.open('x',encoding='utf-8') as handle:
        handle.write(experiment_id)
    try:
        if lock.exists():raise FileExistsError(lock)
        log=LOCK_DIR/f'{experiment_id}.launcher.log'
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu['index'])
        with log.open('x',encoding='utf-8') as handle:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.run_research',
                '--config',CONFIGS[experiment_id],'--archive',str(archive)],
                env=env,stdout=handle,stderr=subprocess.STDOUT,
                start_new_session=os.name!='nt')
        write_json(lock,{'experiment_id':experiment_id,'gpu':gpu['index'],'pid':child.pid,
            'config_hash':cfg.identity(),'git_commit':git_commit(),
            'archive_sha256':sha256(archive),'log':str(log)})
        return {'state':'RUNNING','pid':child.pid,'gpu':gpu['index'],'log':str(log)}
    except Exception:
        gpu_lock.unlink(missing_ok=True)
        raise


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    r=sub.add_parser('release');r.add_argument('--id',required=True)
    q=sub.add_parser('launch');q.add_argument('--id',required=True);q.add_argument('--gpu',type=int,required=True)
    q.add_argument('--archive',required=True)
    args=p.parse_args()
    result=(status() if args.action=='status' else
            release(args.id) if args.action=='release' else launch(args.id,args.gpu,args.archive))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
