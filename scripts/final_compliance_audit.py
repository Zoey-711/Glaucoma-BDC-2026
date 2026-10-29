"""Read-only collection of local compliance evidence. Never runs training/inference."""
import json
import subprocess
from pathlib import Path

import pandas as pd

from scripts.accept_t4 import preflight
from scripts.s01_qc import guard
from src.model import verified_weights
from src.submission import validate_submission
from src.utils import sha256, write_json

OUT=Path('outputs/final_compliance_audit')


def git(*args):
    return subprocess.check_output(['git',*args],text=True,errors='replace')


def main():
    if OUT.exists():raise FileExistsError('Audit output exists; preserve prior evidence')
    OUT.mkdir(parents=True)
    result={'scope':'local workspace, tracked Git history, supplied PDF files; no T4/server access',
            'git_commit':git('rev-parse','HEAD').strip(),'frozen_files_verified':guard(),
            'rules':{},'weights':{},'loaded_model_files':{},'submission':{},'git':{}}
    for name,path in [('main','F:/计挑赛/大数据规则/aa80bececc2ee40e69cde5f8f2a0eaec.pdf'),
                      ('supplement','F:/计挑赛/大数据规则/b23d6a1f19602da2a9bd4048108175d0.pdf')]:
        p=Path(path);result['rules'][name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)}
    for label,path in [('B02_EfficientNet_B0','configs/weights_b02.local.json'),
                       ('G01_G02_M02_ConvNeXt_Tiny','configs/weights_g01.local.json')]:
        m=verified_weights(path)
        result['weights'][label]={'manifest':path,'architecture':m['architecture'],'dataset':m['dataset'],
            'source_url':m['source_url'],'model_card_url':m['training_description_url'],
            'license_url':m['license_url'],'license_text_path':m['license_file'],
            'license_sha256':m['license_sha256'],'model_card_path':m['model_card_file'],
            'model_card_sha256':m['model_card_sha256'],'upstream_license_path':m.get('upstream_license_file'),
            'upstream_license_sha256':m.get('upstream_license_sha256'),
            'weight_bytes':Path(m['path']).stat().st_size,'weight_sha256':m['sha256'],
            'medical_pretraining':m['medical_pretraining'],
            'redistribution_declared':m['allows_competition_and_redistribution'],
            'gated_or_nc_declared':m['non_commercial_or_gated']}
    paths={'B02':'outputs/runs/B02/best.pt','G01':'outputs/runs/G01/best.pt',
           'G02':'results/G02/seed_42/fold_none/best.pth',
           'M02b':'results/M02b/seed_42/fold_none/model.json',
           'M02c':'results/M02c/seed_42/fold_none/best.pth',
           'M02d':'results/M02d/seed_42/fold_none/best.pth',
           'L03b':'results/L03b/seed_42/fold_none/best.pt'}
    for name,p in paths.items():
        f=Path(p);result['loaded_model_files'][name]={'path':p,'bytes':f.stat().st_size,
            'mb_decimal':round(f.stat().st_size/1e6,6),'sha256':sha256(f),
            'under_500mb_single_file':f.stat().st_size<=500_000_000,
            'scope':'local checkpoint only; final deployed bundle not assembled'}
    submission=Path('submission/round_01_B02/submission.csv')
    meta=json.loads(Path(str(submission)+'.metadata.json').read_text(encoding='utf-8'))
    raw_test=pd.read_csv('data/official/glaucoma_first/test.csv',dtype=str,keep_default_na=False)
    expected=raw_test.Image_name
    frame=validate_submission(submission,expected,meta['threshold'])
    test_manifest='data/official/manifests/test.csv'
    pre=preflight(submission,paths['B02'],test_manifest)
    raw=submission.read_bytes()
    result['submission']={'path':str(submission),'rows':len(frame),'columns':list(frame.columns),
        'raw_test_columns':list(raw_test.columns),'utf8_without_bom':not raw.startswith(b'\xef\xbb\xbf'),
        'matches_raw_test_order':list(frame.Image_name)==list(expected),'probability_min':float(frame.Probability.min()),
        'probability_max':float(frame.Probability.max()),
        'pred_label_matches_threshold':True,'submission_sha256':sha256(submission),
        'metadata_sha256':sha256(str(submission)+'.metadata.json'),
        'metadata_checkpoint_matches_local':meta['checkpoint_sha256']==result['loaded_model_files']['B02']['sha256'],
        'metadata_csv_matches_local':meta['submission_sha256']==sha256(submission),
        'metadata_uploaded_flag':meta.get('uploaded'),
        'test_manifest_sha256':sha256(test_manifest),'t4_preflight':pre}
    tracked=git('ls-files').splitlines()
    history=[line.split(' ',1)[1] for line in git('rev-list','--objects','--all').splitlines() if ' ' in line]
    forbidden=['data/official/','data/pretrained/','results/','outputs/runs/','submission/round_']
    suspicious_suffix=('.pem','.key','.p12','.safetensors','.pth','.pt','.onnx')
    def flagged(names):return [n for n in names if any(s in n.replace('\\','/') for s in forbidden) or n.lower().endswith(suspicious_suffix)]
    result['git']={'tracked_file_count':len(tracked),'tracked_forbidden_paths':flagged(tracked),
        'historical_forbidden_paths':flagged(history),'branch':git('branch','--show-current').strip(),
        'remote':git('remote','get-url','origin').strip()}
    # Do not read or print potential secret values; scan only tracked paths and headers.
    scan=subprocess.run(['git','grep','-I','-l','-e','BEGIN OPENSSH PRIVATE KEY',
                         '-e','BEGIN RSA PRIVATE KEY','-e','github_pat_','-e','ghp_'],
                        capture_output=True,text=True,errors='replace')
    result['git']['tracked_secret_indicator_paths']=scan.stdout.splitlines()
    result['git']['secret_scan_exit_code']=scan.returncode
    result['t4_report_exists']=any(Path(p).is_file() for p in ['outputs/cost_t4/report.json','outputs/final_t4/report.json'])
    result['final_material_zip_exists']=bool(list(Path('.').glob('*最终提交材料.zip')))
    result['organizer_collaboration_permission_file_found']=False
    write_json(OUT/'local_evidence.json',result)
    print(json.dumps({'evidence':str(OUT/'local_evidence.json'),
        'frozen_files_verified':result['frozen_files_verified'],
        'submission_rows':len(frame),'tracked_forbidden':len(result['git']['tracked_forbidden_paths']),
        'history_forbidden':len(result['git']['historical_forbidden_paths'])},ensure_ascii=False))


if __name__=='__main__':main()
