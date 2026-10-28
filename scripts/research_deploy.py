import argparse
import json

from src.research.deployment import compare_precisions,local_latency,certify_t4_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--val-manifest',default='data/official/manifests/valid.csv')
    p.add_argument('--root',default='data/official');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--latency',action='store_true',help='4090/other local timing, never T4 certification')
    p.add_argument('--t4-report',help='Validate an actual cost.py NVIDIA T4 report')
    a=p.parse_args()
    result=(certify_t4_report(a.checkpoint,a.t4_report) if a.t4_report else
            local_latency(a.checkpoint,a.val_manifest,a.root,a.output,a.count) if a.latency else
            compare_precisions(a.checkpoint,a.val_manifest,a.root,a.output,a.count))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
