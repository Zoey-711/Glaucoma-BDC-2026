import argparse
import json

from src.research.queue import run_queue


def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',default='configs/research/first_batch_8gpu.yaml')
    p.add_argument('--archive');p.add_argument('--execute',action='store_true')
    p.add_argument('--status-file',default='outputs/research_queue/status.json')
    p.add_argument('--retry-failed',action='store_true',help='Retry failed jobs from validated last.pth')
    a=p.parse_args()
    print(json.dumps(run_queue(a.matrix,a.archive,not a.execute,a.status_file,a.retry_failed),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
