import argparse

from src.research.analysis import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('result_dirs',nargs='*')
    p.add_argument('--val',default='data/official/manifests/valid.csv')
    p.add_argument('--output',default='outputs/research_analysis');a=p.parse_args()
    analyze(a.result_dirs,a.val,a.output)


if __name__=='__main__':main()
