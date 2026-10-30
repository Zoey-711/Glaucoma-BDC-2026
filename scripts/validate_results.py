import argparse
import json

from src.research.results import validate_result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--official-val',default='data/official/manifests/valid.csv')
    p.add_argument('--config');p.add_argument('--official-train',default='data/official/manifests/train.csv');a=p.parse_args()
    print(json.dumps(validate_result(a.directory,a.official_val,a.config,
        official_train=a.official_train),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
