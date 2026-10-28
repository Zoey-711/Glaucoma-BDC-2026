import argparse
import json

from src.research.results import package_result,validate_result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True)
    p.add_argument('--official-val',default='data/official/manifests/valid.csv');a=p.parse_args()
    validate_result(a.directory,a.official_val)
    print(json.dumps({'bundle':str(package_result(a.directory,a.output))}))


if __name__=='__main__':main()
