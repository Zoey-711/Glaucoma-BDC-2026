import argparse

from src.research.folds import create_folds,assemble_oof


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create');c.add_argument('--train-manifest',default='data/official/manifests/train.csv')
    c.add_argument('--reviewed-groups',required=True);c.add_argument('--output',required=True)
    o=sub.add_parser('oof');o.add_argument('--fold-dirs',nargs=5,required=True)
    o.add_argument('--train-manifest',default='data/official/manifests/train.csv');o.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='create':print(create_folds(a.train_manifest,a.reviewed_groups,a.output))
    else:print(assemble_oof(a.fold_dirs,a.train_manifest,a.output))


if __name__=='__main__':main()
