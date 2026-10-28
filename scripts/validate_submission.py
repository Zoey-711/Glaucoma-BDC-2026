import argparse
import pandas as pd
from src.submission import validate_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--submission', required=True)
    p.add_argument('--sample', required=True, help='Official sample CSV (Image_name column)')
    p.add_argument('--threshold', type=float, default=.5)
    a = p.parse_args()
    ids = pd.read_csv(a.sample, dtype={'Image_name': str}).Image_name
    validate_submission(a.submission, ids, a.threshold)
    print('Submission valid')


if __name__ == '__main__':
    main()
