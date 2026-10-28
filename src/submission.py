from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ['Image_name', 'Probability', 'Pred_Label']


def validate_submission(path, expected_ids, threshold=.5):
    df = pd.read_csv(path, dtype={'Image_name': str}, keep_default_na=False, encoding='utf-8')
    if list(df.columns) != COLUMNS or df.empty:
        raise ValueError('Wrong submission columns or empty submission')
    if df.Image_name.duplicated().any() or list(df.Image_name) != list(expected_ids):
        raise ValueError('Image IDs/order do not match official manifest/sample submission')
    p = pd.to_numeric(df.Probability, errors='raise').to_numpy()
    labels = pd.to_numeric(df.Pred_Label, errors='raise').to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not np.array_equal(labels, (p >= threshold).astype(int)):
        raise ValueError('Pred_Label disagrees with frozen threshold')
    return df


def write_row(writer, name, probability, threshold):
    writer.writerow([name, format(probability, '.17g'), int(probability >= threshold)])
