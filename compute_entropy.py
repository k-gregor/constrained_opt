import pandas as pd
import numpy as np


def __compute_entropy_for_row(row):
    total_sum = row.sum()
    if total_sum > 0:
        row = row/total_sum
        return -row * np.log2(row, out=np.zeros_like(row), where=(row != 0))  # avoid division by 0
    else:
        return 0*row


def compute_entropy(df: pd.DataFrame):
    """
    Computes the Shannon entropy of a dataframe. We assume the values of the df to contain the counts, so we sum up to get the total count by which we divide.
    """
    return df.apply(__compute_entropy_for_row, axis=1).sum(axis=1)
