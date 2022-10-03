"""
Created on September 7, 2022

@authors: Philipp Deibert
"""
import numpy as np
from scipy.stats import rankdata


def empirical_cdf(data: np.ndarray) -> np.ndarray:
    """TODO."""
    # empirical cumulative distribution function (step function that increases by 1/N at each unique data step in order)
    # here: done using scipy's 'rankdata' function (preferred over numpy's argsort due to tie-breaking)

    nan_mask = np.isnan(data)

    # rank data values from min to max
    ecd = rankdata(data, axis=0, method='max').astype(float)

    # set nan values to 0
    ecd[nan_mask] = 0

    # normalize rank values (not counting nan entries) to get ecd values
    n_entries = (~nan_mask).sum(axis=0, keepdims=True)
    n_entries[n_entries == 0] = 1

    # normalize rank values (not counting nan entries) to get ecd values
    ecd /= n_entries

    return ecd