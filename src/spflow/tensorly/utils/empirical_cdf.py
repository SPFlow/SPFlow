"""Algorithm to compute the empirical cummulative distribution function (CDF) for input data.

Typical usage example:

    ecdf_values = empirical_cdf(data)
"""
import numpy as np
import tensorly as tl
from scipy.stats import rankdata


def empirical_cdf(data: tl.tensor) -> tl.tensor:
    """Computes the empirical cummulative distribution function (CDF) for specified input data.

    Returns the values of all input data according to the empirical cummulative distribution function (CDF) computed from said data.

    Args:
        data:
            Two-dimensional NumPy array containing empirical input values. Each row is regarded as a sample and each column as a different feature.
            All columns (i.e., features) are ranked independently. Missing entries (i.e., NaN) are ignored and are not counted towards the empirical
            CDF of the corresponding feature.

    Returns:
        Numpy array containing the empirical CDF values for the specified input.
    """
    # empirical cumulative distribution function (step function that increases by 1/N at each unique data step in order)
    # here: done using scipy's 'rankdata' function (preferred over numpy's argsort due to tie-breaking)

    nan_mask = np.isnan(data)

    # rank data values from min to max
    ecd = rankdata(data, axis=0, method="max").astype(float)

    # set nan values to 0
    ecd[nan_mask] = 0

    # normalize rank values (not counting nan entries) to get ecd values
    n_entries = tl.sum(~nan_mask, axis=0, keepdims=True)
    n_entries[n_entries == 0] = 1

    # normalize rank values (not counting nan entries) to get ecd values
    ecd /= n_entries

    return ecd
