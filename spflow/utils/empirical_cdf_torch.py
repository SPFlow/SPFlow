"""Algorithm to compute the empirical cumulative distribution function (CDF) for input data.

This module provides an efficient PyTorch-based implementation of the empirical
cumulative distribution function (ECDF) for multi-dimensional data. The ECDF
computes the proportion of data points less than or equal to each value,
handling missing values (NaN) appropriately.
"""

import torch
from torch import Tensor


def empirical_cdf(data: Tensor) -> Tensor:
    """Computes the empirical cumulative distribution function (CDF) for input data.

    Returns the empirical CDF values for each element in the input data,
    computed independently for each feature column. Missing values (NaN) are
    ignored and not counted towards the empirical CDF computation.

    Args:
        data: Two-dimensional Tensor containing empirical input values. Each
            row represents a sample and each column represents a different
            feature. All columns are ranked independently. Missing entries
            (NaN) are ignored and not counted towards the empirical CDF of
            the corresponding feature.

    Returns:
        Tensor containing the empirical CDF values for the input data,
        with the same shape as the input tensor.
    """
    # empirical cumulative distribution function (step function that increases by 1/N at each unique data step in order)
    # here: done using scipy's 'rankdata' function (preferred over numpy's argsort due to tie-breaking)

    nan_mask = torch.isnan(data)

    # rank data values from min to max
    ecd = torch.argsort(torch.argsort(data, dim=0), dim=0).float() + 1

    # set nan values to 0
    ecd[nan_mask] = 0

    # normalize rank values (not counting nan entries) to get ecd values
    n_entries = torch.sum(~nan_mask, dim=0, keepdim=True)
    n_entries[n_entries == 0] = 1

    # normalize rank values (not counting nan entries) to get ecd values
    ecd /= n_entries

    return ecd
