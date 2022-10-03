"""
Created on September 26, 2022

@authors: Philipp Deibert
"""
from spflow.torch.utils.rankdata import rankdata
import torch


def empirical_cdf(data: torch.Tensor) -> torch.Tensor:
    """TODO."""
    # empirical cumulative distribution function (step function that increases by 1/N at each unique data step in order)
    # here: done using scipy's 'rankdata' function (preferred over numpy's argsort due to tie-breaking)

    nan_mask = torch.isnan(data)

    # rank data values from min to max (with tie-breaking)
    ecd = rankdata(data, method='max').type(torch.get_default_dtype())

    # set nan values to 0
    ecd[nan_mask] = 0

    # normalize rank values (not counting nan entries) to get ecd values
    n_entries = (~nan_mask).sum(dim=0, keepdims=True)
    n_entries[n_entries == 0] = 1

    # normalize rank values (not counting nan entries) to get ecd values
    ecd /= n_entries

    return ecd