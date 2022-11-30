"""Algorithm to compute Pearson correlation coefficients for a data set.

Typical usage example:

    coeffs = corrcoef(data)
"""
from typing import Optional

import torch


def corrcoef(data: torch.Tensor) -> torch.Tensor:
    r"""Computes the Pearson correlation coefficients for a given data set.

    Returns the correlation coefficients computed from a specified data set according to:

    .. math::

        \frac{\text{Cov}(x)_{ij}}{\sigma_i\sigma_j}

    Args:
        data:
            Two-dimensional PyTorch tensor representing the data set. Each row is regarded as a sample.

    Returns:
        NumPy array containing the Pearson correlation coefficients.

    Raises:
        ValueError: invalid arguments.
    """
    if torch.is_complex(data):
        raise ValueError(
            "Computing correlation coefficients for complex data is not supported."
        )

    # compute covariance matrix
    cov = torch.cov(data.T, correction=1)

    # if there are non-finite (i.e., Nan or Inf) or zero values
    if torch.any(~torch.isfinite(cov)):
        raise ValueError(
            "Encountered non-finite values in covariance matrix during computation of correlation coefficients."
        )
    if torch.any(cov == 0):
        raise ValueError(
            "Encountered zero values in covariance matrix during computation of correlation coefficients."
        )

    # extract standard deviations per feature
    # TODO: test if there are any underflows resulting in invalid entries
    std = torch.sqrt(torch.diag(cov))

    # divide covariances by product of respective standard deviations
    cov /= torch.outer(std, std)

    return cov
