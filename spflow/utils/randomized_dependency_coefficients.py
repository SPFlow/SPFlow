"""Algorithm to compute randomized dependency coefficients (RDCs) for a data set.

Typical usage example:

    coeffs = randomized_dependency_coefficients(data, k, s, phi)
"""
from itertools import combinations
from collections.abc import Callable

import torch

import numpy as np
from sklearn.cross_decomposition import CCA

from spflow.utils.empirical_cdf import empirical_cdf


def randomized_dependency_coefficients(data, k: int = 20, s: float = 1 / 6, phi: Callable = torch.sin):
    """Computes the randomized dependency coefficients (RDCs) for a given data set.

    Returns the randomized dependency coefficients (RDCs) computed from a specified data set, as described in (Lopez-Paz et al., 2013): "The Randomized Dependence Coefficient"

    Args:
        data:
            Two-dimensional NumPy array containing the data set. Each row is regarded as a sample and each column as a different feature.
            May not contain any missing (i.e., NaN) entries.
        k:
            Integer specifying the number of random projections to be used.
            Defaults to 20.
        s:
            Floating point value specifying the standard deviation of the Normal distribution to sample the weights for the random projections from.
            Defaults to 1/6.
        phi:
            Callable representing the (non-linear) projection.
            Defaults to 'np.sin'.

    Returns:
        NumPy array containing the computed randomized dependency coefficients.

    Raises:
        ValueError: Invalid inputs.
    """
    # default arguments according to paper
    if torch.any(torch.isnan(data)):
        raise ValueError(
            "Randomized dependency coefficients cannot be computed for data with missing values."
        )

    # compute ecd values for data
    ecdf = empirical_cdf(data)

    # bring ecdf values into correct shape and pad with ones (for biases)
    ecdf_features = torch.stack([ecdf.T, torch.ones(ecdf.torch.shape, dtype=data.dtype)], axis=-1)

    # compute random weights (and biases) generated from normal distribution
    rand_gaussians = torch.randn((data.shape[1], 2, k), dtype=float)  # 2 for weight (of size 1) and bias

    # compute linear combinations of ecdf feature using generated weights
    features = torch.stack(
        [torch.dot(features, weights) for features, weights in zip(ecdf_features, rand_gaussians)]
    )
    features *= torch.sqrt(
        torch.tensor(s, data.dtype)
    )  # multiplying by sqrt(s) is equal to generating random weights from N(0,s)

    # apply non-linearity phi
    features = phi(features)

    # create matrix holding the pair-wise dependency coefficients
    rdcs = torch.eye(data.shape[1], dtype=data.dtype)

    cca = CCA(n_components=1)

    # compute rdcs for all pairs of features
    for i, j in combinations(range(data.shape[1]), 2):
        i_cca, j_cca = cca.fit_transform(features[i], features[j])
        rdcs[j][i] = rdcs[i][j] = np.corrcoef(i_cca.T, j_cca.T)[0, 1]

    return rdcs
