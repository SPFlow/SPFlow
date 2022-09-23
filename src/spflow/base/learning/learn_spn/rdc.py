from typing import Callable, Dict, Any
import numpy as np
from itertools import combinations
from scipy.stats import rankdata
from sklearn.cross_decomposition import CCA


def empirical_cdf(data: np.ndarray):
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


def randomized_dependency_coefficients(data: np.ndarray, k: int=20, s: float=1/6, phi: Callable=np.sin):
    """TODO."""
    # default arguments according to paper
    if np.any(np.isnan(data)):
        raise ValueError("Randomized dependency coefficients cannot be computed for data with missing values.")

    # compute ecd values for data
    ecdf = empirical_cdf(data)

    # bring ecdf values into correct shape and pad with ones (for biases)
    ecdf_features = np.stack([ecdf.T, np.ones(ecdf.T.shape)], axis=-1)
    
    # compute random weights (and biases) generated from normal distribution
    rand_gaussians = np.random.randn(data.shape[1], 2, k) # 2 for weight (of size 1) and bias

    # compute linear combinations of ecdf feature using generated weights
    features = np.stack([np.dot(features, weights) for features, weights in zip(ecdf_features, rand_gaussians)])
    features *= np.sqrt(s) # multiplying by sqrt(s) is equal to generating random weights from N(0,s)
    
    # apply non-linearity phi
    features = phi(features)

    # create matrix holding the pair-wise dependency coefficients
    rdcs = np.eye(data.shape[1])

    cca = CCA(n_components=1)

    # compute rdcs for all pairs of features    
    for i,j in combinations(range(data.shape[1]), 2):
        i_cca, j_cca = cca.fit_transform(features[i], features[j])
        rdcs[j][i] = rdcs[i][j] = np.corrcoef(i_cca.T, j_cca.T)[0, 1]

    return rdcs