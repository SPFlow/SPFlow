"""Algorithm to compute randomized dependency coefficients (RDCs) for a data set.

Typical usage example:

    coeffs = randomized_dependency_coefficients(data, k, s, phi)
"""
from itertools import combinations
from typing import Callable

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from spflow.utils.empirical_cdf import empirical_cdf
from spflow.utils.complex import complex_max, complex_min, complex_ge, complex_le


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
    torch.manual_seed(0)
    run_np_verion = False
    #data = data[:, :2]
    if run_np_verion == True:
        rdc_numpy(data[:,0].numpy(), data[:,1].numpy(), f=np.sin, k=20, s=1 / 6., n=1)


    # compute ecd values for data
    ecdf = empirical_cdf(data)

    # bring ecdf values into correct shape and pad with ones (for biases)
    ecdf_features = torch.stack([ecdf.T, torch.ones(ecdf.T.shape, dtype=data.dtype)], axis=-1)

    # compula transformation finished and equal to numpy version

    # compute random weights (and biases) generated from normal distribution
    rand_gaussians = s * torch.randn((data.shape[1], 2, k), dtype=torch.float)  # 2 for weight (of size 1) and bias
    # torch.bmm(ecdf_features, rand_gaussians)  # this is the same as the following loop
    #ToDo: compare to github repo rdc
    # compute linear combinations of ecdf feature using generated weights
    #features = torch.stack([torch.dot(features, weights) for features, weights in zip(ecdf_features, rand_gaussians)])
    features = torch.bmm(ecdf_features, rand_gaussians)

    #features *= torch.sqrt(
    #    torch.tensor(s).type(data.dtype)
    #)  # multiplying by sqrt(s) is equal to generating random weights from N(0,s)

    # apply non-linearity phi
    features = phi(features)


    # create matrix holding the pair-wise dependency coefficients
    rdcs = torch.eye(data.shape[1], dtype=data.dtype)

    cca = CCA(n_components=1)

    # compute rdcs for all pairs of features
    use_pytorch_cca = False
    if use_pytorch_cca == True:
        #rdcs = []
        for i, j in combinations(range(data.shape[1]), 2):

            rdcs[j][i] = rdcs[i][j] = pytorch_cca2(features[i], features[j])
    else:
        for i, j in combinations(range(data.shape[1]), 2):
            i_cca, j_cca = cca.fit_transform(features[i], features[j])
            rdcs[j][i] = rdcs[i][j] = np.corrcoef(i_cca.T, j_cca.T)[0, 1]

    return rdcs

"""
Implements the Randomized Dependence Coefficient
David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf

http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
"""

from scipy.stats import rankdata

def rdc_numpy(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc_numpy(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    #Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    #Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    Rx = s*torch.randn(X.shape[1], k).numpy()
    Ry = s*torch.randn(Y.shape[1], k).numpy()

    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # till here equal to pytorch version

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))

def pytorch_cca(fX, fY, k=20):
    C = torch.cov(torch.cat([fX, fY], dim=1).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = torch.linalg.eigvals(torch.matmul(torch.matmul(torch.linalg.pinv(Cxx), Cxy),
                                        torch.matmul(torch.linalg.pinv(Cyy), Cyx)))
        if torch.is_complex(eigs):
            if not (torch.all(torch.isreal(eigs)) and
                    complex_le(torch.tensor(0, dtype=torch.cfloat),  complex_min(eigs)) and
                    complex_le(complex_max(eigs), torch.tensor(1, dtype=torch.cfloat))):
                ub -= 1
                k = (ub + lb) // 2
                continue

        # Binary search if k is too large
        else:
            if not (torch.all(torch.isreal(eigs)) and
                    0 <= torch.min(eigs) and
                    torch.max(eigs) <= 1):
                ub -= 1
                k = (ub + lb) // 2
                continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return torch.sqrt(complex_max(eigs) if torch.is_complex(eigs) else torch.max(eigs))

def pytorch_cca2(fX, fY, k=20):
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return torch.tensor(np.sqrt(np.max(eigs)))
