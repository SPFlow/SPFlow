"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
import numpy.ma as ma
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian


def nearest_sym_psd(A: np.ndarray) -> np.ndarray:
    # compute closest positive semi-definite matrix as described in (Higham, 1988) https://www.sciencedirect.com/science/article/pii/0024379588902236
    # based on MATLAB implementation found here: https://mathworks.com/matlabcentral/fileexchange/42885-nearestspd?s_tid=mwa_osa_a and this Python port: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    def is_pd(A: np.ndarray) -> np.ndarray:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    # make sure matrix is symmetric
    B = (A + A)/2

    # compute symmetric polar factor of B from SVD (which is symmetric positive definite)
    U, s, _ = np.linalg.svd(B)
    H = np.dot(U, np.dot(np.diag(s), U.T))
    
    # compute closest symmetric positive semi-definite matrix to A in Frobenius norm (see paper linked above)
    A_hat = (B+H)/2
    # again, make sure matrix is symmetric
    A_hat = (A_hat + A_hat.T)/2

    # check if matrix is actually symmetric positive-definite
    if is_pd(A_hat):
        return A_hat

    # else fix it
    spacing = np.spacing(np.linalg.norm(A_hat))
    I = np.eye(A.shape[0])
    k = 1

    while not is_pd(A_hat):
        # compute smallest real part eigenvalue
        min_eigval = np.min(np.real(np.linalg.eigvalsh(A_hat)))
        # adjust matrix
        A_hat += I*(-min_eigval*(k**2) + spacing)
        k += 1

    return A_hat


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: MultivariateGaussian, data: np.ndarray, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if np.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'MultivariateGaussian'.")

    # NaN entries (no information)
    nan_mask = np.isnan(scope_data)

    if np.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    
    if nan_strategy is None and np.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            # TODO: correct? or only use entries where all information is available?
            scope_data = ma.masked_array(scope_data, mask=nan_mask)
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'MultivariateGaussian'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # calculate mean and standard deviation from data
    mean_est = ma.mean(scope_data, axis=0).data
    cov_est = ma.cov(scope_data.T, ddof=1 if bias_correction else 0).data

    if len(leaf.scope.query) == 1:
        cov_est = cov_est.reshape(1,1)
    
    # edge case (if all values are the same, not enough samples or very close to each other)
    for i in range(scope_data.shape[1]):
        if np.isclose(cov_est[i][i], 0):
            cov_est[i][i] = 1e-8
    
    # sometimes numpy returns a matrix with negative eigenvalues (i.e., not a valid positive semi-definite matrix)
    if np.any(np.linalg.eigvalsh(cov_est) < 0):
        # compute nearest symmetric positive semidefinite matrix
        cov_est = nearest_sym_psd(cov_est)
    
    # set parameters of leaf node
    leaf.set_params(mean=mean_est, cov=cov_est)