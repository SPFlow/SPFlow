"""Contains learning methods for ``MultivariateGaussian`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Callable, Optional, Union

import numpy as np
import numpy.ma as ma

from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.base.utils.nearest_sym_pd import nearest_sym_pd
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: MultivariateGaussian,
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``MultivariateGaussian`` node parameters in the ``base`` backend.

    Estimates the mean and covariance matrix :math:`\mu` and :math:`\Sigma` of a Multivariate Gaussian distribution from data, as follows:

    .. math::

        \mu^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i\\
        \Sigma^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})(x_i-\mu^{\*})^T

    or

    .. math::

        \Sigma^{\*}=\frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})(x_i-\mu^{\*})^T

    if bias correction is used, where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set
    
    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional NumPy array containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another NumPy array of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = np.ones(data.shape[0])

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    # reshape weights
    weights = weights.reshape(-1, 1)

    if check_support:
        if np.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'MultivariateGaussian'.")

    # NaN entries (no information)
    nan_mask = np.isnan(scope_data)

    if np.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and np.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )

    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            pass  # handle it during computation
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'MultivariateGaussian'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    if nan_strategy == "ignore":
        n_total = (weights * ~nan_mask).sum(axis=0)
        # compute mean of available data
        mean_est = np.sum(weights * np.nan_to_num(scope_data, nan=0.0), axis=0) / n_total
        # compute covariance of full samples only!
        full_sample_mask = (~nan_mask).sum(axis=1) == scope_data.shape[1]
        cov_est = np.cov(
            scope_data[full_sample_mask].T,
            aweights=weights[full_sample_mask].squeeze(-1),
            ddof=1 if bias_correction else 0,
        )
    else:
        n_total = weights.sum(axis=0)
        # calculate mean and standard deviation from data
        mean_est = np.sum(weights * scope_data, axis=0) / n_total
        cov_est = np.cov(
            scope_data.T,
            aweights=weights.squeeze(-1),
            ddof=1 if bias_correction else 0,
        )

    if len(leaf.scope.query) == 1:
        cov_est = cov_est.reshape(1, 1)

    # edge case (if all values are the same, not enough samples or very close to each other)
    for i in range(scope_data.shape[1]):
        if np.isclose(cov_est[i][i], 0):
            cov_est[i][i] = 1e-8

    # sometimes numpy returns a matrix with negative eigenvalues (i.e., not a valid positive semi-definite matrix)
    if np.any(np.linalg.eigvalsh(cov_est) < 0):
        # compute nearest symmetric positive semidefinite matrix
        cov_est = nearest_sym_pd(cov_est)

    # set parameters of leaf node
    leaf.set_params(mean=mean_est, cov=cov_est)
