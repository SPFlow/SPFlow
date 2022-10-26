# -*- coding: utf-8 -*-
"""Contains inference methods for ``MultivariateGaussian`` nodes for SPFlow in the 'base' backend.
"""
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian

from typing import Optional
import numpy as np


@dispatch(memoize=True)  # type: ignore
def log_likelihood(node: MultivariateGaussian, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    r"""Computes log-likelihoods for ``MultivariateGaussian`` node given input data.

    Log-likelihood for ``MultivariateGaussian`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right))

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # number of marginalized random variables per instance
    n_marg = marg_ids.sum(axis=-1)

    # in case of partially marginalized instances
    if any((n_marg > 0) & (n_marg < len(node.scope.query))):
        raise ValueError(f"Partial marginalization not yet supported for MultivariateGaussian.")

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~n_marg.astype(bool)])

    # TODO: suppress checks
    if not valid_ids.all():
        raise ValueError(
            f"Encountered data instances that are not in the support of the MultivariateGaussian distribution."
        )
    
    if(node.mean is None):
        raise ValueError("Encountered 'None' value for MultivariateGaussian mean vector during inference.")
    if(node.cov is None):
        raise ValueError("Encountered 'None' value for MultivariateGaussian covariance matrix during inference.")

    # compute probabilities for all non-marginalized instances
    probs[~n_marg.astype(bool), 0] = node.dist.logpdf(x=data[~n_marg.astype(bool)])

    return probs