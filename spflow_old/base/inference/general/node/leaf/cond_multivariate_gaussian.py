"""Contains inference methods for ``CondMultivariateGaussian`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import numpy as np

from spflow.base.structure.general.node.leaf.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: CondMultivariateGaussian,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondMultivariateGaussian`` node given input data in the ``base`` backend.

    Log-likelihood for ``CondMultivariateGaussian`` is given by the logarithm of its probability distribution function (PDF):

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
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'mean', 'cov'
    mean, cov = node.retrieve_params(data, dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1), dtype=node.dtype)

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

    if check_support:
        # create masked based on distribution's support
        valid_ids = node.check_support(data[~n_marg.astype(bool)], is_scope_data=True)

        if not valid_ids.all():
            raise ValueError(
                f"Encountered data instances that are not in the support of the MultivariateGaussian distribution."
            )

    if mean is None:
        raise ValueError("Encountered 'None' value for MultivariateGaussian mean vector during inference.")
    if cov is None:
        raise ValueError(
            "Encountered 'None' value for MultivariateGaussian covariance matrix during inference."
        )

    # compute probabilities for all non-marginalized instances
    probs[~n_marg.astype(bool), 0] = node.dist(mean=mean, cov=cov).logpdf(x=data[~n_marg.astype(bool)])

    return probs
