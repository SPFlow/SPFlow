"""Contains inference methods for ``Gaussian`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import numpy as np

from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: Gaussian,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``Gaussian`` node in the ``base`` backend given input data.

    Log-likelihood for ``Gaussian`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2}))

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

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

    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data).sum(axis=-1).astype(bool)

    if check_support:
        # create masked based on distribution's support
        valid_ids = node.check_support(
            data[~marg_ids], is_scope_data=True
        ).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Gaussian distribution."
            )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = node.dist.logpdf(x=data[~marg_ids])

    return probs
