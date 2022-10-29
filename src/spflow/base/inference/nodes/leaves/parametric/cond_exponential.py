# -*- coding: utf-8 -*-
"""Contains inference methods for ``CondExponential`` nodes for SPFlow in the ``base`` backend.
"""
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
)

from typing import Optional
import numpy as np


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: CondExponential,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondExponential`` node given input data in the ``base`` backend.

    Log-likelihood for ``CondExponential`` is given by the logarithm of its probability distribution function (PDF):

    .. math::
        
        \log(\text{PDF}(x)) = \begin{cases} \log(\lambda e^{-\lambda x}) & \text{if } x > 0\\
                                            \log(0)                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter

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

    # retrieve value for 'l'
    l = node.retrieve_params(data, dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data).sum(axis=-1).astype(bool)

    if check_support:
        # create masked based on distribution's support
        valid_ids = node.check_support(data[~marg_ids]).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Exponential distribution."
            )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = node.dist(l=l).logpdf(x=data[~marg_ids])

    return probs
