# -*- coding: utf-8 -*-
"""Contains inference methods for ``CondBinomialLayer`` leaves for SPFlow in the 'base' backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.structure.layers.leaves.parametric.cond_binomial import CondBinomialLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(layer: CondBinomialLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondBinomialLayer`` leaves in the 'base' backend given input data.

    Log-likelihood for ``CondBinomialLayer`` is given by the logarithm of its individual probability mass functions (PMFs):
    
    .. math::

        \log(\text{PMF}(k)) = \log(\binom{n}{k}p^k(1-p)^{n-k})

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

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
        
    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'p'
    p_values = layer.retrieve_params(data, dispatch_ctx)

    for node, p in zip(layer.nodes, p_values):
        dispatch_ctx.update_args(node, {'p': p})

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in layer.nodes], axis=1)
