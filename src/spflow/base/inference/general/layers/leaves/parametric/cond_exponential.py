# -*- coding: utf-8 -*-
"""Contains inference methods for ``CondExponentialLayer`` leaves for SPFlow in the ``base`` backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.base.structure.general.layers.leaves.parametric.cond_exponential import (
    CondExponentialLayer,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondExponentialLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondExponentialLayer`` leaves in the ``base`` backend given input data.

    Log-likelihood for ``CondExponentialLayer`` is given by the logarithm of its individual probability distribution functions (PDFs):

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
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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

    # retrieve value for 'l'
    l_values = layer.retrieve_params(data, dispatch_ctx)

    for node, l in zip(layer.nodes, l_values):
        dispatch_ctx.update_args(node, {"l": l})

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in layer.nodes
        ],
        axis=1,
    )