"""Contains inference methods for ``CondNegativeBinomialLayer`` leaves for SPFlow in the ``base`` backend.
"""
from typing import Optional

import numpy as np

from spflow.base.structure.general.layer.leaf.cond_negative_binomial import (
    CondNegativeBinomialLayer,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondNegativeBinomialLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondNegativeBinomialLayer`` leaves in the ``base`` backend given input data.

    Log-likelihood for ``CondNegativeBinomialLayer`` is given by the logarithm of its individual probability mass functions (PMFs):

    .. math::

        \log(\text{PMF}(k)) = \log(\binom{k+n-1}{n-1}p^n(1-p)^k)

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

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

    # retrieve value for 'p'
    p_values = layer.retrieve_params(data, dispatch_ctx)

    for node, p in zip(layer.nodes, p_values):
        dispatch_ctx.update_args(node, {"p": p})

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
