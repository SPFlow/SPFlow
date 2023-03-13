"""Contains inference methods for ``CondPoissonLayer`` leaves for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl

from spflow.tensorly.structure.general.layers.leaves.parametric.cond_poisson import (
    CondPoissonLayer,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondPoissonLayer,
    data: tl.tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> tl.tensor:
    r"""Computes log-likelihoods for ``CondPoissonLayer`` leaves in the ``base`` backend given input data.

    Log-likelihood for ``CondPoissonLayer`` is given by the logarithm of its individual probability mass functions (PMFs):

    .. math::

        \log(\text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!})

    where
        - :math:`k` is the number of occurrences
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
    return tl.concatenate(
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
