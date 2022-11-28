"""Contains inference methods for ``CondGeometricLayer`` leaves for SPFlow in the ``base`` backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.base.structure.general.layers.leaves.parametric.cond_geometric import (
    CondGeometricLayer,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondGeometricLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondGeometricLayer`` leaves in the ``base`` backend given input data.

    Log-likelihood for ``CondGeometricLayer`` is given by the logarithm of its individual probability distribution functions (PDFs):

    .. math::

        \log(\text{PMF}(k)) =  \log(p(1-p)^{k-1})

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

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
