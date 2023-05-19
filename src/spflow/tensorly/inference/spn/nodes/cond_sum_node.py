"""Contains inference methods for SPN-like conditional sum nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl
from scipy.special import logsumexp  # type: ignore
from spflow.tensorly.utils.helper_functions import T, tl_logsumexp
from spflow.tensorly.structure.spn.nodes.cond_sum_node import CondSumNode
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    sum_node: CondSumNode,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> T:
    """Computes log-likelihoods for conditional SPN-like sum node given input data in the ``base`` backend.

    Log-likelihood for sum node is the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_node:
            Sum node to perform inference for.
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
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'weights'
    weights = sum_node.retrieve_params(data, dispatch_ctx)

    # compute child log-likelihoods
    child_lls = tl.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_node.children
        ],
        axis=1,
    )
    weighted_inputs = child_lls + tl.log(weights)
    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return tl_logsumexp(weighted_inputs, axis=-1, keepdims=True)
