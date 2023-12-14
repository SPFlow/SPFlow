"""Contains inference methods for SPN-like Hadamard layer for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.tensorly.structure.spn.layer.hadamard_layer import HadamardLayer
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    hadamard_layer: HadamardLayer,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> T:
    """Computes log-likelihoods for SPN-like element-wise product layers in the 'base' backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        hadamard_layer:
            Hadamard layer to perform inference for.
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

    # compute child log-likelihoods
    child_lls = tl.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in hadamard_layer.children
        ],
        axis=1,
    )

    # set placeholder values
    hadamard_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return tl.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in hadamard_layer.nodes
        ],
        axis=1,
    )
