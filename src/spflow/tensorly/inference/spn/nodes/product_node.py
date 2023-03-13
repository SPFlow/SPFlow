"""Contains inference methods for SPN-like product nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl

from spflow.tensorly.structure.spn.nodes.product_node import ProductNode
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    product_node: ProductNode,
    data: tl.tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> tl.tensor:
    """Computes log-likelihoods for SPN-like product nodes in the ``base`` backend given input data.

    Log-likelihood for product node is the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_node:
            Product node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
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
            for child in product_node.children
        ],
        axis=1,
    )

    # multiply child log-likelihoods together (sum in log-space)
    return tl.sum(child_lls,axis=1, keepdims=True)
