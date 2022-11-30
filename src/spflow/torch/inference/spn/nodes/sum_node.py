"""Contains inference methods for SPN-like sum nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.nodes.sum_node import SumNode


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: SumNode,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like product nodes in the ``torch`` backend given input data.

    Log-likelihood for product node is the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_node:
            Product node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    inputs = torch.hstack(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in node.children()
        ]
    )

    # weight inputs in log-space
    weighted_inputs = inputs + node.weights.log()

    return torch.logsumexp(weighted_inputs, dim=-1, keepdims=True)
