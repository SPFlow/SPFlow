"""Contains inference methods for SPN-like product layer for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.layers.product_layer import ProductLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    product_layer: ProductLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like product layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_layer:
            Product layer to perform inference for.
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
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in product_layer.children()
        ],
        dim=1,
    )

    # multiply childen (sum in log-space)
    return child_lls.sum(dim=1, keepdims=True).repeat((1, product_layer.n_out))
