"""Contains inference methods for SPN-like partition layer for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.layers.partition_layer import PartitionLayer

from typing import Optional
import torch


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: PartitionLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like partition layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        partition_layer:
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
            for child in partition_layer.children()
        ],
        dim=1,
    )

    # compute all combinations of input indices
    partition_indices = torch.tensor_split(
        torch.arange(0, partition_layer.n_in),
        torch.cumsum(torch.tensor(partition_layer.partition_sizes), dim=0)[:-1],
    )
    indices = torch.cartesian_prod(*partition_indices)

    # multiply children (sum in log-space)
    return child_lls[:, indices].sum(dim=-1)
