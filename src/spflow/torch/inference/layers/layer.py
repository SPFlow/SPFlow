"""
Created on August 10, 2022

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer


@dispatch(memoize=True)
def log_likelihood(sum_layer: SPNSumLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children()], dim=1)

    weighted_lls = child_lls.unsqueeze(1) + sum_layer.weights.log()

    return torch.logsumexp(weighted_lls, dim=-1)


@dispatch(memoize=True)
def log_likelihood(product_layer: SPNProductLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in product_layer.children()], dim=1)

    # multiply childen (sum in log-space)
    return child_lls.sum(dim=1, keepdims=True).repeat((1, product_layer.n_out))


@dispatch(memoize=True)
def log_likelihood(partition_layer: SPNPartitionLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in partition_layer.children()], dim=1)

    # compute all combinations of indices
    partition_indices = torch.tensor_split(torch.arange(0, partition_layer.n_out), torch.cumsum(torch.tensor(partition_layer.partition_sizes), dim=0)[:-1])
    indices = torch.cartesian_prod(*partition_indices)

    # multiply children (sum in log-space)
    return child_lls[:, indices].sum(dim=-1)