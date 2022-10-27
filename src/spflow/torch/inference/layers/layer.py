# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like layer for SPFlow in the ``torch`` backend.
"""
import torch
import numpy as np
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer, SPNHadamardLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(sum_layer: SPNSumLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like sum layers in the ``torch`` backend given input data.

    Log-likelihoods for sum nodes are the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_layer:
            Sum layer to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children()], dim=1)

    weighted_lls = child_lls.unsqueeze(1) + sum_layer.weights.log()

    return torch.logsumexp(weighted_lls, dim=-1)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(product_layer: SPNProductLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like product layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_layer:
            Product layer to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in product_layer.children()], dim=1)

    # multiply childen (sum in log-space)
    return child_lls.sum(dim=1, keepdims=True).repeat((1, product_layer.n_out))


@dispatch(memoize=True)  # type: ignore
def log_likelihood(partition_layer: SPNPartitionLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like partition layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        partition_layer:
            Product layer to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in partition_layer.children()], dim=1)

    # compute all combinations of input indices
    partition_indices = torch.tensor_split(torch.arange(0, partition_layer.n_in), torch.cumsum(torch.tensor(partition_layer.partition_sizes), dim=0)[:-1])
    indices = torch.cartesian_prod(*partition_indices)

    # multiply children (sum in log-space)
    return child_lls[:, indices].sum(dim=-1)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(partition_layer: SPNHadamardLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like element-wise product layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        hadamard_layer:
            Hadamard layer to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    children = list(partition_layer.children())
    partitions = np.split(children, np.cumsum(partition_layer.modules_per_partition[:-1]))

    # compute child log-likelihoods
    partition_lls = [torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in partition.tolist()], dim=1) for partition in partitions]

    # pad partition lls to correct shape (relevant for partitions of total output size 1)
    partition_lls = [torch.nn.functional.pad(lls, (0, partition_layer.n_out-lls.shape[1]), mode='replicate') for lls in partition_lls]

    # multiply element-wise (sum element-wise in log-space)
    return torch.stack(partition_lls).sum(dim=0)