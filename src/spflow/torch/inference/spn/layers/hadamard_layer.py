# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like Hadamard layer for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)

from spflow.torch.structure.spn.layers.hadamard_layer import HadamardLayer

from typing import Optional
import numpy as np
import torch


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: HadamardLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for SPN-like element-wise product layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        hadamard_layer:
            Hadamard layer to perform inference for.
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

    children = list(partition_layer.children())
    partitions = np.split(
        children, np.cumsum(partition_layer.modules_per_partition[:-1])
    )

    # compute child log-likelihoods
    partition_lls = [
        torch.concat(
            [
                log_likelihood(
                    child,
                    data,
                    check_support=check_support,
                    dispatch_ctx=dispatch_ctx,
                )
                for child in partition.tolist()
            ],
            dim=1,
        )
        for partition in partitions
    ]

    # pad partition lls to correct shape (relevant for partitions of total output size 1)
    partition_lls = [
        torch.nn.functional.pad(
            lls, (0, partition_layer.n_out - lls.shape[1]), mode="replicate"
        )
        for lls in partition_lls
    ]

    # multiply element-wise (sum element-wise in log-space)
    return torch.stack(partition_lls).sum(dim=0)
