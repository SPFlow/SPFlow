"""Contains inference methods for SPN-like partition layer for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch
import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import T, tl_array_split, tl_cartesian_product

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.tensorly.structure.spn.layers_layerbased.partition_layer import PartitionLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: PartitionLayer,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> T:
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
    child_lls = tl.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in partition_layer.children
        ],
        axis=1,
    )

    # compute all combinations of input indices
    partition_indices = tl_array_split(
        tl.arange(0, partition_layer.n_in),
        np.cumsum(tl.tensor(partition_layer.partition_sizes, dtype=int), axis=0)[:-1],
    )
    indices = tl.tensor(tl_cartesian_product(*partition_indices),dtype=int)

    # multiply children (sum in log-space)
    return tl.sum(child_lls[:, indices], axis=-1)
