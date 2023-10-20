"""Contains inference methods for SPN-like Hadamard layer for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import T, tl_tolist, tl_pad_edge

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.tensorly.structure.spn.layers_layerbased.hadamard_layer import HadamardLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: HadamardLayer,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> T:
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

    children = partition_layer.children
    partitions = np.split(children, np.cumsum(partition_layer.modules_per_partition[:-1]))

    # compute child log-likelihoods
    partition_lls = [
        tl.concatenate(
            [
                log_likelihood(
                    child,
                    data,
                    check_support=check_support,
                    dispatch_ctx=dispatch_ctx,
                )
                for child in tl_tolist(partition)
            ],
            axis=1,
        )
        for partition in partitions
    ]

    # pad partition lls to correct shape (relevant for partitions of total output size 1)
    partition_lls = [
        tl_pad_edge(lls, (0, partition_layer.n_out - lls.shape[1]))
        for lls in partition_lls
    ]

    # multiply element-wise (sum element-wise in log-space)
    return tl.sum(tl.stack(partition_lls), axis=0)
