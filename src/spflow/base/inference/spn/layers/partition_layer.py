"""Contains inference methods for SPN-like partition layer for SPFlow in the ``base`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.base.structure.spn.layers.partition_layer import PartitionLayer

from typing import Optional
import numpy as np


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: PartitionLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    """Computes log-likelihoods for SPN-like partition layers in the 'base' backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        partition_layer:
            Product layer to perform inference for.
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
    child_lls = np.concatenate(
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

    # set placeholder values
    partition_layer.set_placeholders(
        "log_likelihood", child_lls, dispatch_ctx, overwrite=False
    )

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in partition_layer.nodes
        ],
        axis=1,
    )
