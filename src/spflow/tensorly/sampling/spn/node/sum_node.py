"""Contains sampling methods for SPN-like sum nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional, Union

import tensorly as tl
import numpy as np
import torch
from ....utils.helper_functions import tl_unique, tl_unsqueeze, T

from spflow.tensorly.inference.module import log_likelihood
from spflow.tensorly.structure.spn.node.sum_node import SumNode
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)

@dispatch  # type: ignore
def sample(
    node: SumNode,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> T:
    """Samples from SPN-like sum nodes in the ``base`` backend given potential evidence.

    Samples from each input proportionally to its weighted likelihoods given the evidence.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        node:
            Sum node to sample from.
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, tl.shape(data)[0])

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = tl.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in node.children
        ],
        axis=1,
    )

    # take child likelihoods into account when sampling
    sampling_weights = node.weights + child_lls[sampling_ctx.instance_ids]

    # sample branch for each instance id
    # this solution is based on a trick described here: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
    cum_sampling_weights = tl.cumsum(sampling_weights,axis=1)
    random_choices = tl_unsqueeze(tl.random.random_tensor(tl.shape(sampling_weights)[0], 1, device=node.device), axis=1)
    branches = tl.sum((cum_sampling_weights < random_choices),axis=1)

    # group sampled branches
    for branch in tl.tensor(tl_unique(branches), dtype=int, device=node.device):
        # group instances by sampled branch
        branch_instance_ids = tl.tensor(sampling_ctx.instance_ids, dtype=int, device=node.device)[branches == branch].tolist()
        # get corresponding child and output id for sampled branch
        child_ids, output_ids = node.input_to_output_ids([branch])

        # sample from child module
        sample(
            node.children[child_ids[0]],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                branch_instance_ids,
                [[output_ids[0]] for _ in range(len(branch_instance_ids))],
            ),
        )

    return data
