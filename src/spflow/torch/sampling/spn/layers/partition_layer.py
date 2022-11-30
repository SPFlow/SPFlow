"""Contains sampling methods for SPN-like partition layers for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.structure.spn.layers.partition_layer import PartitionLayer
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
from typing import Optional


@dispatch  # type: ignore
def sample(
    partition_layer: PartitionLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    """Samples from SPN-like partition layers in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        partition_layer:
            Partition layer to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # all nodes in sum layer have same scope
    if any([len(out) != 1 for out in sampling_ctx.output_ids]):
        raise ValueError("'PartitionLayer only allows single output sampling.")

    # TODO: precompute indices
    partition_indices = torch.tensor_split(
        torch.arange(0, partition_layer.n_in),
        torch.cumsum(torch.tensor(partition_layer.partition_sizes), dim=0)[:-1],
    )
    input_ids_per_node = torch.cartesian_prod(*partition_indices)

    children = list(partition_layer.children())

    # sample accoding to sampling_context
    for node_id, instances in sampling_ctx.group_output_ids(
        partition_layer.n_out
    ):

        # get input ids for this node
        input_ids = input_ids_per_node[node_id]
        child_ids, output_ids = partition_layer.input_to_output_ids(
            input_ids.tolist()
        )

        # group by child ids
        for child_id in np.unique(child_ids):

            child_output_ids = np.array(output_ids)[
                np.array(child_ids) == child_id
            ].tolist()

            # sample from partition node
            sample(
                children[child_id],
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
                sampling_ctx=SamplingContext(
                    instances, [child_output_ids for _ in instances]
                ),
            )

    return data
