"""Contains sampling methods for conditional SPN-like layers for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample
from spflow.torch.structure.spn.layers.cond_sum_layer import CondSumLayer


@dispatch  # type: ignore
def sample(
    sum_layer: CondSumLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    """Samples from conditional SPN-like sum layers in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Samples from each input proportionally to its weighted likelihoods given the evidence.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        sum_layer:
            Sum layer to sample from.
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
        raise ValueError("'CondSumLayer only allows single output sampling.")

    # retrieve value for 'weights'
    weights = sum_layer.retrieve_params(data, dispatch_ctx)

    # create mask for instane ids
    instance_ids_mask = torch.zeros(data.shape[0]).bool()
    instance_ids_mask[sampling_ctx.instance_ids] = True

    # compute log likelihoods for sum "nodes"
    partition_ll = torch.concat(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_layer.chs
        ],
        dim=1,
    )

    children = list(sum_layer.chs)

    for node_id, instances in sampling_ctx.group_output_ids(sum_layer.n_out):

        # sample branches
        input_ids = torch.multinomial(weights[node_id] * partition_ll[instances].exp(), num_samples=1).flatten()

        # get correct child id and corresponding output id
        child_ids, output_ids = sum_layer.input_to_output_ids(input_ids)

        # group by child ids
        for child_id in torch.unique(torch.tensor(child_ids)):

            child_instance_ids = torch.tensor(instances)[torch.tensor(child_ids) == child_id].tolist()
            child_output_ids = torch.tensor(output_ids)[torch.tensor(child_ids) == child_id].unsqueeze(1).tolist()

            # sample from partition node
            sample(
                children[child_id],
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
                sampling_ctx=SamplingContext(child_instance_ids, child_output_ids),
            )

    return data
