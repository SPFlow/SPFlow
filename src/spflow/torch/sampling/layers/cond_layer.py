"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.layers.cond_layer import SPNCondSumLayer
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
from typing import Optional


@dispatch
def sample(sum_layer: SPNCondSumLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # all nodes in sum layer have same scope
    if any([len(out) != 1 for out in sampling_ctx.output_ids]):
        raise ValueError("'SPNCondSumLayer only allows single output sampling.")
    
    # retrieve value for 'weights'
    weights = sum_layer.retrieve_params(data, dispatch_ctx)

    # create mask for instane ids
    instance_ids_mask = torch.zeros(data.shape[0]).bool()
    instance_ids_mask[sampling_ctx.instance_ids] = True

    # compute log likelihoods for sum "nodes"
    partition_ll = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children()], dim=1)

    children = list(sum_layer.children())

    for node_id, instances in sampling_ctx.group_output_ids(sum_layer.n_out):

        # sample branches
        input_ids = torch.multinomial(weights[node_id]*partition_ll[instances].exp(), num_samples=1).flatten()

        # get correct child id and corresponding output id
        child_ids, output_ids = sum_layer.input_to_output_ids(input_ids)

        # group by child ids
        for child_id in torch.unique(torch.tensor(child_ids)):

            child_instance_ids = torch.tensor(instances)[torch.tensor(child_ids) == child_id].tolist()
            child_output_ids = torch.tensor(output_ids)[torch.tensor(child_ids) == child_id].unsqueeze(1).tolist()

            # sample from partition node
            sample(children[child_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(child_instance_ids, child_output_ids))

    return data