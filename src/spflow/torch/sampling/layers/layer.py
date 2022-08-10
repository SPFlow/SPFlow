"""
Created on August 08, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
from typing import Optional


@dispatch
def sample(sum_layer: SPNSumLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # all leafs in a _TorchRegionLayer have same scope, so only one output can be sampled simultaneously
    if any([len(out) != 1 for out in sampling_ctx.output_ids]):
        raise ValueError("'SPNSumLayer only allows single output sampling.")

    # create mask for instane ids
    instance_ids_mask = torch.zeros(data.shape[0]).bool()
    instance_ids_mask[sampling_ctx.instance_ids] = True

    # compute log likelihoods for sum "nodes"
    partition_ll = torch.concat([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children()], dim=1)

    for node_ids, indices in sampling_ctx.group_output_ids():

        if(node_ids):
            node_id = node_ids[0]
        else:
            node_id = 0

        # sample branches
        input_ids = torch.multinomial(sum_layer.weights[node_id]*partition_ll[indices].exp(), num_samples=1)

        # get correct child id and corresponding output id
        # child_ids, output_ids = zip(*sum_layer.input_to_output_id(input_ids))

        child_ids = []
        output_ids = []

        # TODO: can be optimized by batch processing input_to_output_id
        for input_id in input_ids:
            child_id, output_id = sum_layer.input_to_output_id(input_id)
            child_ids.append(child_id)
            output_ids.append(output_id)

        children = list(sum_layer.children())

        # group by child ids
        for child_id in np.unique(child_ids):

            child_instance_ids = torch.tensor(indices)[torch.tensor(child_ids) == child_id].tolist()
            child_output_ids = np.array(output_ids)[np.array(child_ids) == child_id].tolist()

            # sample from partition node
            sample(children[child_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(child_instance_ids, child_output_ids))

    return data


@dispatch
def sample(product_layer: SPNProductLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log-likelihoods of this module (needed to initialize log-likelihood cache for placeholder)
    log_likelihood(product_layer, data, dispatch_ctx=dispatch_ctx)

    # sample accoding to sampling_context
    # TODO
    for node_ids in torch.unique(sampling_ctx.output_ids, dim=0):
        if(len(node_ids) != 1):
            raise ValueError("Too many output ids specified for outputs over same scope.")

        node_id = node_ids[0]
        node_instance_ids = torch.tensor(sampling_ctx.instance_ids)[torch.tensor(np.where(sampling_ctx.output_ids == node_ids)[0])].tolist()

        sample(product_layer.nodes[node_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(node_instance_ids, [[] for i in node_instance_ids]))

    return data