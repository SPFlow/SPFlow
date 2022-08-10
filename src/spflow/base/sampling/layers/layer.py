"""
Created on August 08, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.base.structure.layers.layer import SPNSumLayer, SPNProductLayer
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
from typing import Optional


@dispatch
def sample(sum_layer: SPNSumLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log-likelihoods of this module (needed to initialize log-likelihood cache for placeholder)
    log_likelihood(sum_layer, data, dispatch_ctx=dispatch_ctx)

    # sample accoding to sampling_context
    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if(len(node_ids) != 1):
            raise ValueError("Too many output ids specified for outputs over same scope.")

        # single node id
        node_id = node_ids[0]
        node_instance_ids = np.array(sampling_ctx.instance_ids)[np.where(sampling_ctx.output_ids == node_ids)[0]].tolist()

        sample(sum_layer.nodes[node_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(node_instance_ids, [[] for i in node_instance_ids]))

    return data


@dispatch
def sample(product_layer: SPNProductLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log-likelihoods of this module (needed to initialize log-likelihood cache for placeholder)
    log_likelihood(product_layer, data, dispatch_ctx=dispatch_ctx)

    # sample accoding to sampling_context
    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if(len(node_ids) != 1):
            raise ValueError("Too many output ids specified for outputs over same scope.")

        node_id = node_ids[0]
        node_instance_ids = np.array(sampling_ctx.instance_ids)[np.where(sampling_ctx.output_ids == node_ids)[0]].tolist()

        sample(product_layer.nodes[node_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(node_instance_ids, [[] for i in node_instance_ids]))

    return data