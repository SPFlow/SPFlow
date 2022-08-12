"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.module import Module, NestedModule

import torch
import numpy as np
from typing import Optional


@dispatch
def sample(module: Module, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return sample(module, 1, dispatch_ctx=dispatch_ctx)


@dispatch
def sample(module: Module, n: int, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = SamplingContext(list(range(n)))
    data = torch.full((n, max(module.scope.query)+1), float("nan"))

    return sample(module, data, dispatch_ctx=dispatch_ctx, sampling_ctx=sampling_ctx)


@dispatch
def sample(placeholder: NestedModule.Placeholder, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""    
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])
    
    # dictionary to hold the 
    sampling_ids_per_child = [([],[]) for _ in placeholder.host.children()]

    for instance_id, output_ids in zip(sampling_ctx.instance_ids, sampling_ctx.output_ids):
        # convert ids to actual child and output ids of host module
        child_ids_actual, output_ids_actual = placeholder.input_to_output_ids(output_ids)

        for child_id in np.unique(child_ids_actual):
            sampling_ids_per_child[child_id][0].append(instance_id)
            sampling_ids_per_child[child_id][1].append(np.array(output_ids_actual)[child_ids_actual == child_id].tolist())

    # sample from children
    for child_id, (instance_ids, output_ids) in enumerate(sampling_ids_per_child):
        if(len(instance_ids) == 0):
            continue
        sample(placeholder.host.children[child_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(instance_ids, output_ids))

    return data