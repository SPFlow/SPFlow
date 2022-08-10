"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.module import Module, NestedModule

import torch
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
    
    sampling_ctx_per_child = {}

    # TODO: could potentially be done more efficiently via grouping
    for instance_id, instance_output_ids in zip(sampling_ctx.instance_ids, sampling_ctx.output_ids):

        output_per_child = {}
        
        # iterate over actual child and output ids
        if instance_output_ids == []:
            # all children    
            for _, ids in placeholder.input_to_output_id_dict.items():
                output_per_child[ids[0]] = [ids[1]]
        else:
            for child_id, output_id in [placeholder.host.input_to_output_id(output_id) for output_id in instance_output_ids]:

                # sort output ids per child id
                if(child_id in output_per_child):
                    output_per_child[child_id].append(output_id)
                else:
                    output_per_child[child_id] = [output_id]
        
        # append (or create) sampling contexts
        for child_id, output_ids in output_per_child.items():
            if(child_id) in sampling_ctx_per_child:
                sampling_ctx_per_child[child_id].instance_ids.append(instance_id)
                sampling_ctx_per_child[child_id].output_ids.append(output_ids)
            else:
                sampling_ctx_per_child[child_id] = SamplingContext([instance_id], [output_ids])
    
    host_children = list(placeholder.host.children())

    # sample from children
    for child_id, child_sampling_ctx in sampling_ctx_per_child.items():
        sample(host_children[child_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=child_sampling_ctx)

    return data