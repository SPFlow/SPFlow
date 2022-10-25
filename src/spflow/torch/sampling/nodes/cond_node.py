"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.nodes.cond_node import SPNCondSumNode
from spflow.torch.inference.nodes.cond_node import log_likelihood
from spflow.torch.sampling.module import sample

import torch
from typing import Optional


@dispatch
def sample(node: SPNCondSumNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:

    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = torch.concat(
        [log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children()], dim=1
    )

    # retrieve value for 'weights'
    weights = node.retrieve_params(data, dispatch_ctx)

    # take child likelihoods into account when sampling
    sampling_weights = weights + child_lls[sampling_ctx.instance_ids]

    # sample branch for each instance id
    branches = torch.multinomial(sampling_weights, 1).squeeze(1)

    # group sampled branches
    for branch in branches.unique():

        # group instances by sampled branch
        branch_instance_ids = torch.tensor(sampling_ctx.instance_ids)[branches == branch].tolist()

        # get corresponding child and output id for sampled branch
        child_ids, output_ids = node.input_to_output_ids([branch.item()])

        # sample from child module
        sample(
            list(node.children())[child_ids[0]], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(branch_instance_ids, [[output_ids[0]] for _ in range(len(branch_instance_ids))])
        )

    return data