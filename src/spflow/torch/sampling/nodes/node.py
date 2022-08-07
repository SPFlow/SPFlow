"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNLeafNode
from spflow.torch.sampling.module import sample

import torch
from typing import Optional


@dispatch
def sample(node: SPNSumNode, data: torch.Tensor, dispatch_ct: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:

    if any(len(child) != 1 for child in node.children()):
        raise NotImplementedError(
            f"Sampling from multi-output child modules not yet supported for TorchSumNode."
        )

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = torch.concat(
        [log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children()], dim=1
    )

    # take child likelihoods into account when sampling
    sampling_weights = node.weights + child_lls[sampling_ctx.instance_ids]

    # sample branch for each instance id
    branches = torch.multinomial(sampling_weights, 1).squeeze(1)

    # group sampled branches
    for branch in branches.unique():

        # select corresponding instance ids
        branch_instance_ids = torch.tensor(sampling_ctx.instance_ids)[branches == branch].tolist()

        # sample from child module
        sample(
            list(node.children())[branch], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(instance_ids=branch_instance_ids, output_ids=[[0] for _ in range(len(branch_instance_ids))])
        )

    return data


@dispatch
def sample(node: SPNProductNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:

    for child in node.children():
        sample(child, data, dispatch_ctx=dispatch_ctx, sampling_ctx=sampling_ctx)

    return data