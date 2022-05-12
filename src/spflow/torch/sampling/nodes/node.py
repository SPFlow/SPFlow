"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.torch.structure.nodes import TorchSumNode, TorchProductNode
from spflow.torch.inference import log_likelihood
from typing import Dict, List


@dispatch(TorchSumNode, torch.Tensor, ll_cache=dict, instance_ids=list)  # type: ignore[no-redef]
def sample(
    node: TorchSumNode, data: torch.Tensor, ll_cache: Dict, instance_ids: List[int]
) -> torch.Tensor:

    if any(len(child) != 1 for child in node.children()):
        raise NotImplementedError(
            f"Sampling from multi-output child modules not yet supported for TorchSumNode."
        )

    # TODO: replace 'instance_ids' with 'sampling_context'
    # TODO: support multi-output modules

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = torch.concat(
        [log_likelihood(child, data, cache=ll_cache) for child in node.children()], dim=1
    )

    # take child likelihoods into account when sampling
    sampling_weights = node.weights + child_lls[instance_ids]

    # sample branch for each instance id
    branches = torch.multinomial(sampling_weights, 1).squeeze(1)

    # group sampled branches
    for branch in branches.unique():

        # select corresponding instance ids
        branch_instance_ids = torch.tensor(instance_ids)[branches == branch].tolist()

        # sample from child module
        sample(
            list(node.children())[branch], data, ll_cache=ll_cache, instance_ids=branch_instance_ids
        )

    return data


@dispatch(TorchProductNode, torch.Tensor, ll_cache=dict, instance_ids=list)  # type: ignore[no-redef]
def sample(
    node: TorchProductNode, data: torch.Tensor, ll_cache: Dict, instance_ids: List[int]
) -> torch.Tensor:

    for child in node.children():
        sample(child, data, ll_cache=ll_cache, instance_ids=instance_ids)

    return data
