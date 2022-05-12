"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.torch.structure.nodes.leaves.parametric import TorchLogNormal
from typing import Dict, List


@dispatch(TorchLogNormal, torch.Tensor, ll_cache=dict, instance_ids=list)  # type: ignore[no-redef]
def sample(
    leaf: TorchLogNormal, data: torch.Tensor, ll_cache: Dict, instance_ids: List[int]
) -> torch.Tensor:

    # TODO: replace 'instance_ids' with 'sampling_context'

    if any([i >= data.shape[0] for i in instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # TODO: what to do in case of instance ids that are already specified (i.e. not nan)?
    marg_ids = (torch.isnan(data[:, leaf.scope]) == len(leaf.scope)).squeeze(1)

    instance_ids_mask = torch.zeros(data.shape[0])
    instance_ids_mask[instance_ids] = 1

    sampling_ids = marg_ids & instance_ids_mask.bool()

    data[sampling_ids, leaf.scope] = leaf.dist.sample((sampling_ids.sum(),))

    return data
