"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.base.sampling.sampling_context import SamplingContext
from spflow.torch.structure.module import TorchModule
from spflow.torch.structure.nodes.leaves.parametric import TorchGaussian
from typing import Dict, List


@dispatch(TorchGaussian, torch.Tensor, ll_cache=dict, sampling_ctx=SamplingContext)  # type: ignore[no-redef]
def sample(
    leaf: TorchGaussian, data: torch.Tensor, ll_cache: Dict[TorchModule, torch.Tensor], sampling_ctx: SamplingContext
) -> torch.Tensor:

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # TODO: what to do in case of instance ids that are already specified (i.e. not nan)? warning? exception?
    marg_ids = (torch.isnan(data[:, leaf.scope]) == len(leaf.scope)).squeeze(1)

    instance_ids_mask = torch.zeros(data.shape[0])
    instance_ids_mask[sampling_ctx.instance_ids] = 1

    sampling_ids = marg_ids & instance_ids_mask.bool().to(leaf.mean.device)

    data[sampling_ids, leaf.scope] = leaf.dist.sample((sampling_ids.sum(),)).to(leaf.mean.device)

    return data
