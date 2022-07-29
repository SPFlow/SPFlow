"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch
from spflow.base.sampling.sampling_context import SamplingContext  # type: ignore
from spflow.torch.structure.module import TorchModule
from spflow.torch.structure.nodes.leaves.parametric import TorchExponential
from typing import Dict


@dispatch(TorchExponential, torch.Tensor, ll_cache=dict, instance_ids=list)  # type: ignore[no-redef]
def sample(
    leaf: TorchExponential, data: torch.Tensor, ll_cache: Dict[TorchModule, torch.Tensor], sampling_ctx: SamplingContext
) -> torch.Tensor:

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # TODO: what to do in case of instance ids that are already specified (i.e. not nan)?
    marg_ids = (torch.isnan(data[:, leaf.scope]) == len(leaf.scope)).squeeze(1)

    instance_ids_mask = torch.zeros(data.shape[0])
    instance_ids_mask[sampling_ctx.instance_ids] = 1

    sampling_ids = marg_ids & instance_ids_mask.bool().to(leaf.l_aux.device)

    data[sampling_ids, leaf.scope] = leaf.dist.sample((sampling_ids.sum(),)).to(leaf.l_aux.device)

    return data
