"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.base.sampling.sampling_context import SamplingContext
from spflow.torch.structure.module import TorchModule
from spflow.torch.structure.nodes.leaves.parametric import TorchHypergeometric
from typing import Dict


@dispatch(TorchHypergeometric, torch.Tensor, ll_cache=dict, sampling_ctx=SamplingContext)  # type: ignore[no-redef]
def sample(
    leaf: TorchHypergeometric, data: torch.Tensor, ll_cache: Dict[TorchModule, torch.Tensor], sampling_ctx: SamplingContext
) -> torch.Tensor:
    raise NotImplementedError
