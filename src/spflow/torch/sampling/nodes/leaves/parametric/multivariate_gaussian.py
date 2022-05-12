"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.torch.structure.nodes.leaves.parametric import TorchMultivariateGaussian
from typing import Dict, List


@dispatch(TorchMultivariateGaussian, torch.Tensor, ll_cache=dict, instance_ids=list)  # type: ignore[no-redef]
def sample(
    leaf: TorchMultivariateGaussian, data: torch.Tensor, ll_cache: Dict, instance_ids: List[int]
) -> torch.Tensor:
    raise NotImplementedError
