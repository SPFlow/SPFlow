import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchMultivariateGaussian


@dispatch(TorchMultivariateGaussian, torch.Tensor, cache=dict)
@memoize(TorchMultivariateGaussian)
def log_likelihood(leaf: TorchMultivariateGaussian, data: torch.Tensor, cache: Dict = {}):
    return leaf(data)
