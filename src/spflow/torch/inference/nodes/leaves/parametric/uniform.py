import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchUniform


@dispatch(TorchUniform, torch.Tensor, cache=dict)
@memoize(TorchUniform)
def log_likelihood(leaf: TorchUniform, data: torch.Tensor, cache: Dict = {}):
    return leaf(data)
