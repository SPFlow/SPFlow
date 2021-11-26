import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchPoisson


@dispatch(TorchPoisson, torch.Tensor, cache=dict)
@memoize(TorchPoisson)
def log_likelihood(leaf: TorchPoisson, data: torch.Tensor, cache: Dict = {}):
    return leaf(data)
