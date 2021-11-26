import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchExponential


@dispatch(TorchExponential, torch.Tensor, cache=dict)
@memoize(TorchExponential)
def log_likelihood(leaf: TorchExponential, data: torch.Tensor, cache: Dict = {}):
    return leaf(data)
