from multipledispatch import dispatch  # type: ignore
import torch
from typing import Dict
from spflow.base.memoize import memoize
from spflow.base.inference import log_likelihood
from spflow.torch.structure.module import TorchModule


@dispatch(TorchModule, torch.Tensor, cache=dict)
@memoize(TorchModule)
def likelihood(module: TorchModule, data: torch.Tensor, cache: Dict = {}) -> torch.Tensor:
    return torch.exp(log_likelihood(module, data, cache=cache))
