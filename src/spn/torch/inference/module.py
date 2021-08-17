from multipledispatch import dispatch  # type: ignore
import torch
from spn.torch.structure.module import TorchModule


@dispatch(TorchModule, torch.Tensor)
def log_likelihood(module: TorchModule, data: torch.Tensor) -> torch.Tensor:
    return module.forward(data)


@dispatch(TorchModule, torch.Tensor)
def likelihood(module: TorchModule, data: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_likelihood(module))
