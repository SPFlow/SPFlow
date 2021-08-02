from .rat_spn import TorchRatSpn
from multipledispatch import dispatch  # type: ignore
import torch


@dispatch(TorchRatSpn, torch.Tensor)
def log_likelihood(rat_spn: TorchRatSpn, data: torch.Tensor) -> torch.Tensor:
    return rat_spn.forward(data)


@dispatch(TorchRatSpn, torch.Tensor)
def likelihood(rat_spn: TorchRatSpn, data: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_likelihood(rat_spn, data))
