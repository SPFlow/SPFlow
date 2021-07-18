from .rat_spn import TorchRatSpn
from multipledispatch import dispatch
import torch


@dispatch(TorchRatSpn, torch.Tensor)
def log_likelihood(rat_spn: TorchRatSpn, data: torch.Tensor) -> torch.Tensor:
    # make sure data has double precision
    data = data.double()
    return rat_spn.forward(data)


@dispatch(TorchRatSpn, torch.Tensor)
def likelihood(rat_spn: TorchRatSpn, data: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_likelihood(rat_spn, data))
