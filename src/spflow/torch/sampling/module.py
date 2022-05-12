"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from spflow.torch.structure.module import TorchModule


@dispatch(TorchModule)  # type: ignore[no-redef]
def sample(module: TorchModule) -> torch.Tensor:
    return sample(module, 1)


@dispatch(TorchModule, int)  # type: ignore[no-redef]
def sample(module: TorchModule, n: int) -> torch.Tensor:
    return sample(
        module,
        torch.full((n, max(module.scope) + 1), float("nan")),
        ll_cache={},
        instance_ids=list(range(n)),
    )
