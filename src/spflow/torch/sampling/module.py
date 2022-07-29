"""
Created on May 10, 2022

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch
from spflow.base.sampling.sampling_context import SamplingContext
from spflow.torch.structure.module import TorchModule
from typing import Dict, Optional


@dispatch(TorchModule, ll_cache=dict, sampling_ctx=SamplingContext)  # type: ignore[no-redef]
def sample(module: TorchModule, ll_cache: Optional[Dict[TorchModule, torch.Tensor]]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    
    if ll_cache is None:
        ll_cache = {}
    if sampling_ctx is None:
        sampling_ctx = SamplingContext([0])

    return sample(module, max(sampling_ctx.instance_ids)+1, ll_cache=ll_cache, sampling_ctx=sampling_ctx)


@dispatch(TorchModule, int, ll_cache=dict, sampling_ctx=SamplingContext)  # type: ignore[no-redef]
def sample(module: TorchModule, n: int, ll_cache: Optional[Dict[TorchModule, torch.Tensor]]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    
    if ll_cache is None:
        ll_cache = {}
    if sampling_ctx is None:
        sampling_ctx = SamplingContext(list(range(n)))
    
    data = torch.full((n, max(module.scope)+1), float("nan"))

    return sample(module, data, ll_cache=ll_cache, sampling_ctx=sampling_ctx)


@dispatch(TorchModule, torch.Tensor, ll_cache=dict)  # type: ignore[no-redef]
def sample(module: TorchModule, data: torch.Tensor, ll_cache: Optional[Dict[TorchModule, torch.Tensor]]=None) -> torch.Tensor:

    if ll_cache is None:
        ll_cache = {}
    
    n = data.shape[0]

    return sample(module, data, ll_cache=ll_cache, sampling_ctx=SamplingContext(list(range(n))))


@dispatch(TorchModule, torch.Tensor, sampling_context=SamplingContext)  # type: ignore[no-redef]
def sample(module: TorchModule, data: torch.Tensor, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:

    if sampling_ctx is None:
        n = data.shape[0]
        sampling_ctx = SamplingContext(list(range(n)))

    return sample(module, data, ll_cache={}, sampling_ctx=sampling_ctx)


@dispatch(TorchModule, torch.Tensor)  # type: ignore[no-redef]
def sample(module: TorchModule, data: torch.Tensor) -> torch.Tensor:
    return sample(module, data, ll_cache={}, sampling_ctx=SamplingContext(list(range(data.shape[0]), [0 for _ in range(data.shape[0])])))