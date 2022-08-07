"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.module import Module

import torch
from typing import Optional


@dispatch
def sample(module: Module, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    return sample(module, 1, dispatch_ctx=dispatch_ctx)


@dispatch
def sample(module: Module, n: int, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    
    sampling_ctx = SamplingContext(list(range(n)))
    data = torch.full((n, max(module.scope.query)+1), float("nan"))

    return sample(module, data, dispatch_ctx=dispatch_ctx, sampling_ctx=sampling_ctx)