import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.module import Module


@dispatch(memoize=True)
def log_likelihood(module: Module, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError()


@dispatch
def likelihood(module: Module, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return torch.exp(log_likelihood(module, data, dispatch_ctx=dispatch_ctx))