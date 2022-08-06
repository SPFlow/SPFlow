import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.module import Module


@dispatch(memoize=True)
def log_likelihood(module: Module, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    raise NotImplementedError()


@dispatch
def likelihood(module: Module, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    return np.exp(log_likelihood(module, data, dispatch_ctx=dispatch_ctx))