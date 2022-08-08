"""
Created on August 08, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.module import Module

import numpy as np
from typing import Optional


@dispatch
def sample(module: Module, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    return sample(module, 1, dispatch_ctx=dispatch_ctx)


@dispatch
def sample(module: Module, n: int, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    
    sampling_ctx = SamplingContext(list(range(n)))
    data = np.full((n, max(module.scope.query)+1), float("nan"))

    return sample(module, data, dispatch_ctx=dispatch_ctx, sampling_ctx=sampling_ctx)