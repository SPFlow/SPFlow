"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric

import torch
from typing import Optional


@dispatch
def sample(leaf: Hypergeometric, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    raise NotImplementedError()
