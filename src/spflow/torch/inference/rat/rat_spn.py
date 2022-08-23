"""
Created on August 23, 2022

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.rat.rat_spn import RatSPN


@dispatch(memoize=True)
def log_likelihood(rat_spn: RatSPN, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return log_likelihood(rat_spn.root_node, data, dispatch_ctx=dispatch_ctx)