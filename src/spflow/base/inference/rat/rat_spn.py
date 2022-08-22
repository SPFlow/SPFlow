"""
Created on August 22, 2022

@authors: Philipp Deibert
"""
import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.structure.rat.rat_spn import RatSPN


@dispatch(memoize=True)
def log_likelihood(rat_spn: RatSPN, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return log_likelihood(rat_spn.root_node, data, dispatch_ctx=dispatch_ctx)