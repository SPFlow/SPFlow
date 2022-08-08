"""
Created on August 08, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric

import numpy as np
from typing import Optional


@dispatch
def sample(leaf: Hypergeometric, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:
    """TODO"""
    raise NotImplementedError()