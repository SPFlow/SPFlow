"""
Created on September 25, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer


@dispatch(memoize=True)
def maximum_likelihood_estimation(layer: UniformLayer, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = torch.hstack([data[:, scope.query] for scope in layer.scopes_out])

    if torch.any(~layer.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'UniformLayer'.")

    # do nothing since there are no learnable parameters
    pass


@dispatch(memoize=True)
def em(layer: UniformLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # update parameters through maximum weighted likelihood estimation (NOTE: simply for checking support)
    maximum_likelihood_estimation(layer, data, bias_correction=False, dispatch_ctx=dispatch_ctx)