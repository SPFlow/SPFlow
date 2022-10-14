"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: Hypergeometric, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if torch.any(~leaf.check_support(data[:, leaf.scope.query])):
        raise ValueError("Encountered values outside of the support for 'Hypergeometric'.")

    # do nothing since there are no learnable parameters
    pass


@dispatch(memoize=True)
def em(leaf: Hypergeometric, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # update parameters through maximum weighted likelihood estimation (NOTE: simply for checking support)
    maximum_likelihood_estimation(leaf, data, bias_correction=False, dispatch_ctx=dispatch_ctx)