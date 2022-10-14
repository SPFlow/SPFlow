"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform


# TODO: MLE dispatch context?


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: Uniform, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    if torch.any(~leaf.check_support(data[:, leaf.scope.query])):
        raise ValueError("Encountered values outside of the support for 'Uniform'.")

    # do nothing since there are no learnable parameters
    pass


@dispatch
def em(leaf: Uniform, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # update parameters through maximum weighted likelihood estimation (NOTE: simply for checking support)
    maximum_likelihood_estimation(leaf, data, bias_correction=False)