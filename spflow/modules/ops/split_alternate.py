from typing import Optional, Union, Callable, Optional

import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.meta.dispatch import (
    DispatchContext,
    init_default_dispatch_context,
    SamplingContext,
    init_default_sampling_context,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.modules.module import Module
from spflow.modules.ops.split import Split


class SplitAlternate(Split): # ToDo: make abstract and implement concrete classes

    def __init__(self, inputs: Module, dim: int = 1, num_splits: Optional[int] = 2):
        """
        Split a single module along a given dimension.

        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)
        #self.inputs = nn.ModuleList([inputs])

        #self.dim = dim
        #self.num_splits = num_splits


    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> list[Scope]:
        scopes = self.inputs[0].feature_to_scope
        feature_to_scope = []
        for i in range(self.num_splits):
             sub_scopes = scopes[i::self.num_splits]
             feature_to_scope.append(sub_scopes)
        return feature_to_scope


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: SplitAlternate,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> list[Tensor]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = log_likelihood(module.inputs[0], data, check_support, dispatch_ctx)
    # Split the tensor along the specified dimension
    # Get the size of the specified dimension
    size = lls.size(module.dim)

    # Create indices for the split
    indices = torch.arange(size, device=lls.device) % module.num_splits

    # Create masks for each split
    masks = [indices == i for i in range(module.num_splits)]

    # Use advanced indexing to extract slices
    split_lls = [lls.index_select(module.dim, torch.nonzero(mask, as_tuple=True)[0]) for mask in masks]

    return split_lls


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: SplitAlternate,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, Module]:
    # Initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError
