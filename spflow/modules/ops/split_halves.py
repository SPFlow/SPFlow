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


class SplitHalves(Split): # ToDo: make abstract and implement concrete classes

    def __init__(self, inputs: Module, dim: int = 1, num_splits: Optional[int] = 2, split_func: Optional[Callable[[torch.Tensor], list[torch.Tensor]]] = None):
        """
        Split a single module along a given dimension. This implementation splits the features consecutively.
        Example:
            If num_splits=2, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5]
            - Split 0: features [0, 1, 2]
            - Split 1: features [3, 4, 5]
            If num_splits=3, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5]
            - Split 0: features [0, 1]
            - Split 1: features [2, 3]
            - Split 2: features [4, 5]


        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
            num_splits: Number of splits along the given dimension.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)


    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> list[Scope]:
        scopes = self.inputs[0].feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        feature_to_scope = []
        for i in range(self.num_splits):
             sub_scopes = scopes[i*num_scopes_per_chunk:(i+1)*num_scopes_per_chunk]
             feature_to_scope.append(sub_scopes)
        return feature_to_scope




@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: SplitHalves,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> list[Tensor]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = log_likelihood(module.inputs[0], data, check_support, dispatch_ctx)

    lls_split = lls.split(module.inputs[0].out_features // module.num_splits, dim=module.dim)

    return lls_split





@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: SplitHalves,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, Module]:
    # Initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError
