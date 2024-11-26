from typing import Optional, Union

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


class Split(Module):
    def __init__(self, inputs: Module, dim: int = -1):
        """
        Split a single module along a given dimension.

        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
        """
        super().__init__()
        self.inputs = nn.ModuleList(inputs)
        raise NotImplementedError

    @property
    def out_features(self) -> int:
        raise NotImplementedError

    @property
    def out_channels(self) -> int:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Split,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = log_likelihood(module.inputs, data, check_support, dispatch_ctx)
    num_splits = 2  # TODO: implement specific split
    lls_split = lls.split(module.inputs.out_features // num_splits, dim=module.dim)

    return lls_split


@dispatch  # type: ignore
def sample(
    module: Split,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])
    raise NotImplementedError


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: Split,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, Module]:
    # Initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError
