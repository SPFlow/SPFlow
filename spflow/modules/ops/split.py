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


class Split(Module):
    def __init__(self, inputs: Module, dim: int = 1, num_splits: Optional[int] = 2, split_func: Optional[Callable[[torch.Tensor], list[torch.Tensor]]] = None):
        """
        Split a single module along a given dimension.

        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
        """
        super().__init__()
        self.inputs = nn.ModuleList([inputs])

        self.dim = dim
        self.num_splits = num_splits
        self.split_func = split_func


    @property
    def out_features(self) -> int:
        #if self.dim == 1:
        #    return self.inputs[0].out_features // self.num_splits
        #else:
        return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        #if self.dim == 2:
        #    return self.inputs[0].out_channels // self.num_splits
        #else:
        return self.inputs[0].out_channels

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Split,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> list[Tensor]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = log_likelihood(module.inputs[0], data, check_support, dispatch_ctx)
    if module.split_func is not None:
        lls_split = module.split_func(lls)
    else:
        lls_split = lls.split(module.inputs[0].out_features // module.num_splits, dim=module.dim)

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

    # Expand mask and channels to match input module shape
    mask = sampling_ctx.mask.expand(data.shape[0], module.inputs[0].out_features)
    channel_index = sampling_ctx.channel_index.expand(data.shape[0], module.inputs[0].out_features)
    sampling_ctx.update(channel_index=channel_index, mask=mask)

    sample(
        module.inputs[0],
        data,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
    return data


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
