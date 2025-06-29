from typing import Optional, Union, Callable, Optional
from abc import abstractmethod
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

# abstract split module
class Split(Module):

    def __init__(self, inputs: Module, dim: int = 1, num_splits: Optional[int] = 2):
        """
        Base Split module to split a single module along a given dimension.

        Args:
            inputs:
            dim: Split dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
            num_splits: Number of splits along the given dimension.
        """
        super().__init__()
        self.inputs = nn.ModuleList([inputs])

        self.dim = dim
        self.num_splits = num_splits
        self.num_repetitions = self.inputs[0].num_repetitions

    @property
    def out_features(self) -> int:
        return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        return self.inputs[0].out_channels

    def get_out_shapes(self, event_shape):
        """
        Get the output shapes of the split operation based on the input event shape.
        """
        split_size = event_shape[self.dim]
        quotient = split_size // self.num_splits
        remainder = split_size % self.num_splits
        if self.dim == 0:
            if remainder == 0:
                return [(quotient, event_shape[1])] * self.num_splits
            else:
                return [(quotient, event_shape[1])] * (self.num_splits-1) + [(remainder, event_shape[1])]

        else:
            if remainder == 0:
                return [(event_shape[0], quotient)] * self.num_splits
            else:
                return [(event_shape[0], quotient)] * (self.num_splits - 1) + [(event_shape[1], remainder)]



    @property
    @abstractmethod
    def feature_to_scope(self) -> list[Scope]:
        pass

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

    # compute module scope (same for all outputs)
    module_scope = module.scope

    mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))

    # Node scope is only being partially marginalized
    if mutual_rvs:
        # marginalize child modules

        marg_child_module = marginalize(module.inputs[0], marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
        if marg_child_module:
            if prune and marg_child_module.out_features == 1:
                return marg_child_module
            else:
                return module.__class__(inputs=marg_child_module, dim=module.dim, num_splits=module.num_splits)

        # if all children were marginalized, return None
        else:
            return None

        # if only a single input survived marginalization, return it if pruning is enabled
    else:
        return module