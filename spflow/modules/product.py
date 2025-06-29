from typing import Optional, Union

import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat


class Product(Module):
    """
    A product module that calculates the product over the feature dimension of its input modules.
    """

    def __init__(self, inputs: Union[Module, list[Module]]) -> None:
        """
        Args:
            inputs: Single input module or list of modules. The product is over the feature dimension of the input.
        """
        super().__init__()

        # If inputs is a list, ensure concatenation along the feature dimension
        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=1)
        else:
            self.inputs = inputs

        # Scope of this product module is equal to the scope of its only input
        self.scope = self.inputs.scope
        self.num_repetitions = self.inputs.num_repetitions

    @property
    def out_channels(self) -> int:
        return self.inputs.out_channels

    @property
    def out_features(self) -> int:
        return 1

    @property
    def feature_to_scope(self) -> list[Scope]:
        return [Scope.join_all(self.inputs.feature_to_scope)]


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: Product,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[Product, Module, None]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope
    marg_child = None
    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        # passing this loop means marginalizing over the whole scope of this branch
        pass
    # node scope is being partially marginalized
    elif mutual_rvs:
        # marginalize child modules
        marg_child_layer = marginalize(layer.inputs, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

        # if marginalized child is not None
        if marg_child_layer:
            marg_child = marg_child_layer

    else:
        marg_child = layer.inputs

    if marg_child is None:
        return None

    elif prune and marg_child.out_features == 1:
        return marg_child
    else:
        return Product(inputs=marg_child)


@dispatch  # type: ignore
def sample(
    module: Product,
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
    mask = sampling_ctx.mask.expand(data.shape[0], module.inputs.out_features)
    channel_index = sampling_ctx.channel_index.expand(data.shape[0], module.inputs.out_features)
    sampling_ctx.update(channel_index=channel_index, mask=mask)

    sample(
        module.inputs,
        data,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    product_layer: Product,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    ll = log_likelihood(
        product_layer.inputs,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )

    # multiply childen (sum in log-space)
    return torch.sum(ll, dim=1, keepdim=True)
