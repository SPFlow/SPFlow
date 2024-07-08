from typing import Optional, Union

import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.meta.dispatch import (
    DispatchContext,
    init_default_dispatch_context,
    init_default_sampling_context,
    SamplingContext,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.modules.module import Module


class Stack(Module):
    def __init__(self, inputs: list[Module]):
        super().__init__()
        self.inputs = nn.ModuleList(inputs)

        # Check that all inputs have the same shape
        if not all([module.out_features == self.inputs[0].out_features for module in self.inputs]):
            raise ValueError("All inputs must have the same number of features.")

        self._out_channels = max([module.out_channels for module in self.inputs])
        if not all([module.out_channels in (1, self._out_channels) for module in self.inputs]):
            raise ValueError(
                "All inputs must have the same number of channels or a single channel which is then broadcast."
            )
        if not Scope.all_equal([module.scope for module in self.inputs]):
            raise ValueError("All inputs must have the same scope.")

        # Set scope
        self._scope = self.inputs[0].scope

    @property
    def out_features(self) -> int:
        return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Stack,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = []
    for input_module in module.inputs:
        ll = log_likelihood(input_module, data, check_support, dispatch_ctx)

        # Check if we need to expand to enable broadcasting along channels
        if ll.shape[2] == 1:
            ll = ll.expand(-1, -1, module.out_channels)

        lls.append(ll)

    # Concatenate log likelihoods
    return torch.stack(lls, dim=-1)


@dispatch  # type: ignore
def sample(
    module: Stack,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    """
    Unlike other modules, Stack return samples for each stack input, i.e. the output is a tensor of shape (batch,
    out_features, len(inputs)).
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    samples = []
    for i in range(len(module.inputs)):
        x = sample(
            module.inputs[i],
            data.clone(),  # Clone data to avoid modifying the original data
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx.copy(),
        )
        samples.append(x)

    samples = torch.stack(samples, dim=-1)
    return samples


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: Stack,
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
        inputs = []
        # marginalize child modules
        for input_module in module.inputs:
            marg_child_module = marginalize(input_module, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child_module:
                inputs.append(marg_child_module)

        # if all children were marginalized, return None
        if len(inputs) == 0:
            return None

        # if only a single input survived marginalization, return it if pruning is enabled
        if prune and len(inputs) == 1:
            return inputs[0]

        return Stack(inputs=inputs)
    else:
        return module
