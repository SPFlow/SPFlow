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


class Cat(Module):
    def __init__(self, inputs: list[Module], dim: int = -1):
        """
        Concatenation of multiple modules along a given dimension.

        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
        """
        super().__init__()
        self.inputs = nn.ModuleList(inputs)
        self.dim = dim

        if self.dim == 1:
            # Check if all inputs have the same number of channels
            if not all([module.out_channels == self.inputs[0].out_channels for module in self.inputs]):
                raise ValueError("All inputs must have the same number of channels.")

            # Check that all scopes are disjoint
            if not Scope.all_pairwise_disjoint([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have disjoint scopes.")

            # Scope is the join of all input scopes
            for inp in self.inputs:
                self._scope = self._scope.join(inp.scope)

        elif self.dim == 2:
            # Check if all inputs have the same number of features and scopes
            if not all([module.out_features == self.inputs[0].out_features for module in self.inputs]):
                raise ValueError("All inputs must have the same number of features.")
            if not Scope.all_equal([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have the same scope.")

            # Scope is the same as all inputs
            self._scope = self.inputs[0].scope
        else:
            raise ValueError("Invalid dimension for concatenation.")

    @property
    def out_features(self) -> int:
        if self.dim == 1:
            return sum([module.out_features for module in self.inputs])
        else:
            return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        if self.dim == 2:
            return sum([module.out_channels for module in self.inputs])
        else:
            return self.inputs[0].out_channels

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Cat,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get log likelihoods for all inputs
    lls = []
    for input_module in module.inputs:
        lls.append(log_likelihood(input_module, data, check_support, dispatch_ctx))

    # Concatenate log likelihoods
    return torch.cat(lls, dim=module.dim)


@dispatch  # type: ignore
def sample(
    module: Cat,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if module.dim == 1:
        split_size = module.out_features // len(module.inputs)
        channel_index_per_module = sampling_ctx.channel_index.split(split_size, dim=module.dim)
        mask_per_module = sampling_ctx.mask.split(split_size, dim=module.dim)
    elif module.dim == 2:
        # Concatenation happens at out_channels
        # Therefore, we need to use modulo to get the correct output_ids
        channel_index_per_module = []
        mask_per_module = []

        # Get split assignments
        split_size = module.out_channels // len(module.inputs)
        split_assignment = sampling_ctx.channel_index // split_size
        for i, _ in enumerate(module.inputs):
            oids = sampling_ctx.channel_index
            oids_mod = oids.remainder(split_size)
            channel_index_per_module.append(oids_mod)
            mask = split_assignment == i & sampling_ctx.mask
            mask_per_module.append(mask)


    else:
        raise ValueError("Invalid dimension for concatenation.")

    # Iterate over inputs
    for i in range(len(module.inputs)):
        input_module = module.inputs[i]
        sampling_ctx_copy = sampling_ctx.copy()
        sampling_ctx_copy.update(channel_index=channel_index_per_module[i], mask=mask_per_module[i])

        sample(
            input_module,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx_copy,
        )

    return data


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: Cat,
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

        return Cat(inputs=inputs, dim=module.dim)
    else:
        return module
