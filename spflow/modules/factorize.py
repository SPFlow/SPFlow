from itertools import product
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
from spflow.modules.base_product import BaseProduct, _get_input_log_likelihoods
from spflow.modules.module import Module
from spflow.modules import Product
import numpy as np


class Factorize(BaseProduct):

    def __init__(
        self,
        inputs: list[Module],
        depth: int,
        num_repetitions: int,
    ) -> None:
        r"""Initializes ``OuterProduct`` module.

        Args:
            inputs:
                Can be either a Module or a list of Modules.
                The scopes for all child modules need to be pair-wise disjoint.

                (1) If `inputs` is a list of Modules, they have to be of disjoint scopes and have equal number of features or a single feature wich will the be broadcast.

                Example shapes:
                    inputs = ((3, 4), (3, 5))
                    output = (3, 4 * 5)

                    inputs = ((3, 4), (3, 1))
                    output = (3, 4 * 1)  # broadcasted

                    inputs = ((3, 4), (1, 5))
                    output = (3, 4 * 5)  # broadcasted


        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(inputs=inputs)

        self.depth = depth
        self.num_repetitions = num_repetitions
        self.indices = self._factorize(depth, num_repetitions)# shape: [num_features_in, num_features_out, num_repetitions]

    @property
    def out_channels(self) -> int:
        """Returns the number of output nodes for this module."""
        return self.inputs[0].out_channels

    @property
    def out_features(self) -> int:
        return 2 ** self.depth

    @property
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs[0].feature_to_scope

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        return self.unraveled_channel_indices[output_ids]

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        return mask.unsqueeze(-1).expand(-1, -1, len(self.inputs))

    def _factorize(self, depth, num_repetitions):
        scope = self.inputs[0].scope
        num_features = len(scope.query)
        num_features_out = 2 ** depth
        assert num_features >= num_features_out
        cardinality = int(np.floor(num_features / num_features_out))
        group_sizes = np.ones(num_features_out, dtype=int) * cardinality
        rest = num_features - cardinality * num_features_out
        for i in range(rest):
            group_sizes[i] += 1
        np.random.shuffle(group_sizes)
        scopes = torch.zeros(num_features, num_features_out, num_repetitions)
        for r in range(num_repetitions):
            idxs = torch.randperm(n=num_features)
            offset = 0
            for o in range(num_features_out):
                group_size = group_sizes[o]
                low = offset
                high = offset + group_size
                offset = high
                if o == num_features_out - 1:
                    high = num_features
                scopes[idxs[low:high], o, r] = 1

        return scopes.to(torch.int32)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Factorize,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    lls = _get_input_log_likelihoods(module, data, check_support, dispatch_ctx) # lls[0] shape: [batch_size, num_features, num_channel]

    indices = module.indices.to(module.device) # shape: [n_in_f, n_out_f, num_repetitions]

    # sum over input feature dimension
    output = torch.sum(lls[0].unsqueeze(2) * indices.unsqueeze(0).unsqueeze(3), dim=1)

    return output

@dispatch  # type: ignore
def sample(
    module: Factorize,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # gather indices for specific repetitions
    rep_indices = sampling_ctx.repetition_idx.view(-1,1,1,1).expand(-1, module.indices.shape[0], module.indices.shape[1], -1)
    indices = module.indices.unsqueeze(0).expand(data.shape[0], -1, -1, -1).to(module.device)
    indices = torch.gather(indices, dim=-1, index=rep_indices).squeeze(-1)

    channel_index = torch.sum(sampling_ctx.channel_index.unsqueeze(1) * indices, dim=-1)

    mask = torch.sum(sampling_ctx.mask.unsqueeze(1) * indices, dim=-1).bool()

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
    layer: Factorize,
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
        marg_child_layer = marginalize(layer.inputs[0], marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

        # if marginalized child is not None
        if marg_child_layer:
            marg_child = marg_child_layer

    else:
        marg_child = layer.inputs[0]

    if marg_child is None:
        return None

    # ToDo: check if this is correct / prune if only one scope is left?
    elif prune and marg_child.out_features == 1:
        return marg_child
    else:
        return Factorize(inputs=[marg_child], depth=layer.depth, num_repetitions=layer.num_repetitions)



