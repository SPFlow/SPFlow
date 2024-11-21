from itertools import product
from typing import Optional, Union

import torch
from torch import Tensor

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.base_product import BaseProduct, _get_input_log_likelihoods
from spflow.modules.module import Module
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
        """
        if len(inputs) == 1:
            assert num_splits is not None and num_splits > 1

        self.num_splits = num_splits

        # Store unraveled channel indices
        unraveled_channel_indices = list(product(*[list(range(self._max_out_channels)) for _ in self.inputs]))
        self.register_buffer(
            name="unraveled_channel_indices",
            tensor=torch.tensor(unraveled_channel_indices),
        )
        """
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.indices = self._factorize(depth, num_repetitions) # shape: [num_features, num_groups, num_repetitions] other formulation: [num_features_in, num_features_out, num_repetitions]

    @property
    def out_channels(self) -> int:
        """Returns the number of output nodes for this module."""
        return self.inputs[0].out_channels

    @property
    def out_features(self) -> int:
        return 2 ** self.depth

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
    indices = module.indices # shape: [num_features, num_groups, num_repetitions] other formulation: [num_features_in, num_features_out, num_repetitions]


    output = torch.einsum('fgr,bfcr->bgcr', indices.float(), lls[0]) # shape: [batch_size, num_groups, num_channel, num_repetitions]


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
    # ToDo: generalize for repepitions / adapd sampling context
    data = (data.unsqueeze(-1)* module.indices[...,0]).sum(-1)
    return data

    """
    scopes = module.indices.permute(2, 0, 1)
    rnge_in = torch.arange(module.out_features, device=data.device)
    scopes = (scopes * rnge_in).sum(-1).long()
    indices_in_gather = sampling_ctx.channel_index.gather(dim=1, index=scopes)
    indices_in_gather = indices_in_gather.view(sampling_ctx.channel_index.shape[0], 1, -1, 1)

    indices_in_gather = indices_in_gather.expand(-1, data.shape[1], -1, -1)
    indices_in_gather = indices_in_gather.repeat(1, 1, module.inputs[0].out_features, 1)
    samples = data.gather(dim=-1, index=indices_in_gather)
    samples.squeeze_(-1)
    """


