from itertools import product
from typing import Optional, Union

import torch
from torch import Tensor

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.base_product import BaseProduct, _get_input_log_likelihoods
from spflow.modules.module import Module


class OuterProduct(BaseProduct):
    def __init__(
        self,
        inputs: list[Module],
        num_splits: Optional[int] = None,
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

        if len(inputs) == 1:
            assert num_splits is not None and num_splits > 1

        self.num_splits = num_splits

        # Store unraveled channel indices
        unraveled_channel_indices = list(product(*[list(range(self._max_out_channels)) for _ in self.inputs]))
        self.register_buffer(
            name="unraveled_channel_indices",
            tensor=torch.tensor(unraveled_channel_indices),
        )

    @property
    def out_channels(self) -> int:
        """Returns the number of output nodes for this module."""
        ocs = 1
        for inp in self.inputs:
            ocs *= inp.out_channels
        return ocs

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        return self.unraveled_channel_indices[output_ids]

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        return mask.unsqueeze(-1).expand(-1, -1, len(self.inputs))


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: OuterProduct,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    lls = _get_input_log_likelihoods(module, data, check_support, dispatch_ctx)

    if len(module.inputs) == 1:  # If input is a single module, perform binary split manually
        # TODO: padding if module.inputs[0].out_features % module.num_splits != 0
        lls = lls[0].split(module.inputs[0].out_features // module.num_splits, dim=1)

    # Compute the outer sum of pairwise log-likelihoods
    # [b, n, m1] + [b, n, m2] -> [b, n, m1, 1] + [b, n, 1, m2]  -> [b, n, m1, m2] -> [b, n, 1, m1*m2] ...
    output = lls[0].unsqueeze(2)
    for i in range(1, len(lls)):
        output = output + lls[i].unsqueeze(3)
        output = output.view(output.size(0), output.size(1), 1, -1)

    # View as [b, n, m1 * m2]
    output = output.view(output.size(0), module.out_features, module.out_channels)

    return output
