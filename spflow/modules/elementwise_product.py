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


class ElementwiseProduct(BaseProduct):
    def __init__(
        self,
        inputs: Union[Module, tuple[Module, Module], list[Module]],
    ) -> None:
        r"""Initializes ``ElementwiseProduct`` object.

        Args:
            inputs:
                List of Modules.
                The scopes for all child modules need to be pair-wise disjoint.

                (1) If `inputs` is a list of Modules, they have to be of disjoint scopes and have equal number of features or a single feature wich will the be broadcast and an equal number of channels or a single channel which will be broadcast.

                Example shapes:
                    inputs = ((3, 4), (3, 4))
                    output = (3, 4)

                    inputs = ((3, 4), (3, 1))
                    output = (3, 4)  # broadcasted

                    inputs = ((3, 4), (1, 4))
                    output = (3, 4)  # broadcasted

                    inputs = ((3, 1), (1, 4))
                    output = (3, 4)  # broadcasted


        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(inputs=inputs)

        # Check if all inputs either have equal number of out_channels or 1
        if not all(inp.out_channels in (1, self._max_out_channels) for inp in self.inputs):
            raise ValueError(
                f"Inputs must have equal number of channels or one of them must be '1', but were {[inp.out_channels for inp in self.inputs]}"
            )

    @property
    def out_channels(self) -> int:
        """Returns the number of output nodes for this module."""
        # Max since one of the inputs can also only have a single output channel which is then broadcasted
        return self._max_out_channels

    def map_out_channels_to_in_channels(self, index: Tensor) -> Tensor:
        cids = []
        for inp in self.inputs:
            if inp.out_channels == 1:
                cids.append(torch.zeros_like(index))
            else:
                cids.append(index)
        return torch.stack(cids, dim=-1)

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        masks = []
        for inp in self.inputs:
            if inp.out_channels == 1:
                masks.append(torch.full_like(mask, fill_value=True))
            else:
                masks.append(mask)
        return torch.stack(masks, dim=-1)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: ElementwiseProduct,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    lls = _get_input_log_likelihoods(module, data, check_support, dispatch_ctx)

    # Check if we need to expand to enable broadcasting along channels
    for i, ll in enumerate(lls):
        if ll.shape[2] == 1:
            lls[i] = ll.expand(-1, -1, module.out_channels)

    # Compute the elementwise sum of left and right split
    output = torch.sum(torch.stack(lls, dim=-1), dim=-1)

    # View as [b, n, m]
    output = output.view(output.size(0), module.out_features, module.out_channels)

    return output
