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
        split_method: Optional[str] = None,
        split_indices: Optional[
            Union[tuple[list[int], list[int]], torch.IntTensor, tuple[torch.IntTensor, torch.IntTensor]]
        ] = None,
    ) -> None:
        r"""Initializes ``ElementwiseProduct`` object.

        Args:
            inputs:
                Can be either a Module or a list of Modules.
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

                (2) If `inputs` is a single Module, the input is split into two equal-sized subsets. This can be either done by specifying the `split_indices` argument or by specifying the `split_method` argument.

                Example shapes:
                    inputs = (4, 3), split_method = "split_indices", split_indices = ([0, 1], [2, 3])
                    splits = [(2, 3), (2, 3)]
                    output = (2, 3)

                    inputs = (4, 3), split_method = "random", split_indices = None
                    splits = [(2, 3), (2, 3)]  # random split
                    output = (2, 3)

            split_method:
                Method to split the input into two equal-sized subsets.
                Possible values are "split_indices" and "random".
                Defaults to "split_indices".

            split_indices:
                Indices to split the input into two equal-sized subsets.
                If split_method is set to "split_indices", this argument is required.
                Defaults to None.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(inputs=inputs, split_method=split_method, split_indices=split_indices)

    @property
    def out_channels(self) -> int:
        """Returns the number of output nodes for this module."""
        if self.has_single_input:
            return self.inputs.out_channels
        else:
            return self.inputs[0].out_channels


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

    # Compute the elementwise sum of left and right split
    output = sum(lls)

    # View as [b, n, m]
    output = output.view(output.size(0), module.out_features, module.out_channels)

    return output
