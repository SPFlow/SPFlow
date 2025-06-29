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
from spflow.meta.data import Scope
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_halves import SplitHalves
from spflow.modules.ops.split_alternate import SplitAlternate
import time

class OuterProduct(BaseProduct):
    def __init__(
        self,
        inputs: list[Module],
        num_splits: Optional[int] = 2,
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

        if len(self.inputs) == 1:
            assert num_splits is not None and num_splits > 1

        self.num_splits = num_splits

        # Store unraveled channel indices
        if self.input_is_split:
            unraveled_channel_indices = list(product(*[list(range(self._max_out_channels)) for _ in range(self.num_splits)]))
        else:
            unraveled_channel_indices = list(product(*[list(range(self._max_out_channels)) for _ in self.inputs]))
        self.register_buffer(
            name="unraveled_channel_indices",
            tensor=torch.tensor(unraveled_channel_indices),
        )
        self.check_shapes()

    def check_shapes(self, inputs=None):
        """
        Checks if the list of two-dimensional shapes satisfies the given conditions.
        """
        if inputs is None:
            inputs = self.inputs

        if self.input_is_split:
            assert self.num_splits == inputs[0].num_splits, "num_splits must be the same for all inputs"
            shapes = inputs[0].get_out_shapes((self.out_features, self.out_channels))
        else:
            shapes = [(inp.out_features, inp.out_channels) for inp in inputs]

        if not shapes:
            return False  # No shapes to check

        # Extract dimensions
        dim0_values = [shape[0] for shape in shapes]
        dim1_values = [shape[1] for shape in shapes]

        # Check if all shapes have the same first dimension
        if len(set(dim0_values)) == 1:
            return True

        # Check if all shapes have the same second dimension
        if len(set(dim1_values)) == 1:
            return True

        # Check if all but one of the first dimensions are 1
        if dim0_values.count(1) == len(dim0_values) - 1:
            return True

        # Check if all but one of the second dimensions are 1
        if dim1_values.count(1) == len(dim1_values) - 1:
            return True

        # If none of the conditions are satisfied
        raise ValueError(f"the shapes of the inputs { shapes } are not broadcastable")

    @property
    def out_channels(self) -> int:
        ocs = 1
        for inp in self.inputs:
            ocs *= inp.out_channels
        if len(self.inputs) == 1:
            ocs = ocs ** self.num_splits
        return ocs

    @property
    def out_features(self) -> int:
        if self.input_is_split:
            return int(self.inputs[0].out_features // self.num_splits)
        else:
            return self.inputs[0].out_features

    @property
    def feature_to_scope(self) -> list[Scope]:
        if isinstance(self.inputs, Split):
            scope_lists = self.inputs.feature_to_scope
        else:
            scope_lists = [module.feature_to_scope for module in self.inputs]

        outer_product = list(product(*scope_lists))

        feature_to_scope = []
        for scopes in outer_product:
            feature_to_scope.append(Scope.join_all(scopes))
        return feature_to_scope

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        if self.input_is_split:
            if isinstance(self.inputs[0], SplitHalves):
                return self.unraveled_channel_indices[output_ids].permute(0,2,1).flatten(1,2).unsqueeze(-1)
            elif isinstance(self.inputs[0], SplitAlternate):
                return self.unraveled_channel_indices[output_ids].view(-1, self.inputs[0].out_features).unsqueeze(-1)
            else:
                raise NotImplementedError("Other Split types are not implemented yet.")
        else:
            return self.unraveled_channel_indices[output_ids]


    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        num_inputs = len(self.inputs) if not self.input_is_split else self.num_splits
        if self.input_is_split:
            if isinstance(self.inputs[0], SplitHalves):
                return mask.unsqueeze(-1).repeat(1, 1, num_inputs).permute(0,2,1).flatten(1,2).unsqueeze(-1)
            elif isinstance(self.inputs[0], SplitAlternate):
                return mask.unsqueeze(-1).repeat(1, 1, num_inputs).view(-1, self.inputs[0].out_features).unsqueeze(-1)
            else:
                raise NotImplementedError("Other Split types are not implemented yet.")
        else:
            return mask.unsqueeze(-1).repeat(1, 1, num_inputs)


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

    # Compute the outer sum of pairwise log-likelihoods
    # [b, n, m1] + [b, n, m2] -> [b, n, m1, 1] + [b, n, 1, m2]  -> [b, n, m1, m2] -> [b, n, 1, m1*m2] ...

    output = lls[0]
    for i in range(1, len(lls)):
        output = output.unsqueeze(3) + lls[i].unsqueeze(2)
        if output.ndim == 4:
            output = output.view(output.size(0), module.out_features, -1)
        elif output.ndim == 5:
            output = output.view(output.size(0), module.out_features, -1, module.num_repetitions)
        else:
            raise ValueError("Invalid number of dimensions")


    # View as [b, n, m1 * m2, r]
    if module.num_repetitions is None:
        output = output.view(output.size(0), module.out_features, module.out_channels)
    else:
        output = output.view(output.size(0), module.out_features, module.out_channels, module.num_repetitions)
    return output
