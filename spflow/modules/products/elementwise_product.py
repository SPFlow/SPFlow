from __future__ import annotations

import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_alternate import SplitAlternate
from spflow.modules.ops.split_halves import SplitHalves
from spflow.modules.products.base_product import BaseProduct
from spflow.utils.cache import Cache, init_cache


class ElementwiseProduct(BaseProduct):
    """Elementwise product with automatic broadcasting.

    Multiplies inputs element-wise with broadcasting support. All input scopes
    must be pairwise disjoint. Commonly used in RAT-SPN architectures.
    """

    def __init__(
        self,
        inputs: Module | tuple[Module, Module] | list[Module],
        num_splits: int | None = 2,
    ) -> None:
        """Initialize elementwise product.

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

        self.num_repetitions = self.inputs[0].num_repetitions

        if self.num_splits == None:
            self.num_splits = num_splits

        self.check_shapes()

    def check_shapes(self):
        """Check if input shapes are compatible for broadcasting."""
        inputs = self.inputs

        if self.input_is_split:
            if self.num_splits != inputs[0].num_splits:
                raise ValueError("num_splits must be the same for all inputs")
            shapes = inputs[0].get_out_shapes((self.out_features, self.out_channels))
        else:
            shapes = [(inp.out_features, inp.out_channels) for inp in inputs]

        if not shapes:
            return False  # No shapes to check

        # Extract dimensions
        dim0_values = [shape[0] for shape in shapes]
        dim1_values = [shape[1] for shape in shapes]

        # Condition 1: All shapes are the same
        if all(shape == shapes[0] for shape in shapes):
            return True

        # Condition 2: dim0 is the same and dim1 is the same or has the value 1
        if len(set(dim0_values) - {1}) == 1:
            return True

        if len(set(dim1_values) - {1}) == 1:
            return True

        # Condition 4: In dim0 every value except one has the value 1,
        # and in dim1 every value except one has the value 1
        dim0_non_ones = [value for value in dim0_values if value != 1]
        dim1_non_ones = [value for value in dim1_values if value != 1]

        if len(dim0_non_ones) <= 1 and len(dim1_non_ones) <= 1:
            return True

        # If none of the conditions are satisfied
        raise ValueError(f"the shapes of the inputs {shapes} are not broadcastable")

    @property
    def out_channels(self) -> int:
        # Max since one of the inputs can also only have a single output channel which is then broadcasted
        return self._max_out_channels

    @property
    def out_features(self) -> int:
        if self.inputs[0].out_features == 1:
            return 1
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

        feature_to_scope = []
        # Group elements by index
        grouped_scopes = list(zip(*scope_lists))
        for scopes in grouped_scopes:
            feature_to_scope.append(Scope.join_all(scopes))
        return feature_to_scope

    def map_out_channels_to_in_channels(self, index: Tensor) -> Tensor:
        if self.input_is_split:
            num_splits = self.num_splits
            if isinstance(self.inputs[0], SplitHalves):
                return index.repeat((1, num_splits)).unsqueeze(-1)
            elif isinstance(self.inputs[0], SplitAlternate):
                return index.repeat_interleave(num_splits, dim=1).unsqueeze(-1)
            else:
                raise NotImplementedError("Other Split types are not implemented yet.")
        else:
            num_splits = len(self.inputs)
            return index.unsqueeze(-1).repeat(1, 1, num_splits)

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        if self.input_is_split:
            num_splits = self.num_splits
            return mask.repeat((1, num_splits)).unsqueeze(-1)
        else:
            num_splits = len(self.inputs)
            return mask.unsqueeze(-1).repeat(1, 1, num_splits)

    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood by element-wise summing inputs."""
        # initialize cache
        cache = init_cache(cache)

        lls = self._get_input_log_likelihoods(data, cache)

        # Check if we need to expand to enable broadcasting along channels
        for i, ll in enumerate(lls):
            if ll.shape[2] == 1:
                if ll.ndim == 4:
                    lls[i] = ll.expand(-1, -1, self.out_channels, -1)
                else:
                    lls[i] = ll.expand(-1, -1, self.out_channels)

        # Compute the elementwise sum of left and right split
        output = torch.sum(torch.stack(lls, dim=-1), dim=-1)

        # View as [b, n, m, r]
        if output.ndim == 4:
            output = output.view(output.size(0), self.out_features, self.out_channels, -1)
        else:
            output = output.view(output.size(0), self.out_features, self.out_channels)

        return output
