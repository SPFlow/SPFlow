from __future__ import annotations

from itertools import product

import numpy as np
import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_alternate import SplitAlternate
from spflow.modules.ops.split_halves import SplitHalves
from spflow.modules.products.base_product import BaseProduct
from spflow.utils.cache import Cache, cached


class OuterProduct(BaseProduct):
    """Outer product creating all pairwise channel combinations.

    Computes Cartesian product of input channels. Output channels equal
    product of input channels. All input scopes must be pairwise disjoint.

    Attributes:
        unraveled_channel_indices (Tensor): Mapping from output to input channel pairs.
    """

    def __init__(
        self,
        inputs: list[Module],
        num_splits: int | None = 2,
    ) -> None:
        """Initialize outer product.

        Args:
            inputs: Modules with pairwise disjoint scopes.
            num_splits: Number of splits for input operations.
        """
        super().__init__(inputs=inputs)

        if len(self.inputs) == 1:
            if num_splits is None or num_splits <= 1:
                raise ValueError("num_splits must be at least 2 when input is a single module")

        self.num_splits = num_splits

        # Store unraveled channel indices
        if self.input_is_split:
            unraveled_channel_indices = list(
                product(*[list(range(self._max_out_channels)) for _ in range(self.num_splits)])
            )
        else:
            unraveled_channel_indices = list(
                product(*[list(range(self._max_out_channels)) for _ in self.inputs])
            )
        self.register_buffer(
            name="unraveled_channel_indices",
            tensor=torch.tensor(unraveled_channel_indices),
        )
        self.check_shapes()

    def check_shapes(self, inputs=None):
        """Check if input shapes are compatible for outer product.

        Args:
            inputs: Input modules to check shapes for. If None, uses self.inputs.

        Returns:
            bool: True if shapes are compatible, False if no shapes to check.

        Raises:
            ValueError: If input shapes are not broadcastable.
        """
        if inputs is None:
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
        raise ValueError(f"the shapes of the inputs {shapes} are not broadcastable")

    @property
    def out_channels(self) -> int:
        ocs = 1
        for inp in self.inputs:
            ocs *= inp.out_channels
        if len(self.inputs) == 1:
            ocs = ocs**self.num_splits
        return ocs

    @property
    def out_features(self) -> int:
        if self.input_is_split:
            return int(self.inputs[0].out_features // self.num_splits)
        else:
            return self.inputs[0].out_features

    @property
    def feature_to_scope(self) -> np.ndarray:
        result_scopes = []
        for r in range(self.num_repetitions):
            if isinstance(self.inputs, Split):
                scope_lists = self.inputs.feature_to_scope[:, r]
            else:
                scope_lists = [module.feature_to_scope[:, r] for module in self.inputs]

            outer_product = list(product(*scope_lists))

            feature_to_scope = []
            for joined_scopes in outer_product:
                feature_to_scope.append(Scope.join_all(joined_scopes))

            result_scopes.append(feature_to_scope)

        # Transpose from (num_repetitions, num_features) to (num_features, num_repetitions)
        return np.array(result_scopes).T

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        """Map output channel indices to input channel indices.

        Args:
            output_ids: Tensor of output channel indices to map.

        Returns:
            Tensor: Mapped input channel indices corresponding to the output channels.

        Raises:
            NotImplementedError: If split type is not supported.
        """
        if self.input_is_split:
            if isinstance(self.inputs[0], SplitHalves):
                return self.unraveled_channel_indices[output_ids].permute(0, 2, 1).flatten(1, 2).unsqueeze(-1)
            elif isinstance(self.inputs[0], SplitAlternate):
                return (
                    self.unraveled_channel_indices[output_ids]
                    .view(-1, self.inputs[0].out_features)
                    .unsqueeze(-1)
                )
            else:
                raise NotImplementedError("Other Split types are not implemented yet.")
        else:
            return self.unraveled_channel_indices[output_ids]

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        """Map output mask to input masks.

        Args:
            mask: Output mask tensor to map to input masks.

        Returns:
            Tensor: Mapped input masks corresponding to the output mask.

        Raises:
            NotImplementedError: If split type is not supported.
        """
        num_inputs = len(self.inputs) if not self.input_is_split else self.num_splits
        if self.input_is_split:
            if isinstance(self.inputs[0], SplitHalves):
                return (
                    mask.unsqueeze(-1).repeat(1, 1, num_inputs).permute(0, 2, 1).flatten(1, 2).unsqueeze(-1)
                )
            elif isinstance(self.inputs[0], SplitAlternate):
                return (
                    mask.unsqueeze(-1)
                    .repeat(1, 1, num_inputs)
                    .view(-1, self.inputs[0].out_features)
                    .unsqueeze(-1)
                )
            else:
                raise NotImplementedError("Other Split types are not implemented yet.")
        else:
            return mask.unsqueeze(-1).repeat(1, 1, num_inputs)

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood via outer sum of pairwise combinations.

        Args:
            data: Input data tensor for computing log likelihood.
            cache: Optional cache for storing intermediate computations.

        Returns:
            Tensor: Log likelihood values with shape [batch_size, out_features, out_channels, num_repetitions].

        Raises:
            ValueError: If output tensor has invalid number of dimensions.
        """
        # initialize cache

        lls = self._get_input_log_likelihoods(data, cache)

        # Compute the outer sum of pairwise log-likelihoods
        # [b, n, m1] + [b, n, m2] -> [b, n, m1, 1] + [b, n, 1, m2]  -> [b, n, m1, m2] -> [b, n, 1, m1*m2] ...

        output = lls[0]
        for i in range(1, len(lls)):
            output = output.unsqueeze(3) + lls[i].unsqueeze(2)
            if output.ndim == 4:
                output = output.view(output.size(0), self.out_features, -1)
            elif output.ndim == 5:
                output = output.view(output.size(0), self.out_features, -1, self.num_repetitions)
            else:
                raise ValueError("Invalid number of dimensions")

        # View as [b, n, m1 * m2, r]
        if self.num_repetitions is None:
            output = output.view(output.size(0), self.out_features, self.out_channels)
        else:
            output = output.view(output.size(0), self.out_features, self.out_channels, self.num_repetitions)
        return output
