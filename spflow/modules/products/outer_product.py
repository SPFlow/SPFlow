from __future__ import annotations

from itertools import product

import numpy as np
import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
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
        self.check_shapes()

        if len(self.inputs) == 1:
            if num_splits is None or num_splits <= 1:
                raise ValueError("num_splits must be at least 2 when input is a single module")

        self.num_splits = num_splits

        # Store unraveled channel indices using in_shape.channels (set by BaseProduct)
        if self.input_is_split:
            unraveled_channel_indices = list(
                product(*[list(range(self.in_shape.channels)) for _ in range(self.num_splits)])
            )
        else:
            unraveled_channel_indices = list(
                product(*[list(range(self.in_shape.channels)) for _ in self.inputs])
            )
        self.register_buffer(
            name="unraveled_channel_indices",
            tensor=torch.tensor(unraveled_channel_indices),
        )

        # Shape computation: compute out_shape based on outer product of channels
        input_features = self.inputs[0].out_shape.features
        if self.input_is_split:
            out_features = int(input_features // self.num_splits)
        else:
            out_features = input_features
        
        # Compute out_channels as product of input channels
        ocs = 1
        for inp in self.inputs:
            ocs *= inp.out_shape.channels
        if len(self.inputs) == 1:
            ocs = ocs**self.num_splits
        out_channels = ocs
        
        self.out_shape = ModuleShape(out_features, out_channels, self.in_shape.repetitions)


    def check_shapes(self):
        """Check if input shapes are compatible for outer product.

        Returns:
            bool: True if shapes are compatible, False if no shapes to check.

        Raises:
            ValueError: If input shapes are not broadcastable.
        """
        # Compute out_features locally
        input_features = self.inputs[0].out_shape.features
        if self.input_is_split:
            out_features = int(input_features // self.num_splits)
        else:
            out_features = input_features
        
        # Compute out_channels as product of input channels
        ocs = 1
        for inp in self.inputs:
            ocs *= inp.out_shape.channels
        if len(self.inputs) == 1:
            ocs = ocs**self.num_splits
        out_channels = ocs

        if self.input_is_split:
            if self.num_splits != self.inputs[0].num_splits:
                raise ValueError("num_splits must be the same for all inputs")
            shapes = self.inputs[0].get_out_shapes((out_features, out_channels))
        else:
            shapes = [(inp.out_shape.features, inp.out_shape.channels) for inp in self.inputs]

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
    def feature_to_scope(self) -> list[Scope]:
        out = []
        for r in range(self.out_shape.repetitions):
            if isinstance(self.inputs, Split):
                scope_lists_r = self.inputs.feature_to_scope[
                    ..., r
                ]  # Shape: (num_features_per_split, num_splits)
                scope_lists_r = [scope_lists_r[:, i] for i in range(self.num_splits)]
            else:
                scope_lists_r = [
                    module.feature_to_scope[..., r] for module in self.inputs
                ]  # Shape: (num_features_per_split, num_splits)

            outer_product_r = list(product(*scope_lists_r))

            feature_to_scope_r = []
            for scopes_r in outer_product_r:
                feature_to_scope_r.append(Scope.join_all(scopes_r))
            out.append(np.array(feature_to_scope_r))
        return np.stack(out, axis=1)

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
                    .view(-1, self.inputs[0].out_shape.features)
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
                    .view(-1, self.inputs[0].out_shape.features)
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
                output = output.view(output.size(0), self.out_shape.features, -1)
            elif output.ndim == 5:
                output = output.view(output.size(0), self.out_shape.features, -1, self.out_shape.repetitions)
            else:
                raise ValueError("Invalid number of dimensions")

        # View as [b, n, m1 * m2, r]
        if self.out_shape.repetitions is None:
            output = output.view(output.size(0), self.out_shape.features, self.out_shape.channels)
        else:
            output = output.view(output.size(0), self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions)
        return output
